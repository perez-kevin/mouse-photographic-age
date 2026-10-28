"""
Mask out the background and crop each image to just the mouse.

  1. Grounding DINO finds the mouse from the text prompt "a mouse."  -> bounding box
  2. SAM traces the outline inside that box                          -> pixel mask
  3. Mask is cleaned (largest region, holes filled, background-coloured rim removed, edge softened inward)
  4. Background is replaced by one flat grey for every image, then cropped and padded to square

Output mirrors the input tree (out/<folder>/<image>), so train_age.py runs on it with --root out.
Also writes out/crop_qc.csv (scores + flags per image) and out/_qc/<folder>.jpg contact sheets
(flagged images get a red border) so you can eyeball the results.

Quick test on a few images per folder:
  pip install torch transformers scipy pandas pillow
  python crop_mice.py --csv labels_id_sex_clean.csv --root /path/to/images --out crops_test --per-folder 8
Full run:
  python crop_mice.py --csv labels_id_sex_clean.csv --root /path/to/images --out crops
"""
import argparse, os, time, traceback
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from scipy import ndimage
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor)


class Cropper:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        self.dp = AutoProcessor.from_pretrained(a.detector)
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained(a.detector).to(dev).eval()
        self.sp = SamProcessor.from_pretrained(a.sam)
        self.sm = SamModel.from_pretrained(a.sam).to(dev).eval()

    @torch.no_grad()
    def detect(self, im):
        """Returns list of (score, [x0,y0,x1,y1]) sorted by score."""
        inp = self.dp(images=im, text=self.a.prompt, return_tensors="pt").to(self.dev)
        out = self.dm(**inp)
        pp, kw = self.dp.post_process_grounded_object_detection, dict(text_threshold=self.a.box_thr,
                                                                      target_sizes=[im.size[::-1]])
        try:                                     # transformers >= 4.51 calls it `threshold`...
            r = pp(out, inp.input_ids, threshold=self.a.box_thr, **kw)[0]
        except TypeError:                        # ...older versions call it `box_threshold`
            r = pp(out, inp.input_ids, box_threshold=self.a.box_thr, **kw)[0]
        dets = [(float(s), [float(v) for v in b]) for s, b in zip(r["scores"].cpu(), r["boxes"].cpu())]
        return sorted(dets, key=lambda d: -d[0])

    @torch.no_grad()
    def segment(self, im, box):
        """Returns (bool mask HxW, SAM's predicted IoU for it). --pick iou: SAM's best predicted IoU;
        --pick fill (sam_pick): the candidate whose area best fills the box, ties -> IoU."""
        inp = self.sp(images=im, input_boxes=[[box]], return_tensors="pt")
        # Apple MPS has no float64; SAM's processor returns the boxes as float64 -> cast to float32
        inp = {k: (v.float() if v.dtype == torch.float64 else v).to(self.dev) for k, v in inp.items()}
        out = self.sm(**inp, multimask_output=True)
        masks = self.sp.image_processor.post_process_masks(
            out.pred_masks.float().cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0]
        iou = out.iou_scores.float().cpu()[0, 0]
        if self.a.pick == "fill":                     # a mouse fills ~45-75 % of its box
            bw, bh = box[2] - box[0], box[3] - box[1]
            fill = np.array([float(m.sum()) / max(1.0, bw * bh) for m in masks])
            k = int(np.argmax(-np.abs(fill - 0.6) + 0.3 * iou.numpy()))
        else:
            k = int(iou.argmax())
        return masks[k].numpy().astype(bool), float(iou[k])


def largest_filled(m):
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)    # largest connected region
    return ndimage.binary_fill_holes(m)


def strip_background_rim(im, m, band_frac, tol):
    """Remove pixels in a thin band along the mask edge whose colour matches the background
    just outside the mask (teal cloth, grey bench...). Only the band is touched, so a white
    mouse on a white bench loses at most `band` pixels of edge, never its body."""
    size = np.sqrt(m.sum())
    band = max(2, int(round(band_frac * size)))
    ring = ndimage.binary_dilation(m, iterations=band + 2) & ~ndimage.binary_dilation(m, iterations=2)
    edge = m & ~ndimage.binary_erosion(m, iterations=band)
    if ring.sum() < 20 or edge.sum() == 0:
        return m, 0.0
    lab = np.asarray(im.convert("LAB"), dtype=np.float32)
    bg = np.median(lab[ring], axis=0)                                   # local background colour
    dist = np.sqrt(((lab - bg) ** 2).sum(-1))
    # also drop edge pixels tinted green/teal/blue/purple (cloth, gloves): fur never has these hues,
    # and fur/cloth mixed pixels at the boundary are not close enough to the pure background colour
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)              # PIL: H, S, V all 0..255
    tinted = (hsv[..., 0] >= 50) & (hsv[..., 0] <= 214) & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 30)
    drop = edge & ((dist < tol) | tinted)
    return largest_filled(m & ~drop), float(drop.sum() / m.sum())


def clean_mask(im, m, a):
    m = largest_filled(m)
    if m.sum() == 0:
        return m, 0.0
    r = int(round(a.dilate * np.sqrt(m.sum())))                         # >0 grows, <0 shrinks the mask
    if r > 0:
        m = ndimage.binary_dilation(m, iterations=r)
    elif r < 0:
        m = largest_filled(ndimage.binary_erosion(m, iterations=-r))
    if a.rim_tol > 0:
        return strip_background_rim(im, m, a.rim_band, a.rim_tol)
    return m, 0.0


def compose(im, m, a):
    """Flat-grey background, edge softened INWARD only (no background pixels blended in),
    crop with margin, pad to square, resize."""
    if a.feather > 0:
        inner = ndimage.binary_erosion(m, iterations=int(np.ceil(a.feather)))
        alpha = ndimage.gaussian_filter(inner.astype(np.float32), a.feather / 2) * m
        alpha = np.clip(alpha / max(alpha.max(), 1e-6), 0, 1)
    else:
        alpha = m.astype(np.float32)
    alpha = Image.fromarray((alpha * 255).astype(np.uint8))
    bg = Image.new("RGB", im.size, (a.fill,) * 3)
    out = Image.composite(im, bg, alpha)
    ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(a.margin * max(x1 - x0, y1 - y0))
    out = out.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))
    return ImageOps.pad(out, (a.size, a.size), color=(a.fill,) * 3)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def process(c, path, a):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")       # phone photos: respect rotation
    if a.max_side and max(im.size) > a.max_side:                        # outputs are small; full-res only slows
        im.thumbnail((a.max_side, a.max_side), Image.LANCZOS)            # down the CPU mask clean-up
    W, H = im.size
    dets = c.detect(im)
    row = {"det_score": dets[0][0] if dets else 0.0, "n_boxes": len(dets), "flags": []}
    if not dets:
        row["flags"].append("no_detection")
        box = [0, 0, W, H]                                              # fall back to whole image for SAM
    else:
        box = dets[0][1]
        if dets[0][0] < a.low_score:
            row["flags"].append("low_score")
        # a second confident, separate box -> possibly two mice (or a hand detected as one)
        if any(s > a.low_score and box_iou(box, b) < 0.3 for s, b in dets[1:]):
            row["flags"].append("multiple_boxes")
    m, iou = c.segment(im, box)
    m, row["rim_removed_frac"] = clean_mask(im, m, a)
    bw, bh = box[2] - box[0], box[3] - box[1]
    row.update(sam_iou=iou, mask_frac_image=m.mean(), mask_frac_box=m.sum() / max(1.0, bw * bh))
    if m.sum() == 0:
        row["flags"].append("empty_mask"); return None, row
    if iou < 0.85: row["flags"].append("low_sam_iou")
    # many source images are already tightly cropped, so a nose or tail at the edge is normal;
    # only flag when the mouse runs along a large part of an edge (body likely cut off)
    edge = max(m[0].mean(), m[-1].mean(), m[:, 0].mean(), m[:, -1].mean())
    row["edge_frac"] = edge
    if edge > 0.25: row["flags"].append("cut_off_at_edge")
    if not 0.25 < row["mask_frac_box"] < 0.95: row["flags"].append("odd_mask_vs_box")
    return compose(im, m, a), row


def contact_sheet(items, path, thumb=160, cols=8):
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (img, label, flagged) in enumerate(items):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 14)
        sheet.paste(img.resize((thumb, thumb)), (x, y))
        if flagged:
            d.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline="red", width=4)
        d.text((x + 2, y + thumb), label[:26], fill="red" if flagged else "black")
    sheet.save(path, quality=85)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="labels_id_sex_clean.csv")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-folder", type=int, default=0, help="only process the first N images per folder (testing)")
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam", default="facebook/sam-vit-huge", help="facebook/sam-vit-base is ~5x faster")
    ap.add_argument("--prompt", default="a mouse.")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--pick", choices=["iou", "fill"], default="iou",
                    help="which of SAM's 3 masks: best predicted IoU, or best box fill (sam_pick: keeps more paws)")
    ap.add_argument("--low-score", type=float, default=0.40, help="flag detections below this score")
    ap.add_argument("--dilate", type=float, default=0.0,
                    help="grow (>0) or shrink (<0) the mask, fraction of mouse size; e.g. -0.005 trims a remaining halo")
    ap.add_argument("--rim-band", type=float, default=0.02,
                    help="width of the edge band checked for background-coloured pixels, fraction of mouse size")
    ap.add_argument("--rim-tol", type=float, default=12.0,
                    help="colour distance (Lab) under which an edge pixel counts as background; 0 = off")
    ap.add_argument("--feather", type=float, default=1.5, help="inward softening of the mask edge (px)")
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--fill", type=int, default=128, help="background grey level")
    ap.add_argument("--size", type=int, default=512, help="output side length (px)")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="shrink input photos to this longest side before processing (0 = keep full size)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--debug", action="store_true", help="stop at the first error")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(a.csv, encoding="utf-8-sig")
    if a.per_folder:
        df = df.groupby("folder").head(a.per_folder)
    import sys, transformers
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.40"):
        sys.exit(f"transformers {transformers.__version__} is too old for Grounding DINO (needs >= 4.40).\n"
                 f"This run uses the Python at {sys.executable}.\n"
                 "Run with the Python environment you used before, or: pip install -U transformers")
    print(f"{len(df)} images, device={dev}, torch {torch.__version__}, transformers {transformers.__version__}. "
          "Loading models...", flush=True)
    c = Cropper(a, dev)
    os.makedirs(os.path.join(a.out, "_qc"), exist_ok=True)

    rows, sheets, t0 = [], {}, time.time()
    for i, (f, folder) in enumerate(zip(df["file"], df["folder"])):
        src, dst = os.path.join(a.root, f), os.path.join(a.out, f)
        if not os.path.exists(src):
            rows.append({"file": f, "flags": "missing_file"}); continue
        if a.skip_existing and os.path.exists(dst):
            continue
        try:
            img, row = process(c, src, a)
        except Exception as e:                                           # keep going; report at the end
            n_err = sum("error" in str(r.get("flags", "")) for r in rows)
            if n_err < 2 or a.debug:                                     # show the full traceback for the first errors
                print(f"\nERROR on {f}:"); traceback.print_exc()
            if a.debug:
                raise
            img, row = None, {"flags": [f"error:{type(e).__name__}: {str(e)[:120]}"]}
        row["file"], row["flags"] = f, ";".join(row["flags"])
        rows.append(row)
        if img is not None:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img.save(dst, quality=95)
            sheets.setdefault(folder, []).append((img, os.path.basename(f), bool(row["flags"])))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df)}  ({(time.time() - t0) / (i + 1):.2f}s/img)", flush=True)

    qc = pd.DataFrame(rows)
    qc.to_csv(os.path.join(a.out, "crop_qc.csv"), index=False)
    for folder, items in sheets.items():                                # cap sheet size; flagged ones first
        items = sorted(items, key=lambda x: not x[2])[:64]
        contact_sheet(items, os.path.join(a.out, "_qc", f"{folder}.jpg"))

    flagged = qc["flags"].fillna("").ne("")
    print(f"\nDone: {len(qc)} images, {flagged.sum()} flagged ({flagged.mean():.0%}).")
    if flagged.any():
        print(qc.loc[flagged, "flags"].str.split(";").explode().value_counts().to_string())
    print(f"QC table: {a.out}/crop_qc.csv   contact sheets: {a.out}/_qc/")
