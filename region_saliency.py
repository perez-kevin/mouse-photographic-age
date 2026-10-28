"""
Quantify saliency maps by body region: head, mid-body, rear, tail, limbs.

The regions are found geometrically from each crop's mask (no extra model):
  * mask   = pixels that differ from the flat grey background of the crops
  * a morphological opening removes thin parts -> body core; the thin leftovers are tail + limbs,
    the tail being the largest, most elongated thin piece
  * the body axis is the main direction of the core (PCA); the end where the tail attaches is the rear,
    the opposite end the head. Head = front 25 % of body length, rear = back 25 %, mid = the rest.
  * if no tail is visible (top views, tail tucked under), --detect-head finds the head directly with
    Grounding DINO (prompt "a mouse head.", same detector as crop_mice.py) and orients the axis by it.
    Mice with BOTH a tail and a detected head give a reliability check: how often the two agree.
  * orientation status per image:  tail | tail+agree | tail+disagree | head-detector | unknown
    Head/rear statistics use tail, tail+agree and head-detector; tail+disagree and unknown are left out.

Heat maps come from saliency_age.py (saliency_maps.npz + saliency_index.csv), in the same frame as the
model input, so the crop is passed through the same padding/resizing before segmenting.

  python region_saliency.py --sal grey=saliency_crops blur=saliency_crops_blur --root /path/to/crops \
                            --csv labels.csv --size 448 --out saliency_regions [--exclude bad_files.txt] \
                            [--detect-head]          # GPU recommended; downloads grounding-dino-base once

Outputs (--out):
  regions_per_image.csv   mean heat (days per pixel) in each region, per image and saliency mode
  regions_summary.csv     mean over images with 95 % bootstrap interval, overall and by age group / coat colour
  regions_qc_<mode>.jpg   segmentation overlays (check these before trusting the numbers)
Heat sign: + = region pushes predicted age UP (looks older), - = pushes it DOWN.
"""
import argparse, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from train_age import tfms, MEAN, STD

ORIENTED = ["tail", "tail+agree", "head-detector"]


class HeadDetector:
    """Grounding DINO, prompt 'a mouse head.'; returns the centre (x, y) of the best head box or None."""

    def __init__(self, name="IDEA-Research/grounding-dino-base", thr=0.25):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.p = AutoProcessor.from_pretrained(name)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(self.dev).eval()
        self.thr = thr
        print(f"head detector {name} on {self.dev}")

    def __call__(self, img, mask):
        im = Image.fromarray(img)
        with self.torch.no_grad():
            inp = self.p(images=im, text="a mouse head.", return_tensors="pt").to(self.dev)
            out = self.m(**inp)
        pp, kw = self.p.post_process_grounded_object_detection, dict(text_threshold=self.thr,
                                                                     target_sizes=[im.size[::-1]])
        try:
            r = pp(out, inp.input_ids, threshold=self.thr, **kw)[0]
        except TypeError:
            r = pp(out, inp.input_ids, box_threshold=self.thr, **kw)[0]
        ys, xs = np.nonzero(mask)
        mouse_box = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        for sc, b in sorted(zip(r["scores"].cpu().tolist(), r["boxes"].cpu().tolist()), key=lambda z: -z[0]):
            x0, y0, x1, y1 = b
            if (x1 - x0) * (y1 - y0) > 0.6 * mouse_box:          # box around the whole mouse, not the head
                continue
            cx, cy = int(np.clip((x0 + x1) / 2, 0, img.shape[1] - 1)), int(np.clip((y0 + y1) / 2, 0, img.shape[0] - 1))
            if mask[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1].mean() < 0.15:
                continue                                          # box mostly off the mouse (clutter)
            return (cx, cy)
        return None


REGIONS = ["head", "mid", "rear", "tail", "limbs"]
COLORS = {"head": (230, 60, 60), "mid": (60, 120, 230), "rear": (240, 180, 40), "tail": (60, 200, 90),
          "limbs": (190, 80, 220), "core": (120, 120, 120)}


def to_uint8(x):
    m = np.array(MEAN)[:, None, None]; s = np.array(STD)[:, None, None]
    return (np.clip(x.numpy() * s + m, 0, 1) * 255).transpose(1, 2, 0).astype(np.uint8)


def largest(m):
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    return lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)


def segment(img, fill=128, tol=14, detector=None):
    """img: HxWx3 uint8 in the model frame. Returns (dict region -> bool mask, orientation status, head point)."""
    m = (np.abs(img.astype(int) - fill).max(-1) > tol)
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_fill_holes(largest(m))
    area = m.sum()
    if area < 200:
        return None, "no mouse", None
    r = max(2, int(round(0.07 * np.sqrt(area))))                     # removes structures thinner than ~2r
    st = ndimage.generate_binary_structure(2, 1)
    core = largest(ndimage.binary_opening(m, structure=st, iterations=r))
    core = ndimage.binary_fill_holes(core)
    thin = m & ~ndimage.binary_dilation(core, iterations=1)
    lab, n = ndimage.label(thin)
    regions = {k: np.zeros_like(m) for k in REGIONS}
    tail_id, best = None, 0
    for i in range(1, n + 1):
        comp = lab == i
        a = comp.sum()
        if a < 0.004 * area:
            continue
        ys, xs = np.nonzero(comp)
        cov = np.cov(np.vstack([xs, ys])) if a > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = np.sqrt(ev[0] / max(ev[1], 1e-6))                       # length / width
        score = a * elong
        if elong > 3 and score > best:
            tail_id, best = i, score
    for i in range(1, n + 1):
        comp = lab == i
        if i == tail_id:
            regions["tail"] = comp
        elif comp.sum() >= 0.002 * area:
            regions["limbs"] |= comp
    # body axis
    ys, xs = np.nonzero(core)
    pts = np.vstack([xs, ys]).astype(float)
    c = pts.mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.cov(pts))
    axis = v[:, np.argmax(w)]
    t = axis @ (pts - c)
    length = t.max() - t.min()
    sign_tail = sign_head = None                                      # +1: head is the +t end
    if tail_id is not None:
        # tail attachment: tail pixels closest to the core
        dist = ndimage.distance_transform_edt(~core)
        ty, tx = np.nonzero(regions["tail"])
        k = np.argsort(dist[ty, tx])[:max(3, len(ty) // 20)]
        t_tail = axis @ (np.vstack([tx[k], ty[k]]).astype(float).mean(1) - c[:, 0])
        sign_tail = -1 if t_tail > 0 else 1
    hp = detector(img, m) if detector is not None else None
    if hp is not None:
        t_head = axis @ (np.array(hp, float) - c[:, 0])
        if abs(t_head) > 0.1 * length:                                # head clearly at one end
            sign_head = 1 if t_head > 0 else -1
    if sign_tail is not None:
        status = "tail" if sign_head is None else ("tail+agree" if sign_head == sign_tail else "tail+disagree")
        sign = sign_tail
    elif sign_head is not None:
        status, sign = "head-detector", sign_head
    else:
        regions["core"] = core
        return regions, "unknown", hp
    t = t * sign
    lo, hi = t.min(), t.max(); L = hi - lo
    head = t >= hi - 0.25 * L
    rear = t <= lo + 0.25 * L
    for name, sel in [("head", head), ("rear", rear), ("mid", ~head & ~rear)]:
        mm = np.zeros_like(m); mm[ys[sel], xs[sel]] = True
        regions[name] = mm
    return regions, status, hp


def overlay(img, regions, label, hp=None):
    out = img.astype(float).mean(2, keepdims=True).repeat(3, 2) * 0.6 + 50
    for k, msk in regions.items():
        if msk.any():
            out[msk] = 0.45 * out[msk] + 0.55 * np.array(COLORS[k])
    im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    if hp is not None:                                                # detected head: white ring
        rr = max(4, img.shape[0] // 40)
        d.ellipse([hp[0] - rr, hp[1] - rr, hp[0] + rr, hp[1] + rr], outline=(255, 255, 255), width=3)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((4, 2), label, fill=(255, 255, 255))
    return im


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sal", nargs="+", required=True, help="name=folder written by saliency_age.py")
    ap.add_argument("--root", required=True, help="crops folder used for the saliency run")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--out", default="saliency_regions")
    ap.add_argument("--exclude", help="text file with one relative image path per line to leave out (bad crops)")
    ap.add_argument("--fill", type=int, default=128)
    ap.add_argument("--detect-head", action="store_true", help="orient tail-less mice with Grounding DINO head boxes")
    ap.add_argument("--detector", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--head-thr", type=float, default=0.25)
    ap.add_argument("--skip-sources", default="PaperImages", help="comma-separated sources to leave out")
    ap.add_argument("--qc-n", type=int, default=48, help="images per QC sheet")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = pd.read_csv(a.csv, encoding="utf-8-sig")
    excl = set(open(a.exclude).read().split()) if a.exclude else set()
    skip = {x.strip() for x in a.skip_sources.split(",") if x.strip()}
    det = HeadDetector(a.detector, a.head_thr) if a.detect_head else None
    tf = tfms(a.size, False)
    rows, seg_cache = [], {}

    for spec in a.sal:
        mode, folder = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        idx = pd.read_csv(os.path.join(folder, "saliency_index.csv")).drop_duplicates("file")
        maps = np.load(os.path.join(folder, "saliency_maps.npz"))
        qc = []
        for r in idx.itertuples():
            if r.file in excl or r.source in skip:
                continue
            if r.file not in seg_cache:
                img = to_uint8(tf(Image.open(os.path.join(a.root, r.file)).convert("RGB")))
                seg_cache[r.file] = (img,) + segment(img, a.fill, detector=det)
            img, regions, status, hp = seg_cache[r.file]
            if regions is None:
                continue
            heat = maps[r.key].astype(np.float32)
            if heat.shape != img.shape[:2]:
                raise SystemExit(f"map size {heat.shape} != image {img.shape[:2]}: use the --size of the saliency run")
            row = dict(mode=mode, file=r.file, source=r.source, age_days=r.age_days, pred_days=r.pred_days,
                       orientation=status, heat_mouse=heat[np.any([*regions.values()], axis=0)].mean())
            for k in REGIONS + ["core"]:
                if k in regions and regions[k].any():
                    row[f"{k}_mean"] = heat[regions[k]].mean()
                    row[f"{k}_px"] = int(regions[k].sum())
            rows.append(row)
            if len(qc) < a.qc_n:
                qc.append(overlay(img, regions, f"{r.source} {r.age_days:.0f}d {status}", hp))
        if qc:
            s = 200; cols = 8; rws = (len(qc) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * s, rws * s), "white")
            for i, im in enumerate(qc):
                sheet.paste(im.resize((s, s)), ((i % cols) * s, (i // cols) * s))
            sheet.save(os.path.join(a.out, f"regions_qc_{mode}.jpg"), quality=90)

    D = pd.DataFrame(rows)
    if D.empty:
        raise SystemExit("no images analysed")
    strain = L.drop_duplicates("file").set_index("file").strain
    D["coat"] = np.where(D.file.map(strain).astype(str).str.contains("BALB"), "white (BALB/c)", "pigmented")
    D["age_group"] = pd.cut(D.age_days, [0, 180, 540, 2000], labels=["<6 mo", "6-18 mo", ">18 mo"]).astype(str)
    D.to_csv(os.path.join(a.out, "regions_per_image.csv"), index=False)

    out = []
    for mode, g in D.groupby("mode"):
        oriented = g[g.orientation.isin(ORIENTED)]
        print(f"\n=== {mode}: {len(g)} images, {len(oriented)} with head/rear orientation ===")
        print(g.orientation.value_counts().to_string())
        both = g.orientation.isin(["tail+agree", "tail+disagree"])
        if both.any():
            print(f"tail and head detector agree on the head end in {(g.orientation[both] == 'tail+agree').mean():.0%} "
                  f"of {both.sum()} mice with both (reliability of head-detector-only orientation)")
        print("oriented, by lab:\n" + g.assign(o=g.orientation.isin(ORIENTED)).groupby("source").o
              .agg(["sum", "size"]).rename(columns={"sum": "oriented", "size": "images"}).to_string())
        for by in [None, "age_group", "coat"]:
            groups = [("all", oriented)] if by is None else list(oriented.groupby(by))
            for gname, x in groups:
                for k in REGIONS:
                    col = f"{k}_mean"
                    if col not in x or x[col].notna().sum() < 3:
                        continue
                    lo, hi = boot_ci(x[col])
                    out.append(dict(mode=mode, split=by or "all", group=gname, region=k, n=int(x[col].notna().sum()),
                                    mean_days_per_px=x[col].mean(), ci_low=lo, ci_high=hi))
                if "head_mean" in x and "mid_mean" in x:
                    dlt = (x.head_mean - x.mid_mean).dropna()
                    lo, hi = boot_ci(dlt)
                    out.append(dict(mode=mode, split=by or "all", group=gname, region="head - mid",
                                    n=len(dlt), mean_days_per_px=dlt.mean(), ci_low=lo, ci_high=hi))
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(a.out, "regions_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    for mode, s in S.groupby("mode"):
        print(f"\n--- {mode}: mean heat per pixel (days; + = makes the mouse look older) ---")
        t = s[s.split.isin(["all", "age_group"])].pivot_table(index="region", columns="group",
                                                               values="mean_days_per_px").round(2)
        print(t.reindex([r for r in REGIONS + ["head - mid"] if r in t.index]).to_string())
    print(f"\ntables and QC sheets in {a.out}/ — check regions_qc_*.jpg first")


if __name__ == "__main__":
    main()
