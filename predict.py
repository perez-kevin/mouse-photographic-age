"""
Photographic age of laboratory mice from your own photos, with the model from the paper.

  python predict.py PHOTO_FOLDER --out predictions.csv
  python predict.py PHOTO_FOLDER --out predictions.csv --yolo MouseAge2_YOLO_Best_050925.pt \
      --model mouse_age_ridge448.joblib --work crops_work

One mouse per photo. Sub-folders are treated as photo SESSIONS (same day, same setup): put each session's
photos in its own folder. For every photo: downscale (longest side 2048 px, rotation tag applied, sRGB)
-> YOLO crop around the mouse -> Grounding DINO + SAM mask, flat grey background -> frozen DINOv2
ViT-B/14 embedding at 448 px (horizontal-flip TTA) -> ridge regression trained on 7 facilities.

USE IT FOR RANKING, NOT FOR ABSOLUTE AGE. In facilities the model had never seen, absolute predictions
were off by several months on average (each facility's photos shift all predictions), and predictions are
compressed towards the middle of the age range. Within a session, the older of two mice was predicted older
in ~65% of pairs (0.5 = chance), more often the larger their age difference. Compare mice photographed
together; treat `pred_age_days` as a score, not as an age.

Output columns: file, session, pred_age_days, rank_in_session (1 = looks youngest in its session),
n_in_session, crop_flags (from the masking step: check those crops in --work/masked/).
GPU or Apple MPS recommended; first run downloads YOLOv5 code, Grounding DINO, SAM ViT-H and DINOv2 (~3 GB).
"""
import argparse, os, sys, types
import numpy as np, pandas as pd, torch, timm, joblib
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from downscale_raw import to_srgb                      # noqa: E402
from train_age import tfms, YEAR                       # noqa: E402
import crop_mice                                       # noqa: E402
from yolo_crop import load_yolo                        # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic")
CAVEAT = ("NOTE: use these predictions to RANK mice photographed in the same session, not as absolute ages "
          "(see predict.py header).")


def find_photos(root):
    out = []
    for d, _, fs in os.walk(root):
        out += [os.path.relpath(os.path.join(d, f), root) for f in sorted(fs)
                if f.lower().endswith(EXT) and not f.startswith(".")]
    return sorted(out)


def yolo_crop(model, im, margin=0.15):
    """Highest-scoring box, enlarged by `margin` of its size on each side (as for the training data)."""
    p = model(im).xyxy[0]
    if p.shape[0] == 0:
        return None
    x1, y1, x2, y2 = map(int, p[0][:4]); w, h = im.size
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return im.crop((max(x1 - dx, 0), max(y1 - dy, 0), min(x2 + dx, w), min(y2 + dy, h)))


def bundled(name):
    """Model file shipped with the code: release/<name> in the repository (or next to this script)."""
    p = os.path.join(HERE, "release", name)
    return p if os.path.exists(p) else os.path.join(HERE, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("photos", help="folder of photos; sub-folders = sessions")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--yolo", default=bundled("MouseAge2_YOLO_Best_050925.pt"))
    ap.add_argument("--model", default=bundled("mouse_age_ridge448.joblib"))
    ap.add_argument("--work", default="predict_work", help="intermediate crops (YOLO and masked) for checking")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    M = joblib.load(a.model)
    files = find_photos(a.photos)
    print(f"{len(files)} photos, device={dev}; model trained on {M['trained_on']['mice']} mice from "
          f"{len(M['trained_on']['facilities'])} facilities\n{CAVEAT}", flush=True)

    # 1-3: downscale, YOLO crop, mask (same settings as the paper's crop_mice.py defaults, --pick iou)
    yolo = load_yolo(a.yolo); yolo.conf = 0.25
    ca = types.SimpleNamespace(detector="IDEA-Research/grounding-dino-base", sam="facebook/sam-vit-huge",
                               prompt="a mouse.", box_thr=0.25, pick="iou", low_score=0.40, dilate=0.0,
                               rim_band=0.02, rim_tol=12.0, feather=1.5, margin=0.08, fill=128, size=512, max_side=1600)
    cropper = crop_mice.Cropper(ca, dev)
    rows = []
    for i, f in enumerate(files):
        im = to_srgb(ImageOps.exif_transpose(Image.open(os.path.join(a.photos, f))))
        im.thumbnail((2048, 2048), Image.LANCZOS)
        y = yolo_crop(yolo, im)
        row = dict(file=f, session=os.path.dirname(f) or ".", crop_flags="")
        if y is None:
            row["crop_flags"] = "no_mouse_found"; rows.append(row); continue
        yp = os.path.join(a.work, "yolo", os.path.splitext(f)[0] + ".jpg"); os.makedirs(os.path.dirname(yp), exist_ok=True)
        y.save(yp, quality=95)
        masked, qc = crop_mice.process(cropper, yp, ca)
        row["crop_flags"] = ";".join(qc["flags"])
        if masked is None:
            rows.append(row); continue
        mp = os.path.join(a.work, "masked", os.path.splitext(f)[0] + ".jpg"); os.makedirs(os.path.dirname(mp), exist_ok=True)
        masked.save(mp, quality=95); row["masked"] = mp; rows.append(row)
        print(f"  cropped {i + 1}/{len(files)} {f}{'  [' + row['crop_flags'] + ']' if row['crop_flags'] else ''}", flush=True)
    del cropper, yolo
    df = pd.DataFrame(rows)
    ok = df.get("masked", pd.Series(dtype=str)).notna()

    # 4-5: embedding + ridge
    net = timm.create_model(M["backbone"], pretrained=True, num_classes=0, img_size=M["size"]).to(dev).eval()
    tf = tfms(M["size"], False); feats = []
    paths = df.masked[ok].tolist()
    with torch.no_grad():
        for s in range(0, len(paths), a.bs):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s + a.bs]]).to(dev)
            feats.append(((net(x) + net(torch.flip(x, dims=[3]))) / 2).float().cpu().numpy())
            print(f"  embedded {min(s + a.bs, len(paths))}/{len(paths)}", flush=True)
    df.loc[ok, "pred_age_days"] = M["model"].predict(np.concatenate(feats)) * YEAR if paths else []
    df["rank_in_session"] = df.groupby("session").pred_age_days.rank(method="average")
    df["n_in_session"] = df.groupby("session").pred_age_days.transform("count")
    df.drop(columns=["masked"], errors="ignore").round(1).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {ok.sum()} of {len(df)} photos predicted"
          + (f"; {(~ok).sum()} without a usable crop (see crop_flags)" if (~ok).any() else "") + f"\n{CAVEAT}")


if __name__ == "__main__":
    main()
