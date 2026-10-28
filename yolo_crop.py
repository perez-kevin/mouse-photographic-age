"""
Crop each photo to the mouse with the fine-tuned YOLOv5n detector (step 2 of the pipeline).

  python yolo_crop.py downscaled_photos yolo_crops --weights MouseAge2_YOLO_Best_050925.pt

Highest-scoring box, enlarged by --margin of its width/height on each side, clipped to the photo.
Walks sub-folders and mirrors them in the output; photos without a detection are listed at the end.
(Same settings as used for the paper; the detector was fine-tuned on 247 hand-labelled photos.)
"""
import argparse, os
from pathlib import Path
import torch
from PIL import Image


def load_yolo(weights):
    """YOLOv5 detector via torch.hub. PyTorch >= 2.6 loads checkpoints with weights_only=True by default, which
    rejects YOLOv5's pickled model class; the detector file is our own, so it is loaded with weights_only=False."""
    orig = torch.load
    torch.load = lambda *args, **kw: orig(*args, **{**kw, "weights_only": kw.get("weights_only") or False})
    try:
        return torch.hub.load("ultralytics/yolov5", "custom", path=weights, trust_repo=True)
    finally:
        torch.load = orig


def crop_mouse(img, model, margin=0.15):
    p = model(img).xyxy[0]
    if p.shape[0] == 0:
        return None
    x1, y1, x2, y2 = map(int, p[0][:4]); w, h = img.size
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return img.crop((max(x1 - dx, 0), max(y1 - dy, 0), min(x2 + dx, w), min(y2 + dy, h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_folder")
    ap.add_argument("output_folder")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "MouseAge2_YOLO_Best_050925.pt"))
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    model = load_yolo(a.weights)
    model.conf = a.conf
    src = Path(a.input_folder)
    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("."))
    missed = []
    for i, p in enumerate(imgs):
        c = crop_mouse(Image.open(p).convert("RGB"), model, a.margin)
        if c is None:
            missed.append(p); continue
        out = Path(a.output_folder) / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True); c.save(out)
        if i % 50 == 0:
            print(f"  {i + 1}/{len(imgs)}", flush=True)
    print(f"mouse found in {len(imgs) - len(missed)} of {len(imgs)} photos")
    for p in missed[:20]:
        print("  no detection:", p)


if __name__ == "__main__":
    main()
