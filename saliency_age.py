"""
Saliency maps for the age model (frozen DINOv2 + ridge): which image regions push the
predicted age UP (red, "looks older") or DOWN (blue, "looks younger").

Method: occlusion. A small window is slid over the image; at each position that region is
replaced and the change in predicted age (days) is recorded:
    heat = prediction(original) - prediction(region replaced)
so +50 at a pixel means "this region adds ~50 days to the prediction".

Two replacement modes, which answer different questions:
  --mode grey  region replaced by flat grey (= the crop background). "What happens if this part
               of the mouse is not there?" Also changes the outline, so body shape/size cues show up.
  --mode blur  region replaced by a blurred copy of itself. Outline and colour stay, fine texture
               goes. "Where does fine detail (fur, greying, eyes, skin) matter?"

The age model explaining an image is fitted WITHOUT that image's lab (same as --split source),
so maps show what the model uses on data it has never seen.

Usage (GPU strongly recommended, ~1-3 s per image on a T4):
  python saliency_age.py --root /path/to/crops --csv labels_id_sex_clean.csv --out saliency_crops
  python saliency_age.py --root /path/to/crops ... --mode blur --out saliency_crops_blur
  python saliency_age.py --root /path/to/originals ... --out saliency_raw      # background used?
  python saliency_age.py --root ... --files "Vivo_070725_Cropped/x.jpg,Miller_raw_d_c/y.jpg"

Outputs in --out:
  saliency_<lab>.jpg     per lab, images spread over the lab's age range: image | map
  saliency_worst.jpg     the images the model gets most wrong
  saliency_maps.npz      raw maps (days per pixel, 224x224) + saliency_index.csv
"""
import argparse, os, types
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from train_age import load_table, get_embeddings, tfms, amp_dtype, MEAN, STD, YEAR
import time


def linear_head(X, y):
    """Fit scaler+ridge, return it as a single linear map on raw embeddings (w, b), in years."""
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(X, y)
    sc, rd = reg[0], reg[1]
    w = rd.coef_ / sc.scale_
    b = rd.intercept_ - (sc.mean_ * w).sum()
    return w.astype(np.float32), float(b)


@torch.no_grad()
def occlusion_map(model, w, b, x, a, dev):
    """x: (3,S,S) normalized tensor. Returns (prediction in days, heat map SxS in days)."""
    S = x.shape[-1]
    mean = torch.tensor(MEAN).view(3, 1, 1); std = torch.tensor(STD).view(3, 1, 1)
    if a.mode == "grey":
        filler = ((torch.full((3, 1, 1), a.fill / 255) - mean) / std).expand_as(x)
    else:
        k = 2 * int(a.win * 0.6) + 1
        filler = TF.gaussian_blur(x[None], kernel_size=k, sigma=a.win / 3)[0]
    x, filler, wt = x.to(dev), filler.to(dev), torch.tensor(w, device=dev)

    dt = amp_dtype(dev) if a.amp else None

    def predict(batch, amp=True):
        with torch.autocast(dev, dtype=dt or torch.float32, enabled=amp and dt is not None):
            f = model(batch)
            if a.flip:
                f = (f + model(torch.flip(batch, dims=[3]))) / 2
        return (f.float() @ wt + b) * YEAR

    starts = list(range(0, S - a.win + 1, a.stride))
    if starts[-1] != S - a.win:
        starts.append(S - a.win)
    pos = [(i, j) for i in starts for j in starts]
    base_exact = predict(x[None], amp=False)[0].item()     # full precision: the reported prediction
    base = predict(x[None])[0].item()                       # same precision as the occluded copies
    heat = torch.zeros(S, S, device=dev); cnt = torch.zeros(S, S, device=dev)
    for c in range(0, len(pos), a.bs):
        chunk = pos[c:c + a.bs]
        xs = x[None].repeat(len(chunk), 1, 1, 1)
        for k, (i, j) in enumerate(chunk):
            xs[k, :, i:i + a.win, j:j + a.win] = filler[:, i:i + a.win, j:j + a.win]
        d = base - predict(xs)
        for k, (i, j) in enumerate(chunk):
            heat[i:i + a.win, j:j + a.win] += d[k]; cnt[i:i + a.win, j:j + a.win] += 1
    return base_exact, (heat / cnt.clamp(min=1)).cpu().numpy()


def to_uint8(x):
    mean = np.array(MEAN)[:, None, None]; std = np.array(STD)[:, None, None]
    return (np.clip(x.numpy() * std + mean, 0, 1) * 255).transpose(1, 2, 0).astype(np.uint8)


def overlay(img, heat, vmax):
    v = np.clip(heat / max(vmax, 1e-6), -1, 1)
    col = np.where(v[..., None] > 0, np.array([215, 35, 35]), np.array([35, 85, 215]))
    alpha = (np.abs(v) ** 0.8 * 0.7)[..., None]
    base = img.mean(2, keepdims=True).repeat(3, 2)          # greyscale base so colours read clearly
    return (base * (1 - alpha) + col * alpha).astype(np.uint8)


def sheet(items, path, title, cols=3):
    """items: list of (img, heat, label). Each item = image tile + map tile."""
    if not items:
        return
    S = items[0][0].shape[0]
    rows = (len(items) + cols - 1) // cols
    W, H = cols * (2 * S + 12), rows * (S + 30) + 28
    out = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(out)
    d.text((6, 6), title + "   red = pushes predicted age UP, blue = DOWN (scale per image)", fill="black")
    for n, (img, heat, label) in enumerate(items):
        r, c = divmod(n, cols)
        x0, y0 = c * (2 * S + 12), 28 + r * (S + 30)
        vmax = np.percentile(np.abs(heat), 99)
        out.paste(Image.fromarray(img), (x0, y0))
        out.paste(Image.fromarray(overlay(img, heat, vmax)), (x0 + S, y0))
        d.text((x0 + 2, y0 + S + 2), f"{label}  scale ±{vmax:.0f}d", fill="black")
    out.save(path, quality=92)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="labels_id_sex_clean.csv")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="saliency")
    ap.add_argument("--mode", choices=["grey", "blur"], default="grey")
    ap.add_argument("--per-lab", type=int, default=6, help="images per lab, spread over its age range")
    ap.add_argument("--worst", type=int, default=12, help="also map the N largest errors")
    ap.add_argument("--files", default="", help="comma-separated relative paths to map instead")
    ap.add_argument("--win", type=int, default=0, help="occlusion window in px (default: 1/8 of --size)")
    ap.add_argument("--stride", type=int, default=0, help="step in px (default: half the window)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="full precision for the occlusion runs")
    ap.add_argument("--fill", type=int, default=128, help="grey level for --mode grey (crop background)")
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.win = a.win or max(8, round(a.size / 8))             # same relative window at any resolution
    a.stride = a.stride or max(4, a.win // 2)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    df = load_table(a.csv, a.root).reset_index(drop=True)
    torch.manual_seed(0)                      # only matters with --no-pretrained (testing)
    X = get_embeddings(df, a, dev)
    y = df.age_days.values / YEAR

    # one age head per lab, fitted on the OTHER labs; held-out predictions for everyone
    heads, pred = {}, np.zeros(len(df))
    for lab in df.source.unique():
        tr, te = (df.source != lab).values, (df.source == lab).values
        heads[lab] = linear_head(X[tr], y[tr])
        w, b = heads[lab]
        pred[te] = (X[te] @ w + b) * YEAR
    df["pred_days"] = pred
    print(f"held-out MAE {np.abs(pred - df.age_days).mean():.0f} days (should match train_age --split source)")

    # choose images
    groups = {}
    if a.files:
        want = [f.strip() for f in a.files.split(",") if f.strip()]
        groups["selected"] = df.index[df.file.isin(want)].tolist()
    else:
        rng = np.random.default_rng(a.seed)
        for lab, g in df.groupby("source"):
            g = g.sort_values("age_days")
            q = np.linspace(0, len(g) - 1, min(a.per_lab, len(g))).round().astype(int)
            groups[lab] = g.index[q].tolist()
        if a.worst:
            groups["worst"] = (df.pred_days - df.age_days).abs().sort_values(ascending=False).index[:a.worst].tolist()

    torch.manual_seed(0)
    model = timm.create_model(a.backbone, pretrained=not a.no_pretrained, num_classes=0,
                              img_size=a.size).to(dev).eval()
    tf = tfms(a.size, False)
    maps, index = {}, []
    n_total, n_done, t_start = sum(len(v) for v in groups.values()), 0, time.time()
    n_win = len(range(0, a.size - a.win + 1, a.stride)) ** 2
    print(f"{n_total} images; occlusion window {a.win}px, step {a.stride}px (~{n_win} positions per image), "
          f"device={dev}, mixed precision={'on' if a.amp and amp_dtype(dev) else 'off'}", flush=True)
    for name, idx in groups.items():
        items = []
        for i in idx:
            r = df.loc[i]
            x = tf(Image.open(r.path).convert("RGB"))
            w, b = heads[r.source]
            p, heat = occlusion_map(model, w, b, x, a, dev)
            if a.flip and abs(p - r.pred_days) > 5:   # maps must explain the same prediction as the evaluation
                raise SystemExit(f"prediction mismatch for {r.file}: {p:.0f} vs {r.pred_days:.0f} days. "
                                 "Were the cached embeddings made with a different model/size? Delete emb_*.npy and rerun.")
            key = f"img{i}"
            maps[key] = heat.astype(np.float16)
            index.append(dict(key=key, group=name, file=r.file, source=r.source, folder=r.folder,
                              age_days=r.age_days, pred_days=round(p), heat_sum_pos=float(heat[heat > 0].sum()),
                              heat_sum_neg=float(heat[heat < 0].sum())))
            items.append((to_uint8(x), heat, f"{r.source} true {r.age_days:.0f}d pred {p:.0f}d"))
            n_done += 1
            el = time.time() - t_start
            print(f"  {name}: {r.file}  true {r.age_days:.0f}  pred {p:.0f}   "
                  f"[{n_done}/{n_total}, {el / n_done:.1f}s/img, ~{el / n_done * (n_total - n_done) / 60:.0f} min left]",
                  flush=True)
        sheet(items, os.path.join(a.out, f"saliency_{name}.jpg"), f"{name} - occlusion ({a.mode})")
    np.savez_compressed(os.path.join(a.out, "saliency_maps.npz"), **maps)
    pd.DataFrame(index).to_csv(os.path.join(a.out, "saliency_index.csv"), index=False)
    print(f"done: {a.out}/saliency_*.jpg")
