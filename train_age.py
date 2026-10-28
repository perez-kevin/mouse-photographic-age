"""
Predict lab-mouse age (days) from photos.

Two stages, both evaluated with splits that never put the same mouse in train and test:

  embed     Frozen pretrained backbone (DINOv2 by default) -> image embeddings -> ridge regression.
            Fast (minutes, runs on CPU), strong baseline for ~2.6k images. Start here.
  finetune  Fine-tune the whole network end to end (needs a GPU). Use once the baseline works.

Evaluation schemes (--split):
  mouse     5-fold GroupKFold by mouse id: same labs in train and test (optimistic).
  source    Leave-one-lab-out: train on all other labs, test on the held-out lab.
            This is the honest test of whether the model reads the mouse rather than the
            background/lab setup.

Reference point printed every run: MAE obtained from the *folder name alone* (no pixels).
An image model is only learning something about the mouse if it clearly beats that number,
especially in the --split source setting.

Usage
  pip install torch torchvision timm scikit-learn pandas pillow
  python train_age.py embed    --csv labels_id_sex_clean.csv --root /path/to/images
  python train_age.py embed    --csv ... --root ... --split source
  python train_age.py finetune --csv ... --root ... --epochs 25
  python train_age.py finetune --csv ... --root ... --full --out model.pt   # train on everything
"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, timm
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
YEAR = 365.25


# ---------------------------------------------------------------- data
def load_table(csv, root):
    df = pd.read_csv(csv, encoding="utf-8-sig")
    df["path"] = [os.path.join(root, f) for f in df["file"]]
    ok = df["path"].map(os.path.exists)
    if (~ok).any():
        print(f"WARNING: {(~ok).sum()} of {len(df)} images not found under {root}, e.g. {df.path[~ok].iloc[0]}")
    df = df[ok].reset_index(drop=True)
    df["mouse"] = df["source"] + "|" + df["id"].astype(str)   # ids are only unique within a lab
    df["train_ok"] = training_mask(df)
    return df


CONTROL_WORDS = {"", "nan", "none", "control", "ctrl", "vehicle", "untreated", "wt", "wild-type", "wildtype"}


def training_mask(df):
    """Rows allowed in TRAINING. Everything else (treated, progeria, ...) is still predicted, by models
    that never saw it, but never trained on and not used in the main evaluation.
    Uses `training` or `use_for_training` (yes/no) if present, plus `treatment` / `genotype` if present."""
    ok = pd.Series(True, index=df.index)
    for col in ["training", "use_for_training"]:
        if col in df:
            ok &= ~df[col].astype(str).str.strip().str.lower().isin(["no", "n", "false", "0"])
    for col in ["treatment", "genotype"]:
        if col in df:
            ok &= df[col].fillna("").astype(str).str.strip().str.lower().isin(CONTROL_WORDS)
    return ok.values


def restrict(folds, ok):
    """Remove non-training rows from every training fold (they stay in the test folds)."""
    return [(tr[ok[tr]], te) for tr, te in folds]


def eval_view(df, pred, folds):
    """Rows usable for the main evaluation, with folds re-indexed to those rows."""
    ok = df["train_ok"].values
    new = np.cumsum(ok) - 1
    f2 = [(new[tr[ok[tr]]], new[te[ok[te]]]) for tr, te in folds]
    return df[ok].reset_index(drop=True), pred[ok], f2


class PadSquare:
    """Pad to square with edge color so the whole mouse survives the resize (no center-crop cutting it)."""
    def __call__(self, im):
        w, h = im.size
        s = max(w, h)
        fill = tuple(int(c) for c in np.asarray(im)[[0, -1]][:, [0, -1]].reshape(-1, 3).mean(0))
        return ImageOps.pad(im, (s, s), color=fill)


def tfms(size, train):
    if train:
        return T.Compose([
            PadSquare(), T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
            # strong color jitter + grayscale: discourages reading lab-specific background color/lighting
            T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomGrayscale(0.2),
            T.ToTensor(), T.Normalize(MEAN, STD), T.RandomErasing(p=0.25)])
    return T.Compose([PadSquare(), T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


IMG_CACHE = {}   # path -> small PIL image; filled once by preload(), shared by DataLoader workers


def preload(paths, max_side):
    """Decode every image once and keep a copy shrunk to max_side px (enough for training at --size).
    Avoids re-reading and re-decoding all photos every epoch, which is slow on Colab's 2 CPU cores."""
    todo = [p for p in dict.fromkeys(paths) if p not in IMG_CACHE]
    t0 = time.time()
    for k, p in enumerate(todo):
        im = Image.open(p).convert("RGB")
        im.thumbnail((max_side, max_side), Image.BILINEAR)
        IMG_CACHE[p] = im
        if (k + 1) % 500 == 0:
            print(f"  cached {k + 1}/{len(todo)} images", flush=True)
    if todo:
        print(f"  cached {len(todo)} images at <= {max_side}px in {time.time() - t0:.0f}s", flush=True)


class Imgs(Dataset):
    def __init__(self, paths, y, tf):
        self.p, self.y, self.tf = list(paths), np.asarray(y, np.float32), tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        im = IMG_CACHE.get(self.p[i])
        im = im.copy() if im is not None else Image.open(self.p[i]).convert("RGB")
        return self.tf(im), self.y[i]


def splits(df, how):
    if how == "mouse":
        return list(GroupKFold(5).split(df, groups=df["mouse"]))
    return list(LeaveOneGroupOut().split(df, groups=df["source"]))


# ---------------------------------------------------------------- reporting
def folder_only_baseline(df, folds, mask):
    p = np.zeros(len(df))
    for tr, te in folds:
        m = df.iloc[tr].groupby("folder").age_days.mean()
        p[te] = df.iloc[te].folder.map(m).fillna(df.age_days.iloc[tr].mean()).values
    return np.abs(p - df.age_days)[mask].mean()


def report(df, pred, folds, tag):
    mask = ~np.isnan(pred)                                    # rows that were actually predicted
    d = df.assign(pred=pred, err=lambda x: (x.pred - x.age_days).abs())[mask]
    # several photos of the same mouse at the same age -> average them (usually lowers error)
    per_mouse = d.groupby(["mouse", "age_days"]).agg(pred=("pred", "mean"), src=("source", "first")).reset_index()
    per_mouse["err"] = (per_mouse.pred - per_mouse.age_days).abs()
    r = np.corrcoef(d.pred, d.age_days)[0, 1]
    print(f"\n=== {tag} ===")
    print(f"MAE per image : {d.err.mean():6.1f} days   (r = {r:.3f})")
    print(f"MAE per mouse : {per_mouse.err.mean():6.1f} days   (photos of one mouse/timepoint averaged)")
    print(f"Folder-name-only baseline (same splits): {folder_only_baseline(df, folds, mask):6.1f} days")
    print(f"Predict-the-median baseline:             {np.abs(d.age_days - df.age_days.median()).mean():6.1f} days")
    print("\nPer lab:")
    print(d.groupby("source").agg(n=("err", "size"), MAE=("err", "mean"),
                                  age_min=("age_days", "min"), age_max=("age_days", "max")).round(1).to_string())
    print("\nBy age bin:")
    bins = pd.cut(d.age_days, [0, 120, 300, 500, 700, 900, 1100])
    print(d.groupby(bins, observed=True).agg(n=("err", "size"), MAE=("err", "mean"),
                                             mean_pred=("pred", "mean")).round(1).to_string())


# ---------------------------------------------------------------- stage 1: frozen embeddings
@torch.no_grad()
def embed(df, a, dev):
    m = timm.create_model(a.backbone, pretrained=not a.no_pretrained, num_classes=0, img_size=a.size).to(dev).eval()
    dl = DataLoader(Imgs(df.path, df.age_days, tfms(a.size, False)), batch_size=a.bs, num_workers=a.workers)
    out = []
    for i, (x, _) in enumerate(dl):
        x = x.to(dev)
        f = (m(x) + m(torch.flip(x, dims=[3]))) / 2          # horizontal-flip TTA
        out.append(f.float().cpu())
        if i % 10 == 0:
            print(f"  embedded {min((i + 1) * a.bs, len(df))}/{len(df)}", flush=True)
    return torch.cat(out).numpy()


def get_embeddings(df, a, dev):
    # cache keyed by image root + exact file list, so cropped and raw images never share embeddings
    import hashlib
    key = hashlib.md5((os.path.abspath(a.root) + "\n".join(df.path)).encode()).hexdigest()[:8]
    cache = f"emb_{a.backbone.replace('/', '_')}_{a.size}_{key}.npy"
    if os.path.exists(cache) and len(np.load(cache)) == len(df):
        print(f"using cached embeddings {cache}")
        return np.load(cache)
    X = embed(df, a, dev); np.save(cache, X)
    return X


def run_embed(df, a, dev):
    X = get_embeddings(df, a, dev)
    y = df.age_days.values / YEAR
    folds = restrict(splits(df, a.split), df["train_ok"].values)
    pred = np.zeros(len(df))
    for tr, te in folds:
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))
        reg.fit(X[tr], y[tr]); pred[te] = reg.predict(X[te]) * YEAR
    report(*eval_view(df, pred, folds), f"frozen {a.backbone} + ridge, split={a.split}")
    save_predictions(df, pred, a, "embed")


def save_predictions(df, pred, a, mode):
    out = a.pred_out or f"oof_{mode}_{a.split}{'_' + a.tag if a.tag else ''}.csv"
    d = df.drop(columns=["path"]).assign(pred_days=pred, model=f"{mode}:{a.backbone}:{a.size}:{a.tag}",
                                         split=a.split, root=a.root)
    d.to_csv(out, index=False)
    n_extra = int((~df["train_ok"]).sum())
    print(f"\nheld-out predictions: {out}"
          + (f"  ({n_extra} images of treated/progeria/excluded mice predicted but not trained on)" if n_extra else ""))


# ---------------------------------------------------------------- stage 2: fine-tuning
def make_model(a):
    return timm.create_model(a.backbone, pretrained=not a.no_pretrained, num_classes=1, img_size=a.size, drop_path_rate=0.1)


def amp_dtype(dev):
    """bfloat16 only on GPUs with native support (Ampere or newer: A100, L4, RTX 30xx+).
    torch.cuda.is_bf16_supported() also says True on a T4, where bf16 is merely emulated and very slow."""
    if dev != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def ridge_head(X, y):
    """Frozen-feature ridge regression (exactly what `embed` uses) as one linear layer (w, b), in years."""
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(X, y)
    sc, rd = reg[0], reg[1]
    w = rd.coef_ / sc.scale_
    return w.astype(np.float32), float(rd.intercept_ - (sc.mean_ * w).sum())


def freeze_first_blocks(m, k):
    """Freeze patch embedding + the first k transformer blocks (ViT); only later blocks + head train."""
    if k <= 0:
        return
    for n, p in m.named_parameters():
        if n.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed")):
            p.requires_grad = False
        if n.startswith("blocks."):
            if int(n.split(".")[1]) < k:
                p.requires_grad = False


def train_one(df_tr, a, dev, df_va=None, eta=None, init_head=None):
    m = make_model(a).to(dev)
    y = df_tr.age_days.values / YEAR
    dl = DataLoader(Imgs(df_tr.path, y, tfms(a.size, True)), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    freeze_first_blocks(m, a.freeze_blocks)
    head = [p for n, p in m.named_parameters() if n.startswith("head") and p.requires_grad]
    body = [p for n, p in m.named_parameters() if not n.startswith("head") and p.requires_grad]
    hm = a.head_lr_mult if a.head_lr_mult is not None else (1.0 if init_head is not None else 10.0)
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * hm}], weight_decay=0.05)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * hm], total_steps=steps, pct_start=0.1)
    with torch.no_grad():
        if init_head is not None:                             # LP-FT: start from the frozen-feature ridge solution
            w, b = init_head
            m.get_classifier().weight.copy_(torch.tensor(w, device=dev)[None])
            m.get_classifier().bias.fill_(b)
        else:                                                 # start head at the mean age
            m.get_classifier().bias.fill_(float(y.mean()))
    if df_va is not None and init_head is not None:
        print(f"  start (frozen features + ridge) val MAE "
              f"{np.abs(predict(m, df_va, a, dev) - df_va.age_days.values).mean():.1f} d", flush=True)
    loss_fn = nn.SmoothL1Loss(beta=0.1)                       # ~L1 in years: robust, optimizes MAE
    dt = amp_dtype(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)
    for ep in range(a.epochs):
        m.train(); t0, tot = time.time(), 0.0
        for x, t in dl:
            x, t = x.to(dev, non_blocking=True), t.to(dev)
            with torch.autocast(dev, dtype=dt or torch.float32, enabled=dt is not None):
                loss = loss_fn(m(x).squeeze(1).float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x)
        secs = time.time() - t0
        msg = f"  epoch {ep + 1:2d}/{a.epochs} loss {tot / (len(dl) * a.bs):.4f} ({secs:.0f}s)"
        if eta is not None and ep == 0:
            total = secs * a.epochs * eta / 3600
            msg += f"  -> whole run roughly {total:.1f} h ({eta} models x {a.epochs} epochs)"
        if df_va is not None and ((ep + 1) % 5 == 0 or ep + 1 == a.epochs):
            msg += f"  val MAE {np.abs(predict(m, df_va, a, dev) - df_va.age_days.values).mean():.1f} d"
        print(msg, flush=True)
    return m


@torch.no_grad()
def predict(m, df, a, dev):
    m.eval()
    dl = DataLoader(Imgs(df.path, df.age_days, tfms(a.size, False)), batch_size=a.bs, num_workers=a.workers)
    out = []
    for x, _ in dl:
        x = x.to(dev)
        dt = amp_dtype(dev)
        with torch.autocast(dev, dtype=dt or torch.float32, enabled=dt is not None):
            p = (m(x) + m(torch.flip(x, dims=[3]))) / 2
        out.append(p.float().squeeze(1).cpu())
    return torch.cat(out).numpy() * YEAR


def run_finetune(df, a, dev):
    if a.full:
        if a.cache_px:
            preload(df.path[df["train_ok"]], a.cache_px)
        m = train_one(df[df["train_ok"]].reset_index(drop=True), a, dev)
        torch.save({"state_dict": m.state_dict(), "backbone": a.backbone, "size": a.size}, a.out)
        print(f"saved {a.out}"); return
    ok = df["train_ok"].values
    folds = restrict(splits(df, a.split), ok)
    if a.folds:
        folds = folds[:a.folds]
    if a.cache_px:
        preload(df.path, a.cache_px)
    # a fold whose test rows can never be trained on (e.g. all literature images) needs no model of its own:
    # every other fold's model also never saw them, so they get the average of those models' predictions
    X_lp = get_embeddings(df, a, dev) if a.lp_init else None   # cached frozen features (same as `embed`)
    deferred = [te for tr, te in folds if not ok[te].any()]
    deferred = np.concatenate(deferred) if deferred else np.array([], int)
    folds_run = [(tr, te) for tr, te in folds if ok[te].any()]
    if a.only:
        want = [s.strip() for s in a.only.split(",")]
        folds_run = [(tr, te) for tr, te in folds_run if set(df.source.iloc[te]) & set(want)]
        deferred = np.array([], int)
        print(f"--only: running {len(folds_run)} fold(s) for {', '.join(want)} (diagnostic run; other labs not predicted)")
    if len(deferred):
        print(f"{len(deferred)} images in folds with nothing trainable "
              f"({', '.join(sorted(df.source.iloc[deferred].unique()))}): predicted by the other folds' models")
    pred = np.full(len(df), np.nan)
    extra = np.zeros(len(deferred))
    for k, (tr, te) in enumerate(folds_run):
        print(f"\nfold {k + 1}/{len(folds_run)}: train {len(tr)}, test {len(te)} "
              f"({', '.join(sorted(df.source.iloc[te].unique()))})")
        va = te[ok[te]]
        init = ridge_head(X_lp[tr], df.age_days.values[tr] / YEAR) if a.lp_init else None
        m = train_one(df.iloc[tr], a, dev, df.iloc[va] if len(va) else None,
                      eta=len(folds_run) if k == 0 else None, init_head=init)
        pred[te] = predict(m, df.iloc[te], a, dev)
        if len(deferred):
            extra += predict(m, df.iloc[deferred], a, dev) / len(folds_run)
        del m; torch.cuda.empty_cache() if dev == "cuda" else None
    if len(deferred):
        pred[deferred] = extra
    tag = f"fine-tuned {a.backbone}, split={a.split}" + (" [LP-FT]" if a.lp_init else "") + \
          (f" [first {a.freeze_blocks} blocks frozen]" if a.freeze_blocks else "")
    if a.only:
        done = ~np.isnan(pred)
        d0 = df.assign(pred=pred)[done & df["train_ok"].values]
        print(f"\n=== {tag} — diagnostic folds only ===")
        print(d0.groupby("source").apply(lambda g: pd.Series({
            "n": len(g), "MAE": (g.pred - g.age_days).abs().mean(), "pred_sd": g.pred.std(), "true_sd": g.age_days.std(),
            "spearman": g[["pred", "age_days"]].corr(method="spearman").iloc[0, 1]}), include_groups=False).round(2).to_string())
        return
    report(*eval_view(df, pred, folds), tag)
    save_predictions(df, pred, a, "finetune")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["embed", "finetune"])
    ap.add_argument("--csv", default="labels_id_sex_clean.csv")
    ap.add_argument("--root", required=True, help="folder that contains the per-session image folders")
    ap.add_argument("--split", choices=["mouse", "source"], default="mouse")
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m",
                    help="any timm model, e.g. convnext_small.fb_in22k_ft_in1k, vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--folds", type=int, default=0, help="run only the first N folds (quick check)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--cache-px", type=int, default=-1,
                    help="finetune: keep all images in memory shrunk to this size (default 1.3 x --size; 0 = off)")
    ap.add_argument("--full", action="store_true", help="finetune: train on all data and save weights")
    ap.add_argument("--out", default="mouse_age_model.pt")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (pipeline testing only)")
    ap.add_argument("--tag", default="", help="label added to the predictions file name, e.g. crops224")
    ap.add_argument("--lp-init", action="store_true",
                    help="finetune: start each fold from the frozen-feature ridge solution (LP-FT); use a low --lr")
    ap.add_argument("--freeze-blocks", type=int, default=0, help="finetune: keep the first N transformer blocks frozen")
    ap.add_argument("--head-lr-mult", type=float, default=None,
                    help="head learning rate = lr x this (default 10, or 1 with --lp-init)")
    ap.add_argument("--only", default="", help="finetune: run only the folds that hold out these labs, e.g. Malavolta,IRCAN")
    ap.add_argument("--pred-out", default="", help="explicit file name for the held-out predictions")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if a.cache_px < 0:
        a.cache_px = int(round(a.size * 1.3))
    if dev == "cpu" and a.mode == "finetune":
        print("WARNING: fine-tuning on CPU will be very slow; use a GPU or start with `embed`.")
    df = load_table(a.csv, a.root)
    print(f"{len(df)} images, {df.mouse.nunique()} mice, {df.source.nunique()} labs, device={dev}")
    if dev == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, mixed precision: {str(amp_dtype(dev)).replace('torch.', '')}, "
              f"data loader workers: {a.workers}")
    if (~df["train_ok"]).any():
        print(f"  {(~df['train_ok']).sum()} images ({df.loc[~df['train_ok'], 'mouse'].nunique()} mice) excluded from "
              "training (treated / progeria / use_for_training=no); they are predicted by models that never saw them")
    torch.manual_seed(0); np.random.seed(0)
    (run_embed if a.mode == "embed" else run_finetune)(df, a, dev)
