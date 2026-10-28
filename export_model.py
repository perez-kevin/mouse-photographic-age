"""
Fit the final age model on ALL training mice (no held-out fold) and save it for predict.py.

  python export_model.py --csv labels_id_sex_clean.csv --root crops_sampick --out mouse_age_ridge448.joblib

Same model as the paper: frozen DINOv2 ViT-B/14 at 448 px, horizontal-flip TTA, StandardScaler + RidgeCV
(alphas logspace(-1, 4, 20)), target age in years. Reuses the embedding cache written by train_age.py
(run it from the same folder).
"""
import argparse, types
import numpy as np, torch, joblib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from train_age import load_table, get_embeddings, YEAR

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="mouse_age_ridge448.joblib")
ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
ap.add_argument("--size", type=int, default=448)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
df = load_table(a.csv, a.root)
X = get_embeddings(df, types.SimpleNamespace(backbone=a.backbone, size=a.size, root=a.root, bs=32,
                                              workers=a.workers, no_pretrained=False), dev)
ok = df.train_ok.values
reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(X[ok], df.age_days.values[ok] / YEAR)
joblib.dump(dict(model=reg, backbone=a.backbone, size=a.size, target="age_years",
                 trained_on=dict(images=int(ok.sum()), mice=int(df[ok].mouse.nunique()),
                                 facilities=sorted(df[ok].source.unique()),
                                 age_days=[float(df[ok].age_days.min()), float(df[ok].age_days.max())])), a.out)
print(f"saved {a.out}: {ok.sum()} images, {df[ok].mouse.nunique()} mice, alpha={reg[-1].alpha_:.3g}")
