"""
Original vs cropped images: same model, same images, same splits, side by side.

  python compare_crops.py --raw /path/to/original_images --crops /path/to/crops --split source
  python compare_crops.py --raw ... --crops ... --split mouse --qc /path/to/crops/crop_qc.csv --drop-flagged

What it reports (each for ORIGINAL and CROPPED, plus the difference CROPPED - ORIGINAL with a 95%
interval from resampling mice, so the difference is tested on the same mice):

  1. Session recognition: how accurately the photo session (folder) and the lab can be predicted from the
     image features. This is the shortcut cropping is meant to remove, so LOWER is better here.
  2. Age: within-folder concordance (0.5 = chance) and overall MAE.
  3. Sex: within-folder AUC (a reference task; it should survive cropping if the model looks at the mouse).
  4. Age concordance by age gap and by age of the younger mouse (does cropping help the hard old-vs-old pairs?)
  5. Per-folder age concordance.

Expect: session recognition drops a lot; within-folder concordance stays similar or improves
(the mouse is still there); overall MAE may go up with --split mouse (the folder shortcut is gone)
and should improve or stay similar with --split source. If within-folder concordance drops sharply,
look at the crops: either masks cut off mouse parts, or the model relied on the background.

Embeddings are cached per image folder by train_age.py, so reruns are fast.
"""
import argparse, warnings, types
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from train_age import load_table, get_embeddings, splits, YEAR
from compare_tasks import within_folder_concordance, oof

warnings.filterwarnings("ignore", category=FutureWarning)


def embeddings_for(csv, root, a, dev):
    df = load_table(csv, root)
    aa = types.SimpleNamespace(**{**vars(a), "root": root})
    X = get_embeddings(df, aa, dev)
    return df.reset_index(drop=True), X


def paired_boot(fn, mouse, n, seed=0):
    """fn(weights) -> (value_raw, value_crop). Returns 95% interval of crop - raw, resampling mice."""
    rng = np.random.default_rng(seed)
    mice, inv = np.unique(mouse, return_inverse=True)
    d = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice))[inv].astype(float)
        r, c = fn(w)
        d.append(c - r)
    return np.nanpercentile(d, [2.5, 97.5])


def session_recognition(X, labels, mouse):
    """Balanced accuracy of predicting `labels` from features, 5-fold grouped by mouse."""
    y = pd.factorize(labels)[0]
    p = np.zeros(len(y), int)
    for tr, te in GroupKFold(5).split(X, groups=mouse):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return balanced_accuracy_score(y, p), 1 / len(np.unique(y))


def gap_table(age_days, preds, folder):
    rows = []
    for f in np.unique(folder):
        i = np.where(folder == f)[0]
        a = age_days[i]
        u, v = np.triu_indices(len(i), 1)
        gap = np.abs(a[u] - a[v])
        keep = gap > 0
        r = {"gap": gap[keep], "younger": np.minimum(a[u], a[v])[keep]}
        for name, p in preds.items():
            s = np.sign(a[u] - a[v]) * np.sign(p[i][u] - p[i][v])
            r[name] = ((s + 1) / 2)[keep]
        rows.append(pd.DataFrame(r))
    r = pd.concat(rows)
    r["gap_days"] = pd.cut(r.gap, [0, 100, 200, 300, 500, 1100])
    r["younger_mouse"] = pd.cut(r.younger, [0, 180, 400, 1100], labels=["<6 mo", "6-13 mo", ">13 mo"])
    t = r.groupby(["younger_mouse", "gap_days"], observed=True)[list(preds)].mean()
    t["pairs"] = r.groupby(["younger_mouse", "gap_days"], observed=True).size()
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="labels_id_sex_clean.csv")
    ap.add_argument("--raw", required=True, help="folder with the original images")
    ap.add_argument("--crops", required=True, help="folder written by crop_mice.py (same relative paths)")
    ap.add_argument("--split", choices=["mouse", "source"], default="source")
    ap.add_argument("--qc", help="crop_qc.csv from crop_mice.py")
    ap.add_argument("--drop-flagged", action="store_true", help="exclude images whose crop was flagged")
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print("== original images"); d_raw, X_raw = embeddings_for(a.csv, a.raw, a, dev)
    print("== cropped images");  d_crp, X_crp = embeddings_for(a.csv, a.crops, a, dev)

    # keep only images present in BOTH sets (and optionally only unflagged crops)
    common = set(d_raw.file) & set(d_crp.file)
    if a.qc and a.drop_flagged:
        qc = pd.read_csv(a.qc)
        bad = set(qc.loc[qc["flags"].fillna("").ne(""), "file"])
        print(f"dropping {len(common & bad)} images with flagged crops")
        common -= bad
    ir = d_raw.index[d_raw.file.isin(common)].values
    order = d_raw.file.iloc[ir].values
    ic = d_crp.reset_index().set_index("file").loc[order, "index"].values
    df, Xr, Xc = d_raw.iloc[ir].reset_index(drop=True), X_raw[ir], X_crp[ic]
    print(f"\n{len(df)} images in both sets ({len(d_raw) - len(df)} originals without a usable crop), "
          f"{df.mouse.nunique()} mice, split={a.split}")

    folds = splits(df, a.split)
    age = df.age_days.values / YEAR
    fol, mouse = df.folder.values, df.mouse.values

    # ---- 1. session recognition (the shortcut)
    print("\n=== 1. How well can the photo session / lab be recognised? (lower = less shortcut) ===")
    for lab in ["folder", "source"]:
        br, ch = session_recognition(Xr, df[lab].values, mouse)
        bc, _ = session_recognition(Xc, df[lab].values, mouse)
        print(f"  {lab:7s} balanced accuracy: original {br:.3f}   cropped {bc:.3f}   (chance {ch:.3f})")

    # ---- 2. age
    pr, pc = oof(Xr, age, folds, "age"), oof(Xc, age, folds, "age")
    wr, wc = within_folder_concordance(age, pr, fol), within_folder_concordance(age, pc, fol)
    ci = paired_boot(lambda w: (within_folder_concordance(age, pr, fol, w),
                                within_folder_concordance(age, pc, fol, w)), mouse, a.boot)
    mr, mc = np.abs(pr - age).mean() * YEAR, np.abs(pc - age).mean() * YEAR
    ci_mae = paired_boot(lambda w: (np.average(np.abs(pr - age), weights=w) * YEAR,
                                    np.average(np.abs(pc - age), weights=w) * YEAR), mouse, a.boot)
    print("\n=== 2. Age ===")
    print(f"  within-folder concordance: original {wr:.3f}   cropped {wc:.3f}   "
          f"difference {wc - wr:+.3f}  [95% {ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  MAE (days):                original {mr:.0f}     cropped {mc:.0f}     "
          f"difference {mc - mr:+.0f}  [95% {ci_mae[0]:+.0f}, {ci_mae[1]:+.0f}]")

    # ---- 3. sex (reference task)
    s = df.sex.notna().values
    if s.sum() > 50:
        ds = df[s].reset_index(drop=True)
        fs = splits(ds, a.split)
        male = (ds.sex == "Male").astype(int).values
        sr, sc = oof(Xr[s], male, fs, "sex"), oof(Xc[s], male, fs, "sex")
        ok = ~np.isnan(sr) & ~np.isnan(sc)
        f2, m2 = ds.folder.values[ok], ds.mouse.values[ok]
        ar = within_folder_concordance(male[ok], sr[ok], f2)
        ac = within_folder_concordance(male[ok], sc[ok], f2)
        ci_s = paired_boot(lambda w: (within_folder_concordance(male[ok], sr[ok], f2, w),
                                      within_folder_concordance(male[ok], sc[ok], f2, w)), m2, a.boot)
        print("\n=== 3. Sex (reference) ===")
        print(f"  within-folder AUC:         original {ar:.3f}   cropped {ac:.3f}   "
              f"difference {ac - ar:+.3f}  [95% {ci_s[0]:+.3f}, {ci_s[1]:+.3f}]")

    # ---- 4. age by gap
    print("\n=== 4. Age concordance by age gap and age of the younger mouse ===")
    print(gap_table(df.age_days.values, {"original": pr, "cropped": pc}, fol)
          .round(3).to_string())

    # ---- 5. per folder
    per = []
    for f in np.unique(fol):
        i = fol == f
        per.append(dict(folder=f, n=i.sum(), n_ages=len(np.unique(age[i])),
                        original=within_folder_concordance(age[i], pr[i], fol[i]),
                        cropped=within_folder_concordance(age[i], pc[i], fol[i])))
    per = pd.DataFrame(per).set_index("folder")
    per["diff"] = per.cropped - per.original
    print("\n=== 5. Age concordance per folder (NaN = single age) ===")
    print(per.round(2).to_string())

    out = f"compare_crops_{a.split}.csv"
    df.assign(pred_age_original=pr * YEAR, pred_age_cropped=pc * YEAR).to_csv(out, index=False)
    print(f"\nper-image predictions: {out}")
