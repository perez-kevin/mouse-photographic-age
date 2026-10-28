"""
Compare how well the same image features predict AGE vs SEX.

Both tasks use the same frozen embeddings (cached by train_age.py), the same images
(only those with sex recorded) and the same train/test splits (grouped by mouse).

Why a special metric: in this dataset both age AND sex are partly given away by the folder
(several folders are single-sex, and each folder covers a narrow age range). A plain accuracy
or MAE therefore partly measures "can the model recognise the photo session". The headline
metric here is WITHIN-FOLDER CONCORDANCE, which only compares mice photographed in the same session:

  age: for two mice in the same folder with different ages, how often is the older one predicted older?
  sex: for a male and a female in the same folder, how often is the male given the higher "male" score?
       (this is the within-folder ROC AUC)

Both are on the same scale: 0.5 = chance, 1.0 = perfect. Folder-only prediction scores exactly 0.5,
so any lift above 0.5 has to come from the mouse itself (or from within-session photo differences).
95% intervals come from resampling mice (bootstrap), so repeated photos of one mouse are not
counted as independent evidence.

Usage
  python compare_tasks.py --root /path/to/images                    # 5-fold split by mouse
  python compare_tasks.py --root /path/to/images --split source     # leave-one-lab-out
"""
import argparse, warnings
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from train_age import load_table, get_embeddings, splits, YEAR

warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn API-change notices


# ---------------------------------------------------------------- metrics
def within_folder_concordance(y, p, folder, w=None):
    """Pairwise concordance computed only among pairs in the same folder with different y.
    Ties in prediction count 1/2. w = per-row weights (used for the bootstrap)."""
    w = np.ones(len(y)) if w is None else w
    num = den = 0.0
    for f in np.unique(folder):
        i = np.where((folder == f) & (w > 0))[0]
        if len(i) < 2:
            continue
        dy = np.sign(y[i][:, None] - y[i][None, :])
        dp = np.sign(p[i][:, None] - p[i][None, :])
        W = np.triu(w[i][:, None] * w[i][None, :], 1) * (dy != 0)
        num += (W * (dy * dp + 1) / 2).sum()
        den += W.sum()
    return num / den if den else np.nan


def n_pairs(y, folder):
    return sum(int(np.triu(np.sign(y[folder == f][:, None] - y[folder == f][None, :]) != 0, 1).sum())
               for f in np.unique(folder))


def bootstrap_ci(y, p, folder, mouse, n=200, seed=0):
    """Resample mice with replacement; each mouse's photos move together."""
    rng = np.random.default_rng(seed)
    mice, inv = np.unique(mouse, return_inverse=True)
    vals = []
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice))
        vals.append(within_folder_concordance(y, p, folder, counts[inv].astype(float)))
    return np.nanpercentile(vals, [2.5, 97.5])


def oof(X, y, folds, kind):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        if kind == "sex" and len(np.unique(y[tr])) < 2:
            continue
        if kind == "age":
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))
            m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
        else:
            m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 12), max_iter=3000,
                                                                     scoring="roc_auc", class_weight="balanced"))
            m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def folder_only(df, y, folds):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = pd.Series(y[tr]).groupby(df.folder.values[tr]).mean()
        p[te] = df.folder.iloc[te].map(m).fillna(y[tr].mean()).values
    return p


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="labels_id_sex_clean.csv")
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", choices=["mouse", "source"], default="mouse")
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    df = load_table(a.csv, a.root)
    X = get_embeddings(df, a, dev)                       # same cache as train_age.py (all images)
    keep = df.sex.notna().values
    df, X = df[keep].reset_index(drop=True), X[keep]
    print(f"{len(df)} images with sex recorded, {df.mouse.nunique()} mice, split={a.split}")

    folds = splits(df, a.split)
    age = df.age_days.values / YEAR
    male = (df.sex == "Male").astype(int).values
    fol, mouse = df.folder.values, df.mouse.values

    p_age, p_sex = oof(X, age, folds, "age"), oof(X, male, folds, "sex")
    f_age, f_sex = folder_only(df, age, folds), folder_only(df, male, folds)
    ok = ~np.isnan(p_sex) & ~np.isnan(p_age)

    rows = []
    for task, y, p, fb in [("age", age, p_age, f_age), ("sex", male, p_sex, f_sex)]:
        y, p, fb, fo, mo = y[ok], p[ok], fb[ok], fol[ok], mouse[ok]
        c = within_folder_concordance(y, p, fo)
        lo, hi = bootstrap_ci(y, p, fo, mo, a.boot)
        if task == "age":
            overall = f"MAE {np.abs(p - y).mean() * YEAR:.0f} d (folder-only {np.abs(fb - y).mean() * YEAR:.0f} d)"
        else:
            overall = (f"AUC {roc_auc_score(y, p):.3f} (folder-only {roc_auc_score(y, fb):.3f}), "
                       f"bal.acc {balanced_accuracy_score(y, p > .5):.3f}")
        rows.append(dict(task=task, within_folder=c, ci_low=lo, ci_high=hi,
                         pairs=n_pairs(y, fo), overall=overall))
    res = pd.DataFrame(rows).set_index("task")

    print("\n=== Within-folder concordance (0.5 = chance, 1 = perfect; folder-only scores 0.5) ===")
    print(res[["within_folder", "ci_low", "ci_high", "pairs"]].round(3).to_string())
    print("\n=== Overall metrics (these include the folder shortcut) ===")
    for t, r in res.iterrows():
        print(f"  {t}: {r.overall}")

    # per-folder breakdown: only folders where a task can be scored
    per = []
    for f in np.unique(fol[ok]):
        i = ok & (fol == f)
        per.append(dict(folder=f, n=i.sum(), n_ages=len(np.unique(age[i])),
                        male_frac=male[i].mean().round(2),
                        age_conc=within_folder_concordance(age[i], p_age[i], fol[i]),
                        sex_auc=within_folder_concordance(male[i], p_sex[i], fol[i])))
    per = pd.DataFrame(per).set_index("folder")
    print("\n=== Per folder (NaN = task not scorable there: single age or single sex) ===")
    print(per.round(2).to_string())

    both = per.dropna()
    if len(both):
        print(f"\nFolders where both tasks can be scored ({len(both)}, {both.n.sum()} images): "
              f"age {both.age_conc.mean():.3f} vs sex {both.sex_auc.mean():.3f} (unweighted mean)")

    out = f"compare_tasks_{a.split}.csv"
    df.assign(pred_age_days=p_age * YEAR, pred_male_prob=p_sex).to_csv(out, index=False)
    print(f"\nper-image predictions: {out}")
