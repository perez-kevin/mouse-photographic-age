"""
Score the human age ratings (from make_rating_task.py) and compare them with the model on the same images.

  python score_ratings.py --key rating_task/answer_key.csv --ratings ratings_Anna.csv ratings_Ben.csv
  python score_ratings.py --key ... --ratings ratings_*.csv --model oof_finetune_source_crops224.csv

Per rater (and, with 2+ raters, the raters' average; with --model, the model):
  MAE in days, median error, bias (mean predicted - true), Spearman correlation with true age,
  within-session concordance (pairs of mice from the same session with different ages: is the older
  one rated older? 0.5 = chance), the same by age gap, and seconds per photo.
Self-consistency: each rater's answers on the hidden repeats (same mouse, same age).
Agreement between raters (pairwise Spearman, mean absolute difference).
With --model: paired differences rater - model on exactly the same images, 95% interval from
resampling mice.

Scoring uses the 100 unique mice; the hidden repeats are only used for self-consistency.
Model predictions are matched by file name; use the held-out file of the final model
(leave-one-lab-out), so the model has never seen these labs, like the raters.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

MONTH = 30.44


def concordance(y, p, group, w=None):
    w = np.ones(len(y)) if w is None else w
    num = den = 0.0
    for g in np.unique(group):
        i = np.where((group == g) & (w > 0))[0]
        if len(i) < 2:
            continue
        dy = np.sign(y[i][:, None] - y[i][None, :]); dp = np.sign(p[i][:, None] - p[i][None, :])
        W = np.triu(w[i][:, None] * w[i][None, :], 1) * (dy != 0)
        num += (W * (dy * dp + 1) / 2).sum(); den += W.sum()
    return num / den if den else np.nan


def boot(fn, mouse, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    mice, inv = np.unique(mouse, return_inverse=True)
    v = [fn(np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice))[inv].astype(float))
         for _ in range(n)]
    return np.nanpercentile(v, [2.5, 97.5])


def metrics(d, col, n_boot):
    y, p = d.age_days.values, d[col].values
    sess = d.in_session_group.values.astype(bool)
    f, m = d.folder.values, d.mouse.values
    c = concordance(y[sess], p[sess], f[sess])
    lo, hi = boot(lambda w: concordance(y[sess], p[sess], f[sess], w[sess]), m, n_boot)
    e = np.abs(p - y)
    elo, ehi = boot(lambda w: np.average(e, weights=w), m, n_boot)
    return dict(n=len(d), MAE_days=e.mean(), MAE_ci_low=elo, MAE_ci_high=ehi, median_AE_days=np.median(e),
                bias_days=(p - y).mean(), spearman=spearmanr(p, y)[0],
                within_session_conc=c, conc_ci_low=lo, conc_ci_high=hi,
                scored_pairs=int(sum((np.triu(np.sign(y[sess][f[sess] == g][:, None] - y[sess][f[sess] == g][None, :]) != 0, 1)).sum()
                                     for g in np.unique(f[sess]))))


def gap_rows(d, col, name):
    rows = []
    s = d[d.in_session_group.astype(bool)]
    for f, g in s.groupby("folder"):
        a, p = g.age_days.values, g[col].values
        i, j = np.triu_indices(len(g), 1); k = a[i] != a[j]
        c = (np.sign(a[i] - a[j]) * np.sign(p[i] - p[j]) + 1) / 2
        rows.append(pd.DataFrame({"gap": np.abs(a[i] - a[j])[k], "c": c[k]}))
    r = pd.concat(rows)
    r["gap_days"] = pd.cut(r.gap, [0, 100, 200, 300, 500, 1100]).astype(str)
    t = r.groupby("gap_days").c.agg(["mean", "size"])
    return pd.DataFrame({"rater": name, "gap_days": t.index, "concordance": t["mean"].values, "pairs": t["size"].values})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--ratings", nargs="+", required=True, help="ratings_*.csv files (wildcards ok)")
    ap.add_argument("--model", help="held-out predictions file with `file` and a pred column")
    ap.add_argument("--model-col", default=None, help="prediction column (default pred_days / first pred_age*)")
    ap.add_argument("--out", default="human_baseline")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220)

    key = pd.read_csv(a.key)
    key["repeat_of_id"] = key.repeat_of_id.fillna("")
    files = sorted({f for pat in a.ratings for f in glob.glob(pat)})
    R = pd.concat([pd.read_csv(f).assign(source_file=os.path.basename(f)) for f in files], ignore_index=True)
    R["rater"] = R.rater.astype(str).str.strip()
    R["pred_days"] = R.age_months * MONTH

    raters = []
    for name, r in R.groupby("rater"):
        info = r.iloc[0]
        missing = r.age_months.isna().sum()
        print(f"{name}: {len(r) - missing}/{len(r)} answered | {info.role} | {info.years_experience} years | "
              f"worked in a source lab: {info.worked_in_source_lab} | median {r.seconds.median():.1f} s per photo")
        raters.append(name)

    wide = key.merge(R.pivot_table(index="image_id", columns="rater", values="pred_days").reset_index(),
                     on="image_id", how="left")
    uniq = wide[wide.repeat_of_id == ""].copy()

    cols = list(raters)
    if len(raters) > 1:
        uniq["raters_mean"] = uniq[raters].mean(axis=1); cols.append("raters_mean")
    if a.model:
        M = pd.read_csv(a.model)
        mc = a.model_col or ("pred_days" if "pred_days" in M else next(c for c in M.columns if c.startswith("pred_age")))
        M = M.groupby("file", as_index=False)[mc].mean().rename(columns={mc: "model"})
        uniq = uniq.merge(M, on="file", how="left")
        nm = uniq.model.isna().sum()
        if nm:
            print(f"WARNING: model has no prediction for {nm} of {len(uniq)} rated images (compared on the rest)")
        cols.append("model")

    # ---- accuracy
    res = []
    for c in cols:
        d = uniq[uniq[c].notna()]
        res.append(dict(rater=c, **metrics(d, c, a.boot)))
    res = pd.DataFrame(res)
    res.to_csv(f"{a.out}/accuracy.csv", index=False)
    print("\n=== Accuracy on the unique mice (days; concordance 0.5 = chance) ===")
    print(res.round(3).to_string(index=False))

    # ---- by age band
    band = []
    for c in cols:
        d = uniq[uniq[c].notna()]
        for b, g in d.groupby("band", observed=True):
            band.append(dict(rater=c, band=b, n=len(g), MAE_days=np.abs(g[c] - g.age_days).mean(),
                             bias_days=(g[c] - g.age_days).mean()))
    band = pd.DataFrame(band)
    order = ["<4 mo", "4-9 mo", "9-15 mo", "15-21 mo", ">21 mo"]
    band.to_csv(f"{a.out}/by_age_band.csv", index=False)
    print("\n=== MAE (days) by true age band ===")
    print(band.pivot(index="band", columns="rater", values="MAE_days").reindex(order).round(0).to_string())

    gaps = pd.concat([gap_rows(uniq[uniq[c].notna()], c, c) for c in cols])
    gaps.to_csv(f"{a.out}/concordance_by_gap.csv", index=False)
    print("\n=== Within-session concordance by age gap ===")
    print(gaps.pivot(index="gap_days", columns="rater", values="concordance").round(2).to_string())

    # ---- self-consistency on hidden repeats
    rep = wide[wide.repeat_of_id != ""][["image_id", "repeat_of_id", "repeat_type"] + raters]
    orig = wide.set_index("image_id")[raters]
    cons = []
    for name in raters:
        x = rep[name].values; y = orig.loc[rep.repeat_of_id, name].values
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum():
            cons.append(dict(rater=name, repeats=int(ok.sum()),
                             mean_abs_diff_months=np.abs(x[ok] - y[ok]).mean() / MONTH,
                             spearman=spearmanr(x[ok], y[ok])[0] if ok.sum() > 2 else np.nan))
    if cons:
        cons = pd.DataFrame(cons); cons.to_csv(f"{a.out}/self_consistency.csv", index=False)
        print("\n=== Self-consistency on hidden repeats (same mouse, same age) ===")
        print(cons.round(2).to_string(index=False))

    # ---- agreement between raters
    if len(raters) > 1:
        ag = []
        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                x, y = uniq[raters[i]], uniq[raters[j]]; ok = x.notna() & y.notna()
                ag.append(dict(pair=f"{raters[i]} vs {raters[j]}", spearman=spearmanr(x[ok], y[ok])[0],
                               mean_abs_diff_months=(x[ok] - y[ok]).abs().mean() / MONTH))
        ag = pd.DataFrame(ag); ag.to_csv(f"{a.out}/agreement.csv", index=False)
        print("\n=== Agreement between raters ===")
        print(ag.round(2).to_string(index=False))

    # ---- paired comparison with the model
    if a.model:
        rows = []
        for c in [c for c in cols if c != "model"]:
            d = uniq[uniq[c].notna() & uniq.model.notna()]
            y, h, m_ = d.age_days.values, d[c].values, d.model.values
            s, f, mo = d.in_session_group.values.astype(bool), d.folder.values, d.mouse.values
            dm = np.abs(h - y).mean() - np.abs(m_ - y).mean()
            dmi = boot(lambda w: np.average(np.abs(h - y), weights=w) - np.average(np.abs(m_ - y), weights=w), mo, a.boot)
            dc = concordance(y[s], h[s], f[s]) - concordance(y[s], m_[s], f[s])
            dci = boot(lambda w: concordance(y[s], h[s], f[s], w[s]) - concordance(y[s], m_[s], f[s], w[s]), mo, a.boot)
            rows.append(dict(rater=c, n=len(d), MAE_rater_minus_model=dm, MAE_ci_low=dmi[0], MAE_ci_high=dmi[1],
                             conc_rater_minus_model=dc, conc_ci_low=dci[0], conc_ci_high=dci[1]))
        cmp_ = pd.DataFrame(rows); cmp_.to_csv(f"{a.out}/rater_vs_model.csv", index=False)
        print("\n=== Rater minus model on the same images (negative MAE / positive concordance = rater better) ===")
        print(cmp_.round(3).to_string(index=False))

    uniq.to_csv(f"{a.out}/per_image.csv", index=False)
    print(f"\ntables in {a.out}/")


if __name__ == "__main__":
    main()
