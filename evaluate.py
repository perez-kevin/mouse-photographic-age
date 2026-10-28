"""
One evaluation for every age model. Takes held-out prediction files and writes all the numbers
used in the paper, at two levels:

  image    every photo counted separately
  session  photos of the same mouse in the same session (same folder, same age) averaged first
           -> the honest unit when some mice were photographed several times per session

Usage
  python evaluate.py oof_finetune_source_crops224.csv
  python evaluate.py frozen=oof_embed_source.csv finetuned=oof_finetune_source_crops224.csv --out eval_main
  python evaluate.py original=compare_crops_source.csv:pred_age_original cropped=compare_crops_source.csv:pred_age_cropped

  name=path[:column]   column defaults to pred_days (or the first column starting with "pred_age")
  The first file is the reference: later files get paired differences vs it (same images only).

Main evaluation uses only mice allowed in training (controls). Treated / progeria / excluded mice
(train_ok = False, or marked in use_for_training / treatment / genotype) are NOT summarised here
on purpose: choose the final model on control performance first, then look at them.
They do get an age-adjusted age gap in age_gap_<name>.csv, ready for the biomarker analyses.

Writes to --out (default eval/):
  summary.csv            headline metrics per model and level (+ paired differences vs the first model)
  per_lab_<name>.csv     n, MAE, bias (offset), within-lab rank correlation, age range
  gap_table_<name>.csv   within-session concordance by age gap x age of the younger mouse
  longitudinal_<name>.csv  same mouse photographed at several ages: is the later photo predicted older?
  calibration_<name>.csv   MAE after correcting each lab's offset with k known-age mice
  age_gap_<name>.csv     per mouse x session: true age, predicted age, raw gap, age-adjusted gap (ALL mice)
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from compare_tasks import within_folder_concordance
from train_age import training_mask

warnings.filterwarnings("ignore", category=RuntimeWarning)
YEAR = 365.25


# ---------------------------------------------------------------- loading
def load_pred(spec):
    name, rest = spec.split("=", 1) if "=" in spec else (os.path.splitext(os.path.basename(spec))[0], spec)
    path, col = (rest.rsplit(":", 1) if ":" in rest and not os.path.exists(rest) else (rest, None))
    d = pd.read_csv(path)
    if col is None:
        col = "pred_days" if "pred_days" in d else next(c for c in d.columns if c.startswith("pred_age"))
    d = d.rename(columns={col: "pred"})
    if "mouse" not in d:
        d["mouse"] = d["source"].astype(str) + "|" + d["id"].astype(str)
    if "train_ok" not in d:
        d["train_ok"] = training_mask(d)
    d["train_ok"] = d["train_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    d = d[d.pred.notna()].reset_index(drop=True)
    split = d["split"].iloc[0] if "split" in d else None
    return name, d, split


def session_level(d):
    keys = ["mouse", "folder", "age_days"]
    first = {c: "first" for c in ["source", "strain", "sex", "train_ok", "treatment", "genotype", "id"] if c in d}
    s = d.groupby(keys, as_index=False).agg(pred=("pred", "mean"), n_photos=("pred", "size"),
                                            file=("file", "first"), **{k: (k, v) for k, v in first.items()})
    return s


# ---------------------------------------------------------------- metrics
def boot_ci(fn, mouse, n=200, seed=0):
    rng = np.random.default_rng(seed)
    mice, inv = np.unique(mouse, return_inverse=True)
    v = [fn(np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice))[inv].astype(float))
         for _ in range(n)]
    return np.nanpercentile(v, [2.5, 97.5])


def folder_only_mae(d, split):
    """Baseline: predict the mean age of the same folder (other mice) or, for unseen labs, of the other labs."""
    if split == "source":
        tot, n = d.groupby("source").age_days.agg(["sum", "size"]).pipe(lambda t: (t["sum"], t["size"]))
        base = (tot.sum() - d.source.map(tot)) / (n.sum() - d.source.map(n))
    else:
        g = d.groupby("folder").age_days
        own = d.groupby(["folder", "mouse"]).age_days.transform("sum")
        cnt = d.groupby(["folder", "mouse"]).age_days.transform("size")
        base = (g.transform("sum") - own) / (g.transform("size") - cnt)
        base = base.fillna(d.age_days.mean())
    return np.abs(base - d.age_days).mean()


def headline(d, split, boot):
    y, p, f, m = d.age_days.values, d.pred.values, d.folder.values, d.mouse.values
    err = np.abs(p - y)
    c = within_folder_concordance(y, p, f)
    lo, hi = boot_ci(lambda w: within_folder_concordance(y, p, f, w), m, boot)
    return dict(n=len(d), mice=d.mouse.nunique(), MAE=err.mean(), median_AE=np.median(err),
                r=np.corrcoef(p, y)[0, 1], spearman=spearmanr(p, y)[0],
                R2=1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                within_session_conc=c, conc_ci_low=lo, conc_ci_high=hi,
                MAE_folder_only=folder_only_mae(d, split), MAE_median_guess=np.abs(y - np.median(y)).mean())


def per_lab(d):
    rows = []
    for lab, g in d.groupby("source"):
        rows.append(dict(source=lab, n=len(g), mice=g.mouse.nunique(), n_ages=g.age_days.nunique(),
                         age_min=g.age_days.min(), age_max=g.age_days.max(),
                         MAE=np.abs(g.pred - g.age_days).mean(), bias=(g.pred - g.age_days).mean(),
                         spearman_within_lab=spearmanr(g.pred, g.age_days)[0] if g.age_days.nunique() > 1 else np.nan,
                         within_session_conc=within_folder_concordance(g.age_days.values, g.pred.values, g.folder.values)))
    return pd.DataFrame(rows)


def gap_table(d):
    rows = []
    for f, g in d.groupby("folder"):
        a, p = g.age_days.values, g.pred.values
        i, j = np.triu_indices(len(g), 1)
        gap = np.abs(a[i] - a[j]); k = gap > 0
        s = np.sign(a[i] - a[j]) * np.sign(p[i] - p[j])
        rows.append(pd.DataFrame({"gap": gap[k], "younger": np.minimum(a[i], a[j])[k], "c": ((s + 1) / 2)[k]}))
    r = pd.concat(rows)
    r["gap_days"] = pd.cut(r.gap, [0, 100, 200, 300, 500, 1100]).astype(str)
    r["younger_mouse"] = pd.cut(r.younger, [0, 180, 400, 1100], labels=["<6 mo", "6-13 mo", ">13 mo"]).astype(str)
    return r.groupby(["younger_mouse", "gap_days"]).c.agg(concordance="mean", pairs="size").reset_index()


def longitudinal(s, boot):
    """Same mouse at several ages (session level)."""
    m = s.groupby("mouse").age_days.nunique()
    L = s[s.mouse.isin(m[m > 1].index)]
    if L.empty:
        return pd.DataFrame(), {}
    pairs, slopes = [], []
    for mouse, g in L.groupby("mouse"):
        g = g.sort_values("age_days")
        a, p, f = g.age_days.values, g.pred.values, g.folder.values
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if a[j] > a[i]:
                    pairs.append(dict(mouse=mouse, source=g.source.iloc[0], age_1=a[i], age_2=a[j],
                                      pred_1=p[i], pred_2=p[j], same_session=f[i] == f[j],
                                      correct=float(np.sign(p[j] - p[i]) > 0) + 0.5 * (p[j] == p[i])))
        if len(np.unique(a)) > 1:
            slopes.append(np.polyfit(a, p, 1)[0])
    P = pd.DataFrame(pairs)
    mice = P.mouse.values
    lo, hi = boot_ci(lambda w: np.average(P.correct, weights=w) if w.sum() else np.nan, mice, boot)
    summ = dict(mice=P.mouse.nunique(), pairs=len(P), later_predicted_older=P.correct.mean(),
                ci_low=lo, ci_high=hi, pairs_across_sessions=float((~P.same_session).mean()),
                median_slope=np.median(slopes), frac_positive_slope=float(np.mean(np.array(slopes) > 0)),
                median_real_gap_days=float((P.age_2 - P.age_1).median()),
                median_predicted_gap_days=float((P.pred_2 - P.pred_1).median()))
    return P, summ


def calibration(s, ks=(0, 1, 3, 5, 10), reps=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for lab, g in s.groupby("source"):
        mice = g.mouse.unique()
        for k in ks:
            if k and k > len(mice) // 2:
                continue
            maes = []
            for _ in range(reps if k else 1):
                cal = rng.choice(mice, k, replace=False) if k else []
                c, t = g[g.mouse.isin(cal)], g[~g.mouse.isin(cal)]
                off = (c.age_days - c.pred).mean() if k else 0.0
                maes.append(np.abs(t.pred + off - t.age_days).mean())
            rows.append(dict(source=lab, k_mice=k, MAE=np.mean(maes), n_eval=len(g)))
    r = pd.DataFrame(rows)
    overall = (r.assign(w=r.MAE * r.n_eval).groupby("k_mice")[["w", "n_eval"]].sum()
               .pipe(lambda t: (t.w / t.n_eval).rename("MAE")).reset_index())
    overall["source"] = "ALL (weighted)"
    return pd.concat([r, overall], ignore_index=True)


def age_gap(s_all):
    """Age-adjusted gap: predicted age minus the prediction expected for a CONTROL mouse of that true age
    in the same lab (linear fit on controls). Removes the regression-to-the-mean artefact."""
    out = []
    ctrl_all = s_all[s_all.train_ok]
    glob = np.polyfit(ctrl_all.age_days, ctrl_all.pred, 1)
    for lab, g in s_all.groupby("source"):
        c = g[g.train_ok]
        if c.age_days.nunique() >= 3 and len(c) >= 10:
            coef, how = np.polyfit(c.age_days, c.pred, 1), "lab fit"
        elif len(c):
            coef = glob.copy(); coef[1] += (c.pred - np.polyval(glob, c.age_days)).mean(); how = "global slope + lab offset"
        else:
            coef, how = glob, "global fit (no controls in lab)"
        g = g.assign(expected_pred=np.polyval(coef, g.age_days), adjust_method=how)
        out.append(g)
    o = pd.concat(out)
    o["gap_raw_days"] = o.pred - o.age_days
    o["gap_adj_days"] = o.pred - o.expected_pred
    cols = [c for c in ["mouse", "id", "source", "folder", "strain", "sex", "treatment", "genotype", "train_ok",
                        "age_days", "n_photos", "pred", "expected_pred", "gap_raw_days", "gap_adj_days",
                        "adjust_method", "file"] if c in o]
    return o[cols].rename(columns={"pred": "pred_days", "file": "example_file"})


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="name=path[:column] or just path")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--split", choices=["source", "mouse"], help="override if the file has no `split` column")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 200)

    runs = [load_pred(p) for p in a.preds]
    summary, ref = [], None
    for name, d, split in runs:
        split = a.split or split or "source"
        ev = d[d.train_ok].reset_index(drop=True)
        levels = {"image": ev, "session": session_level(ev)}
        print(f"\n################ {name}  ({split} split; {len(ev)} control images, "
              f"{(~d.train_ok).sum()} excluded images not evaluated here)")
        for lvl, x in levels.items():
            h = headline(x, split, a.boot)
            summary.append(dict(model=name, level=lvl, split=split, **h))
            print(f"  {lvl:8s} n={h['n']:5d}  MAE {h['MAE']:6.1f} d (median {h['median_AE']:5.1f})  r {h['r']:.3f}  "
                  f"within-session conc {h['within_session_conc']:.3f} [{h['conc_ci_low']:.3f}, {h['conc_ci_high']:.3f}]  "
                  f"| baselines: folder-only {h['MAE_folder_only']:.0f} d, median guess {h['MAE_median_guess']:.0f} d")
        s = levels["session"]
        pl = per_lab(s); pl.to_csv(f"{a.out}/per_lab_{name}.csv", index=False)
        print("\n  per lab (session level):\n" + pl.round(2).to_string(index=False).replace("\n", "\n  "))
        gt = gap_table(s); gt.to_csv(f"{a.out}/gap_table_{name}.csv", index=False)
        print("\n  within-session concordance by age gap (session level):\n  "
              + gt.pivot(index="younger_mouse", columns="gap_days", values="concordance")
                  .reindex(["<6 mo", "6-13 mo", ">13 mo"]).round(2).to_string().replace("\n", "\n  "))
        P, ls = longitudinal(s, a.boot)
        if ls:
            P.to_csv(f"{a.out}/longitudinal_{name}.csv", index=False)
            print(f"\n  same mouse at several ages: {ls['mice']} mice, {ls['pairs']} pairs "
                  f"({ls['pairs_across_sessions']:.0%} across different sessions)")
            print(f"    later photo predicted older: {ls['later_predicted_older']:.3f} "
                  f"[{ls['ci_low']:.3f}, {ls['ci_high']:.3f}]; median real gap {ls['median_real_gap_days']:.0f} d, "
                  f"median predicted gap {ls['median_predicted_gap_days']:.0f} d; "
                  f"median slope {ls['median_slope']:.2f} (1 = perfect)")
            summary[-1].update({f"long_{k}": v for k, v in ls.items()})
        cal = calibration(s); cal.to_csv(f"{a.out}/calibration_{name}.csv", index=False)
        print("\n  MAE after correcting each lab's offset with k known-age mice (session level):\n  "
              + cal.pivot(index="source", columns="k_mice", values="MAE").round(0).to_string().replace("\n", "\n  "))
        ag = age_gap(session_level(d)); ag.to_csv(f"{a.out}/age_gap_{name}.csv", index=False)
        print(f"\n  age gaps for all {len(ag)} mouse-sessions (incl. excluded mice): {a.out}/age_gap_{name}.csv")

        # paired differences vs the first model, same images
        if ref is None:
            ref = (name, d)
        else:
            rn, rd = ref
            for lvl in ["image", "session"]:
                A = rd[rd.train_ok]; B = d[d.train_ok]
                if lvl == "session":
                    A, B = session_level(A), session_level(B)
                    key = ["mouse", "folder", "age_days"]
                else:
                    key = ["file"]
                M = A.merge(B[key + ["pred"]], on=key, suffixes=("_a", "_b"))
                y, pa, pb, f, mo = M.age_days.values, M.pred_a.values, M.pred_b.values, M.folder.values, M.mouse.values
                dc = within_folder_concordance(y, pb, f) - within_folder_concordance(y, pa, f)
                dci = boot_ci(lambda w: within_folder_concordance(y, pb, f, w) - within_folder_concordance(y, pa, f, w), mo, a.boot)
                dm = np.abs(pb - y).mean() - np.abs(pa - y).mean()
                dmi = boot_ci(lambda w: np.average(np.abs(pb - y), weights=w) - np.average(np.abs(pa - y), weights=w), mo, a.boot)
                summary.append(dict(model=f"{name} - {rn}", level=lvl, n=len(M), MAE=dm, MAE_ci_low=dmi[0], MAE_ci_high=dmi[1],
                                    within_session_conc=dc, conc_ci_low=dci[0], conc_ci_high=dci[1]))
                print(f"\n  vs {rn} ({lvl}, {len(M)} shared): MAE {dm:+.1f} d [{dmi[0]:+.1f}, {dmi[1]:+.1f}], "
                      f"within-session conc {dc:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")

    pd.DataFrame(summary).to_csv(f"{a.out}/summary.csv", index=False)
    print(f"\nall tables in {a.out}/")


if __name__ == "__main__":
    main()
