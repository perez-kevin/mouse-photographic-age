"""
Is the photographic age gap biologically meaningful?

Inputs
  --gap      age_gap_<model>.csv written by evaluate.py (one row per mouse x session, all mice,
             with the age-adjusted gap `gap_adj_days` = predicted age - prediction expected for a
             control mouse of the same true age in the same lab)
  --labels   the label table (to link frailty rows to photos/sessions)
  --frailty  frailty_all.xlsx: id, filename, Age, time-to-death, dead, Weight (g), score, source, type

Analyses (each written to --out as CSV and summarised on screen)
  Primary population: controls. Secondary: all mice incl. treated (frailty: not adjusted for treatment;
  survival: adjusted for treatment).
  A. Frailty / vitality and body weight vs age gap, per lab, adjusted for age (and sex if both present),
     standard errors clustered by mouse (repeated photos). Then a direction-aligned, within-lab
     standardised version (higher = worse health) combined across labs (fixed-effect meta-analysis).
  B. Survival from the photo: Cox model stratified by lab, adjusted for age (and sex), clustered by mouse.
     Does the gap predict death beyond chronological age? Beyond the lab's own frailty/vitality score?
     Sensitivity: first photo per mouse only.
  C. Groups excluded from training (drugs, progeroid strains) vs CONTROLS PHOTOGRAPHED IN THE SAME SESSION:
     difference in age gap (days), adjusted for age, clustered by mouse; Holm-adjusted p across groups.
  D. Literature images: predictions per image and within-figure ranking (needs a `paper` or `figure`
     column in the label table to know which images belong together; falls back to folder).

Primary analyses use control mice only (train_ok = True). Run this ONCE, on the model chosen beforehand
from control performance in evaluate.py.

  pip install statsmodels lifelines
  python biomarkers.py --gap eval_main/age_gap_finetuned.csv --labels labels_id_sex_clean.csv \
                       --frailty frailty_all.xlsx --out biomarkers
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr, norm

warnings.filterwarnings("ignore")
CONTROL = {"", "nan", "none", "control", "ctrl", "vehicle", "untreated", "wt", "wild-type", "wildtype"}


# ---------------------------------------------------------------- data
def load(a):
    g = pd.read_csv(a.gap)
    g["train_ok"] = g["train_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    L = pd.read_csv(a.labels, encoding="utf-8-sig")
    F = pd.read_excel(a.frailty) if a.frailty.endswith(("xlsx", "xls")) else pd.read_csv(a.frailty)
    F = F.rename(columns={"Weight (g)": "weight", "time-to-death": "ttd", "Age": "age_days"})
    n0 = len(F)
    F = F.drop_duplicates(["source", "filename", "age_days"])
    F.loc[F.score <= 0, "score"] = np.nan                       # 0 = missing
    F.loc[F.weight <= 0, "weight"] = np.nan
    F["age_days"] = F.age_days.astype(float)
    # link frailty row -> photo -> session (mouse, folder, age)
    k = F.merge(L[["source", "image", "age_days", "folder", "id"]].rename(columns={"image": "filename", "id": "id_photo"}),
                on=["source", "filename", "age_days"], how="inner")
    k["mouse"] = k.source.astype(str) + "|" + k.id_photo.astype(str)
    k = k.drop_duplicates(["mouse", "folder", "age_days"])
    print(f"frailty sheet: {n0} rows ({n0 - len(F)} duplicates dropped); {len(k)} linked to a photographed session")
    d = g.merge(k[["mouse", "folder", "age_days", "score", "type", "weight", "ttd", "dead"]],
                on=["mouse", "folder", "age_days"], how="left")
    unl = len(k) - len(k.merge(g[["mouse", "folder", "age_days"]], on=["mouse", "folder", "age_days"]))
    if unl > 0:
        print(f"  note: {unl} linked frailty rows have no prediction in the gap file (e.g. crop missing)")
    empty = int((k.score.isna() & k.weight.isna()).sum())
    if empty > 0:
        print(f"  note: {empty} linked frailty rows have neither score nor weight (survival only)")
    d["gap100"] = d.gap_adj_days / 100
    d["age100"] = d.age_days / 100
    d["mouse_code"] = pd.factorize(d.mouse)[0]
    d["treatment"] = d.get("treatment", pd.Series("", index=d.index)).fillna("").astype(str)
    d["treated"] = ~d.treatment.str.strip().str.lower().isin(CONTROL) & ~d.source.isin(a.paper_sources.split(","))
    return d


def ols(formula, data):
    m = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data.mouse_code})
    return m


def coef_row(m, term, **extra):
    ci = m.conf_int().loc[term]
    return dict(**extra, estimate=m.params[term], ci_low=ci[0], ci_high=ci[1], p=m.pvalues[term], n=int(m.nobs))


def meta(rows):
    """Fixed-effect inverse-variance meta-analysis of per-lab estimates."""
    b = np.array([r["estimate"] for r in rows]); se = np.array([(r["ci_high"] - r["ci_low"]) / (2 * 1.96) for r in rows])
    w = 1 / se ** 2; est = (w * b).sum() / w.sum(); s = np.sqrt(1 / w.sum())
    Q = (w * (b - est) ** 2).sum(); I2 = max(0.0, (Q - (len(b) - 1)) / Q) if Q > 0 else 0.0
    return dict(estimate=est, ci_low=est - 1.96 * s, ci_high=est + 1.96 * s, p=2 * norm.sf(abs(est / s)),
                n=int(sum(r["n"] for r in rows)), heterogeneity_I2=I2)


def sex_term(x):
    t = " + C(sex)" if "sex" in x and x.sex.nunique() > 1 else ""
    return t + (" + treated" if "treated" in x and x.treated.nunique() > 1 else "")


# ---------------------------------------------------------------- A. frailty / weight
def frailty_weight(d, population="controls"):
    rows, std_score, std_weight = [], [], []
    c = d[d.train_ok] if population == "controls" else d[d.train_ok | d.treated]
    c = c.assign(treated=False)                       # frailty: treated mice pooled with controls, no treatment term
    for lab, x in c[c.score.notna() | c.weight.notna()].groupby("source"):
        typ = x.type.dropna().iloc[0] if x.type.notna().any() else "score"
        sgn = -1 if str(typ).lower().startswith("vital") else 1         # aligned: higher = worse health
        xs = x[x.score.notna()].copy()
        if len(xs) > 10:
            m = ols(f"score ~ gap100 + age100{sex_term(xs)}", xs)
            r0 = smf.ols(f"score ~ age100{sex_term(xs)}", xs).fit().rsquared
            rows.append(coef_row(m, "gap100", outcome=f"{typ} score (raw units per 100 d gap)", lab=lab, population=population,
                                 mice=xs.mouse.nunique(), R2_age_only=r0, R2_with_gap=m.rsquared))
            xs["z"] = sgn * (xs.score - xs.score.mean()) / xs.score.std()
            r = coef_row(ols(f"z ~ gap100 + age100{sex_term(xs)}", xs), "gap100",
                         outcome="health deficit (SD, aligned) per 100 d gap", lab=lab, mice=xs.mouse.nunique(),
                         population=population)
            rows.append(r); std_score.append(r)
            ra = spearmanr(smf.ols(f"score ~ age100{sex_term(xs)}", xs).fit().resid,
                           smf.ols(f"gap100 ~ age100{sex_term(xs)}", xs).fit().resid)[0]
            rows[-2]["partial_spearman"] = ra
        xw = x[x.weight.notna()].copy()
        if len(xw) > 10:
            xw["wz"] = xw.groupby("sex" if "sex" in xw and xw.sex.notna().all() else "source").weight.transform(
                lambda v: (v - v.mean()) / v.std())
            r = coef_row(ols(f"wz ~ gap100 + age100{sex_term(xw)}", xw), "gap100",
                         outcome="body weight (SD within sex) per 100 d gap", lab=lab, mice=xw.mouse.nunique(),
                         population=population)
            rows.append(r); std_weight.append(r)
    if len(std_score) > 1:
        rows.append(dict(outcome="health deficit (SD, aligned) per 100 d gap", lab="ALL (meta-analysis)",
                         population=population, **meta(std_score)))
    if len(std_weight) > 1:
        rows.append(dict(outcome="body weight (SD within sex) per 100 d gap", lab="ALL (meta-analysis)",
                         population=population, **meta(std_weight)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B. survival
def survival(d, population="controls", exclude=()):
    from lifelines import CoxPHFitter
    pop = d.train_ok if population == "controls" else (d.train_ok | d.treated)
    s = d[pop & d.ttd.notna() & d.dead.notna() & ~d.source.isin(exclude)].copy()
    s["treated"] = s.treated.astype(int)
    if s.empty:
        return pd.DataFrame()
    s["time"] = s.ttd + 0.5                                   # deaths recorded at the photo -> tiny positive time
    s["event"] = s.dead.astype(int)
    s["female"] = (s.get("sex", "") == "Female").astype(int)
    labs_scores = s.groupby("source").score.transform(lambda v: (v - v.mean()) / v.std())
    sgn = np.where(s.type.astype(str).str.lower().str.startswith("vital"), -1, 1)
    s["deficit_z"] = sgn * labs_scores
    rows = []

    def fit(data, covs, label, strata=True, cluster=True):
        cols = ["time", "event"] + covs + (["source"] if strata else []) + (["mouse_code"] if cluster else [])
        x = data[cols].dropna()
        covs_used = [c for c in covs if x[c].nunique() > 1]
        x = x[["time", "event"] + covs_used + (["source"] if strata else []) + (["mouse_code"] if cluster else [])]
        cph = CoxPHFitter()
        cph.fit(x, "time", "event", strata=["source"] if strata else None,
                cluster_col="mouse_code" if cluster else None)
        sm = cph.summary
        for cv in covs_used:
            rows.append(dict(population=population, model=label, term=cv, HR=sm.loc[cv, "exp(coef)"], ci_low=sm.loc[cv, "exp(coef) lower 95%"],
                             ci_high=sm.loc[cv, "exp(coef) upper 95%"], p=sm.loc[cv, "p"],
                             n=len(x), events=int(x.event.sum()), mice=data.loc[x.index, "mouse"].nunique(),
                             concordance=cph.concordance_index_))

    base = ["age100"] + (["female"] if s.female.nunique() > 1 else []) + (["treated"] if s.treated.nunique() > 1 else [])
    fit(s, base, "1 age only")
    fit(s, base + ["gap100"], "2 age + photo age gap")
    if s.deficit_z.notna().sum() > 20:
        fit(s[s.deficit_z.notna()], base + ["deficit_z"], "3 age + frailty/vitality")
        fit(s[s.deficit_z.notna()], base + ["deficit_z", "gap100"], "4 age + frailty/vitality + photo age gap")
    first = s.sort_values("age_days").drop_duplicates("mouse")
    fit(first, base + ["gap100"], "5 sensitivity: first photo per mouse", cluster=False)
    for lab, x in s.groupby("source"):
        if x.event.sum() >= 10:
            fit(x, base + ["gap100"], f"6 per lab: {lab}", strata=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C. excluded groups vs same-session controls
def group_effects(d, paper_sources):
    x = d[~d.source.isin(paper_sources)].copy()
    tr = x.treatment.str.strip().str.lower()
    strain_ctrl = x[x.train_ok].groupby("folder").strain.agg(lambda s: s.mode().iloc[0])
    x["group"] = np.where(~tr.isin(CONTROL), x.treatment,
                          np.where(x.strain != x.folder.map(strain_ctrl), "strain: " + x.strain.astype(str), "control"))
    x.loc[x.train_ok, "group"] = "control"
    rows = []
    for f, g in x.groupby("folder"):
        groups = [k for k in g.group.unique() if k != "control"]
        if not groups or (g.group == "control").sum() < 5:
            continue
        for k in groups + (["ANY non-control (pooled)"] if len(groups) > 1 else []):
            sub = g[(g.group == "control") | ((g.group == k) if k != "ANY non-control (pooled)" else (g.group != "control"))].copy()
            sub["is_group"] = (sub.group != "control").astype(int)
            m = ols("gap_adj_days ~ is_group + age100", sub)
            rows.append(coef_row(m, "is_group", folder=f, group=k,
                                 mice_group=sub[sub.is_group == 1].mouse.nunique(),
                                 mice_control=sub[sub.is_group == 0].mouse.nunique(),
                                 mean_gap_group=sub[sub.is_group == 1].gap_adj_days.mean(),
                                 mean_gap_control=sub[sub.is_group == 0].gap_adj_days.mean()))
    r = pd.DataFrame(rows)
    if len(r):
        # Holm correction within each session's family of comparisons (e.g. the 7 drugs in one study);
        # a separate question in another session (e.g. TERC vs its controls) is its own family
        r["p_holm"] = np.nan
        for f, idx in r[~r.group.str.startswith("ANY")].groupby("folder").groups.items():
            r.loc[idx, "p_holm"] = multipletests(r.loc[idx, "p"], method="holm")[1]
    return r.rename(columns={"estimate": "gap_difference_days"})


# ---------------------------------------------------------------- D. literature images
def paper_images(d, labels_path, paper_sources, group_col):
    p = d[d.source.isin(paper_sources)].copy()
    if p.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    L = pd.read_csv(labels_path, encoding="utf-8-sig")
    col = group_col if group_col in L else ("figure" if "figure" in L else ("paper" if "paper" in L else "folder"))
    if col != "folder":
        p = p.merge(L[["file", col]].rename(columns={"file": "example_file"}), on="example_file", how="left")
    p["figure_group"] = p[col].map(lambda v: str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
    rows = []
    for f, g in p.groupby("figure_group"):
        a, q = g.age_days.values, g.pred_days.values
        i, j = np.triu_indices(len(g), 1); k = a[i] != a[j]
        conc = (((np.sign(a[i] - a[j]) * np.sign(q[i] - q[j]))[k] + 1) / 2).mean() if k.any() else np.nan
        rows.append(dict(figure_group=f, images=len(g), ages=sorted(set(a.round())), age_pairs=int(k.sum()),
                         within_figure_concordance=conc, MAE_days=np.abs(q - a).mean()))
    keep = [c for c in ["example_file", "figure_group", "treatment", "age_days", "pred_days", "gap_raw_days", "gap_adj_days"] if c in p]
    # treated vs control within the same figure (descriptive: few images per group)
    by_trt = (p.groupby(["figure_group", "age_days", "treatment"])
                .agg(images=("pred_days", "size"), mean_pred_days=("pred_days", "mean"),
                     mean_gap_raw_days=("gap_raw_days", "mean")).reset_index())
    return p[keep], pd.DataFrame(rows), by_trt


# ---------------------------------------------------------------- main
def show(title, df, cols):
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("  (no data)"); return
    print("  " + df[[c for c in cols if c in df]].round(3).to_string(index=False).replace("\n", "\n  "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frailty", required=True)
    ap.add_argument("--out", default="biomarkers")
    ap.add_argument("--paper-sources", default="PaperImages", help="source name(s) of literature images")
    ap.add_argument("--survival-exclude", default="", help="comma-separated sources left out of the survival models (B)")
    ap.add_argument("--paper-col", default="paper", help="label-table column grouping literature images by figure")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220)
    d = load(a)
    d.to_csv(f"{a.out}/merged_sessions.csv", index=False)
    papers = [s.strip() for s in a.paper_sources.split(",")]

    A = pd.concat([frailty_weight(d, "controls"), frailty_weight(d, "all mice, not adjusted for treatment")],
                  ignore_index=True); A.to_csv(f"{a.out}/A_frailty_weight.csv", index=False)
    show("A. Frailty / vitality and weight vs photographic age gap (controls; per 100 days of gap, age-adjusted)",
         A, ["population", "outcome", "lab", "mice", "n", "estimate", "ci_low", "ci_high", "p", "partial_spearman",
             "R2_age_only", "R2_with_gap", "heterogeneity_I2"])
    surv_excl = [x.strip() for x in a.survival_exclude.split(",") if x.strip()]
    def survival_safe(pop):
        try:
            return survival(d, pop, surv_excl)
        except Exception as e:                          # e.g. lifelines ConvergenceError with very few events
            print(f"WARNING: survival ({pop}) not estimated: {type(e).__name__}: {str(e)[:120]}")
            return pd.DataFrame()
    B = pd.concat([survival_safe("controls"), survival_safe("all mice, adjusted for treatment")],
                  ignore_index=True); B.to_csv(f"{a.out}/B_survival.csv", index=False)
    show("B. Survival from the photo (Cox, stratified by lab; HR per 100 days of gap / per SD of deficit / per 100 days of age)",
         B, ["population", "model", "term", "HR", "ci_low", "ci_high", "p", "mice", "events", "concordance"])
    C = group_effects(d, papers); C.to_csv(f"{a.out}/C_groups_vs_same_session_controls.csv", index=False)
    show("C. Treated / progeroid groups vs controls of the same session (age gap difference, days)",
         C, ["folder", "group", "mice_group", "mice_control", "gap_difference_days", "ci_low", "ci_high", "p", "p_holm"])
    Dimg, Dfig, Dtrt = paper_images(d, a.labels, papers, a.paper_col)
    if len(Dimg):
        Dimg.to_csv(f"{a.out}/D_paper_images.csv", index=False); Dfig.to_csv(f"{a.out}/D_paper_figures.csv", index=False)
        Dtrt.to_csv(f"{a.out}/D_paper_treatments.csv", index=False)
        show("D. Literature images, within-figure ranking", Dfig,
             ["figure_group", "images", "ages", "age_pairs", "within_figure_concordance", "MAE_days"])
        if Dtrt.groupby(["figure_group", "age_days"]).treatment.nunique().max() > 1:
            show("D2. Literature images: treated vs control at the same age in the same paper (descriptive, few images)",
                 Dtrt, ["figure_group", "age_days", "treatment", "images", "mean_pred_days", "mean_gap_raw_days"])
    print(f"\nall tables in {a.out}/")


if __name__ == "__main__":
    main()
