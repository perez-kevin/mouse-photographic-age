"""
Build single figures from evaluate.py output, without the inputs the other figures need.

  python figs_only.py --figs 2 --data RUN_FOLDER --out figures
  python figs_only.py --figs 5 --data RUN_FOLDER --out figures      (needs biomarkers_embed/)
  python figs_only.py --figs 1 --data RUN_FOLDER --crops-root CROPS --raw-root RAW --yolo-root YOLO --pipeline FILE
RUN_FOLDER needs labels_id_sex_clean.csv and eval_res/ (frozen448_source; Fig. 2 also frozen448_mice).
Fig. 1 options are the same as in make_figures.py.
"""
import argparse, os
import pandas as pd
import make_figures as mf

ap = argparse.ArgumentParser()
ap.add_argument("--figs", nargs="+", choices=["1", "2", "5"], required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="figures")
ap.add_argument("--crops-root"); ap.add_argument("--raw-root"); ap.add_argument("--yolo-root")
ap.add_argument("--examples"); ap.add_argument("--pipeline"); ap.add_argument("--rotate")
ap.add_argument("--pipeline-tiles", nargs=3)
ap.add_argument("--survival-exclude", default="EPITERNA", help="Fig. 5: sources left out of the survival panels")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
p = lambda *x: os.path.join(a.data, *x)
r = lambda *x: pd.read_csv(p(*x), encoding="utf-8-sig")
files = dict(labels=("labels_id_sex_clean.csv",), summary=("eval_res", "summary.csv"),
             gap_source=("eval_res", "age_gap_frozen448_source.csv"), gap_mice=("eval_res", "age_gap_frozen448_mice.csv"),
             perlab_source=("eval_res", "per_lab_frozen448_source.csv"), perlab_mice=("eval_res", "per_lab_frozen448_mice.csv"),
             gaptab_source=("eval_res", "gap_table_frozen448_source.csv"),
             long_source=("eval_res", "longitudinal_frozen448_source.csv"),
             calib_source=("eval_res", "calibration_frozen448_source.csv"),
             A=("biomarkers_embed", "A_frailty_weight.csv"), B=("biomarkers_embed", "B_survival.csv"),
             C=("biomarkers_embed", "C_groups_vs_same_session_controls.csv"), merged=("biomarkers_embed", "merged_sessions.csv"))
D = {k: mf.anonymise(r(*f)) for k, f in files.items() if os.path.exists(p(*f))}
if "1" in a.figs:
    mf.fig1(D, a.out, a.crops_root, a.raw_root, a.yolo_root, a.examples, a.pipeline, a.rotate, a.pipeline_tiles)
if "2" in a.figs:
    mf.fig2(D, a.out)
if "5" in a.figs:                                       # Fig. 5 is drawn by fig4() (biomarkers)
    mf.fig4(D, a.out, "all", "controls", [mf.F(x.strip()) for x in a.survival_exclude.split(",") if x.strip()])
pd.DataFrame(mf.KEY).to_csv(os.path.join(a.out, f"key_numbers_fig{''.join(a.figs)}.csv"), index=False)
print(f"done: {a.out}/")
