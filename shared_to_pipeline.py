"""
Turn the shared dataset (labels.csv, frailty.csv) into the input format of the analysis scripts.

  python shared_to_pipeline.py data_sharing --out inputs
  -> inputs/labels_id_sex_clean.csv   (file, age_days, folder, image, sex, id, strain, source, training, treatment, cohort)
  -> inputs/frailty_all.csv           (id, filename, Age, time-to-death, dead, Weight (g), score, source, type)
  -> inputs/rating_task/              (answer_key.csv with folder / source / mouse, ratings_rater*.csv)

Uses only the shared files: facility names come from the published facility key (FACILITY in make_figures.py),
and mice outside training that are not telomerase-deficient (strain TERC) are flagged as treated, which is how
the frailty analysis includes them. Then run the pipeline with --root data_sharing/crops_masked.
"""
import argparse, os
import pandas as pd

NAME = {1: "Malavolta", 2: "IRCAN", 3: "Vivo", 4: "EPITERNA", 5: "IFOM", 6: "Miller", 7: "IOR"}

ap = argparse.ArgumentParser()
ap.add_argument("shared", help="data_sharing folder")
ap.add_argument("--out", default="inputs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

S = pd.read_csv(os.path.join(a.shared, "labels.csv"))
cohort = S.cohort.fillna("") if "cohort" in S else pd.Series("", index=S.index)
L = pd.DataFrame({"file": S.file, "age_days": S.age_days, "folder": S.session, "image": S.file.str.split("/").str[-1],
                  "sex": S.sex, "id": S.mouse_id, "strain": S.strain, "source": S.facility.map(NAME), "training": S.training,
                  "treatment": ((S.training == "no") & (S.strain != "TERC")).map({True: "treated", False: ""}), "cohort": cohort})
L.to_csv(os.path.join(a.out, "labels_id_sex_clean.csv"), index=False, encoding="utf-8-sig")
print(f"labels: {len(L)} photos, {(L.source + '|' + L.id.astype(str)).nunique()} mice, {(L.treatment == 'treated').sum()} treated")

fp = os.path.join(a.shared, "frailty.csv")
if os.path.exists(fp):
    F = pd.read_csv(fp)
    out = pd.DataFrame({"id": F.mouse_id, "filename": F.file.str.split("/").str[-1], "Age": F.age_days,
                        "time-to-death": F.days_to_death, "dead": F.dead, "Weight (g)": F.weight_g, "score": F.score,
                        "source": F.facility.map(NAME), "type": F.score_type})
    out.to_csv(os.path.join(a.out, "frailty_all.csv"), index=False)
    print(f"frailty: {len(out)} rows")

rp = os.path.join(a.shared, "ratings")
if os.path.isdir(rp):
    import shutil
    os.makedirs(os.path.join(a.out, "rating_task"), exist_ok=True)
    K = pd.read_csv(os.path.join(rp, "answer_key.csv"))
    K["folder"] = K.session; K["source"] = K.facility.map(NAME); K["mouse"] = K.source + "|" + K.mouse_id.astype(str)
    K.to_csv(os.path.join(a.out, "rating_task", "answer_key.csv"), index=False)
    for f in os.listdir(rp):
        if f.startswith("ratings_"):
            shutil.copyfile(os.path.join(rp, f), os.path.join(a.out, "rating_task", f))
    print(f"ratings: answer key {len(K)} rows + {sum(f.startswith('ratings_') for f in os.listdir(rp))} raters")
