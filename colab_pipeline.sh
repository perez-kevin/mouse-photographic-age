#!/bin/bash
# Full rebuild on a Colab GPU, from masked crops to every figure:
#   embeddings + ridge (448 source / mouse, 224) -> evaluate -> biomarkers -> sex & background controls
#   -> human baseline -> occlusion saliency (grey, blur) -> region saliency -> make_figures.py
#   bash colab_pipeline.sh            full run          (outputs in /content/work/run)
#   SMOKE=1 bash colab_pipeline.sh    ~12 mice per facility, few bootstraps, just checks every step runs
#   BLUR=1 bash colab_pipeline.sh     also blur-occlusion saliency (region tables only; +~15 min)
#   CROPS / YOLO_DIR: masked / YOLO crop folders under /content/work; PIPE: Fig. 1a photo; ROTATE: Fig. 1b rotation file;
#   FRAILTY: frailty file in inputs/ (frailty_all.csv from shared_to_pipeline.py); BOOT_CROPS: bootstraps for step 7;
#   PIPE_TILES: "a.png b.png c.png" hand-cropped Fig. 1a images (photo with box, YOLO crop, mask), paths under /content/work
# Expects /content/work with the scripts, inputs/ (labels, frailty_all.xlsx, rating_task/), $CROPS (masked
# crops, default crops_sampick from colab_sampick.sh), yolo_crops/ and raw_vivo/ (the Fig. 1a photo).
set -euo pipefail
W=/content/work; OUT=$W/${OUT_NAME:-run}; [ -n "${SMOKE:-}" ] && OUT=$W/smoke
export HF_HOME=/content/hf
mkdir -p "$OUT"; cd "$OUT"
LABELS=$W/inputs/labels_id_sex_clean.csv; BOOT=200; HBOOT=1000; PERLAB=30
if [ -n "${SMOKE:-}" ]; then
  BOOT=5; HBOOT=5; PERLAB=2
  python - "$LABELS" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig"); d["m"] = d.source + "|" + d.id.astype(str)
keep = d.groupby("source").m.apply(lambda s: s.drop_duplicates().sample(min(12, s.nunique()), random_state=0)).explode()
d[d.m.isin(set(keep))].drop(columns="m").to_csv("labels_smoke.csv", index=False, encoding="utf-8-sig")
print("smoke labels:", d.m.isin(set(keep)).sum(), "images")
EOF
  LABELS=$OUT/labels_smoke.csv
fi
cp "$LABELS" labels_id_sex_clean.csv          # make_figures reads it from --data
S=$W; SAM=$W/${CROPS:-crops_sampick}; YOLO=$W/${YOLO_DIR:-yolo_crops}
step() { echo; echo "=== $(date +%T) $*"; }

# exact software versions of this run (for reproducing the figures)
{ python --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; pip freeze; } > environment.txt
python -c "import torch, timm, transformers, sklearn; print('torch', torch.__version__, '| timm', timm.__version__, '| transformers', transformers.__version__, '| scikit-learn', sklearn.__version__)"
step "1/12 embed 448, leave-one-facility-out"
python $S/train_age.py embed --csv labels_id_sex_clean.csv --root $SAM --split source --size 448 --workers 8 --pred-out oof_frozen448_source.csv
step "2/12 embed 448, 5-fold by mouse (reuses cached embeddings)"
python $S/train_age.py embed --csv labels_id_sex_clean.csv --root $SAM --split mouse --size 448 --workers 8 --pred-out oof_frozen448_mice.csv
step "2b export the final model (all training mice) for predict.py"
python $S/export_model.py --csv labels_id_sex_clean.csv --root $SAM --out mouse_age_ridge448.joblib --workers 8
step "3/12 embed 224, leave-one-facility-out (Supp. Fig. 1)"
python $S/train_age.py embed --csv labels_id_sex_clean.csv --root $SAM --split source --size 224 --workers 8 --pred-out oof_frozen224.csv
step "4/12 evaluate"
python $S/evaluate.py frozen448_source=oof_frozen448_source.csv frozen448_mice=oof_frozen448_mice.csv \
    frozen224=oof_frozen224.csv --out eval_res --boot $BOOT
step "5/12 biomarkers"
python $S/biomarkers.py --gap eval_res/age_gap_frozen448_source.csv --labels labels_id_sex_clean.csv \
    --frailty $W/inputs/${FRAILTY:-frailty_all.xlsx} --out biomarkers_embed --survival-exclude EPITERNA
step "6/12 age vs sex (224)"
python $S/compare_tasks.py --csv labels_id_sex_clean.csv --root $SAM --split source --workers 8 --boot $BOOT
step "7/12 masked vs YOLO crop with background (224)"
python $S/compare_crops.py --csv labels_id_sex_clean.csv --raw $YOLO --crops $SAM --split source --workers 8 --boot ${BOOT_CROPS:-$BOOT}
step "8/12 human baseline (all raters)"
python $S/score_ratings.py --key $W/inputs/rating_task/answer_key.csv --ratings "$W/inputs/rating_task/ratings_*.csv" \
    --model oof_frozen448_source.csv --out human_baseline --boot $HBOOT
step "9/12 occlusion saliency, grey (448, $PERLAB per facility)"
python $S/saliency_age.py --csv labels_id_sex_clean.csv --root $SAM --mode grey --size 448 --per-lab $PERLAB --workers 8 --out saliency_grey
SAL="grey=saliency_grey"
if [ -n "${BLUR:-}" ]; then        # blur maps only feed the region tables; the figures use grey
  step "10/12 occlusion saliency, blur"
  python $S/saliency_age.py --csv labels_id_sex_clean.csv --root $SAM --mode blur --size 448 --per-lab $PERLAB --workers 8 --out saliency_blur
  SAL="$SAL blur=saliency_blur"
fi
step "11/12 region saliency"
python $S/region_saliency.py --sal $SAL --root $SAM --csv labels_id_sex_clean.csv \
    --size 448 --out saliency_regions
step "12/12 figures"
python $S/make_figures.py --data . --regions saliency_regions/regions_per_image.csv --sal grey=saliency_grey \
    --crops-root $SAM --raw-root $W/raw_vivo --yolo-root $YOLO --pipeline ${PIPE:-Vivo-211026_cropped/MOU_000417_2.jpg} \
    --survival-exclude EPITERNA ${ROTATE:+--rotate $W/$ROTATE} ${PIPE_TILES:+--pipeline-tiles $PIPE_TILES} --out figures
step "zip results (no embedding caches)"
(cd $W && rm -f results.zip && zip -qr results.zip $(basename $OUT) -x "*.npy")
step "PIPELINE FINISHED: $OUT"
