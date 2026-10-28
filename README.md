# Photographic age of laboratory mice

Code for *Photographic age gap is associated with frailty and survival in laboratory mice*
(Perez et al., in preparation).

A frozen self-supervised vision model (DINOv2 ViT-B/14) reads a photograph of a mouse, and a ridge
regression turns its features into a predicted age. The model was trained on ~2,650 photographs of
~1,200 control mice from seven facilities and always tested on facilities it had never seen.

**What it is good for:** ranking mice photographed together (same session, same setup) by how old they
look, and the *photographic age gap* (predicted age minus the age expected for a control mouse of that age
in that facility) as a cheap, non-invasive biomarker.
**What it is not:** an absolute age clock. Each facility's photographs shift all predictions (by up to
several months), and predictions are compressed towards the middle of the age range.

## Use the model on your own photos

```bash
pip install -r requirements.txt
python predict.py my_photos/ --out predictions.csv
```

- `environment_paper.txt` lists the exact package versions (and GPU) used for the paper's results.
- One mouse per photo, viewed from above or the side, whole body visible.
- Put each photo **session** (same day, same camera and background) in its own sub-folder of `my_photos/`.
- The two models made for this paper come with the repository in `release/` and are found automatically:
  `MouseAge2_YOLO_Best_050925.pt` (mouse detector, 3.9 MB) and `mouse_age_ridge448.joblib` (age model: ridge on
  DINOv2 features, fitted on 2,651 photos of 1,205 control mice from 7 facilities, ages 35–1,037 days). They are also
  archived with the dataset on Zenodo. Use `--yolo` / `--model` to point to other copies. The age model was saved with
  scikit-learn 1.6.1; newer versions load it with a version warning. The first run downloads YOLOv5,
  Grounding DINO, SAM ViT-H and DINOv2 (~3 GB). A GPU (CUDA) or Apple-silicon GPU (MPS) is strongly recommended.

`predictions.csv` has one row per photo: `pred_age_days`, `rank_in_session` (1 = looks youngest in its
session), `n_in_session`, and `crop_flags` from the masking step. The intermediate crops are written to
`predict_work/` so you can check that each mouse was cut out correctly (`predict_work/masked/`).

> **Use `pred_age_days` as a score for ranking mice within a session, not as an age.**
> In facilities the model had never seen, the older of two mice photographed together was predicted older in
> about 65% of pairs (50% = chance), and in almost all pairs more than ~16 months apart. Absolute errors were
> around 5 months on average.

## Pipeline (as in the paper)

| Step | Script | What it does |
|---|---|---|
| 1 | `downscale_raw.py` | applies the camera rotation tag, converts to sRGB, longest side ≤ 2048 px |
| 2 | `yolo_crop.py` | YOLOv5n fine-tuned on 247 hand-labelled photos → crop around the mouse (+15 % margin) |
| 3 | `crop_mice.py` | Grounding DINO ("a mouse.") → SAM ViT-H mask (of SAM's 3 masks, the one with the best predicted IoU) → clean-up → flat grey background, 512 px square |
| 4 | `train_age.py embed` | frozen DINOv2 ViT-B/14 at 448 px, flip TTA → StandardScaler + RidgeCV; held-out predictions, leave-one-facility-out (`--split source`) or 5-fold by mouse (`--split mouse`) |
| 5 | `evaluate.py` | MAE, correlation, within-session concordance (bootstrap CI over mice), longitudinal pairs, calibration, age gap per mouse and session |
| 6 | `biomarkers.py` | age gap vs frailty / vitality and body weight (OLS, clustered by mouse, meta-analysis), survival (Cox, stratified by facility), groups vs same-session controls |
| 7 | `compare_tasks.py`, `compare_crops.py` | age vs sex from the same features; masked vs unmasked crops |
| 8 | `make_rating_task.py`, `score_ratings.py` | human-expert rating task and scoring against the model |
| 9 | `saliency_age.py`, `region_saliency.py` | occlusion saliency (grey / blur) and per-body-region summaries |
| 10 | `make_figures.py` | all figures, tables and `key_numbers.csv` from the result files |
| – | `export_model.py` | fits the final model on all training mice → `mouse_age_ridge448.joblib` (used by `predict.py`) |
| – | `colab_pipeline.sh` | runs steps 4–10 end to end on a GPU (written for Google Colab, `/content/work`) |

### Label table

All scripts read one CSV (`--csv`), one row per photo:

| column | meaning |
|---|---|
| `file` | path of the crop relative to `--root`, e.g. `SessionA/mouse12_1.jpg` |
| `age_days` | age at the photo |
| `folder` | photo session (pairs for within-session concordance are formed inside a folder) |
| `id`, `source` | mouse identifier (unique within a facility) and facility |
| `sex`, `strain` | optional, used for the sex / strain analyses |
| `training` | `yes` / `no`: `no` = predicted by models that never saw it, not trained on, not in the main evaluation |
| `treatment` | optional; anything other than empty / `Control` is excluded from training |

Photos of the same mouse in the same session are averaged before metrics. The mouse key is `source|id`.

### Example: rebuild the results

```bash
python downscale_raw.py raw/ downscaled/
python yolo_crop.py downscaled/ yolo_crops/
python crop_mice.py --csv labels.csv --root yolo_crops --out crops
python train_age.py embed --csv labels.csv --root crops --split source --size 448 --pred-out oof_frozen448_source.csv
python train_age.py embed --csv labels.csv --root crops --split mouse  --size 448 --pred-out oof_frozen448_mice.csv
python evaluate.py frozen448_source=oof_frozen448_source.csv frozen448_mice=oof_frozen448_mice.csv --out eval_res
python biomarkers.py --gap eval_res/age_gap_frozen448_source.csv --labels labels.csv --frailty frailty.xlsx --out biomarkers_embed
```
or, on a GPU machine with everything in `/content/work`: `bash colab_pipeline.sh` (`SMOKE=1` for a quick check).

## Notes

- Fine-tuning the network end to end, or with a ridge-initialised head and frozen early blocks, did not
  transfer to new facilities as well as the frozen model; only the frozen model is provided.
- Hardware: embeddings run on CPU, CUDA or Apple MPS (fp32 on MPS). Set `HF_HOME` to a local (not
  cloud-synced) folder: model files on synced folders can get corrupted.
- The photographs are not included in this repository (see Data availability in the paper).

## Citation

Perez K. et al. Photographic age gap is associated with frailty and survival in laboratory mice. In preparation.

## Licence

Code and the age model (`mouse_age_ridge448.joblib`): MIT (see `LICENSE`). The mouse detector
(`MouseAge2_YOLO_Best_050925.pt`) is fine-tuned from YOLOv5 and therefore under AGPL-3.0. The dataset (Zenodo) is
CC BY 4.0. YOLOv5, Grounding DINO, SAM and DINOv2 are downloaded at run time under their own licences.
