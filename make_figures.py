"""
Build all manuscript figures and supplementary tables from the saved result files.
Nothing is re-trained here: every panel reads a CSV written by evaluate.py, biomarkers.py,
compare_tasks.py, compare_crops.py, score_ratings.py or region_saliency.py.

  python make_figures.py --data Archive --regions regions_per_image.csv --out figures

--data is the folder with:
  eval_res/                  evaluate.py outputs (frozen448_source, frozen448_mice, frozen224)
  biomarkers_embed/          biomarkers.py outputs
  compare_tasks_source.csv   age vs sex predictions (leave-one-lab-out)
  compare_crops_source.csv   cropped vs original predictions (leave-one-lab-out)
  human_baseline/            score_ratings.py outputs
  labels_id_sex_clean.csv
Optional: --regions regions_per_image.csv (region_saliency.py)
          --crops-root / --raw-root   photos for Fig. 1 (pipeline, examples) and Fig. 6 (saliency examples)
          --sal grey=DIR blur=DIR     saliency_age.py outputs for Fig. 6a
          --finetune eval_main        frozen vs fine-tuned evaluation (Fig. S2)

Outputs (--out): fig1..fig6 (fig4 = sex/strain) and figS1 (resolution) as .pdf + .png, tables/*.csv, key_numbers.csv
(every number quoted in the text, with its source file, so the text can be checked against it).
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------- style
BLUE, ORANGE, AQUA, GREY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#d6d5cf"
INK, INK2 = "#0b0b0b", "#52514e"
MM = 1 / 25.4
FULL, HALF = 183 * MM, 89 * MM
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
})
# Facilities are anonymised in every figure and table (numbered by number of photos, largest first);
# the key is written to tables/facility_key.csv for the authors only - the text cites the sources.
FACILITY = {"Malavolta": "Facility 1", "IRCAN": "Facility 2", "Vivo": "Facility 3", "EPITERNA": "Facility 4",
            "IFOM": "Facility 5", "Miller": "Facility 6", "IOR": "Facility 7", "PaperImages": "Published images"}
F = lambda name: FACILITY.get(name, name)
LAB_ORDER = [F(k) for k in FACILITY if k != "PaperImages"]


def anonymise(df):
    """Replace lab names in the columns that are displayed."""
    df = df.copy()
    for c in ["source", "lab", "group"]:
        if c in df and df[c].dtype == object:
            df[c] = df[c].map(lambda v: F(v) if isinstance(v, str) else v)
    if "model" in df and df["model"].dtype == object:
        df["model"] = df["model"].map(lambda v: v if not isinstance(v, str) else
                                      next((v.replace(k, F(k)) for k in FACILITY if k in v), v))
    return df
KEY = []                                                             # key numbers for the text


def key(name, value, source, note=""):
    KEY.append(dict(quantity=name, value=value, source=source, note=note))


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def placeholder(ax, text):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle((0, (3, 2))); s.set_edgecolor(GREY)
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK2, transform=ax.transAxes, wrap=True)


# ---------------------------------------------------------------- statistics
def pairs_within(group, y):
    """All pairs (i<j) in the same group with different y. Returns i, j, sign(y_i - y_j)."""
    I, J = [], []
    for _, idx in pd.Series(np.arange(len(group))).groupby(np.asarray(group)):
        idx = idx.values
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        I.append(idx[a]); J.append(idx[b])
    if not I:
        return np.array([], int), np.array([], int), np.array([])
    I, J = np.concatenate(I), np.concatenate(J)
    s = np.sign(np.asarray(y)[I] - np.asarray(y)[J])
    k = s != 0
    return I[k], J[k], s[k]


def concordance(y, p, group, mouse=None, n_boot=1000, seed=0):
    """Within-group pairwise concordance (0.5 = chance); 95 % CI from resampling mice."""
    I, J, s = pairs_within(group, y)
    if len(I) == 0:
        return np.nan, np.nan, np.nan, 0
    p = np.asarray(p, float)
    c = (s * np.sign(p[I] - p[J]) + 1) / 2
    est = c.mean()
    if mouse is None or n_boot == 0:
        return est, np.nan, np.nan, len(I)
    mice, inv = np.unique(np.asarray(mouse), return_inverse=True)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice))[inv].astype(float)
        ww = w[I] * w[J]
        bs.append((ww * c).sum() / ww.sum() if ww.sum() else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi, len(I)


def boot_mean(v, groups=None, n=2000, seed=0):
    """Mean with 95 % CI; resamples `groups` (e.g. labs) when given, else individual values."""
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    if groups is None:
        v = v[~np.isnan(v)]
        b = rng.choice(v, (n, len(v))).mean(1)
    else:
        groups = np.asarray(groups); G = {g: v[groups == g] for g in np.unique(groups)}
        keys = list(G); b = []
        for _ in range(n):
            x = np.concatenate([G[k] for k in rng.choice(keys, len(keys))])
            b.append(np.nanmean(x))
    return np.nanmean(v), *np.nanpercentile(b, [2.5, 97.5])


def perf_sessions(d):
    """Restrict an age_gap_*.csv table to the evaluation set (control mice, real labs)."""
    d = d[d.source != "PaperImages"]
    if "train_ok" in d:
        d = d[d.train_ok.astype(str).str.lower().isin(["true", "1"])]
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- images
def pad_square(path, size=300, fill=(255, 255, 255), rotate=0):
    """Whole image, aspect kept, centred on a square of `fill` (the figure background): no crop, no stretch.
    rotate: degrees clockwise (e.g. 90 to stand a sideways mouse upright)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if rotate:
        im = im.rotate(-rotate, expand=True)
    return np.asarray(ImageOps.pad(im, (size, size), color=fill, method=Image.LANCZOS))


def model_frame(path, size=448):
    """Same framing as the model input (train_age.tfms): pad to square with the corner colour, resize."""
    from PIL import Image, ImageOps
    im = Image.open(path).convert("RGB")
    w, h = im.size; a = np.asarray(im)
    fill = tuple(int(c) for c in a[[0, -1]][:, [0, -1]].reshape(-1, 3).mean(0))
    return np.asarray(ImageOps.pad(im, (max(w, h), max(w, h)), color=fill).resize((size, size), Image.BILINEAR))


def heat_overlay(img, heat, vmax):
    """Same colouring as saliency_age.py: greyscale photo, red = pushes age up, blue = down."""
    v = np.clip(heat / max(vmax, 1e-6), -1, 1)
    col = np.where(v[..., None] > 0, np.array([215, 35, 35]), np.array([35, 85, 215]))
    alpha = (np.abs(v) ** 0.8 * 0.7)[..., None]
    base = img.mean(2, keepdims=True).repeat(3, 2)
    return (base * (1 - alpha) + col * alpha).astype(np.uint8)


def show(ax, img, label=None, color=INK):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if label:
        ax.set_title(label, fontsize=6, color=color, pad=2)


class RawIndex:
    """Find the original photo for a label `file` under raw_root: same relative path if it exists,
    otherwise the file with the same name (any extension) whose folder name best matches."""

    def __init__(self, root):
        self.root, self.by_stem = root, {}
        for d, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    self.by_stem.setdefault(os.path.splitext(f)[0].lower(), []).append(os.path.join(d, f))

    def find(self, rel):
        p = os.path.join(self.root, rel)
        if os.path.exists(p):
            return p
        cands = self.by_stem.get(os.path.splitext(os.path.basename(rel))[0].lower(), [])
        if not cands:
            return None
        folder = rel.split("/")[0].lower()
        return max(cands, key=lambda c: len(os.path.commonprefix([folder, os.path.basename(os.path.dirname(c)).lower()])))


def load_thumb(path, side=600):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); im.thumbnail((side, side))
    return np.asarray(im)


def pick_examples(L, ok_path, override=None):
    """Youngest and oldest control mouse per facility (first photo at that age), or the files listed in
    `override` (text file, one relative path per line). ok_path(file) -> path or None. {facility: [(path, age)]}"""
    L = L[(L.source != F("PaperImages"))].copy()
    if "training" in L:
        L = L[~L.training.astype(str).str.lower().isin(["no", "false", "0"])]
    out = {}
    if override:
        for f in [l.strip() for l in open(override) if l.strip()]:
            r = L[L.file == f]; p = ok_path(f)
            if len(r) and p:
                out.setdefault(r.source.iloc[0], []).append((p, r.age_days.iloc[0], r.strain.iloc[0]))
    else:
        for lab, g in L.groupby("source"):
            g = g[g.strain.ne("TERC")]
            g = g[g.strain == g.strain.value_counts().index[0]]  # same strain for young and old: the facility's main one
            g = g.sort_values(["age_days", "file"])
            picks = []
            for part in (g, g.iloc[::-1]):                      # youngest first, then oldest
                for f, age, st in zip(part.file, part.age_days, part.strain):
                    p = ok_path(f)
                    if p:
                        picks.append((p, age, st)); break
            out[lab] = picks
    return {k: sorted(v, key=lambda x: x[1]) for k, v in out.items()}


def load_square(path, side=400):
    """Square tile: centre crop of the photo (for full photos; pad YOLO crops with model_frame instead)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(ImageOps.fit(im, (side, side), Image.LANCZOS))


def load_resized(path, side=400):
    """Square tile: the whole image resized to side x side (aspect ratio not kept)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im.resize((side, side), Image.LANCZOS))


def locate_crop(photo, yolo, thumb_w):
    """Where the YOLO crop sits in the photo (template matching), as (x, y, w, h) in thumbnail pixels."""
    import cv2
    from PIL import Image, ImageOps
    P = np.asarray(ImageOps.exif_transpose(Image.open(photo)).convert("L"), np.float32)
    Y = np.asarray(Image.open(yolo).convert("L"), np.float32)
    s = 0.25                                                   # match at quarter size: fast, plenty accurate here
    Ps, Ys = cv2.resize(P, None, fx=s, fy=s), cv2.resize(Y, None, fx=s, fy=s)
    if Ys.shape[0] > Ps.shape[0] or Ys.shape[1] > Ps.shape[1]:
        return None
    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(Ps, Ys, cv2.TM_CCOEFF_NORMED))
    if score < 0.8:
        return None
    f = thumb_w / P.shape[1]
    return (x / s * f, y / s * f, Y.shape[1] * f, Y.shape[0] * f)


def draw_pipeline(ax, raw=None, yolo=None, crop=None, raw_box=None, same_height=False):
    """photo -> YOLO square crop -> mask -> frozen DINOv2 -> ridge -> age, drawn in axes coordinates."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [("Photo", "downscaled,\nlongest side ≤2048 px", raw),
             ("YOLO crop", "fine-tuned detector,\nbox around mouse", yolo),
             ("Mask", "Grounding DINO\n+ SAM, flat grey\nbackground", crop),
             ("DINOv2", "ViT-B/14,\nfrozen, 448 px\n768 features", None),
             ("Ridge", "regression\ntrained on other\nfacilities", None),
             ("Age", "predicted age;\ngap = predicted\n− expected\nfor age", None)]
    n = len(steps); rel = np.array([1.35, 1, 1, 1, 1, 1])        # the whole photo gets a wider slot
    gap = 0.008; ws = (0.94 - gap * (n - 1)) * rel / rel.sum()
    img_h = 0.6
    if same_height:      # given (hand-cropped) tiles: same height, each slot as wide as its image needs;
        bb = ax.get_position(); fw, fh = ax.figure.get_size_inches(); phys = (bb.height * fh) / (bb.width * fw)
        asp = [im.shape[1] / im.shape[0] if im is not None else 0 for im in (raw, yolo, crop)]
        box_w = 0.135                                            # text boxes keep at least this width
        img_h = min(0.6, (0.94 - gap * (n - 1) - 3 * box_w - 0.036) / (sum(asp) * phys))
        img_w = [a_ * img_h * phys + 0.012 for a_ in asp]
        ws = np.array(img_w + [(0.94 - gap * (n - 1) - sum(img_w)) / 3] * 3)
    xs = 0.01 + np.concatenate([[0], np.cumsum(ws[:-1] + gap)])
    for k, (x, w, (title, sub, img)) in enumerate(zip(xs, ws, steps)):
        if img is not None:
            h = 0.64 if k == 0 and not same_height else img_h          # photo slightly taller, upwards only
            ia = ax.inset_axes([x + 0.006, 0.36 + (0.6 - h) / 2, w - 0.012, h]); show(ia, img)
            if k == 0 and raw_box is not None:                          # dotted: region cut out as the YOLO crop
                x0, y0, bw, bh = raw_box
                ia.add_patch(plt.Rectangle((x0, y0), bw, bh, fill=False, ec="white", lw=1.6))
                ia.add_patch(plt.Rectangle((x0, y0), bw, bh, fill=False, ec=INK, lw=0.9, ls=(0, (2, 1.5))))
            ax.text(x + w / 2, 0.2, sub, ha="center", va="top", fontsize=5.5, color=INK2)
        else:
            ax.add_patch(plt.Rectangle((x + 0.008, 0.38), w - 0.016, 0.56, fc="#eef3fb" if k < n - 1 else "#fdeee7",
                                       ec=BLUE if k < n - 1 else ORANGE, lw=0.8))
            ax.text(x + w / 2, 0.66, sub, ha="center", va="center", fontsize=5.6, color=INK2)
        ax.text(x + w / 2, 0.28, title, ha="center", va="center", fontsize=6.3,
                fontweight="bold")
        if k < n - 1:
            ax.annotate("", xy=(xs[k + 1] + 0.004, 0.66), xytext=(x + w - 0.002, 0.66),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))


STRAIN_COL = {"C57BL/6": BLUE, "BALB/c": ORANGE, "HET3": AQUA, "TERC": "#eda100"}
SEX_COL = {"Female": "#e87ba4", "Male": "#4a3aa7", "not recorded": LIGHT}


def stacked(ax, t, cols, title, letter):
    labs = list(t.index); y = np.arange(len(labs)); left = np.zeros(len(labs))
    frac = t.div(t.sum(1), axis=0)
    for c in t.columns:
        ax.barh(y, frac[c], left=left, color=cols.get(c, GREY), height=0.62, label=c, ec="white", lw=0.6)
        left += frac[c].values
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50", "100%"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6, handlelength=1, columnspacing=0.8)
    ax.set_title(title, loc="left"); panel(ax, letter, -0.08)


def fig1(D, out, crops_root=None, raw_root=None, yolo_root=None, examples=None, pipeline=None, rotate=None, tiles_a=None):
    L = D["labels"]; L = L[L.source != F("PaperImages")]
    labs = [l for l in LAB_ORDER if l in set(L.source)]
    raw = RawIndex(raw_root) if raw_root else None
    # Fig. 1b tiles: YOLO crops (mouse in its own setting) if given, else masked crops, else photos
    root_b = yolo_root or crops_root
    if root_b:
        tile_path = lambda f: os.path.join(root_b, f) if os.path.exists(os.path.join(root_b, f)) else None
    else:
        tile_path = raw.find if raw else None
    ex = pick_examples(D["labels"], tile_path, examples) if tile_path else {}
    # optional per-tile rotation: text file, one "degrees_clockwise relative/path" per line
    ROT = {}
    if rotate:
        for line in open(rotate):
            if line.strip():
                deg, f = line.strip().split(maxsplit=1); ROT[f] = float(deg)
    fig = plt.figure(figsize=(FULL, 165 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.85, 1.1, 1.35], hspace=0.35)

    # a pipeline, one real mouse followed through the steps
    ax = fig.add_subplot(gs[0]); panel(ax, "a", -0.01, 0.95)
    tiles = [None, None, None]
    cand = L[L.source == labs[0]].sort_values("age_days").file
    order = list(cand.iloc[len(cand) // 2:]) + list(cand.iloc[:len(cand) // 2])
    for f in ([pipeline] if pipeline else []) + order:
        p_raw = raw.find(f) if raw else None
        p_yolo = os.path.join(yolo_root, f) if yolo_root else None
        p_crop = os.path.join(crops_root, f) if crops_root else None
        if all(p is None or os.path.exists(p) for p in (p_raw, p_yolo, p_crop)):
            # whole photo (larger slot); YOLO crop and mask the same square size, YOLO padded white (not stretched)
            tiles = [load_thumb(p_raw) if p_raw else None, pad_square(p_yolo, 300) if p_yolo else None,
                     model_frame(p_crop, 300) if p_crop else None]
            box = locate_crop(p_raw, p_yolo, tiles[0].shape[1]) if p_raw and p_yolo else None
            break
    else:
        box = None
    if tiles_a:          # hand-cropped photo (crop box drawn on it), YOLO crop and mask: used as they are
        from PIL import Image
        tiles = [np.asarray(Image.open(p).convert("RGB")) for p in tiles_a]; box = None
    draw_pipeline(ax, *tiles, raw_box=box, same_height=bool(tiles_a))

    # b example photographs as one mosaic (square tiles, edge to edge): youngest (top) and oldest (bottom)
    #   mouse of the facility's main strain
    ax = fig.add_subplot(gs[1]); panel(ax, "b", -0.01, 1.12)
    T = 300; mosaic = np.full((2 * T, len(labs) * T, 3), 255, np.uint8)
    for j, lab in enumerate(labs):
        for i in range(2):
            if lab in ex and len(ex[lab]) > i:
                p, age, _ = ex[lab][0 if i == 0 else -1]
                # YOLO crops are not square: whole crop, aspect kept, centred on white (the figure background)
                rot = next((d for f, d in ROT.items() if p.endswith(f)), 0)
                mosaic[i * T:(i + 1) * T, j * T:(j + 1) * T] = np.pad(pad_square(p, T - 6, rotate=rot), ((3, 3), (3, 3), (0, 0)),
                                                                      constant_values=255)
    ax.imshow(mosaic, extent=(0, len(labs), 2, 0)); ax.set_xlim(0, len(labs)); ax.set_ylim(2, 0)
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    for j, lab in enumerate(labs):
        st_ = ex[lab][0][2] if lab in ex and ex[lab] else ""
        ax.text(j + 0.5, -0.04, f"{lab}\n{st_}", ha="center", va="bottom", fontsize=6)
        for i in range(2):
            if lab in ex and len(ex[lab]) > i:
                age = ex[lab][0 if i == 0 else -1][1]
                ax.text(j + 0.04, i + 0.95, f"{age / 30.44:.0f} mo", fontsize=6, color="white", va="bottom",
                        bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))

    # c ages, d strain, e sex
    sub = gs[2].subgridspec(1, 3, width_ratios=[1.55, 0.7, 0.7], wspace=0.12)
    ax = fig.add_subplot(sub[0]); panel(ax, "c", -0.3)
    sess = L.drop_duplicates(["source", "id", "age_days"])
    cnt = L.groupby("source").agg(images=("file", "size"), mice=("id", "nunique"))
    for k, lab in enumerate(labs):
        a = sess[sess.source == lab].age_days.values / 30.44
        ax.scatter(a, k + np.random.default_rng(k).uniform(-0.28, 0.28, len(a)), s=3, color=BLUE, alpha=0.35, lw=0)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([f"{l}\n{cnt.loc[l, 'mice']} mice, {cnt.loc[l, 'images']} photos" for l in labs], fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("Age at photo (months)"); ax.set_xlim(0, 36)
    ax.set_title("Ages photographed", loc="left")
    st = pd.crosstab(L.source, L.strain).reindex(labs).fillna(0)
    st = st[[c for c in STRAIN_COL if c in st.columns] + [c for c in st.columns if c not in STRAIN_COL]]
    stacked(fig.add_subplot(sub[1]), st, STRAIN_COL, "Strain (photos)", "d")
    sx = pd.crosstab(L.source, L.sex.fillna("not recorded")).reindex(labs).fillna(0)
    sx = sx[[c for c in SEX_COL if c in sx.columns]]
    stacked(fig.add_subplot(sub[2]), sx, SEX_COL, "Sex (photos)", "e")
    save(fig, out, "fig1_dataset")
    if ex:
        pd.DataFrame([(lab, *x) for lab, v in ex.items() for x in v], columns=["facility", "path", "age_days", "strain"]).to_csv(
            os.path.join(out, "tables", "fig1_example_files.csv"), index=False)
    key("photos (excl. literature)", len(L), "labels"); key("mice", L.drop_duplicates(["source", "id"]).shape[0], "labels")
    key("facilities", len(labs), "labels")
    for s_, n_ in L.groupby("strain").id.nunique().items():
        key(f"mice of strain {s_}", int(n_), "labels")


def load_maps(spec):
    """'grey=dir' -> (index DataFrame, npz)."""
    mode, d = spec.split("=", 1)
    idx = anonymise(pd.read_csv(os.path.join(d, "saliency_index.csv")).drop_duplicates("file"))
    return mode, idx, np.load(os.path.join(d, "saliency_maps.npz"))


def fig6(D, out, regions=None, sal=None, crops_root=None, files=None, n_ex=10, size=448, mode="grey"):
    """Saliency: ONE occlusion mode (grey and blur maps look alike); `mode` picks it."""
    fig = plt.figure(figsize=(FULL, 150 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.2)
    maps = [m for m in (load_maps(s) for s in sal) if m[0] == mode][:1] if sal else []
    oriented = None
    if regions is not None:
        oriented = regions[regions.orientation.isin(["tail", "tail+agree", "head-detector"])]

    # a examples: photo / grey occlusion / blur occlusion, young -> old
    if maps and crops_root:
        idx0 = maps[0][1]
        cand = idx0[[all(f in set(m[1].file) for m in maps) for f in idx0.file]].copy()
        cand["err"] = (cand.pred_days - cand.age_days).abs()
        if files:
            want = [l.strip() for l in open(files) if l.strip()]
            chosen = cand.set_index("file").loc[[w for w in want if w in set(cand.file)]].reset_index()
        else:
            if oriented is not None:
                cand = cand[cand.file.isin(set(oriented.file))]
            cand = cand[cand.err < cand.err.quantile(0.5)].sort_values("age_days")    # well-predicted mice
            q = np.linspace(0, len(cand) - 1, n_ex).round().astype(int)
            chosen = cand.iloc[q]
        chosen = chosen.sort_values("age_days")
        # occlusion maps only (the map is drawn over a greyscale copy of the photo), young -> old, 2 rows
        mode_, idx, npz = maps[0]
        ncol = int(np.ceil(len(chosen) / 2))
        sub = gs[0].subgridspec(2, ncol, wspace=0.04, hspace=0.22)
        last = None
        for j, r in enumerate(chosen.itertuples()):
            img = model_frame(os.path.join(crops_root, r.file), size)
            heat = npz[idx.set_index("file").loc[r.file, "key"]].astype(np.float32)
            a_ = fig.add_subplot(sub[j // ncol, j % ncol])
            show(a_, heat_overlay(img, heat, np.percentile(np.abs(heat), 99)),
                 f"{r.source.replace('Facility', 'Fac.')} · {r.age_days / 30.44:.0f} mo (pred. {r.pred_days / 30.44:.0f})")
            if j == 0:
                panel(a_, "a", -0.12, 1.12)
            last = a_
        pd.DataFrame(chosen).to_csv(os.path.join(out, "tables", "fig6_example_files.csv"), index=False)
        bb = last.get_position()
        fig.text(bb.x1, bb.y0 - 0.012, "red: region makes the mouse look older   blue: younger   (colour scale per map)",
                 ha="right", va="top", fontsize=6, color=INK2)
    else:
        a_ = fig.add_subplot(gs[0]); panel(a_, "a", -0.01)
        placeholder(a_, "Saliency examples: run with --sal grey=DIR blur=DIR --crops-root DIR")

    sub = gs[1].subgridspec(1, 3, width_ratios=[0.8, 1, 1], wspace=0.5)
    # b how regions are defined, on the first example
    ax = fig.add_subplot(sub[0]); panel(ax, "b", -0.1)
    seg_ok = False
    if crops_root and maps and oriented is not None:
        try:
            import region_saliency as RS
            for r in chosen.itertuples():
                img = model_frame(os.path.join(crops_root, r.file), size)
                regs, status, _ = RS.segment(img)
                if regs is not None and status in ("tail", "tail+agree", "head-detector"):
                    ov = img.astype(float).mean(2, keepdims=True).repeat(3, 2) * 0.6 + 50
                    for k_, m_ in regs.items():
                        if k_ in RS.COLORS and k_ != "core" and m_.any():
                            ov[m_] = 0.45 * ov[m_] + 0.55 * np.array(RS.COLORS[k_])
                    show(ax, ov.clip(0, 255).astype(np.uint8)); seg_ok = True
                    break
        except Exception as e:                                       # region_saliency needs torch; never fatal
            print("  region overlay skipped:", e)
    if seg_ok:
        hs = [Line2D([], [], marker="s", ls="", color=np.array(RS.COLORS[k]) / 255, label=k) for k in RS.REGIONS]
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=6, handletextpad=0.1,
                  columnspacing=0.6)
        ax.set_title("Body regions", loc="left")
    else:
        placeholder(ax, "region definition\n(needs --crops-root)")
    if oriented is None:
        for k in (1, 2):
            placeholder(fig.add_subplot(sub[k]), "region analysis\n(--regions not given)")
        save(fig, out, "fig6_saliency"); return
    R = oriented
    ax = fig.add_subplot(sub[1]); panel(ax, "c")
    reg = ["head", "mid", "rear", "tail", "limbs"]
    rows = []
    for k, (mode_, c) in enumerate([(mode, BLUE)]):
        x = R[R["mode"] == mode_]
        for j, r in enumerate(reg):
            v = x.dropna(subset=[r + "_mean"])
            m, lo, hi = boot_mean(v[r + "_mean"].values, v.source.values)
            rows.append(dict(mode=mode_, region=r, n=len(v), mean=m, ci_low=lo, ci_high=hi,
                             labs_negative=int((v.groupby("source")[r + "_mean"].mean() < 0).sum()), labs=v.source.nunique()))
            ax.errorbar(j, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=4, elinewidth=0.9,
                        mec="white", mew=0.5)
    pd.DataFrame(rows).to_csv(os.path.join(out, "tables", "saliency_regions.csv"), index=False)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(5)); ax.set_xticklabels(reg); ax.set_ylabel("Effect on predicted age\n(days per pixel)")
    ax.set_title(f"Effect by region ({R.file.nunique()} mice, {R.source.nunique()} facilities)", loc="left")
    ax = fig.add_subplot(sub[2]); panel(ax, "d")
    rows = []
    for k, (mode_, c) in enumerate([(mode, BLUE)]):
        x = R[R["mode"] == mode_].assign(hm=lambda z: z.head_mean - z.mid_mean)
        for j, ag in enumerate(["<6 mo", "6-18 mo", ">18 mo"]):
            v = x[x.age_group == ag].dropna(subset=["hm"])
            m, lo, hi = boot_mean(v.hm.values, v.source.values)
            rows.append(dict(mode=mode_, age_group=ag, n=len(v), mean=m, ci_low=lo, ci_high=hi))
            ax.errorbar(j, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=4, elinewidth=0.9, mec="white", mew=0.5)
    pd.DataFrame(rows).to_csv(os.path.join(out, "tables", "saliency_head_minus_mid.csv"), index=False)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(3)); ax.set_xticklabels(["<6", "6–18", ">18"]); ax.set_xlabel("Age (months)")
    ax.set_ylabel("Head − mid-body\n(days per pixel)"); ax.set_title("Head vs body by age", loc="left")
    save(fig, out, "fig6_saliency")


def finetune_numbers(d):
    """Fine-tuning is reported in the text only (no figure): numbers for key_numbers.csv."""
    s = pd.read_csv(os.path.join(d, "summary.csv"))
    ses = s[s.level == "session"].set_index("model")
    for m in ses.index:
        if " - " in m:
            r = ses.loc[m]
            key(f"{m} (session MAE, d)", f"{r.MAE:.1f} [{r.MAE_ci_low:.1f}, {r.MAE_ci_high:.1f}]", f"{d}/summary.csv")
            key(f"{m} (within-session concordance)", f"{r.within_session_conc:.3f} [{r.conc_ci_low:.3f}, {r.conc_ci_high:.3f}]",
                f"{d}/summary.csv")
        else:
            r = ses.loc[m]
            key(f"{m}: session MAE / concordance", f"{r.MAE:.0f} d / {r.within_session_conc:.3f}", f"{d}/summary.csv")
    pf = anonymise(pd.read_csv(os.path.join(d, "per_lab_finetuned.csv")))
    for _, r in pf.iterrows():
        key(f"fine-tuned, {r.source}: MAE / concordance", f"{r.MAE:.0f} d / {r.within_session_conc:.3f}", f"{d}/per_lab_finetuned.csv")


def group_rows(S, col, levels, n_boot=1000):
    """Per facility (and all facilities): within-session concordance among pairs of the SAME group,
    MAE, bias, and the age-adjusted photographic gap; 95 % CIs from resampling mice."""
    rows = []
    facs = ["All facilities"] + [l for l in LAB_ORDER if l in set(S.source)]
    for fac in facs:
        x = S if fac == "All facilities" else S[S.source == fac]
        for lev in levels:
            g = x[x[col] == lev]
            if g.mouse.nunique() < 8:
                continue
            c, lo, hi, n_pairs = concordance(g.age_days.values, g.pred_days.values, g.folder.values, g.mouse.values, n_boot)
            m, mlo, mhi = boot_mean(np.abs(g.pred_days - g.age_days).values, g.mouse.values, n=n_boot)
            gm, glo, ghi = boot_mean(g.gap_adj_days.values, g.mouse.values, n=n_boot)
            rows.append(dict(facility=fac, group=lev, mice=g.mouse.nunique(), sessions=len(g), conc=c, conc_lo=lo, conc_hi=hi,
                             pairs=n_pairs, MAE=m, MAE_lo=mlo, MAE_hi=mhi, bias=(g.pred_days - g.age_days).mean(),
                             gap=gm, gap_lo=glo, gap_hi=ghi))
    return pd.DataFrame(rows)


def gap_difference(S, col, a, b, n_boot=2000, seed=0):
    """Age-adjusted photographic gap, group a minus group b, within each facility that has both;
    pooled = mean of within-facility differences weighted by mice; CI from resampling mice within facility."""
    rng = np.random.default_rng(seed); rows = []
    both = [f for f in LAB_ORDER if f in set(S.source) and {a, b} <= set(S[S.source == f][col])]
    per = {}
    for fac in both:
        x = S[S.source == fac]
        ga = x[x[col] == a].groupby("mouse").gap_adj_days.mean().values
        gb = x[x[col] == b].groupby("mouse").gap_adj_days.mean().values
        if min(len(ga), len(gb)) < 5:
            continue
        bs = [rng.choice(ga, len(ga)).mean() - rng.choice(gb, len(gb)).mean() for _ in range(n_boot)]
        per[fac] = (ga.mean() - gb.mean(), np.array(bs), len(ga) + len(gb))
        rows.append(dict(facility=fac, contrast=f"{a} − {b}", diff=ga.mean() - gb.mean(),
                         lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), mice_a=len(ga), mice_b=len(gb)))
    if len(per) > 1:
        w = np.array([v[2] for v in per.values()], float); w /= w.sum()
        d = sum(wi * v[0] for wi, v in zip(w, per.values())); bs = sum(wi * v[1] for wi, v in zip(w, per.values()))
        rows.append(dict(facility="Pooled", contrast=f"{a} − {b}", diff=d, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         mice_a=np.nan, mice_b=np.nan))
    return pd.DataFrame(rows)


def figS2_groups(D, out):
    """Does performance depend on sex or strain? (unseen-facility predictions, control mice)"""
    S = perf_sessions(D["gap_source"])
    S["sex"] = S.sex.fillna("not recorded")
    sexR = group_rows(S, "sex", ["Female", "Male"])
    strR = group_rows(S, "strain", ["C57BL/6", "BALB/c", "HET3"])
    sexD = gap_difference(S, "sex", "Male", "Female")
    strD = pd.concat([gap_difference(S, "strain", "BALB/c", "C57BL/6"), gap_difference(S, "strain", "HET3", "C57BL/6")])
    for name, t in [("performance_by_sex", sexR), ("performance_by_strain", strR), ("gap_by_sex", sexD), ("gap_by_strain", strD)]:
        t.to_csv(os.path.join(out, "tables", f"{name}.csv"), index=False)

    fig, axs = plt.subplots(2, 2, figsize=(FULL * 0.85, 120 * MM), gridspec_kw=dict(hspace=0.75, wspace=0.75))
    def dots(ax, R, cols, title, letter):
        facs = list(dict.fromkeys(R.facility)); y = {f: i for i, f in enumerate(facs)}
        levs = list(cols); off = {l: (k - (len(levs) - 1) / 2) * 0.22 for k, l in enumerate(levs)}
        for lev in levs:
            r = R[R.group == lev]
            if r.empty:
                continue
            yy = np.array([y[f] for f in r.facility]) + off[lev]
            ax.errorbar(r.conc, yy, xerr=[r.conc - r.conc_lo, r.conc_hi - r.conc], fmt="o", color=cols[lev], ms=4,
                        elinewidth=0.9, mec="white", mew=0.5, label=f"{lev}")
        ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
        ax.set_yticks(range(len(facs))); ax.set_yticklabels(facs); ax.invert_yaxis(); ax.set_xlim(0.3, 1.2); ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Within-session concordance"); ax.set_title(title, loc="left"); panel(ax, letter, -0.55)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(levs), handletextpad=0.1, columnspacing=0.8)
    def diffs(ax, R, title, letter, color):
        labels = [f"{r.contrast}: {r.facility}" for r in R.itertuples()]
        forest(ax, labels, R["diff"], R.lo, R.hi, [BLUE if f == "Pooled" else color for f in R.facility],
               bold=[i for i, f in enumerate(R.facility) if f == "Pooled"])
        ax.set_xlabel("Difference in photographic age gap (days)"); ax.set_title(title, loc="left"); panel(ax, letter, -0.75)
        if len(R) < 5:                                   # few rows: keep them close, centred
            pad = (5 - len(R)) / 2; ax.set_ylim(len(R) - 0.5 + pad, -0.5 - pad)
        lim = max(150, np.nanmax(np.abs(R[["lo", "hi"]].values)) * 1.1); ax.set_xlim(-lim, lim)
    dots(axs[0, 0], sexR, {"Female": SEX_COL["Female"], "Male": SEX_COL["Male"]}, "Ranking within each sex", "a")
    diffs(axs[0, 1], sexD, "Do males look older?", "b", GREY)
    dots(axs[1, 0], strR, {k: STRAIN_COL[k] for k in ["C57BL/6", "BALB/c", "HET3"]}, "Ranking within each strain", "c")
    diffs(axs[1, 1], strD, "Do strains look older?", "d", GREY)
    save(fig, out, "fig4_sex_strain")
    for t, nm in [(sexR, "sex"), (strR, "strain")]:
        for r in t[t.facility == "All facilities"].itertuples():
            key(f"{nm} {r.group}: concordance / MAE", f"{r.conc:.3f} [{r.conc_lo:.3f}, {r.conc_hi:.3f}] / {r.MAE:.0f} d ({r.mice} mice)",
                f"tables/performance_by_{nm}.csv")
    for t, nm in [(sexD, "sex"), (strD, "strain")]:
        for r in t.itertuples():
            key(f"gap difference {r.contrast} ({r.facility})", f"{r.diff:.1f} [{r.lo:.1f}, {r.hi:.1f}] d", f"tables/gap_by_{nm}.csv")


def fig2(D, out):
    S, M = perf_sessions(D["gap_source"]), perf_sessions(D["gap_mice"])
    summ = D["summary"]
    fig = plt.figure(figsize=(FULL, 112 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.62)

    # a predicted vs true (leave-one-lab-out, session level)
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a")
    ax.scatter(S.age_days / 30.44, S.pred_days / 30.44, s=3, color=BLUE, alpha=0.3, lw=0)
    ax.plot([0, 36], [0, 36], color=GREY, lw=0.8, ls="--")
    r = summ[(summ.model == "frozen448_source") & (summ.level == "session")].iloc[0]
    ax.text(0.03, 0.97, f"r = {r.r:.2f}\nMAE = {r.MAE:.0f} d\n{int(r.n)} sessions", transform=ax.transAxes,
            va="top", fontsize=6.5)
    ax.set_xlim(0, 36); ax.set_ylim(0, 36); ax.set_aspect("equal")
    ax.set_xlabel("True age (months)"); ax.set_ylabel("Predicted age (months)")
    ax.set_title("Unseen facilities", loc="left")
    key("source split session MAE (d)", round(r.MAE, 1), "eval_res/summary.csv")
    key("source split session r", round(r.r, 3), "eval_res/summary.csv")
    key("source split within-session concordance", f"{r.within_session_conc:.3f} [{r.conc_ci_low:.3f}, {r.conc_ci_high:.3f}]",
        "eval_res/summary.csv")
    key("folder-only baseline MAE (d)", round(r.MAE_folder_only, 1), "eval_res/summary.csv")
    rm = summ[(summ.model == "frozen448_mice") & (summ.level == "session")].iloc[0]
    key("mouse split session MAE (d)", round(rm.MAE, 1), "eval_res/summary.csv")
    key("mouse split within-session concordance", f"{rm.within_session_conc:.3f} [{rm.conc_ci_low:.3f}, {rm.conc_ci_high:.3f}]",
        "eval_res/summary.csv")

    # b per-lab within-session concordance, unseen lab vs unseen mouse
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b")
    ps, pm = D["perlab_source"].set_index("source"), D["perlab_mice"].set_index("source")
    labs = [l for l in LAB_ORDER if l in ps.index]
    y = np.arange(len(labs))
    for yy, lab in zip(y, labs):
        ax.plot([ps.loc[lab, "within_session_conc"], pm.loc[lab, "within_session_conc"]], [yy, yy], color=LIGHT, lw=1.5, zorder=1)
    ax.scatter(ps.loc[labs, "within_session_conc"], y, s=22, color=BLUE, zorder=3, label="unseen facility", ec="white", lw=0.6)
    ax.scatter(pm.loc[labs, "within_session_conc"], y, s=22, color=ORANGE, zorder=3, label="unseen mouse", ec="white", lw=0.6)
    ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([f"{l} ({int(ps.loc[l, 'mice'])})" for l in labs]); ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("Within-session concordance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.1, columnspacing=1)
    ax.set_xlabel("Within-session concordance (0.5 = chance)", labelpad=1)
    ax.set_title("Same-session ranking", loc="left")

    # c concordance by age gap
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    g = D["gaptab_source"].copy()
    bins = ["(0, 100]", "(100, 200]", "(200, 300]", "(300, 500]", "(500, 1100]"]
    lbl = ["≤100", "100–200", "200–300", "300–500", ">500"]
    for grp, col in [("<6 mo", BLUE), ("6-13 mo", ORANGE), (">13 mo", AQUA)]:
        x = g[(g.younger_mouse == grp) & (g.pairs >= 50)].set_index("gap_days").reindex(bins)
        ax.plot(range(5), x.concordance, "-o", color=col, ms=4, lw=1.5, mec="white", mew=0.6,
                label=grp.replace("-", "–"))
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(5)); ax.set_xticklabels(lbl, rotation=30)
    ax.set_xlabel("Age difference within pair (days)"); ax.set_ylabel("Concordance")
    ax.legend(title="younger mouse", loc="upper left", title_fontsize=6.5, ncol=3, columnspacing=0.8, handlelength=1.2,
              borderaxespad=0.2)
    ax.set_ylim(0.4, 1.3); ax.set_yticks([0.4, 0.6, 0.8, 1.0])      # band above 1.0 holds the legend
    ax.set_title("Larger gaps are ranked better", loc="left")

    # d longitudinal: same mouse photographed twice
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "d")
    lo = D["long_source"]
    dx, dy = lo.age_2 - lo.age_1, lo.pred_2 - lo.pred_1
    ax.scatter(dx, dy, s=4, color=BLUE, alpha=0.4, lw=0)
    ax.axhline(0, color=GREY, lw=0.8, ls="--"); ax.plot([0, dx.max()], [0, dx.max()], color=LIGHT, lw=0.8)
    ax.set_xlabel("Real time between photos (days)"); ax.set_ylabel("Change in predicted age (days)")
    ax.text(0.03, 0.97, f"later photo predicted older:\n{r.long_later_predicted_older:.0%} "
            f"[{r.long_ci_low:.0%}–{r.long_ci_high:.0%}] of {int(r.long_pairs)} pairs\n({int(r.long_mice)} mice)",
            transform=ax.transAxes, va="top", fontsize=6.3)
    ax.set_title("Same mouse, repeated photos", loc="left")
    key("longitudinal later-older fraction", f"{r.long_later_predicted_older:.3f} [{r.long_ci_low:.3f}, {r.long_ci_high:.3f}]",
        "eval_res/summary.csv", f"{int(r.long_pairs)} pairs, {int(r.long_mice)} mice")
    key("longitudinal median slope (pred/real)", round(r.long_median_slope, 2), "eval_res/summary.csv")

    # e calibration with k mice of known age from the new lab
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "e")
    c = D["calib_source"]
    allw = c[c.source.str.startswith("ALL")].set_index("k_mice").MAE
    for lab, x in c[~c.source.str.startswith("ALL")].groupby("source"):
        ax.plot(x.k_mice, x.MAE, color=LIGHT, lw=1, label="single facility" if lab == LAB_ORDER[0] else None)
    ax.plot(allw.index, allw.values, "-o", color=BLUE, lw=1.8, ms=4, mec="white", mew=0.6, label="all facilities (weighted)")
    ax.set_xlabel("Mice of known age from the new facility"); ax.set_ylabel("MAE (days)")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.2, borderaxespad=0.2)
    ax.set_title("Calibrating to a new facility", loc="left")
    for k in allw.index:
        key(f"calibration MAE, all labs weighted, k={k}", round(allw[k], 1), "eval_res/calibration_frozen448_source.csv")

    # f bias per lab (the offset that calibration removes)
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "f")
    b = ps.loc[labs, "bias"]
    ax.barh(y, b, color=[ORANGE if v > 0 else BLUE for v in b], height=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel("Mean predicted − true age (days)")
    ax.set_title("Facility offsets (unseen facility)", loc="left")
    save(fig, out, "fig2_age_prediction")


def fig3(D, out):
    fig = plt.figure(figsize=(FULL, 118 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.5)
    T = D["tasks"].dropna(subset=["sex"]).reset_index(drop=True)
    T["male"] = (T.sex == "Male").astype(int)

    # a age vs sex within session, same metric
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a")
    rows = []
    for lab in ["ALL"] + [l for l in LAB_ORDER if l in set(T.source)]:
        x = T if lab == "ALL" else T[T.source == lab]
        ca = concordance(x.age_days.values, x.pred_age_days.values, x.folder.values, x.mouse.values)
        cs = concordance(x.male.values, x.pred_male_prob.values, x.folder.values, x.mouse.values)
        rows.append(dict(lab=lab, age=ca[0], age_lo=ca[1], age_hi=ca[2], age_pairs=ca[3],
                         sex=cs[0], sex_lo=cs[1], sex_hi=cs[2], sex_pairs=cs[3]))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "tables", "age_vs_sex_within_session.csv"), index=False)
    y = np.arange(len(R))
    for col, c, off, lab in [("age", BLUE, -0.15, "age"), ("sex", ORANGE, 0.15, "sex")]:
        ax.errorbar(R[col], y + off, xerr=[R[col] - R[col + "_lo"], R[col + "_hi"] - R[col]], fmt="o", color=c,
                    ms=4, elinewidth=0.9, capsize=0, mec="white", mew=0.5, label=lab)
    ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(["All facilities"] + list(R.lab[1:])); ax.invert_yaxis()
    ax.set_xlabel("Within-session concordance"); ax.set_xlim(0.2, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.2), ncol=2, handletextpad=0.1, columnspacing=0.8)
    ax.set_title("Sex vs age", loc="left")
    a0 = R.iloc[0]
    key("age within-session conc (tasks run)", f"{a0.age:.3f} [{a0.age_lo:.3f}, {a0.age_hi:.3f}]", "compare_tasks_source.csv")
    key("sex within-session conc (= within-session AUC)", f"{a0.sex:.3f} [{a0.sex_lo:.3f}, {a0.sex_hi:.3f}]", "compare_tasks_source.csv")

    # b/c crops vs original photos
    C = D["crops"].reset_index(drop=True)
    rows = []
    for lab in ["ALL"] + [l for l in LAB_ORDER if l in set(C.source)]:
        x = C if lab == "ALL" else C[C.source == lab]
        r = dict(lab=lab)
        for k in ["original", "cropped"]:
            cc = concordance(x.age_days.values, x[f"pred_age_{k}"].values, x.folder.values, x.mouse.values)
            r.update({k: cc[0], k + "_lo": cc[1], k + "_hi": cc[2], k + "_MAE": np.abs(x[f"pred_age_{k}"] - x.age_days).mean()})
        rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "tables", "crops_vs_original.csv"), index=False)
    y = np.arange(len(R))
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b")
    for col, c, off, lab in [("original", GREY, -0.15, "original photo"), ("cropped", BLUE, 0.15, "mouse cropped")]:
        ax.errorbar(R[col], y + off, xerr=[R[col] - R[col + "_lo"], R[col + "_hi"] - R[col]], fmt="o", color=c, ms=4,
                    elinewidth=0.9, capsize=0, mec="white", mew=0.5, label=lab)
    ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(["All facilities"] + list(R.lab[1:])); ax.invert_yaxis()
    ax.set_xlabel("Within-session concordance (age)"); ax.set_xlim(0.3, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.2), ncol=2, handletextpad=0.1, columnspacing=0.8)
    ax.set_title("Cropped vs original photo", loc="left")
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    ax.barh(y - 0.2, R.original_MAE, height=0.38, color=GREY, label="original photo")
    ax.barh(y + 0.2, R.cropped_MAE, height=0.38, color=BLUE, label="mouse cropped")
    ax.set_yticks(y); ax.set_yticklabels(["All facilities"] + list(R.lab[1:])); ax.invert_yaxis()
    ax.set_xlabel("MAE (days)"); ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.2), ncol=2, handlelength=1,
                                           handletextpad=0.4, columnspacing=0.8)
    ax.set_title("Error, cropped vs original", loc="left")
    a0 = R.iloc[0]
    key("crops vs original conc", f"cropped {a0.cropped:.3f} [{a0.cropped_lo:.3f},{a0.cropped_hi:.3f}] vs original {a0.original:.3f} "
        f"[{a0.original_lo:.3f},{a0.original_hi:.3f}]", "compare_crops_source.csv")
    key("crops vs original MAE", f"cropped {a0.cropped_MAE:.0f} vs original {a0.original_MAE:.0f}", "compare_crops_source.csv")

    # d/e/f human raters vs model
    H = D["human"]
    acc = H["accuracy"]; raters = [r for r in acc.rater if r != "model"]
    cols = {r: c for r, c in zip(raters, [ORANGE, AQUA, "#e87ba4", "#4a3aa7"])}; cols["model"] = BLUE
    if "raters_mean" in cols:
        cols["raters_mean"] = INK2
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "d")
    y = np.arange(len(acc))
    ax.errorbar(acc.within_session_conc, y, xerr=[acc.within_session_conc - acc.conc_ci_low, acc.conc_ci_high - acc.within_session_conc],
                fmt="none", ecolor=INK2, elinewidth=0.9)
    ax.scatter(acc.within_session_conc, y, color=[cols.get(r, GREY) for r in acc.rater], s=26, zorder=3, ec="white", lw=0.6)
    ax.axvline(0.5, color=GREY, lw=0.8, ls="--")
    names = {"model": "Model", "raters_mean": "Experts (mean)"}
    names.update({r: f"Expert {i + 1}" for i, r in enumerate([r for r in acc.rater if r not in names])})
    ax.set_yticks(y); ax.set_yticklabels([names[r] for r in acc.rater])
    ax.invert_yaxis(); ax.set_xlim(0.3, 0.8); ax.set_ylim(len(acc) - 0.4, -0.6); ax.set_xlabel("Within-session concordance")
    ax.set_title(f"Same {int(acc.n.iloc[0])} photos", loc="left")
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "e")
    g = H["gap"]; bins = ["(0, 100]", "(100, 200]", "(200, 300]", "(300, 500]", "(500, 1100]"]
    for rname, x in g[g.rater != "raters_mean"].groupby("rater"):
        x = x.set_index("gap_days").reindex(bins)
        ax.plot(range(5), x.concordance, "-o", color=cols.get(rname, GREY), ms=4, lw=1.5, mec="white", mew=0.6, label=names.get(rname, rname))
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(range(5)); ax.set_xticklabels(["≤100", "100–200", "200–300", "300–500", ">500"], rotation=30)
    ax.set_xlabel("Age difference within pair (days)"); ax.set_ylabel("Concordance"); ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper left"); ax.set_title("Concordance by age gap", loc="left")
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "f")
    band = H["band"]; order = ["<4 mo", "4-9 mo", "9-15 mo", "15-21 mo", ">21 mo"]
    rs = [r for r in acc.rater if r != "raters_mean"]; w = 0.8 / len(rs)
    for k, rname in enumerate(rs):
        x = band[band.rater == rname].set_index("band").reindex(order)
        ax.bar(np.arange(5) + (k - (len(rs) - 1) / 2) * w, x.bias_days, width=w * 0.9, color=cols.get(rname, GREY),
               label=names.get(rname, rname))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([o.replace("-", "–") for o in order], rotation=30)
    ax.set_xlabel("True age"); ax.set_ylabel("Mean predicted − true (days)"); ax.legend(loc="lower left")
    ax.set_title("Bias by true age", loc="left")
    cmp_ = H["cmp"]
    for _, rr in cmp_.iterrows():
        key(f"{rr.rater} minus model, concordance", f"{rr.conc_rater_minus_model:.3f} [{rr.conc_ci_low:.3f}, {rr.conc_ci_high:.3f}]",
            "human_baseline/rater_vs_model.csv")
        key(f"{rr.rater} minus model, MAE (d)", f"{rr.MAE_rater_minus_model:.1f} [{rr.MAE_ci_low:.1f}, {rr.MAE_ci_high:.1f}]",
            "human_baseline/rater_vs_model.csv")
    for _, rr in acc.iterrows():
        key(f"{rr.rater} MAE / concordance", f"{rr.MAE_days:.0f} d / {rr.within_session_conc:.3f}", "human_baseline/accuracy.csv")
    if "self" in H:
        for _, rr in H["self"].iterrows():
            key(f"{rr.rater} self-consistency (Spearman, repeats)", round(rr.spearman, 2), "human_baseline/self_consistency.csv")
    save(fig, out, "fig3_what_photos_tell")


def forest(ax, labels, est, lo, hi, colors, ref=0.0, log=False, bold=()):
    y = np.arange(len(labels))
    for yy, e, l, h, c in zip(y, est, lo, hi, colors):
        ax.plot([l, h], [yy, yy], color=c, lw=1.2)
        ax.scatter([e], [yy], s=30 if yy in bold else 18, marker="D" if yy in bold else "o", color=c, zorder=3,
                   ec="white", lw=0.5)
    ax.axvline(ref, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    for t, yy in zip(ax.get_yticklabels(), y):
        if yy in bold:
            t.set_fontweight("bold")


POP = {"all": "all mice, not adjusted for treatment", "controls": "controls"}
POP_SURV = {"all": "all mice, adjusted for treatment", "controls": "controls"}


def fig4(D, out, frailty_pop="all", survival_pop="controls", survival_exclude=()):
    """Health deficit / weight: frailty_pop; survival (Cox + curves): survival_pop.
    'all' = treated mice included with treatment as a covariate; 'controls' = control mice only.
    Both versions are always written to the supplementary tables."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    A, B, M = D["A"], D["B"], D["merged"]
    fig = plt.figure(figsize=(FULL, 112 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.75)

    # a health deficit & body weight per 100 d of photographic age gap (controls)
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a", -0.62)
    population = frailty_pop; a = A[A.population == POP[frailty_pop]]
    rows = []
    for out_name, short in [("health deficit (SD, aligned) per 100 d gap", "Frailty"),
                            ("body weight (SD within sex) per 100 d gap", "Body weight")]:
        x = a[a.outcome == out_name].sort_values("lab", key=lambda c: c.str.startswith("ALL").astype(str) + c)
        if short == "Body weight":      # only Facility 1 has enough mice for an interpretable weight estimate
            x = x[x.lab == F("Malavolta")]
        for _, r in x.iterrows():
            rows.append((f"{short}: {'pooled' if r.lab.startswith('ALL') else r.lab}", r.estimate, r.ci_low, r.ci_high,
                         r.lab.startswith("ALL"), short))
    lab = [r[0] for r in rows]
    forest(ax, lab, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
           [BLUE if r[5] == "Frailty" else GREY for r in rows], bold=[i for i, r in enumerate(rows) if r[4]])
    ax.set_xlabel("SD per 100 days older-looking"); ax.set_xlim(-0.4, 0.4)
    ax.set_title("Frailty and weight", loc="left")
    m = a[(a.outcome.str.startswith("health")) & a.lab.str.startswith("ALL")].iloc[0]
    key(f"health deficit per 100 d gap ({population}, meta)", f"{m.estimate:.3f} [{m.ci_low:.3f}, {m.ci_high:.3f}], p={m.p:.3g}, I2={m.heterogeneity_I2:.2f}",
        "biomarkers_embed/A_frailty_weight.csv")
    m = a[(a.outcome.str.startswith("body")) & (a.lab == F("Malavolta"))].iloc[0]
    key(f"body weight per 100 d gap ({F('Malavolta')}, {population})", f"{m.estimate:.3f} [{m.ci_low:.3f}, {m.ci_high:.3f}], p={m.p:.3g}",
        "biomarkers_embed/A_frailty_weight.csv")

    # b hazard ratio of gap across survival models
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b", -0.75)
    population = survival_pop; b = B[(B.population == POP_SURV[survival_pop]) & (B.term == "gap100")]
    names = {"2 age + photo age gap": "age + sex" + ("\n+ treatment" if population == "all" else ""), "4 age + frailty/vitality + photo age gap": "+ frailty/vitality",
             "5 sensitivity: first photo per mouse": "first photo only",
             f"6 per lab: {F('Malavolta')}": f"{F('Malavolta')} only"}     # Facility 4 alone: 28 mice, CI uninformative
    if B[B.model.str.startswith("6 per lab")].model.nunique() < 2:  # one facility left: its own row repeats the main one
        names.pop(f"6 per lab: {F('Malavolta')}")
    b = b[b.model.isin(names)].copy(); b["order"] = b.model.map(list(names).index); b = b.sort_values("order")
    forest(ax, [names[m] for m in b.model], b.HR, b.ci_low, b.ci_high,
           [BLUE if i == 0 else GREY for i in range(len(b))], ref=1.0, bold=[0])
    if len(b) < 5:
        pad = (5 - len(b)) / 2; ax.set_ylim(len(b) - 0.5 + pad, -0.5 - pad)
    ax.set_xlabel("Hazard ratio per 100 days older-looking"); ax.set_xlim(0.6, 1.6)
    ax.set_title("Survival after the photo" + (f" ({F('Malavolta')})" if survival_exclude else ""), loc="left")
    for _, r in b.iterrows():
        key(f"HR gap100 ({names[r.model]}; {population})", f"{r.HR:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}], p={r.p:.3g}, n={int(r.n)}, events={int(r.events)}",
            "biomarkers_embed/B_survival.csv")

    # c Kaplan-Meier by photographic age gap (controls, first photo per mouse, labs with survival)
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    s = M[M.ttd.notna() & M.dead.notna() & ~M.source.isin(survival_exclude)]
    if population == "controls":
        s = s[s.treated.astype(str).str.lower() == "false"]
    s = s.sort_values("age_days").drop_duplicates("mouse")
    s = s.assign(q=s.groupby("source").gap_adj_days.transform(lambda v: pd.qcut(v, 3, labels=False)))
    labels = ["younger", "middle", "older"]     # thirds of age gap (said in the legend text)
    for q, c in zip(range(3), [BLUE, GREY, ORANGE]):
        x = s[s.q == q]
        KaplanMeierFitter().fit(x.ttd, x.dead, label=labels[q]).plot_survival_function(ax=ax, ci_show=False, color=c, lw=1.5)
    lr = multivariate_logrank_test(s.ttd, s.q, s.dead)
    ax.set_xlabel("Days after photo"); ax.set_ylabel("Surviving"); ax.legend(loc="upper right", fontsize=5.5, title="looks", title_fontsize=5.5, handlelength=1.2,
                                                                                  labelspacing=0.25, borderaxespad=0.2)
    ax.text(0.03, 0.04, f"log-rank\np = {lr.p_value:.2g}\n{len(s)} mice", transform=ax.transAxes, fontsize=5.8)
    ax.set_title("Survival by photo age gap", loc="left"); ax.set_xlim(0, np.percentile(s.ttd, 99))
    key(f"KM by gap tertile ({population}, first photo)", f"log-rank p={lr.p_value:.3g}, n={len(s)}", "biomarkers_embed/merged_sessions.csv",
        "unadjusted illustration; inference is the Cox model")

    # d TERC vs their own wild-type controls, per generation (labels `cohort`: WTG3/TERCG3/WTG4/TERCG4)
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "d")
    import statsmodels.formula.api as smf
    L = D["labels"]
    t = pd.DataFrame()
    if "cohort" in L:
        # join on (anonymised facility, id): `mouse` keys keep the raw source name
        co = L.dropna(subset=["cohort"]).assign(k=lambda x: x.source + "|" + x.id.astype(str))
        co = co.drop_duplicates("k").set_index("k").cohort
        t = M.assign(k=M.source + "|" + M.id.astype(str))
        t = t[t.k.isin(co.index)].drop_duplicates(["mouse", "folder", "age_days"]).copy()
        t["cohort"] = t.k.map(co)
    if t.empty:
        placeholder(ax, "TERC cohorts: add a `cohort` column (WTG3/TERCG3/WTG4/TERCG4) to the labels")
    else:
        t["terc"] = t.cohort.str.startswith("TERC").astype(int); t["gen"] = t.cohort.str[-2:]; t["age100"] = t.age_days / 100
        rows = []
        for k, (gen, d_) in enumerate(t.groupby("gen")):
            for j, (name, v) in enumerate([("WT", d_[d_.terc == 0].gap_adj_days.values),
                                            (f"TERC {gen}", d_[d_.terc == 1].gap_adj_days.values)]):
                xpos = 3.2 * k + j
                jit = np.random.default_rng(10 * k + j).uniform(-0.15, 0.15, len(v))
                ax.scatter(np.full(len(v), xpos) + jit, v, s=8, color=[GREY, ORANGE][j], alpha=0.7, lw=0)
                ax.plot([xpos - 0.25, xpos + 0.25], [v.mean()] * 2, color=INK, lw=1.5)
            m = smf.ols("gap_adj_days ~ terc + age100", d_).fit(cov_type="HC1")
            lo, hi = m.conf_int().loc["terc"]; p = m.pvalues["terc"]
            rows.append(dict(generation=gen, mice_terc=int(d_.terc.sum()), mice_wt=int((1 - d_.terc).sum()),
                             gap_difference_days=m.params["terc"], ci_low=lo, ci_high=hi, p=p))
            ax.text(3.2 * k + 0.5, 0.98, f"{gen}: {m.params['terc']:+.0f} d\n[{lo:.0f}, {hi:.0f}]\n" +
                    ("p < 0.001" if p < 0.001 else f"p = {p:.2f}"), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.5)
            key(f"TERC {gen} gap difference vs own WT (d)",
                f"{m.params['terc']:.1f} [{lo:.1f}, {hi:.1f}], p={p:.3g}, {int(d_.terc.sum())} vs {int((1 - d_.terc).sum())} mice",
                "tables/terc_by_generation.csv")
        pd.DataFrame(rows).to_csv(os.path.join(out, "tables", "terc_by_generation.csv"), index=False)
        gens = sorted(t.gen.unique())
        ax.set_xticks([3.2 * k + j for k in range(len(gens)) for j in (0, 1)])
        ax.set_xticklabels([f"{n}\n{gen}\n({int(c)})" for k, gen in enumerate(gens)
                            for n, c in [("WT", ((t.gen == gen) & (t.terc == 0)).sum()), ("TERC", ((t.gen == gen) & (t.terc == 1)).sum())]],
                           fontsize=5.5)
        ax.axhline(0, color=GREY, lw=0.6, ls="--"); ax.set_xlim(-0.6, 3.2 * (len(gens) - 1) + 1.6)
        top, bot = t.gap_adj_days.max(), t.gap_adj_days.min(); ax.set_ylim(None, top + 0.6 * (top - bot))
        ax.set_ylabel("Photographic age gap (days)")
    ax.set_title("TERC vs own WT controls", loc="left")

    # e/f health deficit vs photographic age gap, one panel per lab with health scores (illustrates a)
    for k, (labn, typ, sign) in enumerate([(F("Malavolta"), "vitality", -1), (F("EPITERNA"), "frailty", 1)]):
        ax = fig.add_subplot(gs[1, 1 + k]); panel(ax, "ef"[k])
        x = M[(M.source == labn) & (M.type == typ)].dropna(subset=["score"])
        if frailty_pop == "controls":
            x = x[x.treated.astype(str).str.lower() == "false"]
        # residualise the score on age within the lab (treated mice pooled, no treatment term, as in panel a);
        # align direction (higher = worse); SD units
        X = np.column_stack([np.ones(len(x)), x.age_days.values])
        res = sign * (x.score.values - X @ np.linalg.lstsq(X, x.score.values, rcond=None)[0])
        z = res / res.std()
        ax.scatter(x.gap_adj_days, z, s=5, color=BLUE, alpha=0.4, lw=0)
        bb = np.polyfit(x.gap_adj_days, z, 1); xx = np.linspace(x.gap_adj_days.min(), x.gap_adj_days.max(), 50)
        ax.plot(xx, np.polyval(bb, xx), color=INK, lw=1.2)
        ax.set_xlabel("Photographic age gap (days)")
        ax.set_ylabel("Frailty, age-adjusted (SD)")
        ax.set_ylim(-2, 4)
        title = f"{labn}: {typ} score ({x.mouse.nunique()} mice)"
        est = D["A"][(D["A"].population == POP[frailty_pop]) & (D["A"].lab == labn)
                     & (D["A"].outcome == "health deficit (SD, aligned) per 100 d gap")]
        if len(est):
            e = est.iloc[0]
            title += (f"\n{e.estimate:+.2f} SD/100 d [{e.ci_low:.2f}, {e.ci_high:.2f}], "
                      + ("p<0.001" if e.p < 0.001 else f"p={e.p:.3f}"))
        ax.set_title(title, loc="left", fontsize=6.5)
    save(fig, out, "fig5_biomarker")


# ---------------------------------------------------------------- supplementary
def figS1_resolution(D, out):
    """448 vs 224 px: age performance only (the reason 448 was chosen)."""
    a, b = perf_sessions(D["gap_224"]), perf_sessions(D["gap_source"])
    m = a.merge(b, on=["mouse", "folder", "age_days"], suffixes=("_224", "_448"))
    rows = []
    rng = np.random.default_rng(0); mice, inv = np.unique(m.mouse, return_inverse=True)
    I, J, s = pairs_within(m.folder.values, m.age_days.values)
    c224 = (s * np.sign(m.pred_days_224.values[I] - m.pred_days_224.values[J]) + 1) / 2
    c448 = (s * np.sign(m.pred_days_448.values[I] - m.pred_days_448.values[J]) + 1) / 2
    e224, e448 = np.abs(m.pred_days_224 - m.age_days).values, np.abs(m.pred_days_448 - m.age_days).values
    dm, dc = [], []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice))[inv].astype(float)
        ww = w[I] * w[J]
        dm.append(np.average(e448 - e224, weights=w)); dc.append((ww * (c448 - c224)).sum() / ww.sum())
    res = dict(sessions=len(m), MAE_224=e224.mean(), MAE_448=e448.mean(), MAE_diff=e448.mean() - e224.mean(),
               MAE_diff_lo=np.percentile(dm, 2.5), MAE_diff_hi=np.percentile(dm, 97.5),
               conc_224=c224.mean(), conc_448=c448.mean(), conc_diff=c448.mean() - c224.mean(),
               conc_diff_lo=np.percentile(dc, 2.5), conc_diff_hi=np.percentile(dc, 97.5))
    pd.DataFrame([res]).to_csv(os.path.join(out, "tables", "resolution_224_vs_448.csv"), index=False)
    key("448 minus 224, session MAE (d)", f"{res['MAE_diff']:.1f} [{res['MAE_diff_lo']:.1f}, {res['MAE_diff_hi']:.1f}]", "eval_res/age_gap_frozen224/448",
        f"{res['MAE_224']:.0f} → {res['MAE_448']:.0f}")
    key("448 minus 224, within-session concordance", f"{res['conc_diff']:.3f} [{res['conc_diff_lo']:.3f}, {res['conc_diff_hi']:.3f}]",
        "eval_res/age_gap_frozen224/448", f"{res['conc_224']:.3f} → {res['conc_448']:.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(FULL * 0.75, 55 * MM), gridspec_kw=dict(wspace=0.55))
    p2, p4 = D["perlab_224"].set_index("source"), D["perlab_source"].set_index("source")
    labs = [l for l in LAB_ORDER if l in p4.index]; y = np.arange(len(labs))
    for ax, col, xl, letter in [(axs[0], "MAE", "MAE (days)", "a"), (axs[1], "within_session_conc", "Within-session concordance", "b")]:
        for yy, l in zip(y, labs):
            ax.plot([p2.loc[l, col], p4.loc[l, col]], [yy, yy], color=LIGHT, lw=1.5)
        ax.scatter(p2.loc[labs, col], y, color=GREY, s=20, zorder=3, label="224 px", ec="white", lw=0.5)
        ax.scatter(p4.loc[labs, col], y, color=BLUE, s=20, zorder=3, label="448 px", ec="white", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels(labs if letter == "a" else []); ax.invert_yaxis(); ax.set_xlabel(xl)
        panel(ax, letter)
    axs[1].axvline(0.5, color=GREY, lw=0.8, ls="--"); axs[0].legend(loc="lower right")
    axs[0].set_title(f"All facilities: {res['MAE_224']:.0f} → {res['MAE_448']:.0f} d", loc="left")
    axs[1].set_title(f"All facilities: {res['conc_224']:.3f} → {res['conc_448']:.3f}", loc="left")
    save(fig, out, "figS1_resolution")


def tables(D, out, regions=None):
    L = D["labels"]; L = L[L.source != "PaperImages"]
    t = L.groupby("source").agg(photos=("file", "size"), mice=("id", "nunique"), sessions=("folder", "nunique"),
                                age_min_d=("age_days", "min"), age_max_d=("age_days", "max"),
                                strains=("strain", lambda s: ", ".join(sorted(s.dropna().unique()))),
                                female_pct=("sex", lambda s: round(100 * (s == "Female").mean())))
    t.reindex([l for l in LAB_ORDER if l in t.index]).to_csv(os.path.join(out, "tables", "table1_dataset.csv"))
    D["A"].to_csv(os.path.join(out, "tables", "supp_frailty_weight.csv"), index=False)
    D["B"].to_csv(os.path.join(out, "tables", "supp_survival_models.csv"), index=False)
    D["perlab_source"].merge(D["perlab_mice"], on="source", suffixes=("_unseen_lab", "_unseen_mouse")).to_csv(
        os.path.join(out, "tables", "supp_per_lab.csv"), index=False)
    if regions is not None:
        g = regions[regions["mode"] == regions["mode"].iloc[0]].copy()
        g["oriented"] = g.orientation.isin(["tail", "tail+agree", "head-detector"])
        g["abs_err"] = (g.pred_days - g.age_days).abs()
        bias = g.groupby("oriented").agg(images=("file", "size"), mean_age_d=("age_days", "mean"),
                                         mean_abs_error_d=("abs_err", "mean"), mean_heat=("heat_mouse", "mean"))
        from scipy.stats import mannwhitneyu
        for c in ["age_days", "abs_err", "heat_mouse"]:
            bias[f"p_{c}"] = mannwhitneyu(g[g.oriented][c], g[~g.oriented][c]).pvalue
        bias.to_csv(os.path.join(out, "tables", "supp_saliency_orientation_bias.csv"))
        g.groupby("source").oriented.agg(["sum", "size"]).rename(columns={"sum": "oriented", "size": "images"}).to_csv(
            os.path.join(out, "tables", "supp_saliency_oriented_per_lab.csv"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--regions", help="regions_per_image.csv from region_saliency.py")
    ap.add_argument("--crops-root", help="cropped photos (Fig. 1 examples, Fig. 6 saliency examples)")
    ap.add_argument("--raw-root", help="original (downscaled) photos: Fig. 1a,b; found by relative path or file name")
    ap.add_argument("--yolo-root", help="YOLO square crops, same relative paths as the labels (Fig. 1a)")
    ap.add_argument("--examples", help="text file: relative paths of the Fig. 1b examples (default: youngest/oldest per lab)")
    ap.add_argument("--pipeline-tiles", nargs=3, help="Fig. 1a: images to show as photo, YOLO crop and mask (e.g. hand-cropped)")
    ap.add_argument("--rotate", help='Fig. 1b: text file of "degrees_clockwise relative/path" lines, to turn tiles upright')
    ap.add_argument("--pipeline", help="relative path (label `file`) of the mouse shown in Fig. 1a (default: Facility 1)")
    ap.add_argument("--sal", nargs="+", help="saliency_age.py output folders, e.g. grey=DIR blur=DIR")
    ap.add_argument("--sal-mode", default="grey", help="which occlusion mode Fig. 6 shows (grey or blur)")
    ap.add_argument("--sal-files", help="text file: relative paths of the Fig. 6a examples (default: automatic)")
    ap.add_argument("--n-sal", type=int, default=10, help="number of saliency examples")
    ap.add_argument("--frailty-pop", choices=["all", "controls"], default="all",
                    help="Fig. 4 health deficit / weight: all mice, not adjusted for treatment (default) or controls only")
    ap.add_argument("--survival-exclude", default="", help="sources left out of the survival panels (as in biomarkers.py)")
    ap.add_argument("--survival-pop", choices=["all", "controls"], default="controls",
                    help="Fig. 4 survival: control mice only (default) or all mice adjusted for treatment")
    ap.add_argument("--finetune", help="evaluate.py folder comparing frozen vs fine-tuned (eval_main)")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    p = lambda *x: os.path.join(a.data, *x)
    r = lambda *x: pd.read_csv(p(*x), encoding="utf-8-sig")
    D = dict(labels=r("labels_id_sex_clean.csv"), summary=r("eval_res", "summary.csv"),
             gap_source=r("eval_res", "age_gap_frozen448_source.csv"), gap_mice=r("eval_res", "age_gap_frozen448_mice.csv"),
             gap_224=r("eval_res", "age_gap_frozen224.csv"),
             perlab_source=r("eval_res", "per_lab_frozen448_source.csv"), perlab_mice=r("eval_res", "per_lab_frozen448_mice.csv"),
             perlab_224=r("eval_res", "per_lab_frozen224.csv"),
             gaptab_source=r("eval_res", "gap_table_frozen448_source.csv"),
             long_source=r("eval_res", "longitudinal_frozen448_source.csv"),
             calib_source=r("eval_res", "calibration_frozen448_source.csv"),
             A=r("biomarkers_embed", "A_frailty_weight.csv"), B=r("biomarkers_embed", "B_survival.csv"),
             C=r("biomarkers_embed", "C_groups_vs_same_session_controls.csv"), merged=r("biomarkers_embed", "merged_sessions.csv"),
             tasks=r("compare_tasks_source.csv"), crops=r("compare_crops_source.csv"),
             human=dict(accuracy=r("human_baseline", "accuracy.csv"), gap=r("human_baseline", "concordance_by_gap.csv"),
                        band=r("human_baseline", "by_age_band.csv"), cmp=r("human_baseline", "rater_vs_model.csv")))
    if os.path.exists(p("human_baseline", "self_consistency.csv")):
        D["human"]["self"] = r("human_baseline", "self_consistency.csv")
    regions = anonymise(pd.read_csv(a.regions)) if a.regions else None
    for k, v in list(D.items()):
        D[k] = {kk: anonymise(vv) for kk, vv in v.items()} if isinstance(v, dict) else anonymise(v)
    pd.DataFrame([(k, v) for k, v in FACILITY.items()], columns=["source", "label"]).to_csv(
        os.path.join(a.out, "tables", "facility_key.csv"), index=False)

    fig1(D, a.out, a.crops_root, a.raw_root, a.yolo_root, a.examples, a.pipeline, a.rotate, a.pipeline_tiles); fig2(D, a.out); fig3(D, a.out)
    fig4(D, a.out, a.frailty_pop, a.survival_pop, [F(x.strip()) for x in a.survival_exclude.split(",") if x.strip()])
    fig6(D, a.out, regions, a.sal, a.crops_root, a.sal_files, a.n_sal, mode=a.sal_mode)
    figS1_resolution(D, a.out)
    figS2_groups(D, a.out)
    if a.finetune:
        finetune_numbers(a.finetune)
    tables(D, a.out, regions)
    pd.DataFrame(KEY).to_csv(os.path.join(a.out, "key_numbers.csv"), index=False)
    print(f"done: {a.out}/ (figures, tables/, key_numbers.csv)")


if __name__ == "__main__":
    main()
