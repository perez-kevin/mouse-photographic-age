"""
Build a human age-rating task: one self-contained HTML file per rater + a hidden answer key.

  python make_rating_task.py --csv labels_id_sex_clean.csv --crops /path/to/crops \
      --raters "Rater1,Rater2,Rater3" --out rating_task

Sample design (all raters see the same images, each in their own random order):
  * one image per mouse (no mouse counted twice)
  * ~60% of images come in groups from the same photo session, spread over that session's ages
    -> lets you compute within-session concordance for humans exactly as for the model
  * the rest fills the age bands (<4, 4-9, 9-15, 15-21, >21 months) evenly, spreading over labs/strains
  * --repeats hidden repeats (another photo of the same mouse at the same age if available,
    otherwise the same image), placed far apart -> each rater's self-consistency
  * excluded: treated mice (column `treatment` not empty/"control", or `use_for_training` == no),
    images whose crop was flagged (with --qc), and any --exclude-sources

Outputs in --out:
  rating_<rater>.html   send one to each rater; works offline in any browser, no installation;
                        progress is kept in the browser if they close it; they download a CSV at the end
  answer_key.csv        KEEP PRIVATE: image id -> file, true age, session, repeat info
"""
import argparse, base64, hashlib, io, json, os, random
import numpy as np, pandas as pd
from PIL import Image

MONTH = 30.44
BANDS = [0, 4, 9, 15, 21, 99]                      # months
BAND_NAMES = ["<4 mo", "4-9 mo", "9-15 mo", "15-21 mo", ">21 mo"]


def load(a):
    df = pd.read_csv(a.csv, encoding="utf-8-sig")
    df["mouse"] = df["source"].astype(str) + "|" + df["id"].astype(str)
    df["path"] = [os.path.join(a.crops, f) for f in df["file"]]
    n0 = len(df)
    df = df[df["path"].map(os.path.exists)]
    print(f"{len(df)} of {n0} images have a crop")
    if "treatment" in df:
        t = df["treatment"].fillna("").astype(str).str.strip().str.lower()
        df = df[t.isin(["", "control", "ctrl", "vehicle", "none"])]
    for col in ["training", "use_for_training"]:
        if col in df:
            df = df[~df[col].astype(str).str.strip().str.lower().isin(["no", "n", "false", "0"])]
    if a.qc:
        qc = pd.read_csv(a.qc)
        bad = set(qc.loc[qc["flags"].fillna("").ne(""), "file"])
        df = df[~df["file"].isin(bad)]
    if a.exclude_sources:
        df = df[~df["source"].isin([s.strip() for s in a.exclude_sources.split(",")])]
    df = df.copy()
    df["band"] = pd.cut(df.age_days / MONTH, BANDS, labels=BAND_NAMES, right=False)
    print(f"{len(df)} eligible images, {df.mouse.nunique()} mice")
    return df.reset_index(drop=True)


def select(df, a, rng):
    # one candidate photo per mouse
    pool = df.sample(frac=1, random_state=a.seed).drop_duplicates("mouse").copy()
    chosen = []

    # 1) session groups: sessions with the most distinct ages, images spread over their ages
    n_sess_total = int(round(a.n * a.session_frac))
    per = max(3, n_sess_total // max(a.sessions, 1))
    ages_per_folder = pool.groupby("folder").age_days.nunique().sort_values(ascending=False)
    folders = [f for f in ages_per_folder.index if ages_per_folder[f] >= 3]
    # prefer different labs among the chosen sessions
    by_lab, picked = {}, []
    for f in folders:
        lab = pool.loc[pool.folder == f, "source"].iloc[0]
        by_lab.setdefault(lab, []).append(f)
    while len(picked) < a.sessions and any(by_lab.values()):
        for lab in list(by_lab):
            if by_lab[lab] and len(picked) < a.sessions:
                picked.append(by_lab[lab].pop(0))
    for f in picked:
        g = pool[pool.folder == f].sort_values("age_days")
        idx = np.unique(np.linspace(0, len(g) - 1, min(per, len(g))).round().astype(int))
        chosen += g.index[idx].tolist()

    # 2) fill age bands evenly, spreading over labs
    target = a.n / len(BAND_NAMES)
    rest = pool.drop(index=chosen)
    while len(chosen) < a.n and len(rest):
        sel = pool.loc[chosen]
        counts = sel.band.value_counts().reindex(BAND_NAMES, fill_value=0)
        avail = [b for b in BAND_NAMES if (rest.band == b).any()]
        if not avail:
            break
        b = min(avail, key=lambda x: counts[x] - target)          # band with the biggest deficit
        cand = rest[rest.band == b]
        lab_counts = sel.source.value_counts()
        cand = cand.assign(lc=cand.source.map(lab_counts).fillna(0))
        cand = cand[cand.lc == cand.lc.min()]                      # least-represented lab first
        i = cand.sample(1, random_state=int(rng.integers(1e9))).index[0]
        chosen.append(i); rest = rest.drop(index=i)

    sel = pool.loc[chosen].copy()
    sel["in_session_group"] = sel.folder.isin(picked)
    return sel, picked


def add_repeats(sel, df, a, rng):
    rows = []
    for i in rng.choice(len(sel), min(a.repeats, len(sel)), replace=False):
        r = sel.iloc[i]
        other = df[(df.mouse == r.mouse) & (df.age_days == r.age_days) & (df.file != r.file)]
        rep = (other.iloc[0] if len(other) else r).copy()
        rep["repeat_of_file"] = r.file
        rep["repeat_type"] = "other_photo" if len(other) else "same_image"
        rows.append(rep)
    return pd.DataFrame(rows)


def encode(path, max_side):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)          # re-encode: no metadata, no filename
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def order_for_rater(items, rng, min_gap=20):
    """Random order with each repeat far from its original."""
    ids = [it["id"] for it in items]
    pairs = {it["id"]: it["pair"] for it in items if it["pair"]}
    for _ in range(2000):
        o = list(rng.permutation(ids))
        pos = {x: k for k, x in enumerate(o)}
        if all(abs(pos[x] - pos[y]) >= min(min_gap, len(o) // 3) for x, y in pairs.items()):
            return o
    return o


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mouse Age Rating</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a65;--line:#dedcd6;--accent:#2f5d50;--accent-ink:#fff;--warn:#a4402c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171716;--card:#222220;--ink:#ecebe6;--muted:#9c9a93;--line:#3a3935;--accent:#7fb3a3;--accent-ink:#10201b;--warn:#e08a74}}
:root[data-theme="dark"]{--bg:#171716;--card:#222220;--ink:#ecebe6;--muted:#9c9a93;--line:#3a3935;--accent:#7fb3a3;--accent-ink:#10201b;--warn:#e08a74}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:720px;margin:0 auto;padding:16px 16px 32px}
h1{font-size:22px;margin:0 0 4px}p{margin:8px 0}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
label{display:block;font-weight:600;margin:14px 0 6px}
input,select{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 12px;width:100%}
input:focus,select:focus,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
button{font:inherit;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:10px 16px;cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:600}
button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:6px 0 16px}.bar>div{height:100%;background:var(--accent);width:0}
.img{background:#808080;border-radius:10px;display:flex;justify-content:center;align-items:center;aspect-ratio:1/1;overflow:hidden;width:min(100%,52vh);margin:0 auto}
.img img{width:100%;height:100%;object-fit:contain;display:block}
.err{color:var(--warn);min-height:1.5em;margin:6px 0 0}
.hidden{display:none}
ol{padding-left:20px}kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:13px}
</style></head><body><main>

<section id="intro" class="card">
  <h1>Mouse age rating</h1>
  <p class="muted">__N__ photos · about 30 minutes · you can stop and continue later</p>
  <ol>
    <li>Each photo shows one laboratory mouse, cut out from its background.</li>
    <li>Estimate its age <b>in months</b>. Ages in this set range from about 1 to 34 months.</li>
    <li>Go with your first impression; there is no feedback.</li>
    <li>At the end, download your results file and send it back.</li>
  </ol>
  <p class="muted">Your answers are saved in this browser as you go. If you close this page, reopen the same file in the same browser to continue.</p>
  <label for="rname">Your name or initials</label><input id="rname" value="__RATER__">
  <label for="role">Role</label>
  <select id="role"><option value="">Choose…</option><option>Animal technician / caretaker</option><option>Veterinarian</option><option>Researcher working with mice</option><option>Researcher, little mouse experience</option><option>Other</option></select>
  <label for="years">Years working with laboratory mice</label><input id="years" type="number" min="0" max="60" step="0.5">
  <label for="lab">Have you worked in any of the labs/facilities that provided these photos?</label>
  <select id="lab"><option value="">Choose…</option><option>No</option><option>Yes</option><option>Not sure</option></select>
  <p class="err" id="introErr"></p>
  <div class="row"><button class="primary" id="start">Start</button></div>
</section>

<section id="task" class="hidden">
  <div class="row" style="justify-content:space-between"><span id="count" class="muted"></span><button id="dl1" class="muted" style="border:0;background:none;padding:0;text-decoration:underline">download answers so far</button></div>
  <div class="bar"><div id="prog"></div></div>
  <div class="img"><img id="photo" alt="Mouse photo"></div>
  <label for="age">Estimated age (months)</label>
  <input id="age" type="number" min="0.5" max="48" step="0.5" inputmode="decimal" placeholder="e.g. 14">
  <p class="err" id="taskErr"></p>
  <div class="row" style="justify-content:space-between;margin-top:8px">
    <button id="back">← Back</button>
    <span class="muted" style="font-size:13px"><kbd>Enter</kbd> next</span>
    <button class="primary" id="next">Next →</button>
  </div>
</section>

<section id="done" class="card hidden">
  <h1>Done, thank you</h1>
  <p>Please download your results and send the file to <b>__CONTACT__</b>.</p>
  <div class="row"><button class="primary" id="dl2">Download results (CSV)</button><button id="review">Go back and review</button></div>
  <p class="muted" id="dlNote"></p>
</section>

</main>
<script>
const TASK="__TASK__", RATER_DEFAULT="__RATER__";
const ITEMS=__ITEMS__;
const KEY="mouse-age-rating-"+TASK;
let st={info:null,i:0,ans:{}};
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}}
function load(){try{const s=localStorage.getItem(KEY);if(s)st=JSON.parse(s)}catch(e){}}
const $=id=>document.getElementById(id);
let shownAt=0;

function show(sec){["intro","task","done"].forEach(s=>$(s).classList.toggle("hidden",s!==sec))}
function render(){
  const it=ITEMS[st.i], a=st.ans[it.id]||{};
  $("photo").src=it.src;
  $("count").textContent=`Photo ${st.i+1} of ${ITEMS.length}`;
  $("prog").style.width=(100*Object.keys(st.ans).length/ITEMS.length)+"%";
  $("age").value=a.age??""; $("taskErr").textContent="";
  $("back").disabled=st.i===0;
  $("next").textContent=st.i===ITEMS.length-1?"Finish":"Next →";
  shownAt=performance.now(); $("age").focus();
}
function record(){
  const it=ITEMS[st.i], v=parseFloat(String($("age").value).replace(",","."));
  if(!isFinite(v)||v<0.5||v>48){$("taskErr").textContent="Please enter an age between 0.5 and 48 months.";return false}
  const prev=st.ans[it.id]||{};
  st.ans[it.id]={age:v,
    secs:Math.round(((prev.secs||0)+(performance.now()-shownAt)/1000)*10)/10,
    position:st.i+1,at:new Date().toISOString()};
  save(); return true;
}
function csv(){
  const info=st.info||{}, esc=s=>`"${String(s??"").replace(/"/g,'""')}"`;
  const head=["task","rater","role","years_experience","worked_in_source_lab","image_id","position","age_months","seconds","answered_at"];
  const rows=ITEMS.map(it=>{const a=st.ans[it.id]||{};return [TASK,info.name,info.role,info.years,info.lab,it.id,a.position,a.age,a.secs,a.at].map(esc).join(",")});
  return head.join(",")+"\n"+rows.join("\n")+"\n";
}
function download(){
  const name=`ratings_${(st.info&&st.info.name||RATER_DEFAULT).replace(/[^A-Za-z0-9_-]+/g,"_")}.csv`;
  const url=URL.createObjectURL(new Blob([csv()],{type:"text/csv"}));
  const a=document.createElement("a");a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),2000);
  $("dlNote").textContent=`Saved as ${name} (usually in your Downloads folder).`;
}
$("start").onclick=()=>{
  const info={name:$("rname").value.trim(),role:$("role").value,years:$("years").value,lab:$("lab").value};
  if(!info.name||!info.role||info.years===""||!info.lab){$("introErr").textContent="Please fill in all four fields.";return}
  st.info=info; save(); show("task"); render();
};
$("next").onclick=()=>{ if(!record())return;
  if(st.i<ITEMS.length-1){st.i++;save();render()}
  else{const miss=ITEMS.findIndex(it=>!st.ans[it.id]);
    if(miss>=0){st.i=miss;save();render();$("taskErr").textContent="This photo has no answer yet."}
    else{show("done")}}
};
$("back").onclick=()=>{ if($("age").value!=="")record(); if(st.i>0){st.i--;save();render()} };
document.addEventListener("keydown",e=>{
  if($("task").classList.contains("hidden"))return;
  if(e.key==="Enter"){e.preventDefault();$("next").click()}
});
$("dl1").onclick=download; $("dl2").onclick=download;
$("review").onclick=()=>{st.i=0;show("task");render()};
load();
if(st.info){ $("rname").value=st.info.name; $("role").value=st.info.role; $("years").value=st.info.years; $("lab").value=st.info.lab;
  if(Object.keys(st.ans).length===ITEMS.length)show("done"); else {show("task");render()} }
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="labels_id_sex_clean.csv")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--raters", required=True, help='comma-separated names, e.g. "Anna,Ben,Chiara"')
    ap.add_argument("--out", default="rating_task")
    ap.add_argument("--n", type=int, default=100, help="unique mice to rate")
    ap.add_argument("--repeats", type=int, default=10, help="hidden repeats for self-consistency")
    ap.add_argument("--sessions", type=int, default=10, help="sessions contributing a group of images")
    ap.add_argument("--session-frac", type=float, default=0.6)
    ap.add_argument("--qc", help="crop_qc.csv: drop images whose crop was flagged")
    ap.add_argument("--exclude-sources", default="", help="comma-separated labs to leave out")
    ap.add_argument("--contact", default="the study organiser", help="who raters send results to")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    df = load(a)
    sel, sessions = select(df, a, rng)
    rep = add_repeats(sel, df, a, rng)
    sel["repeat_of_file"], sel["repeat_type"] = "", ""
    allrows = pd.concat([sel, rep], ignore_index=True)

    # random, uninformative ids shared by all raters
    ids = set()
    while len(ids) < len(allrows):
        ids.add("".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"), 5)))
    allrows["image_id"] = sorted(ids, key=lambda _: rng.random())
    first = dict(zip(sel.file, allrows.image_id[:len(sel)]))
    allrows["repeat_of_id"] = allrows.repeat_of_file.map(first).fillna("")

    print("encoding images…")
    src = {r.image_id: encode(r.path, a.max_side) for r in allrows.itertuples()}
    items = [{"id": r.image_id, "pair": r.repeat_of_id} for r in allrows.itertuples()]
    task = hashlib.md5(("".join(sorted(src))).encode()).hexdigest()[:8]

    for name in [r.strip() for r in a.raters.split(",") if r.strip()]:
        order = order_for_rater(items, np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest(), 16) % 2**32))
        payload = json.dumps([{"id": i, "src": src[i]} for i in order])
        html = (HTML.replace("__ITEMS__", payload).replace("__N__", str(len(order)))
                .replace("__TASK__", task).replace("__RATER__", name.replace('"', ""))
                .replace("__CONTACT__", a.contact))
        fn = os.path.join(a.out, f"rating_{''.join(c if c.isalnum() else '_' for c in name)}.html")
        open(fn, "w", encoding="utf-8").write(html)
        print(f"  {fn}  ({os.path.getsize(fn) / 1e6:.1f} MB)")

    key = allrows[["image_id", "file", "folder", "source", "strain", "sex", "mouse", "age_days", "band",
                   "in_session_group", "repeat_of_id", "repeat_type"]].copy()
    key["in_session_group"] = key["in_session_group"].fillna(False)
    key.to_csv(os.path.join(a.out, "answer_key.csv"), index=False)

    u = allrows[allrows.repeat_type == ""]
    print(f"\n{len(u)} unique mice + {len(rep)} hidden repeats = {len(allrows)} photos per rater")
    print("by age band:", u.band.value_counts().reindex(BAND_NAMES, fill_value=0).to_dict())
    print("by lab:     ", u.source.value_counts().to_dict())
    print("by strain:  ", u.strain.value_counts().to_dict())
    print(f"session groups ({len(sessions)}):", ", ".join(sessions))
    print(f"\nanswer key: {os.path.join(a.out, 'answer_key.csv')}  (keep private)")


if __name__ == "__main__":
    main()
