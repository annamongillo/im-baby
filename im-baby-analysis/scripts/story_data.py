"""Per-language data for the 'pick your language' scrollytelling page."""
import sys, os, re, json
import numpy as np
import pandas as pd

work, freqdir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
A = pd.read_csv(os.path.join(work, "aoa_classified.csv"))
EXCLUDE = "Sign|Malaysian|Tsonga"
A = A[~A.language.str.contains(EXCLUDE)]
A = A[A.lex_class != "other"]
P = A[A.measure == "produces"]
U = A[A.measure == "understands"]

# cross-language median AoA per meaning (needs 8+ languages)
g = P.groupby("uni_lemma")["aoa"].agg(["median", "count"])
gmed = g.loc[g["count"] >= 8, "median"]
nlang = P.groupby("uni_lemma")["language"].nunique()

FREQ = {"English (American)": "en", "English (British)": "en", "English (Australian)": "en",
        "Spanish (Mexican)": "es", "Spanish (European)": "es", "Spanish (Argentinian)": "es", "Spanish (Peruvian)": "es",
        "French (French)": "fr", "French (Quebecois)": "fr", "German (German)": "de", "Italian": "it",
        "Mandarin (Beijing)": "zh_cn", "Mandarin (Taiwanese)": "zh_tw", "Korean": "ko",
        "Portuguese (European)": "pt", "Russian": "ru", "Hebrew": "he", "Turkish": "tr", "Polish": "pl",
        "Dutch": "nl", "Swedish": "sv", "Danish": "da", "Norwegian": "no", "Czech": "cs", "Croatian": "hr",
        "Estonian": "et", "Finnish": "fi", "Hungarian": "hu", "Latvian": "lv", "Slovak": "sk", "Catalan": "ca",
        "Greek (Cypriot)": "el", "Persian": "fa", "Arabic (Saudi)": "ar"}
freq_cache = {}


def freq(code):
    if code not in freq_cache:
        fr = pd.read_csv(os.path.join(freqdir, code, f"{code}_50k.txt"), sep=" ", header=None,
                         names=["w", "c"], quoting=3, keep_default_na=False, dtype={"w": str})
        ranks = {}
        for i, w in enumerate(fr["w"].str.lower(), 1):
            ranks.setdefault(w, i)
        freq_cache[code] = ranks
    return freq_cache[code]


def clean(w):
    w = str(w).lower()
    w = re.sub(r"\(.*?\)", "", w).replace("*", "")
    w = re.split(r"[/,]", w)[0]
    return w.strip()


def clean_variant(v):
    v = re.sub(r"^[A-Z]\s+[A-Za-z]+\s*-\s*\d+\s*", "", v.strip())   # form codes like "A sounds - 8 "
    v = re.sub(r"\.{2,}\s*\d+$", "", v)                             # "cap...193"
    v = re.sub(r"(?<=[^\d\s])\d+$", "", v)                           # "peix2"
    return v.strip()


def variants(w):
    raw = re.sub(r"\s+", " ", str(w).replace("*", "")).strip()
    out = []
    for v in re.split(r"[,，、]", raw):
        v = clean_variant(v)
        if v and v not in out: out.append(v)
    return out


def display(w, uni=""):
    vs = variants(w)
    full = ", ".join(vs) if vs else str(w)
    s = ""
    for v in vs:
        c = re.split(r"/", re.sub(r"[(（].*?[)）]", "", v))[0].strip()
        if c: s = c; break
    if not s: s = full
    u = re.sub(r" \(.*\)$", "", uni)
    if " " in s and u and " " not in u and u.lower() in s.lower().split():
        s = u   # e.g. "oink snort" -> "oink"
    return s, full


CLS = {"nouns": 0, "predicates": 1, "function words": 2, "sounds & routines": 3}
langs, data, summary = [], {}, []
for lang, d in P.groupby("language"):
    if len(d) < 300:
        continue
    d = d.sort_values("aoa").drop_duplicates("uni_lemma")
    u = U[U.language == lang].set_index("uni_lemma")["aoa"]
    code = FREQ.get(lang)
    ranks = freq(code) if code else None
    rows = []
    for r in d.itertuples():
        f = None
        if ranks is not None and isinstance(r.word, str):
            vs = variants(r.word); form = clean(vs[0]) if vs else ""
            multi = bool(re.search(r"[/,]", str(r.word)))
            if form and (" " not in form or code.startswith("zh")) and not (multi and r.lex_class == "sounds & routines"):
                f = ranks.get(form)
        short, full = display(r.word, r.uni_lemma) if isinstance(r.word, str) else (r.uni_lemma, r.uni_lemma)
        rows.append([r.uni_lemma, short if short == full else [short, full],
                     CLS[r.lex_class], round(float(r.aoa), 1),
                     round(float(u[r.uni_lemma]), 1) if r.uni_lemma in u.index and not pd.isna(u[r.uni_lemma]) else None,
                     round(float(gmed[r.uni_lemma]), 1) if r.uni_lemma in gmed.index else None,
                     f])
    data[lang] = rows
    first50 = rows[:50]
    summary.append(dict(lang=lang, n=len(rows), comp=sum(1 for x in rows if x[4] is not None),
                        freq=code is not None,
                        f50_matched=sum(1 for x in first50 if x[6]), f50_top1000=sum(1 for x in first50 if x[6] and x[6] <= 1000)))
    langs.append(lang)

# noun vs predicate gap for all languages (for the comparison step)
cls = P[P.language.isin(langs)].pivot_table(index="language", columns="lex_class", values="aoa", aggfunc="median")
gaps = {l: round(float(cls.loc[l, "predicates"] - cls.loc[l, "nouns"]), 1) for l in langs}

out = dict(langs=langs, data=data, gaps=gaps, nlang={k: int(v) for k, v in nlang.items()},
           classes=list(CLS.keys()))
s = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
open(out_path, "w").write(s)
print(len(langs), "languages,", round(len(s) / 1e6, 2), "MB")
print(pd.DataFrame(summary).to_string(index=False))
