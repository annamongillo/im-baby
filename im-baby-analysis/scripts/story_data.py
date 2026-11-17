"""Per-language data for the 'pick your language' scrollytelling page."""
import sys, os, re, json
import numpy as np
import pandas as pd

work, freqdir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
A = pd.read_csv(os.path.join(work, "aoa_classified.csv"))
EXCLUDE = "Sign|Malaysian|Tsonga"
A = A[~A.language.str.contains(EXCLUDE)]
A = A[A.lex_class != "other"]
# split predicates into the CDI's two categories: action words (verbs) and descriptive words (adjectives)
pred = A.lex_class == "predicates"
A.loc[pred & A.category.str.startswith("action"), "lex_class"] = "verbs"
A.loc[pred & A.category.str.startswith("descriptive"), "lex_class"] = "adjectives"
A = A[A.lex_class != "predicates"]
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


# Some checklists join words with "_": phrases (buenas_noches), reflexives (comer_se) or alternatives (pene_pito).
PHRASE_US = {"por_que": "por qué", "salud_chin_chin": "salud", "siesta_noni_noni": "siesta", "otro_otra_vez": "otro",
             "pileta_de_natación_Pile": "pileta de natación", "por_favor_porfi": "por favor", "sí_afirmación": "sí",
             "bee_mee": "bee", "a_al": "a", "de_del": "de", "hebilla_pelo_gomita": "hebilla", "all_v": "all",
             "tylko_quantifiers": "tylko", "nombre_del_niño_a": "nombre del niño"}
JOINERS = {"de", "del", "al", "te", "ya", "está", "acá", "fritas", "noches", "días", "lado", "dos", "tres", "bien", "lavar"}
def unscore(x):
    def one(t):
        if "_" not in t: return t
        if t in PHRASE_US: return PHRASE_US[t]
        parts = t.split("_")
        if parts[-1] == "se": return "_".join(parts[:-1]).replace("_", " ") + "(se)"
        if any(q in JOINERS for q in parts) or len(parts) > 2: return t.replace("_", " ")
        return parts[0]
    return re.sub(r"[^\s,/]+", lambda m: one(m.group(0)), x)


def display(w, uni=""):
    vs = variants(w)
    full = ", ".join(vs) if vs else str(w)
    s = ""
    for v in vs:
        c = re.split(r"/", re.sub(r"[(（].*?[)）]", "", v))[0].strip()
        if c: s = c; break
    if not s: s = full
    if "_" in s: s, full = unscore(s), unscore(full)
    u = re.sub(r" \(.*\)$", "", uni)
    if " " in s and u and " " not in u and u.lower() in s.lower().split():
        s = u   # e.g. "oink snort" -> "oink"
    return s, full


CLS = {"nouns": 0, "verbs": 1, "function words": 2, "sounds & routines": 3, "adjectives": 4}

# ---- official item glosses from the Wordbank checklist definitions (one CSV per instrument)
import csv, glob as _glob
INSTR = sys.argv[4] if len(sys.argv) > 4 else None


def instr_prefix(lang):
    m = re.match(r"^(.*) \((.*)\)$", lang)
    if not m: return lang.replace(" ", "_")
    base, reg = m.groups()
    return base if base == reg and base != "French" else f"{base}_{reg}"


GLOSS = {}
if INSTR:
    for f in _glob.glob(os.path.join(INSTR, "*", "[[]*[]].csv")):
        d = os.path.basename(os.path.dirname(f))
        with open(f, encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                if (row.get("type") or "") != "word": continue
                g = (row.get("gloss") or "").strip(); u = (row.get("uni_lemma") or "").strip()
                if not g or g == "NA" or not u or u == "NA": continue
                GLOSS.setdefault(d, []).append((u, clean_variant((row.get("definition") or "").replace("*", "")).lower(), g))


def tidy_gloss(g):
    s = re.sub(r"\s*\((not |object|food|animal|action|beverage|verb|noun|adj|description|place|toy)[^)]*\)", "", g).strip()
    if ": " in s: s = s.split(": ", 1)[1].strip()          # "Sheep sound: baa baa" -> "baa baa"
    s = re.sub(r"^\([^)]*\)\s*", "", s)                  # "(Bed) sheet" -> "sheet"
    m = re.match(r"^(.*?)\s*\(([^)]*)\)$", s)               # "au_Au (dog barking)" -> "dog barking"; "chocolate drink (hot or cold)" -> "chocolate drink"
    if m: s = "" if ("_" in m.group(1) or "sound" in m.group(2)) else m.group(1)   # underscore sound spellings: fall back to the meaning label
    s = re.split(r"\s*[/\\]\s*", s)[0].strip()               # first of several alternatives
    if s.count(",") >= 2: s = s.split(",")[0].strip()
    s = s.replace("_", " ")
    if len(s) > 1 and s[0].isupper() and s[1].islower() and s.split()[0] not in ("I", "I'm"): s = s[0].lower() + s[1:]
    return s


def official_gloss(lang, uni, short, full):
    pre = instr_prefix(lang)
    dirs = [d for d in GLOSS if d == pre or d.startswith(pre + "_")]
    cands = [c for d in dirs for c in GLOSS[d] if c[0] == uni]
    if not cands: return None
    forms = [short.lower()] + [v.lower() for v in variants(full)]
    for fm in forms:
        for u, dfn, g in cands:
            if dfn == fm or fm in [x.strip() for x in re.split(r"[/,]", dfn)]: return tidy_gloss(g) or None
    # most common gloss for this meaning in this language
    from collections import Counter
    return tidy_gloss(Counter(g for _, _, g in cands).most_common(1)[0][0]) or None
# Grammatical-number corrections: the checklist gloss names the meaning, but the
# English must match the number of the native form (ojos -> eyes, ojo -> eye).
# Corrected by hand for grammatical number; not a full dictionary audit.
NUMBER_FIX = {
 "French": {"chaussettes":"socks","médicaments":"medicines","yeux":"eyes","fesses":"buttocks","bottes":"boots","boutons":"buttons",
            "noisettes":"hazelnuts","bonbons":"sweets","pantalon":"trousers","photo":"photo","culotte":"underpants",
            "escalier":"stairs","raisin":"grapes","basket":"sneaker","lèvre":"lip","bobo":"boo-boo"},
 "Italian": {"biscotti":"cookies","bollicina":"bubble","pomodori":"tomatoes","cubi":"blocks","pupazzi":"soft toys","carota":"carrot",
             "labbra":"lips","uva":"grapes","sederino":"bottom","bosco":"woods"},
 "Portuguese": {"guardanapos":"napkins","maminhas":"boobs","ervilhas":"peas","lençóis":"sheets","ombros":"shoulders",
                "pestanas":"eyelashes","sobracelhas":"eyebrows","ameixas":"plums","costas":"back",
                "mão":"hand","olho":"eye","meia":"sock","braço":"arm","perna":"leg","chave":"key","flôr":"flower",
                "unha":"nail","bolha":"bubble","pé":"foot","dente":"tooth","médico":"doctor","roupa":"clothes","cubo":"block","bochecha":"cheek","tesoura":"scissors"},
 "Spanish": {"pañales":"diapers","piernas":"legs","botas":"boots","monedas":"coins","cebras":"zebras",
             "pinturas":"paints","pinturitas":"crayons","vitaminas":"vitamins","palomitas":"popcorn",
             "zapatilla":"sneaker","zapato":"shoe","ojo":"eye","calcetín":"sock","brazo":"arm","escalera":"stairs",
             "mano":"hand","oreja":"ear","media":"sock","dedo":"finger","pierna":"leg","arete":"earring",
             "labio":"lip","burbuja":"bubble","pie":"foot","diente":"tooth","bombacha":"panties","fosforo":"match","caballito":"little horse"},
 "Catalan": {"croquetes":"croquettes","cavallets":"merry-go-round"},
 # Polish and Faroese: no official checklist glosses exist,
 # so the meaning label is used, corrected for number.
 "Polish": {"buty":"shoes","klocki":"blocks","kredki":"crayons","skarpetki":"socks","ziemniaki":"potatoes",
            "rękawiczki":"gloves","pantofle":"slippers","kozaki":"boots","truskawki":"strawberries","paznokcie":"nails",
            "grabki":"rake","naleśniki":"pancakes","goście":"guests","imieniny":"name day","pierogi":"dumplings",
            "grzyby":"mushrooms","buraczki":"beets","orzechy":"nuts","witaminy":"vitamins","góry":"mountains",
            "rodzynki":"raisins","frytki":"fries","lody":"ice cream","płatki":"cereal","pan":"mister"},
 "Faroese": {"skógvar":"shoes","bollar":"cups","rosinir":"raisins","gularøtur":"carrots","frikadellur":"meatballs",
             "pannukøkur":"pancakes","inniskógvar":"slippers","vaflur":"waffles","vøttir":"mittens","pulshosur":"socks",
             "nøtir":"nuts","serviettar":"napkins","renniskógvar":"sneakers","boppur":"boob","fastir":"aunt"},
}
def number_fix(lang, short, g):
    for pre, m in NUMBER_FIX.items():
        if lang.startswith(pre):
            k = short.lower()
            if k in m: return m[k]
    return g
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
                     f,
                     number_fix(lang, short, official_gloss(lang, r.uni_lemma, short, full)) if isinstance(r.word, str) else None])
    data[lang] = rows
    first50 = rows[:50]
    summary.append(dict(lang=lang, n=len(rows), comp=sum(1 for x in rows if x[4] is not None),
                        freq=code is not None,
                        f50_matched=sum(1 for x in first50 if x[6]), f50_top1000=sum(1 for x in first50 if x[6] and x[6] <= 1000)))
    langs.append(lang)

# noun vs predicate gap for all languages (for the comparison step)
cls = P[P.language.isin(langs)].pivot_table(index="language", columns="lex_class", values="aoa", aggfunc="median")
gaps = {l: round(float(cls.loc[l, "verbs"] - cls.loc[l, "nouns"]), 1) for l in langs}      # verbs vs nouns
gapsA = {l: round(float(cls.loc[l, "adjectives"] - cls.loc[l, "nouns"]), 1) for l in langs}  # adjectives vs nouns

out = dict(langs=langs, data=data, gaps=gaps, gapsA=gapsA, nlang={k: int(v) for k, v in nlang.items()},
           classes=list(CLS.keys()))
s = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
open(out_path, "w").write(s)
print(len(langs), "languages,", round(len(s) / 1e6, 2), "MB")
print(pd.DataFrame(summary).to_string(index=False))
