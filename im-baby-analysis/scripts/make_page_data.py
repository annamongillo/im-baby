import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

work = sys.argv[1]
A = pd.read_csv(os.path.join(work, "aoa_classified.csv"))
EXCLUDE = "Sign|Malaysian|Tsonga"
P = A[(A.measure == "produces") & ~A.language.str.contains(EXCLUDE)]
r1 = lambda x: None if pd.isna(x) else round(float(x), 1)
out = {}

# headline + consistency
uni = pd.read_csv(os.path.join(work, "universal_words.csv"))
rhos = []
for lang, d in P.groupby("language"):
    others = P[P.language != lang].groupby("uni_lemma")["aoa"].median()
    m = d.set_index("uni_lemma")["aoa"].to_frame().join(others.rename("o"), how="inner")
    if len(m) >= 50:
        rhos.append(spearmanr(m["aoa"], m["o"])[0])
out["stats"] = dict(languages=int(P.language.nunique()), pairs=int(len(P)),
                    universal_n=int(len(uni)), median_rho=round(float(np.median(rhos)), 2))

# 1. earliest / latest universal words
cols = ["uni_lemma", "median_aoa", "q25", "q75", "n_lang", "lex_class"]
out["earliest"] = [dict(w=r.uni_lemma, m=r1(r.median_aoa), lo=r1(r.q25), hi=r1(r.q75), n=int(r.n_lang), c=r.lex_class)
                   for r in uni.nsmallest(30, "median_aoa")[cols].itertuples()]
out["latest"] = [dict(w=r.uni_lemma, m=r1(r.median_aoa), lo=r1(r.q25), hi=r1(r.q75), n=int(r.n_lang), c=r.lex_class)
                 for r in uni.nlargest(12, "median_aoa")[cols].itertuples()]

# 2. frequency vs AoA
fq = pd.read_csv(os.path.join(work, "aoa_vs_frequency.csv"))
en = fq[(fq.language == "English (American)") & fq.freq_rank.notna()]
en = en.sort_values("aoa").drop_duplicates("form")
out["freq_en"] = [dict(w=r.form, a=r1(r.aoa), r=int(r.freq_rank), c=r.lex_class) for r in en.itertuples()]
summ = []
for lang, d in fq.groupby("language"):
    m = d.dropna(subset=["freq_rank"])
    f100 = d.nsmallest(100, "aoa")
    summ.append(dict(lang=lang, rho=round(float(spearmanr(m.aoa, np.log(m.freq_rank))[0]), 2),
                     top1000=int((f100.freq_rank <= 1000).sum()), matched=int(f100.freq_rank.notna().sum())))
out["freq_summary"] = sorted(summ, key=lambda x: x["rho"])

# 3. comprehension vs production
gap = pd.read_csv(os.path.join(work, "comp_prod_gap.csv"))
both = A[~A.language.str.contains(EXCLUDE)].pivot_table(index=["language", "uni_lemma"], columns="measure", values="aoa").dropna()
out["stats"]["gap_median"] = r1((both.produces - both.understands).median())
out["stats"]["gap_pairs"] = int(len(both))
sel = pd.concat([gap.nsmallest(10, "gap"), gap.nlargest(10, "gap")])
out["gap"] = [dict(w=r.uni_lemma, u=r1(r.understands), p=r1(r.produces), g=r1(r.gap), n=int(r.n_lang)) for r in sel.itertuples()]

# 4. nouns vs predicates by language
cls = pd.read_csv(os.path.join(work, "class_by_language.csv"))
cls = cls[~cls.language.str.contains(EXCLUDE)]
nwords = P.groupby("language").size()
cls = cls[cls.language.map(nwords) >= 300].sort_values("pred_minus_noun")
out["classes"] = [dict(lang=r.language, noun=r1(r.nouns), pred=r1(r.predicates), func=r1(r._2), d=r1(r.pred_minus_noun))
                  for r in cls.rename(columns={"function words": "_2"}).itertuples()]

# 5. word explorer
keep = set(uni.uni_lemma)
X = P[P.uni_lemma.isin(keep)]
langs = sorted(X.language.unique())
li = {l: i for i, l in enumerate(langs)}
words = {}
for lemma, d in X.groupby("uni_lemma"):
    words[lemma] = [[li[r.language], r.word if isinstance(r.word, str) else "", r1(r.aoa)] for r in d.sort_values("aoa").itertuples()]
out["explorer"] = dict(langs=langs, words=words, cls={r.uni_lemma: r.lex_class for r in uni.itertuples()})

json.dump(out, open(os.path.join(work, "page_data.json"), "w"), ensure_ascii=False, separators=(",", ":"))
print(json.dumps(out["stats"]), os.path.getsize(os.path.join(work, "page_data.json")))
print(out["freq_summary"])
print(out["classes"][:5], out["classes"][-5:])
