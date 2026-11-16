import sys, os, json, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

work, netdir = sys.argv[1], sys.argv[2]
a = pd.read_csv(os.path.join(work, "aoa.csv"))
a = a[a["aoa"].notna() & ~a["extrapolated"]]

# ---- uni-lemma -> semantic category (majority vote across languages)
cats = []
for f in glob.glob(os.path.join(netdir, "*.json")):
    d = json.load(open(f))
    cats.append(pd.DataFrame({"uni_lemma": d["uni_lemma"], "category": d["category"]}))
cats = pd.concat(cats).dropna()
cat = cats.groupby("uni_lemma")["category"].agg(lambda s: s.value_counts().idxmax())

CLASS = {
    **{c: "nouns" for c in ["animals", "vehicles", "toys", "food_drink", "clothing", "body_parts",
                           "household", "furniture_rooms", "outside", "places", "people"]},
    **{c: "predicates" for c in ["action_words", "descriptive_words", "states",
                                "descriptive_words (adjectives)", "descriptive_words (adverbs)"]},
    **{c: "function words" for c in ["pronouns", "quantifiers", "question_words", "locations",
                                    "connecting_words", "helping_verbs", "time_words", "articles",
                                    "prepositions", "directions", "helping_words", "modal_adverbs",
                                    "classifiers"]},
    **{c: "sounds & routines" for c in ["sounds", "games_routines"]},
}
a["category"] = a["uni_lemma"].map(cat)
a["lex_class"] = a["category"].map(CLASS).fillna("other")
a.to_csv(os.path.join(work, "aoa_classified.csv"), index=False)
print("unmapped categories:", sorted(set(cat.unique()) - set(CLASS)))

p = a[a["measure"] == "produces"]
# exclude sign languages and tiny samples from cross-language summaries
# Malaysian and Tsonga datasets sample older children / few words and track
# poorly with every other language, so leave them out of pooled summaries.
EXCLUDE = ["Sign", "Malaysian", "Tsonga"]
spoken = p[~p["language"].str.contains("|".join(EXCLUDE))]

# ---- 1. universal words
g = spoken.groupby("uni_lemma").agg(n_lang=("language", "nunique"), median_aoa=("aoa", "median"),
                                    q25=("aoa", lambda s: s.quantile(.25)), q75=("aoa", lambda s: s.quantile(.75)),
                                    lex_class=("lex_class", "first"), category=("category", "first"))
uni = g[g["n_lang"] >= 25].sort_values("median_aoa")
uni.to_csv(os.path.join(work, "universal_words.csv"))
print("\n# uni-lemmas in >=25 spoken languages:", len(uni))
print("EARLIEST 40\n", uni.head(40).round(1).to_string())
print("LATEST 15\n", uni.tail(15).round(1).to_string())

# ---- 2. consistency: each language vs leave-one-out cross-language median
rows = []
for lang, d in spoken.groupby("language"):
    others = spoken[spoken["language"] != lang].groupby("uni_lemma")["aoa"].median()
    m = d.set_index("uni_lemma")["aoa"].to_frame().join(others.rename("others"), how="inner")
    if len(m) >= 50:
        rows.append((lang, len(m), spearmanr(m["aoa"], m["others"])[0]))
cons = pd.DataFrame(rows, columns=["language", "n_words", "rho"]).sort_values("rho")
print("\n# consistency with other languages (Spearman rho)\n", cons.round(2).to_string(index=False))

# ---- 3. lexical class by language (median AoA)
cls = spoken.pivot_table(index="language", columns="lex_class", values="aoa", aggfunc="median")
cls["pred_minus_noun"] = cls["predicates"] - cls["nouns"]
cls["func_minus_noun"] = cls["function words"] - cls["nouns"]
cls = cls.sort_values("pred_minus_noun")
cls.to_csv(os.path.join(work, "class_by_language.csv"))
print("\n# median AoA by class\n", cls.round(1).to_string())

# ---- 4. comprehension-production gap
both = a.pivot_table(index=["language", "uni_lemma"], columns="measure", values="aoa").dropna()
both["gap"] = both["produces"] - both["understands"]
gap = both.reset_index().groupby("uni_lemma").agg(n_lang=("language", "nunique"), gap=("gap", "median"),
                                                 understands=("understands", "median"), produces=("produces", "median"))
gap = gap[gap["n_lang"] >= 10].sort_values("gap")
gap.to_csv(os.path.join(work, "comp_prod_gap.csv"))
print("\n# overall median gap (months):", round(both["gap"].median(), 1), "pairs:", len(both))
print("smallest gaps\n", gap.head(15).round(1).to_string())
print("largest gaps\n", gap.tail(15).round(1).to_string())
