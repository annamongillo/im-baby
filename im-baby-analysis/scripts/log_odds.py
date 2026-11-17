"""Weighted log odds (Monroe, Colaresi & Quinn 2008, informative Dirichlet prior) of early words by language.

Counts: for each language and meaning (uni-lemma), the share of children producing it, averaged over every measured month up to 48
(the first four years), scaled to a common 1,000 child-observations so every language weighs the same.
Only meanings measured in at least 20 of the 36 languages are compared, so checklist differences
don't masquerade as language differences; a language-word cell counts only if the word was measured at 24+
months in that language (not just on the infant form). Background for language i = the other 35 languages
(gaps filled with their mean). Prior = one average language's counts.
"""
import sys, re, json, numpy as np, pandas as pd
props, story, out = sys.argv[1:4]
S = json.load(open(story)); L = S["langs"]
d = pd.read_csv(props)
d = d[(d.measure == "produces") & d.language.isin(L) & (d.age <= 48)]
# words that appear only on a checklist's infant form (measured up to ~16-18 months) would look rare next to
# words measured into the third year, so a language-word cell counts only if it was measured at 24+ months
span = d.groupby(["language", "uni_lemma"]).age.max()
keep = span[span >= 24].index
d = d.set_index(["language", "uni_lemma"]).loc[lambda x: x.index.isin(keep)].reset_index()
rate = d.groupby(["language", "uni_lemma"]).prop.mean().unstack()          # language x meaning
core = rate.columns[rate.notna().sum() >= 20]
rate = rate[core]
Y = rate * 1000                                                              # pseudo-counts
alpha = Y.mean(axis=0)                                                       # prior: one average language
a0 = alpha.sum()
res = {}
for lang in L:
    if lang not in Y.index: continue
    yi = Y.loc[lang]; have = yi.notna()
    others = Y.drop(lang)
    bg = others.mean(axis=0)
    yj = bg * len(others)                                                    # gaps filled with the others' mean
    yi, yj, a = yi[have], yj[have], alpha[have]
    ni, nj, a0_ = yi.sum(), yj.sum(), a.sum()
    delta = np.log((yi + a) / (ni + a0_ - yi - a)) - np.log((yj + a) / (nj + a0_ - yj - a))
    z = delta / np.sqrt(1 / (yi + a) + 1 / (yj + a))
    rows = {r[0]: r for r in S["data"][lang]}
    raw = d[d.language == lang].groupby("uni_lemma").words.first()
    def item(u):
        r = rows.get(u)
        word = (r[1] if isinstance(r[1], str) else r[1][0]) if r else None
        if not word and u in raw.index: word = re.split(r"[,/(]", str(raw[u]))[0].strip().replace("_", " ") or None
        return [u, round(float(z[u]), 2), word, (r[7] if r and len(r) > 7 else None), r[2] if r else None,
                round(float(rate.loc[lang, u]) * 100), round(float(bg[u]) / 10)]
    zs = z.sort_values()
    def pick(order):
        out, seen = [], set()
        for u in order:
            it = item(u); key = (it[2] or u).lower()
            if key in seen: continue          # e.g. "pescado" as both fish (animal) and fish (food)
            seen.add(key); out.append(it)
            if len(out) == 10: break
        return out
    top = pick(zs.index[::-1]); used = {(x[2] or x[0]).lower() for x in top}
    bottom = pick([u for u in zs.index if (item(u)[2] or u).lower() not in used])   # a native word can't be both
    res[lang] = {"top": top, "bottom": bottom, "n": int(have.sum())}
json.dump({"meta": {"window": [0, 48], "core": int(len(core))}, "lo": res}, open(out, "w"), ensure_ascii=False)
for l in ["Spanish (Mexican)", "English (American)", "Japanese", "Turkish"]:
    print(l, [(x[2], x[0], x[1], x[5], x[6]) for x in res[l]["top"][:6]], "|", [(x[2], x[0], x[1], x[5], x[6]) for x in res[l]["bottom"][:5]])
print(len(core))
