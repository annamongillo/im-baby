"""Combine Wordbank uni-lemma slices and estimate age of acquisition (AoA).

AoA = age (months) at which a weighted logistic fit of prop ~ age crosses 0.5,
computed per language x uni-lemma x measure (produces / understands), sex = All.
"""
import sys, glob, os, json
import numpy as np
import pandas as pd

src = sys.argv[1]          # .../slices/unilemmas
out = sys.argv[2]          # output dir

index = {d["slug"]: d["uni_lemma"] for d in json.load(open(os.path.join(src, "index.json")))}

frames = []
for slug, lemma in index.items():
    p = os.path.join(src, f"{slug}.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p, dtype={"words": str})
    df = df[df["sex"] == "All"].drop(columns="sex")
    df["uni_lemma"] = lemma
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)

# --- Fix comprehension: slices pool every child who had the item, but only
# Words & Gestures-type forms ask about understanding. Rescale "understands"
# by the number of children whose form measured comprehension at that age.
admins_dir = sys.argv[3]
san = lambda s: "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")
import re
san = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
labels = raw["language"].unique()
prefixes = {lab: san(lab) for lab in labels}
files = [f[:-4] for f in os.listdir(admins_dir) if f.endswith(".csv")]
ncomp = []
for lab, pre in prefixes.items():
    longer = [p for p in prefixes.values() if p != pre and p.startswith(pre + "_")]
    mine = [f for f in files if f.startswith(pre + "_") and not any(f.startswith(l + "_") for l in longer)]
    for f in mine:
        d = pd.read_csv(os.path.join(admins_dir, f + ".csv"), usecols=["age", "comprehension"])
        d = d[d["comprehension"].notna()]
        if len(d):
            c = d.groupby("age").size().rename("n_comp").reset_index()
            c["language"] = lab
            ncomp.append(c)
ncomp = pd.concat(ncomp).groupby(["language", "age"], as_index=False)["n_comp"].sum()
raw = raw.merge(ncomp, on=["language", "age"], how="left")
u = raw["measure"] == "understands"
# Rescaling proved unreliable where form types overlap, so keep only ages at
# which (nearly) every child with the item was on a comprehension form.
clean = raw["n_comp"].fillna(0) >= 0.95 * raw["n"]
raw = raw[~u | (clean & (raw["n"] >= 20))].drop(columns="n_comp")

raw.to_csv(os.path.join(out, "unilemma_props.csv.gz"), index=False, compression="gzip")


def fit_logistic(age, prop, n, iters=50):
    """Weighted binomial GLM via IRLS. Returns (b0, b1) or None."""
    X = np.column_stack([np.ones_like(age), age - age.mean()])
    y = prop
    b = np.zeros(2)
    for _ in range(iters):
        eta = X @ b
        mu = 1 / (1 + np.exp(-eta))
        w = n * mu * (1 - mu) + 1e-9
        z = eta + (y - mu) / (mu * (1 - mu) + 1e-9)
        WX = X * w[:, None]
        try:
            b_new = np.linalg.solve(X.T @ WX + 1e-6 * np.eye(2), WX.T @ z)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new
            break
        b = b_new
    # convert intercept back to raw-age scale
    b0 = b[0] - b[1] * age.mean()
    return b0, b[1]


rows = []
for (lang, lemma, measure), g in raw.groupby(["language", "uni_lemma", "measure"]):
    g = g[g["n"] > 0]
    if g["n"].sum() < 100 or g["age"].nunique() < 4:
        continue
    age = g["age"].to_numpy(float)
    res = fit_logistic(age, g["prop"].to_numpy(float), g["n"].to_numpy(float))
    if res is None:
        continue
    b0, b1 = res
    aoa = -b0 / b1 if b1 > 0 else np.nan
    # most common surface form in this language
    wc = g.dropna(subset=["words"]).groupby("words")["n"].sum()
    word = wc.idxmax() if len(wc) else ""
    rows.append(dict(language=lang, uni_lemma=lemma, measure=measure, word=word,
                     aoa=aoa, slope=b1, n_total=int(g["n"].sum()),
                     age_min=int(age.min()), age_max=int(age.max()),
                     prop_at_max=float(g.loc[g["age"].idxmax(), "prop"])))

aoa = pd.DataFrame(rows)
# keep plausible estimates; flag extrapolations beyond the observed age range
aoa["extrapolated"] = (aoa["aoa"] < aoa["age_min"] - 3) | (aoa["aoa"] > aoa["age_max"] + 6)
aoa.to_csv(os.path.join(out, "aoa.csv"), index=False)
print(raw.shape, aoa.shape)
print(aoa.groupby("measure").size())
print("languages:", aoa["language"].nunique())
