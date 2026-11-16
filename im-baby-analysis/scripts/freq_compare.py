"""Compare babies' age of acquisition with adult (subtitle) word frequency."""
import sys, os, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

work, freqdir = sys.argv[1], sys.argv[2]
a = pd.read_csv(os.path.join(work, "aoa_classified.csv"))
p = a[a["measure"] == "produces"].copy()

LANGS = {"English (American)": "en", "Spanish (Mexican)": "es", "French (French)": "fr",
         "German (German)": "de", "Italian": "it", "Mandarin (Beijing)": "zh_cn",
         "Korean": "ko", "Portuguese (European)": "pt", "Russian": "ru"}


def clean(w):
    w = str(w).lower()
    w = re.sub(r"\(.*?\)", "", w)          # drop parentheticals
    w = w.replace("*", "")
    w = re.split(r"[/,]", w)[0]            # first variant
    return w.strip()


out = []
summary = []
for lang, code in LANGS.items():
    f = os.path.join(freqdir, code, f"{code}_50k.txt")
    fr = pd.read_csv(f, sep=" ", header=None, names=["w", "count"], quoting=3,
                     keep_default_na=False, dtype={"w": str})
    fr["rank"] = np.arange(1, len(fr) + 1)
    rank = dict(zip(fr["w"].str.lower(), fr["rank"]))
    d = p[p["language"] == lang].copy()
    d["form"] = d["word"].map(clean)
    d = d[~d["form"].str.contains(" ") | (code in ("zh_cn",))]
    d["freq_rank"] = d["form"].map(rank)
    d["language_code"] = code
    out.append(d)
    m = d.dropna(subset=["freq_rank"])
    rho = spearmanr(m["aoa"], np.log(m["freq_rank"]))[0]
    first100 = d.nsmallest(100, "aoa")
    summary.append(dict(language=lang, cdi_words=len(d), matched=len(m),
                        rho_aoa_logrank=round(rho, 2),
                        first100_in_adult_top1000=int((first100["freq_rank"] <= 1000).sum()),
                        first100_unmatched=int(first100["freq_rank"].isna().sum())))
    if lang == "English (American)":
        print(d.sort_values("aoa")[["form", "aoa", "freq_rank", "lex_class"]].head(30).to_string())

res = pd.concat(out)
res.to_csv(os.path.join(work, "aoa_vs_frequency.csv"), index=False)
print(pd.DataFrame(summary).to_string(index=False))

en = res[(res["language"] == "English (American)")].dropna(subset=["freq_rank"])
en["resid"] = en["aoa"] - np.poly1d(np.polyfit(np.log(en["freq_rank"]), en["aoa"], 1))(np.log(en["freq_rank"]))
print("\nEnglish: learned much EARLIER than frequency predicts\n",
      en.nsmallest(20, "resid")[["form", "aoa", "freq_rank", "lex_class"]].round(1).to_string(index=False))
print("\nEnglish: learned much LATER than frequency predicts\n",
      en.nlargest(20, "resid")[["form", "aoa", "freq_rank", "lex_class"]].round(1).to_string(index=False))
print("\nEnglish adult top-50 words and whether babies know them:")
fr = pd.read_csv(os.path.join(freqdir, "en", "en_50k.txt"), sep=" ", header=None, names=["w", "c"], nrows=50)
lk = res[res["language"] == "English (American)"].groupby("form")["aoa"].min()
print(", ".join(f"{w}({lk.get(w, np.nan):.0f})" if w in lk.index else w for w in fr["w"]))
