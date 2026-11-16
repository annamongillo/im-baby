# I'm Baby
A data exploration of early-childhood language learning.

# Data Analysis Files

The scripts and outputs behind "I'm Baby" and its data section. Everything is Python (pandas, numpy, scipy) working on public data from Wordbank and FrequencyWords.

## Folder contents

```
scripts/
  build_aoa.py        1. Combine Wordbank word files and estimate age of acquisition
  explore.py          2. Classify words; universal words, consistency, noun/verb gaps, understanding gap
  freq_compare.py     3. Compare babies' word order with adult word frequency
  make_page_data.py   4. Build the data for the first exploration page ("What Babies Say First")
  story_data.py       5. Build the per-language data embedded in the game (story_data.json)
data/
  unilemma_props.csv.gz   share of children saying/understanding each word, by language and month
  aoa.csv                 age-of-acquisition estimates, one row per language x word x measure
  aoa_classified.csv      aoa.csv plus semantic category and word class
  universal_words.csv     meanings found in 25+ languages: median age and spread
  class_by_language.csv   median age by word class, per language
  comp_prod_gap.csv       months between understanding and saying, per meaning
  aoa_vs_frequency.csv    babies' ages joined to adult frequency ranks (9 languages)
  page_data.json          data for the first exploration page
  story_data.json         data embedded in the game (36 languages)
```

## Source data

1. **Wordbank** (Frank, Braginsky, Yurovsky & Marchman), CC BY 4.0. The scripts read the precomputed files published with the Wordbank site:

   ```
   git clone --depth 1 --branch gh-pages https://github.com/langcog/wordbank-datapage
   ```

   They use `slices/unilemmas/` (one CSV per cross-language meaning), `slices/admins/` (one CSV per checklist form, used to identify forms that measure understanding) and `slices/networks/` (word categories).

2. **FrequencyWords** (2018 OpenSubtitles word counts): https://github.com/hermitdave/FrequencyWords. Only the `content/2018/<code>/<code>_50k.txt` files are needed, e.g. `en`, `es`, `fr`, `de`, `it`, `zh_cn`, `zh_tw`, `ko`, `pt`, `ru`, `he`, `tr`, `pl`, `nl`, `sv`, `da`, `no`, `cs`, `hr`, `et`, `fi`, `hu`, `lv`, `sk`, `ca`, `el`, `fa`, `ar`.

## How to reproduce

Requires Python 3 with pandas, numpy and scipy. Paths below assume the two repositories sit next to this folder; adjust as needed.

```
WB=../wordbank-datapage/slices
FW=../FrequencyWords/content/2018
OUT=data

python scripts/build_aoa.py      $WB/unilemmas $OUT $WB/admins
python scripts/explore.py        $OUT $WB/networks
python scripts/freq_compare.py   $OUT $FW
python scripts/make_page_data.py $OUT
python scripts/story_data.py     $OUT $FW $OUT/story_data.json
```

## Methods in brief

- **Age of acquisition (AoA).** For each language, meaning and measure (says / understands), a weighted logistic regression of the share of children with the word against age in months; AoA is the age where the curve crosses 50%. Estimates more than 3 months before or 6 months after the observed ages are flagged `extrapolated` and dropped from summaries.
- **Understanding.** Wordbank pools every child who had a word on their form, but only the younger "Words & Gestures"-type forms ask about understanding. `build_aoa.py` keeps "understands" only at ages where at least 95% of children were on a form that measures it.
- **Exclusions.** Sign languages, the Malaysian datasets and Tsonga are left out of pooled summaries; their samples skew older or cover few words. The game uses the 36 languages with at least 300 estimated words.
- **Word classes.** Wordbank checklist sections are grouped into nouns, predicates (verbs and adjectives), function words, and sounds & routines.
- **Frequency.** A checklist word is matched to its first clean spelling in the language's 50,000 most common subtitle words. Multiword items and sound effects with several spellings are skipped.
- **Clean labels.** `story_data.py` strips form codes (e.g. "A sounds - 8") and numbering suffixes (e.g. "peix2") from display words.
- **Calculated in the game itself.** Some figures in the game's data section are computed in the browser from `story_data.json`: cross-language medians over the 36 game languages, Spearman correlations, the understanding gap (meanings in 10+ languages), and the pace-adjusted "unusually early/late" words. That code is in the game's `index.html`.

## story_data.json format

`data[language]` is a list of words sorted by age, each `[meaning, word, class, age_says, age_understands, cross_language_median, adult_frequency_rank]`, where `word` is either a string or `[short label, all spellings]`, and `class` is 0 nouns, 1 predicates, 2 function words, 3 sounds & routines. Missing values are `null`. `gaps[language]` is the median age of predicates minus nouns, in months.
