# Replicating EDA (Wei & Zou, 2019) — COMP8240 project

Biplap Kharel (60839473), COMP8240 Applications of Data Science, Macquarie University, 2026.

This repository reproduces **EDA: Easy Data Augmentation Techniques for Boosting
Performance on Text Classification Tasks** (Wei & Zou, EMNLP-IJCNLP 2019;
[paper](https://aclanthology.org/D19-1670/), [original code](https://github.com/jasonwei20/eda_nlp))
and applies it to three new datasets, with an LLM-as-judge check of label preservation.

| Dataset | Role | Task | Size used |
|---|---|---|---|
| SST-2 | original (replication) | movie-review sentiment, 2 classes | 6,920 train / 872 dev / 1,821 test |
| AG News | new, existing | news topic, 4 classes | pools of 500 / 2,000 / 5,000; official 7,600 test |
| WNC subjectivity | new, constructed from Pryzant et al. (2020) | biased vs neutral Wikipedia sentence | 10,000 train / 1,000 dev / 2,000 test |
| Christmas tweets | new, built from public tweets (emoji-labelled) | positive vs negative feeling about Christmas | 5,000 tweets: 3,500 train / 500 dev / 1,000 test |

## Layout

```
code/
  eda.py                    original EDA implementation (unchanged, from jasonwei20/eda_nlp)
  augment.py                original command-line augmenter (unchanged)
  eda_cnn_agnews.py         CNN (Kim 2014 as in the paper) + EDA experiment on AG News
  run_eda_cnn.py            same CNN/EDA for SST-2, WNC and the tweets (fixed splits; a k-fold mode is also available);
                            --protocol authors re-creates the settings in the authors' experiment code
  build_wnc_dataset.py      WNC pairs -> biased/neutral classification set
  extract_raw_tweets.py     tweets in the two R files -> one raw CSV (no cleaning)
  preprocess_tweets.py      raw tweets CSV -> cleaned, emoji-labelled 5,000-tweet dataset (+ build log)
  check_emoji_labels.py     emoji labels vs labels from reading the tweets (agreement, Cohen's kappa)
  compare_human_reading.py  the author's own labels on 100 tweets vs the first reading and the emoji labels
  plot_gains.py             summary figure: EDA gain vs training size for all datasets
  make_judge_set.py         blind original/augmented items for the LLM judge
  llm_judge.py              judge prompt (+ optional Anthropic API runner) and scoring
data/
  sst2/  wnc/  tweets/      ready-to-use label<TAB>text files
results/
  sst2/  sst2_authors/  agnews/  wnc/  tweets/   results.csv, table_results.md, figure
  judge/  judge_wnc/        blind items, hidden key, judgements, summary
```

## Setup

Tested with Python 3.10–3.13, TensorFlow 2.21 (CPU).

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
# GloVe 6B 300-d vectors: https://nlp.stanford.edu/projects/glove/  (glove.6B.300d.txt)
```

## Reproduce

**On Windows, one command runs everything:** see [`RUN_ON_WINDOWS.md`](RUN_ON_WINDOWS.md)
(`run_all.bat quick` for a 15-minute test, `run_all.bat` for the full run).
The individual commands (Linux/macOS syntax) are:

```bash
# 1. Replication on SST-2 (paper's Table 3 settings, held-out dev set for early stopping)
python code/run_eda_cnn.py split --name sst2 --glove glove.6B.300d.txt \
  --train data/sst2/train.txt --val data/sst2/dev.txt --test data/sst2/test.txt \
  --sizes 500 2000 5000 full --seeds 5 --max_len 60 --outdir results/sst2

# 1b. Same, with the settings found in the authors' experiment code
python code/run_eda_cnn.py split --protocol authors --name sst2_authors --glove glove.6B.300d.txt \
  --train data/sst2/train.txt --val data/sst2/dev.txt --test data/sst2/test.txt \
  --sizes 500 2000 5000 full --seeds 3 --max_len 50 --epochs 300 --outdir results/sst2_authors

# 2. AG News (downloads from Hugging Face)
python code/eda_cnn_agnews.py --glove glove.6B.300d.txt --eda_dir code \
  --sizes 500 2000 5000 --seeds 5 --outdir results/agnews

# 3. WNC subjectivity (raw pairs: http://bit.ly/bias-corpus, or the mirror
#    https://github.com/erickfm/bias-neutralization data/biased.word.train.csv)
python code/build_wnc_dataset.py --csv biased.word.train.csv --outdir data/wnc
python code/run_eda_cnn.py split --name wnc --glove glove.6B.300d.txt \
  --train data/wnc/train.txt --val data/wnc/dev.txt --test data/wnc/test.txt \
  --sizes 500 2000 5000 --seeds 5 --max_len 50 --outdir results/wnc

# 4. Christmas tweets (RUMgroup collections on GitHub; needs `pip install rdata`)
curl -L -o tweets.Christmas.sampled.RData \
  https://raw.githubusercontent.com/RUMgroup/TweetingChristmas/master/tweets.Christmas.sampled.RData
curl -L -o tweets.Christmas.RDS \
  https://raw.githubusercontent.com/RUMgroup/TweetingChristmasv2/master/data/tweets.Christmas.RDS
python code/extract_raw_tweets.py --rdata tweets.Christmas.sampled.RData --rds tweets.Christmas.RDS \
  --out data/tweets_raw/christmas_tweets_raw.csv
python code/preprocess_tweets.py --raw data/tweets_raw/christmas_tweets_raw.csv \
  --n 5000 --n_test 1000 --n_dev 500 --outdir data/tweets
python code/check_emoji_labels.py --hand data/tweets/label_check_200.csv \
  --out results/tweets/emoji_label_check_200.md
python code/compare_human_reading.py --data data/tweets/human_check_100.csv \
  --out results/tweets/human_check_100.md
python code/run_eda_cnn.py split --name tweets --glove glove.6B.300d.txt \
  --train data/tweets/train.txt --val data/tweets/dev.txt --test data/tweets/test.txt \
  --sizes 500 2000 full --seeds 5 --max_len 40 --outdir results/tweets

# 5. LLM-as-judge (blind label preservation + fluency)
python code/make_judge_set.py --data sst2=data/sst2/train.txt --outdir results/judge
python code/make_judge_set.py --data wnc=data/wnc/train.txt --outdir results/judge_wnc --seed 21
# judge with an API key (optional) ...
python code/llm_judge.py judge --items results/judge/judge_items.csv --out results/judge/judgements.csv
# ... and score
python code/llm_judge.py score --key results/judge/judge_key.csv \
  --judgements results/judge/judgements.csv --outdir results/judge

```

Each experiment writes `results.csv` (one row per run), `table_results.md`
(mean ± std over seeds, gain, paired t-test p) and `fig_accuracy_vs_size.png`.

## Notes

* `results/` holds the run reported in the final report (the third full `run_all.bat` run on a
  Windows laptop, CPU only). Two earlier runs on the same laptop and one on a Linux cloud machine
  are summarised in the report's Table 4 (Section 3.5), which shows how much results vary across
  seeds, runs and hardware; their raw files are not included here.
* EDA's `eda.py` draws from Python sets of words, and Python randomises string hashing in every
  process, so augmented sentences (and therefore results) change from run to run even with fixed
  seeds. Set `PYTHONHASHSEED=0` before running to make runs on one machine repeatable. AG News
  is repeatable regardless, because its augmented sets are cached in `results/agnews/aug_cache.pkl`.

* The GloVe file used in our runs was filtered to alphabetic tokens (317,756 words); results
  with the full file are identical because all text is cleaned to letters only.
* SST-2 was taken from the sentence-level split mirrored at
  `clairett/pytorch-sentiment-classification` (Hugging Face was not reachable from our environment).
* `data/tweets/` is built from two public collections of Christmas tweets (8 Dec 2014 and
  30 Nov 2017) published by RUMgroup on GitHub. Labels come from the authors' own emoji
  (distant supervision); the emoji, links and @usernames are removed from the text.
  `label_check_200.csv` and `hand_labels_450.csv` hold the labels used to check the emoji labels.
  The raw tweets (`data/tweets_raw/`, with usernames) are not committed; `extract_raw_tweets.py`
  re-creates them from the public RUMgroup files.
* The `authors` protocol run (`results/sst2_authors`) caps training at 50 epochs; the authors'
  code has no cap, but with its leaky validation split early stopping rarely triggers.
* The LLM judgements in `results/judge*/judgements.csv` were made by claude-opus-5-5 using
  the prompt in `code/llm_judge.py`, item by item, without access to the key. The original judge
  set also held 125 items from a small Christmas survey that is no longer part of the project;
  those items were removed, so `results/judge/` keeps the original item IDs (with gaps) for SST-2 only.
  The prompt still mentions that survey because it is the exact prompt that was used.
  The WNC judge used the same answer format (label plus 1-5 fluency) with biased (1) / neutral (0)
  in place of positive / negative; its exact prompt wording was not saved. The rules used to read
  the tweets for the emoji-label check are in the docstring of `code/check_emoji_labels.py`.
* `data/tweets/human_check_100.csv` (and the original sheet `tweet_labels_by_Biplap.xlsx`) holds the
  author's own labels for a random 100 of the 200 tweets, made independently with emoji hidden;
  `first_reading` is the LLM's label for the same tweet.

## Licences and attribution

EDA code © Jason Wei (see original repository). SST-2: Socher et al. (2013). AG News: Zhang et
al. (2015). WNC: Pryzant et al. (2020), derived from Wikipedia (CC BY-SA). Christmas tweets:
RUMgroup TweetingChristmas / TweetingChristmasv2 (GitHub, no licence stated); tweet content is
subject to X/Twitter's terms, so only cleaned text without usernames is included here.
