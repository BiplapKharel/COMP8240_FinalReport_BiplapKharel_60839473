#!/usr/bin/env python3
"""
Step 2 of 2: PREPROCESS the raw Christmas tweets into a clean, emoji-labelled
dataset (positive vs negative feeling about Christmas).

Input : the raw CSV written by extract_raw_tweets.py
        (tweet_id, year, created_at, lang, text; text still has emoji, links,
        @usernames and HTML entities)
Output: train.txt, dev.txt, test.txt (label<TAB>text, the format EDA's
        augment.py reads), tweets_clean.csv (tweet id, year, split, label, text)
        and build_log.md (how many tweets each step removed)

Labelling = distant supervision (Go et al., 2009): a tweet containing only
positive emoji/emoticons is labelled 1, only negative ones 0; tweets with both
or neither are dropped. The emoji and emoticons are then REMOVED from the text,
so a classifier has to learn the label from the words.

Processing steps (each counted in the log):
  1. keep English tweets; drop retweets ("RT @...")
  2. drop duplicates and near-duplicates (same words once links, @mentions,
     numbers, punctuation and emoji are removed)
  3. drop advert-like and contest-spam tweets (giveaway, sale, follow me, ...),
     and tweets whose text does not mention Christmas/Xmas
  4. label by emoji; drop tweets with both or no sentiment emoji
  5. clean text: decode HTML entities, remove links and @mentions, drop the
     '#' of hashtags, remove ALL emoji/emoticons and other non-ASCII symbols,
     collapse whitespace
  6. drop tweets with fewer than 4 words after cleaning
  7. sample: all negative tweets + randomly chosen positive tweets up to --n
  8. stratified split into train / dev / test

Usage
  python code/preprocess_tweets.py --raw data/tweets_raw/christmas_tweets_raw.csv \
      --n 5000 --n_test 1000 --n_dev 500 --outdir data/tweets
"""
import argparse
import csv
import html
import os
import random
import re
from collections import Counter

POS = ["\U0001F600", "\U0001F603", "\U0001F604", "\U0001F601", "\U0001F606", "\U0001F60A",
       "☺", "\U0001F642", "\U0001F60D", "\U0001F970", "\U0001F618", "\U0001F63B",
       "❤", "\U0001F495", "\U0001F496", "\U0001F497", "\U0001F493", "\U0001F49E",
       "\U0001F49B", "\U0001F49A", "\U0001F499", "\U0001F49C", "\U0001F917", "\U0001F64C",
       "\U0001F973", "\U0001F389", "\U0001F607"]
NEG = ["\U0001F622", "\U0001F62D", "\U0001F61E", "\U0001F614", "\U0001F61F", "\U0001F625",
       "\U0001F613", "\U0001F62A", "\U0001F494", "\U0001F629", "\U0001F62B", "\U0001F621",
       "\U0001F620", "\U0001F92C", "☹", "\U0001F641", "\U0001F612", "\U0001F615",
       "\U0001F623", "\U0001F616", "\U0001F624"]
# ASCII emoticons: not after a digit (so "5:DC" or "3:30" do not count) and,
# for letter-faces like ":D", not followed by another letter
POS_EMOTICON = re.compile(r"(?<!\d):-?\)|(?<!\d):-?D(?![A-Za-z])|(?<![A-Za-z<])(?:<3+)+(?!\d)")
NEG_EMOTICON = re.compile(r"(?<!\d):'?-?\((?![A-Za-z])")
EMOTICON = re.compile(r"(?<!\d):'?-?[()]|(?<!\d):-?[DOPp](?![A-Za-z])|(?<![A-Za-z<])(?:<3+)+(?!\d)")
ADVERT = re.compile(r"\b(giveaway|win|enter|competition|contest|retweet|rt to|follow|following|"
                    r"follow me|notice me|pick me|sale|% off|shop now|discount|order now|buy now|"
                    r"available now|in stock|book now|livelokai|tixwish)\b", re.I)
LINK = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
NON_ASCII = re.compile(r"[^\x00-\x7f]")


def clean(t):
    t = html.unescape(t)
    t = LINK.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = EMOTICON.sub(" ", t)
    t = t.replace("#", "")
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = NON_ASCII.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="raw CSV from extract_raw_tweets.py")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--n_dev", type=int, default=500)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    log = Counter()

    pool, seen = [], set()
    csv.field_size_limit(10 ** 7)
    with open(args.raw, encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))
    for row in raw:
        text, lang, tid, year = row["text"], row["lang"], row["tweet_id"], int(row["year"])
        log[f"tweets read ({year})"] += 1
        if not text or lang != "en":
            log["dropped: not English / empty"] += 1
            continue
        if text.startswith("RT @"):
            log["dropped: retweet"] += 1
            continue
        # duplicates and near-duplicates (spam copies differing only in numbers,
        # punctuation, emoji, links or @mentions) share the same letters-only key
        key = " ".join(re.sub(r"[^a-z ]", " ", MENTION.sub("", LINK.sub("", text)).lower()).split())
        if key in seen:
            log["dropped: duplicate"] += 1
            continue
        seen.add(key)
        if ADVERT.search(text):
            log["dropped: advert-like"] += 1
            continue
        if not re.search(r"christmas|xmas", text, re.I):
            log["dropped: does not mention Christmas/Xmas"] += 1
            continue
        p = any(e in text for e in POS) or bool(POS_EMOTICON.search(text))
        n = any(e in text for e in NEG) or bool(NEG_EMOTICON.search(text))
        if not (p or n):
            log["dropped: no sentiment emoji"] += 1
            continue
        if p and n:
            log["dropped: both positive and negative emoji"] += 1
            continue
        c = clean(text)
        if len(re.sub(r"[^a-z ]", " ", c.lower()).split()) < args.min_words:
            log[f"dropped: under {args.min_words} words after cleaning"] += 1
            continue
        pool.append((tid, year, 1 if p else 0, c))

    neg = [r for r in pool if r[2] == 0]
    pos = [r for r in pool if r[2] == 1]
    log["labelled pool: positive"] = len(pos)
    log["labelled pool: negative"] = len(neg)
    k_neg = min(len(neg), args.n // 2)
    sample = rng.sample(neg, k_neg) + rng.sample(pos, args.n - k_neg)
    rng.shuffle(sample)

    # stratified split
    split = {"test": [], "dev": [], "train": []}
    for lab in (0, 1):
        rows = [r for r in sample if r[2] == lab]
        frac = len(rows) / len(sample)
        nt, nd = round(args.n_test * frac), round(args.n_dev * frac)
        split["test"] += rows[:nt]
        split["dev"] += rows[nt:nt + nd]
        split["train"] += rows[nt + nd:]
    os.makedirs(args.outdir, exist_ok=True)
    for name, rows in split.items():
        rng.shuffle(rows)
        with open(os.path.join(args.outdir, f"{name}.txt"), "w", encoding="utf-8", newline="\n") as f:
            for _, _, lab, text in rows:
                f.write(f"{lab}\t{text}\n")
    with open(os.path.join(args.outdir, "tweets_clean.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "year", "split", "label", "text"])
        for name, rows in split.items():
            for tid, year, lab, text in rows:
                w.writerow([tid, year, name, lab, text])

    lines = ["# Christmas tweets: build log", "", "| Step | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in log.items()]
    lines += ["", f"Sample: {len(sample)} tweets = all {k_neg} negative + "
              f"{args.n - k_neg} randomly chosen positive (seed {args.seed}).", "",
              "| Split | Tweets | Negative | Positive | 2014 | 2017 |", "|---|---|---|---|---|---|"]
    for name, rows in split.items():
        lines.append(f"| {name} | {len(rows)} | {sum(r[2] == 0 for r in rows)} | "
                     f"{sum(r[2] == 1 for r in rows)} | {sum(r[1] == 2014 for r in rows)} | "
                     f"{sum(r[1] == 2017 for r in rows)} |")
    text = "\n".join(lines)
    open(os.path.join(args.outdir, "build_log.md"), "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
