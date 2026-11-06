#!/usr/bin/env python3
"""
Step 1 of 2: extract the RAW Christmas tweets from the two R data files published
by RUMgroup on GitHub into one CSV. Nothing is cleaned, filtered or labelled here:
the text keeps its emoji, links, @usernames and HTML entities.

  * https://github.com/RUMgroup/TweetingChristmas
      tweets.Christmas.sampled.RData   50,000 tweets, 8 Dec 2014
  * https://github.com/RUMgroup/TweetingChristmasv2
      data/tweets.Christmas.RDS        72,203 tweets, 30 Nov 2017

Output columns: tweet_id, year, created_at, lang, text

The raw file contains usernames, so it is kept out of the public repository
(.gitignore); re-create it with this script. Then run preprocess_tweets.py.

  pip install rdata
  python code/extract_raw_tweets.py \
      --rdata data/tweets_raw/tweets.Christmas.sampled.RData \
      --rds data/tweets_raw/tweets.Christmas.RDS \
      --out data/tweets_raw/christmas_tweets_raw.csv
"""
import argparse
import csv
import os
import warnings

warnings.filterwarnings("ignore")


def decode(b):
    """R stores emoji as CESU-8 surrogate pairs; turn them into normal Unicode."""
    if b is None:
        return None
    if isinstance(b, str):
        return b
    s = b.decode("utf-8", "surrogatepass")
    return s.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def read_r_table(path, columns):
    """Read selected columns of an R data.frame (RDS or RData) without rdata's
    high-level converter, which fails on the CESU-8 emoji."""
    import rdata
    obj = rdata.parser.parse_file(path).object
    if obj.info.type.name == "LIST":           # .RData: pairlist of variables
        obj = obj.value[0]
    names, attr = None, obj.attributes
    while attr is not None and attr.info.type.name == "LIST":
        if attr.tag is not None and attr.tag.value.value == b"names":
            names = [x.value.decode("latin1") for x in attr.value[0].value]
            break
        attr = attr.value[1] if len(attr.value) > 1 else None
    return {c: [None if x.value is None else decode(x.value)
                for x in obj.value[names.index(c)].value] for c in columns}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rdata", required=True, help="TweetingChristmas/tweets.Christmas.sampled.RData")
    ap.add_argument("--rds", required=True, help="TweetingChristmasv2/data/tweets.Christmas.RDS")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "year", "created_at", "lang", "text"])
        for path, year in [(args.rdata, 2014), (args.rds, 2017)]:
            cols = read_r_table(path, ["id_str", "created_at", "lang", "text"])
            for tid, created, lang, text in zip(cols["id_str"], cols["created_at"],
                                                cols["lang"], cols["text"]):
                w.writerow([tid, year, created, lang, text])
                n += 1
            print(f"{path}: {len(cols['text'])} tweets")
    print(f"wrote {n} raw tweets to {args.out}")


if __name__ == "__main__":
    main()
