#!/usr/bin/env python3
"""
Second, human check of the tweet labels.

The author independently labelled a random 100 of the 200 tweets in
data/tweets/label_check_200.csv (emoji hidden, without seeing the first reading
or the emoji label). Labels: 1 = positive feeling about Christmas,
0 = negative feeling, none = no clear feeling about Christmas.

Input:  data/tweets/human_check_100.csv
        columns item, tweet_id, text, emoji_label, first_reading, human_label
Output: agreement and Cohen's kappa for
          * human vs emoji label (tweets the human labelled positive/negative)
          * first reading vs emoji label, on the same 100 tweets
          * human vs first reading (3 classes, clear-vs-none, and on tweets both
            labelled positive/negative)

  python code/compare_human_reading.py --data data/tweets/human_check_100.csv \
      --out results/tweets/human_check_100.md
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.metrics import cohen_kappa_score

    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
    n = len(rows)
    lines = ["# Human check of the tweet labels", "", f"input: {args.data}", f"tweets: {n}", ""]

    for who, name in [("human_label", "human"), ("first_reading", "first reading")]:
        clear = [r for r in rows if r[who] != "none"]
        agree = sum(r[who] == r["emoji_label"] for r in clear) / len(clear)
        k = cohen_kappa_score([r["emoji_label"] for r in clear], [r[who] for r in clear])
        lines.append(f"{name}: no clear feeling {n - len(clear)} ({100 * (n - len(clear)) / n:.0f}%); "
                     f"vs emoji label on {len(clear)} clear tweets: agreement {100 * agree:.1f}%, kappa {k:.3f}")

    h = [r["human_label"] for r in rows]
    f = [r["first_reading"] for r in rows]
    lines += ["", "human vs first reading:",
              f"  3 classes: agreement {100 * sum(a == b for a, b in zip(h, f)) / n:.1f}%, "
              f"kappa {cohen_kappa_score(h, f):.3f}"]
    hc = [a != "none" for a in h]
    fc = [b != "none" for b in f]
    lines.append(f"  clear vs none: agreement {100 * sum(a == b for a, b in zip(hc, fc)) / n:.1f}%, "
                 f"kappa {cohen_kappa_score(hc, fc):.3f}")
    both = [(a, b) for a, b in zip(h, f) if a != "none" and b != "none"]
    lines.append(f"  both gave positive/negative: {len(both)} tweets, "
                 f"agreement {100 * sum(a == b for a, b in both) / len(both):.1f}%")
    labs = ["0", "1", "none"]
    lines += ["", "confusion (rows = human, cols = first reading):", "          0     1  none"]
    for a in labs:
        lines.append(f"  {a:>5} " + " ".join(f"{sum(x == a and y == b for x, y in zip(h, f)):5d}" for b in labs))
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
