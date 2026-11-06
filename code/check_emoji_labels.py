#!/usr/bin/env python3
"""
How good are emoji labels? Compare the automatic emoji label of each tweet with a
label given by reading the tweet with its emoji hidden.

Inputs (either format):
  * data/tweets/label_check_200.csv : 200 tweets drawn at random from the final
    5,000-tweet dataset (emoji already removed), read without seeing their
    emoji label; columns emoji_label, reading_label
  * data/tweets/hand_labels_450.csv : 450 tweets from the 2017 collection read
    with emoji hidden; columns text (with emoji), hand_label
Reading labels: 1 = positive feeling about Christmas, 0 = negative feeling,
none = no clear feeling about Christmas.

For every tweet that has an emoji label (exactly one polarity of emoji), reports
  * how often the reader found no clear feeling at all (label noise of type 1)
  * agreement and Cohen's kappa between emoji label and reading, on the tweets
    the reader did give a positive/negative label

  python code/check_emoji_labels.py --hand data/tweets/label_check_200.csv \
      --out results/tweets/emoji_label_check_200.md
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_tweets import NEG, NEG_EMOTICON, POS, POS_EMOTICON  # noqa: E402


def emoji_label(text):
    p = any(e in text for e in POS) or bool(POS_EMOTICON.search(text))
    n = any(e in text for e in NEG) or bool(NEG_EMOTICON.search(text))
    if p and not n:
        return "1"
    if n and not p:
        return "0"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    rows = list(csv.DictReader(open(args.hand, encoding="utf-8")))
    col = "reading_label" if "reading_label" in rows[0] else "hand_label"
    if "emoji_label" in rows[0]:
        with_emoji = [(r, r["emoji_label"]) for r in rows]
    else:
        with_emoji = [(r, emoji_label(r["text"])) for r in rows]
        with_emoji = [(r, e) for r, e in with_emoji if e is not None]
    none = [r for r, e in with_emoji if r[col] == "none"]
    both = [(r, e) for r, e in with_emoji if r[col] in ("0", "1")]
    a = [e for _, e in both]
    b = [r[col] for r, _ in both]
    agree = sum(x == y for x, y in zip(a, b)) / len(both)
    k = cohen_kappa_score(a, b)
    cm = confusion_matrix(a, b, labels=["0", "1"])
    lines = [
        "# Emoji labels vs reading (emoji hidden)", "",
        f"input: {args.hand}",
        f"hand-labelled tweets: {len(rows)}",
        f"of these, tweets with an emoji label: {len(with_emoji)}",
        f"  reader found no clear feeling about Christmas: {len(none)} "
        f"({100 * len(none) / len(with_emoji):.0f}%)",
        f"  reader gave a positive/negative label: {len(both)}",
        f"    agreement with emoji label: {100 * agree:.1f}%",
        f"    Cohen's kappa: {k:.3f}",
        "    confusion (rows = emoji label, cols = reading):",
        "              neg   pos",
        f"      neg   {cm[0][0]:5d} {cm[0][1]:5d}",
        f"      pos   {cm[1][0]:5d} {cm[1][1]:5d}",
        "", "disagreements:"]
    for (r, e), x, y in zip(both, a, b):
        if x != y:
            lines.append(f"  emoji={x} reading={y}  {r['text'][:120]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
