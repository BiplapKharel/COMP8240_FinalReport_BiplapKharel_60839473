#!/usr/bin/env python3
"""
Build a binary subjectivity-bias classification dataset from the
Wiki Neutrality Corpus (WNC; Pryzant et al., 2020).

Input: biased.word.train.csv (WNC single-word-edit subset; columns include
id, source_raw = biased sentence, target_raw = neutralised sentence).
Original download: http://bit.ly/bias-corpus ; mirror used here:
https://github.com/erickfm/bias-neutralization (data/biased.word.train.csv).

Task: label 1 = sentence as originally written (contains subjective bias,
later removed by a Wikipedia editor); label 0 = the editor's neutralised version.

Design choices
  * ONE side per pair: for each pair we keep either the biased or the
    neutral sentence (chosen at random, 50/50), so no near-duplicate pair
    halves with opposite labels appear anywhere in the data.
  * Pair-disjoint splits: train / dev / test are drawn from different pairs.
  * Light filtering: 8-40 words; drop pairs whose two sides are identical after
    cleaning, and drop exact duplicate sentences.
  * Text cleaned exactly like the other datasets (lowercase, letters only).

Usage
  python build_wnc_dataset.py --csv biased.word.train.csv --outdir data/wnc \
      --n_train 10000 --n_dev 1000 --n_test 2000 --seed 13
"""
import argparse
import csv
import os
import random
import re
import sys

csv.field_size_limit(sys.maxsize)


def clean(s):
    return " ".join(re.sub(r"[^a-z ]", " ", s.lower().replace("'", "")).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_dev", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--min_len", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pairs, seen, stats = [], set(), dict(rows=0, same=0, length=0, dup=0)
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            stats["rows"] += 1
            b, n = clean(r["source_raw"]), clean(r["target_raw"])
            if b == n:
                stats["same"] += 1
                continue
            if not all(args.min_len <= len(x.split()) <= args.max_len for x in (b, n)):
                stats["length"] += 1
                continue
            if b in seen or n in seen:
                stats["dup"] += 1
                continue
            seen.update((b, n))
            pairs.append((r["id"], b, n))
    print(f"read {stats['rows']} pairs; kept {len(pairs)} "
          f"(dropped: identical after cleaning {stats['same']}, length {stats['length']}, "
          f"duplicate {stats['dup']})")

    rng.shuffle(pairs)
    need = args.n_train + args.n_dev + args.n_test
    if len(pairs) < need:
        sys.exit(f"only {len(pairs)} usable pairs, need {need}")
    parts = {"test": pairs[:args.n_test],
             "dev": pairs[args.n_test:args.n_test + args.n_dev],
             "train": pairs[args.n_test + args.n_dev:need]}

    os.makedirs(args.outdir, exist_ok=True)
    for name, ps in parts.items():
        counts = [0, 0]
        with open(os.path.join(args.outdir, f"{name}.txt"), "w", encoding="utf-8") as f, \
             open(os.path.join(args.outdir, f"{name}_ids.tsv"), "w", encoding="utf-8") as g:
            for pid, b, n in ps:
                lab = rng.randint(0, 1)
                f.write(f"{lab}\t{b if lab == 1 else n}\n")
                g.write(f"{pid}\t{lab}\n")
                counts[lab] += 1
        print(f"{name}: {len(ps)} sentences (biased {counts[1]}, neutral {counts[0]})")


if __name__ == "__main__":
    main()
