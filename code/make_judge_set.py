#!/usr/bin/env python3
"""
Build a BLIND evaluation set for the LLM-as-judge label-preservation study.

For each dataset, a class-balanced sample of original sentences is drawn and
each sentence is edited once with EACH of the four EDA operations separately
(SR, RI, RS, RD; alpha = 0.1, the paper's default). Originals are included as
a control. Items are shuffled together and written WITHOUT labels or operation
names, so the judge sees only the text.

Outputs (in --outdir):
  judge_items.csv   item_id, text                (given to the judge)
  judge_key.csv     item_id, dataset, source_id, op, gold   (kept hidden)

Usage
  python make_judge_set.py --data sst2=data/sst2/train.txt \
      --per_class 13 12 --alpha 0.1 --outdir results/judge
"""
import argparse
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eda  # noqa: E402  (original repo code)

OPS = {
    "SR": lambda w, a: eda.synonym_replacement(w, max(1, int(a * len(w)))),
    "RI": lambda w, a: eda.random_insertion(w, max(1, int(a * len(w)))),
    "RS": lambda w, a: eda.random_swap(w, max(1, int(a * len(w)))),
    "RD": lambda w, a: eda.random_deletion(w, a),
}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lab, _, s = line.rstrip("\n").partition("\t")
            s = eda.get_only_chars(s).strip()
            if s:
                rows.append((int(lab), " ".join(s.split())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="name=path")
    ap.add_argument("--per_class", nargs=2, type=int, default=[13, 12],
                    help="number of positive and negative originals per dataset")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)  # eda.py uses the global random module
    os.makedirs(args.outdir, exist_ok=True)

    items = []
    for spec in args.data:
        name, path = spec.split("=", 1)
        rows = load(path)
        pos = [r for r in rows if r[0] == 1 and 6 <= len(r[1].split()) <= 30]
        neg = [r for r in rows if r[0] == 0 and 6 <= len(r[1].split()) <= 30]
        picked = rng.sample(pos, args.per_class[0]) + rng.sample(neg, args.per_class[1])
        for j, (gold, sent) in enumerate(picked):
            src = f"{name}-{j:02d}"
            items.append(dict(dataset=name, source_id=src, op="ORIG", gold=gold, text=sent))
            words = sent.split()
            for op, fn in OPS.items():
                new = " ".join(fn(list(words), args.alpha))
                items.append(dict(dataset=name, source_id=src, op=op, gold=gold, text=new))

    rng.shuffle(items)
    with open(os.path.join(args.outdir, "judge_items.csv"), "w", newline="") as f1, \
         open(os.path.join(args.outdir, "judge_key.csv"), "w", newline="") as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(["item_id", "text"])
        w2.writerow(["item_id", "dataset", "source_id", "op", "gold", "text"])
        for i, it in enumerate(items):
            iid = f"J{i:03d}"
            w1.writerow([iid, it["text"]])
            w2.writerow([iid, it["dataset"], it["source_id"], it["op"], it["gold"], it["text"]])
    print(f"wrote {len(items)} items to {args.outdir}")


if __name__ == "__main__":
    main()
