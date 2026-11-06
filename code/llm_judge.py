#!/usr/bin/env python3
"""
LLM-as-judge for EDA label preservation.

Step 1 (judge): each item in judge_items.csv is shown to an LLM on its own,
with no label and no indication of whether it is original or augmented.
The judge returns a sentiment label (1 positive / 0 negative / U unclear)
and a fluency score 1-5 using the rubric in PROMPT.

  * With ANTHROPIC_API_KEY set:  python llm_judge.py judge --items ... --out judgements.csv
  * Without an API key, the judgements can be produced by pasting PROMPT and
    the items into a chat interface and saving the answers in the same CSV
    format (item_id, judge_label, fluency). This is how the report's numbers
    were produced (see report, Section 6).

Step 2 (score): compare judgements with the hidden key.

  python llm_judge.py score --key judge_key.csv --judgements judgements.csv --outdir .

Outputs: judge_summary.md (per dataset x operation) and judge_merged.csv.
"""
import argparse
import csv
import json
import os
import re
import sys

PROMPT = """You are judging short English texts. Each text is either a movie-review
sentence or a person's answer to "What are you looking forward to this Christmas?".
Some texts may have been automatically edited and can contain odd words or word order.
For the text below, answer in JSON with two fields:
  "label":   1 if the writer's overall sentiment is positive, 0 if negative,
             "U" if you genuinely cannot tell.
  "fluency": an integer 1-5
             5 = natural, fluent English
             4 = one minor slip but reads naturally
             3 = noticeably awkward but meaning is clear
             2 = hard to read; meaning only partly recoverable
             1 = incoherent
Judge the text as written; do not try to guess how it was edited.
Text: "{text}"
JSON:"""


def judge(args):
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()
    rows = list(csv.DictReader(open(args.items, encoding="utf-8")))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "judge_label", "fluency"])
        for r in rows:
            msg = client.messages.create(model=args.model, max_tokens=50, temperature=0,
                                         messages=[{"role": "user",
                                                    "content": PROMPT.format(text=r["text"])}])
            txt = msg.content[0].text
            m = re.search(r"\{.*\}", txt, re.S)
            d = json.loads(m.group(0)) if m else {"label": "U", "fluency": ""}
            w.writerow([r["item_id"], d.get("label", "U"), d.get("fluency", "")])
            print(r["item_id"], d, flush=True)


def score(args):
    key = {r["item_id"]: r for r in csv.DictReader(open(args.key, encoding="utf-8"))}
    jd = {r["item_id"]: r for r in csv.DictReader(open(args.judgements, encoding="utf-8"))}
    merged = []
    for iid, k in key.items():
        j = jd[iid]
        lab = str(j["judge_label"])
        merged.append(dict(k, judge_label=lab, fluency=int(j["fluency"]),
                           preserved=int(lab == str(k["gold"]))))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "judge_merged.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)

    ops = ["ORIG", "SR", "RI", "RS", "RD"]
    lines = ["| Dataset | Operation | n | Judge label = gold (%) | Unclear (%) | Mean fluency (1-5) |",
             "|---|---|---|---|---|---|"]
    for ds in sorted({m["dataset"] for m in merged}) + ["all"]:
        groups = [(op, [m for m in merged if (ds == "all" or m["dataset"] == ds) and m["op"] == op])
                  for op in ops]
        aug = [m for m in merged if (ds == "all" or m["dataset"] == ds) and m["op"] != "ORIG"]
        for op, g in groups + [("all EDA", aug)]:
            n = len(g)
            agree = 100 * sum(m["preserved"] for m in g) / n
            unclear = 100 * sum(m["judge_label"] == "U" for m in g) / n
            flu = sum(m["fluency"] for m in g) / n
            lines.append(f"| {ds} | {op} | {n} | {agree:.1f} | {unclear:.1f} | {flu:.2f} |")
    # Consistency: does the judge give an augmented sentence the SAME label it gave
    # that sentence's original? This separates EDA-induced flips from originals
    # that are themselves ambiguous or mislabelled.
    orig = {m["source_id"]: m["judge_label"] for m in merged if m["op"] == "ORIG"}
    lines += ["", "| Dataset | Augmented items | Same judge label as original (%) |", "|---|---|---|"]
    for ds in sorted({m["dataset"] for m in merged}) + ["all"]:
        aug = [m for m in merged if (ds == "all" or m["dataset"] == ds) and m["op"] != "ORIG"]
        same = sum(m["judge_label"] == orig[m["source_id"]] for m in aug)
        lines.append(f"| {ds} | {len(aug)} | {100 * same / len(aug):.1f} |")
    text = "\n".join(lines)
    open(os.path.join(args.outdir, "judge_summary.md"), "w").write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("judge")
    a.add_argument("--items", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--model", default="claude-opus-5-5")
    b = sub.add_parser("score")
    b.add_argument("--key", required=True)
    b.add_argument("--judgements", required=True)
    b.add_argument("--outdir", required=True)
    args = ap.parse_args()
    (judge if args.cmd == "judge" else score)(args)


if __name__ == "__main__":
    main()
