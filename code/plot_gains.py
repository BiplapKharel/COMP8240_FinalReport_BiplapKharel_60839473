#!/usr/bin/env python3
"""Figure: EDA gain (accuracy points, mean +/- std over seeds) vs training-set size,
for every dataset, with the paper's CNN average as a reference.

Reads results/<name>/results.csv; writes results/fig_gain_by_size.png (+ .pdf).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

ROOT = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SETS = [("sst2", "SST-2 (replication)", "#2a78d6"),
        ("wnc", "WNC bias (new)", "#eb6834"),
        ("agnews", "AG News (new)", "#eda100"),
        ("tweets", "Christmas tweets (new)", "#e87ba4")]
PAPER = {500: 2.1, 2000: 0.8, 5000: 0.6}  # Wei & Zou (2019) Table 2, CNN average


def gains(name):
    d = pd.read_csv(os.path.join(ROOT, name, "results.csv"))
    w = d.pivot_table(index=["n_train", "seed"], columns="eda", values="acc") * 100
    out = []
    for n, g in w.groupby(level=0):
        diff = g[1] - g[0]
        p = ttest_rel(g[1], g[0]).pvalue
        out.append((n, diff.mean(), diff.std(ddof=1), p))
    return out


fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
ax.axhline(0, color="#8a8984", lw=0.8)
ax.plot(list(PAPER), list(PAPER.values()), ls="--", color="#8a8984", lw=1.2, marker="s",
        ms=5, mfc="white", label="Paper, CNN average")
LABEL_AT = {"sst2": (-1, 7, -3), "wnc": (1, 8, 2),
            "agnews": (0, -44, -3), "tweets": (1, 7, 2)}
offsets = {"sst2": 1.0, "wnc": 1.06, "agnews": 0.94, "tweets": 0.89}
for key, label, col in SETS:
    if not os.path.exists(os.path.join(ROOT, key, "results.csv")):
        continue  # e.g. a run without the tweets experiment
    rows = gains(key)
    xs = [r[0] * offsets[key] for r in rows]
    ax.errorbar(xs, [r[1] for r in rows], yerr=[r[2] for r in rows], color=col, lw=1.6,
                capsize=2.5, marker=None, label=label, zorder=3)
    for x, r in zip(xs, rows):  # filled = p < 0.05 (paired t-test), hollow = not significant
        ax.plot(x, r[1], "o", ms=6.5, color=col, mfc=col if r[3] < 0.05 else "white",
                mew=1.6, zorder=4)
    i, dx, dy = LABEL_AT[key]
    ax.annotate(label.split(" (")[0], (xs[i], rows[i][1]), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, color="#33322e")
ax.set_xscale("log")
ax.set_xticks([500, 2000, 3500, 5000, 6920])
ax.set_xticklabels(["500", "2,000", "3,500", "5,000", "6,920\n(SST-2 full)"], fontsize=7.5)
ax.set_xlim(220, 13000)
ax.tick_params(axis="y", labelsize=7.5)
ax.set_xlabel("training sentences (log scale)", fontsize=8)
ax.set_ylabel("EDA gain (accuracy points)", fontsize=8)
ax.set_title("EDA gains are small: none is significant in this run (Windows run 3)",
             fontsize=9, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax.text(0.99, 0.02, "filled = p < 0.05 (paired t-test); bars = ±1 s.d. over seeds",
        transform=ax.transAxes, ha="right", fontsize=6.5, color="#55544f")
ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, f"fig_gain_by_size.{ext}"))
print("saved")
