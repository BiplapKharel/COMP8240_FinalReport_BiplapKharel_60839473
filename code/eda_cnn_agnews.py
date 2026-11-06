#!/usr/bin/env python3
"""
EDA with a CNN classifier on AG News: does augmentation improve accuracy?

Uses the OFFICIAL AG News split (120k train / 7.6k test) from Hugging Face
(fancyzhx/ag_news). A small stratified validation set is carved out of the
train split for early stopping; the official test set is never touched
during training.

For each training size and random seed it trains the paper's CNN twice on the
SAME data subset, once normally and once with EDA augmentation, then reports
test accuracy for both.

Outputs (in --outdir):
  results.csv                  one row per run (acc, epochs, seconds)
  table_results.md             mean +/- std, gain, paired t-test p-value
  fig_accuracy_vs_size.png     left: accuracy vs size; right: EDA gain per size
  split_info.txt               split sizes and settings
  aug_cache.pkl                cached augmented sets (delete to regenerate)

Setup
  git clone https://github.com/jasonwei20/eda_nlp
  pip install tensorflow numpy pandas matplotlib nltk datasets scipy
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
  # GloVe 300-d: glove.6B.300d.txt from https://nlp.stanford.edu/projects/glove/

Usage
  # quick pipeline check (1 seed, ~5 min)
  python eda_cnn_agnews.py --glove glove.6B.300d.txt --eda_dir eda_nlp/code \
      --sizes 500 2000 5000 --seeds 1

  # full run
  python eda_cnn_agnews.py --glove glove.6B.300d.txt --eda_dir eda_nlp/code \
      --sizes 500 2000 5000 --seeds 5

Optional: --train_file / --test_file ("label<TAB>sentence") to skip the
Hugging Face download.
"""
import argparse
import csv
import os
import pickle
import random
import re
import sys
import time

import numpy as np

EDA = None  # the repo's eda module, imported in main()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def recommended(n):
    """Table 3 of the paper: (alpha, n_aug) by training set size."""
    if n <= 500:
        return 0.05, 16
    if n <= 2000:
        return 0.05, 8
    return 0.1, 4


def clean(s):
    """Lowercase, keep letters only (same spirit as the repo's get_only_chars)."""
    return " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split())


def load_tsv(path):
    xs, ys = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            label, _, sent = line.rstrip("\n").partition("\t")
            if label.strip() and clean(sent):
                ys.append(int(label))
                xs.append(clean(sent))
    return xs, ys


def load_ag_news(train_file=None, test_file=None):
    """Return (X_train, Y_train), (X_test, Y_test) with cleaned text."""
    if train_file and test_file:
        return load_tsv(train_file), load_tsv(test_file)
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    out = []
    for split in ("train", "test"):
        xs, ys = [], []
        for ex in ds[split]:
            t = clean(ex["text"])
            if t:
                xs.append(t)
                ys.append(int(ex["label"]))
        out.append((xs, ys))
    return out[0], out[1]


def stratified_val_split(X, Y, val_frac, seed):
    """Carve a stratified validation set out of the train split."""
    rng = random.Random(seed)
    by_class = {}
    for i, y in enumerate(Y):
        by_class.setdefault(y, []).append(i)
    tr_idx, va_idx = [], []
    for idxs in by_class.values():
        rng.shuffle(idxs)
        n_val = max(1, int(val_frac * len(idxs)))
        va_idx += idxs[:n_val]
        tr_idx += idxs[n_val:]
    take = lambda ids: ([X[i] for i in ids], [Y[i] for i in ids])
    return take(tr_idx), take(va_idx)


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
def augment(sents, labels, alpha, n_aug):
    """Return ONLY the augmented variants (originals are added by the caller).
    Every variant keeps its original label."""
    out_s, out_y = [], []
    for s, y in zip(sents, labels):
        try:
            variants = EDA.eda(s, alpha_sr=alpha, alpha_ri=alpha,
                               alpha_rs=alpha, p_rd=alpha, num_aug=n_aug)
        except (ValueError, IndexError):
            continue
        seen = {s}
        for v in variants:
            v = clean(v)
            if v and v not in seen:
                seen.add(v)
                out_s.append(v)
                out_y.append(y)
    return out_s, out_y


# --------------------------------------------------------------------------
# GloVe / encoding
# --------------------------------------------------------------------------
def load_glove(path, needed):
    """Stream the GloVe file and keep only vectors for words in `needed`."""
    vecs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            w, _, rest = line.partition(" ")
            if w in needed:
                vecs[w] = np.asarray(rest.split(), dtype="float32")
    return vecs


def build_vocab(words):
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    return vocab


def embedding_matrix(vocab, glove, seed=0, dim=300):
    E = np.random.RandomState(seed).normal(0, 0.1, (len(vocab), dim)).astype("float32")
    E[0] = 0
    for w, i in vocab.items():
        if w in glove:
            E[i] = glove[w]
    return E


def encode(sents, vocab, max_len):
    X = np.zeros((len(sents), max_len), dtype="int32")
    for i, s in enumerate(sents):
        ids = [vocab.get(w, 1) for w in s.split()][:max_len]
        X[i, :len(ids)] = ids
    return X


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def train_eval(Xtr, ytr, Xva, yva, Xte, yte, n_classes, E, seed, epochs, batch):
    """The paper's CNN (Section 9.1): Conv1D(128, 5) -> max-pool -> Dense(20)."""
    import tensorflow as tf
    from tensorflow.keras import callbacks, initializers, layers, models
    tf.keras.utils.set_random_seed(seed)
    init = initializers.RandomNormal(stddev=0.05)
    inp = layers.Input(shape=(Xtr.shape[1],))
    emb = layers.Embedding(E.shape[0], E.shape[1], trainable=False)
    x = emb(inp)
    emb.set_weights([E])  # works across Keras versions
    x = layers.Conv1D(128, 5, activation="relu", kernel_initializer=init)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(20, activation="relu", kernel_initializer=init)(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_initializer=init)(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    stop = callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                   restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                     batch_size=batch, callbacks=[stop], verbose=0)
    acc = float(model.evaluate(Xte, yte, verbose=0)[1])
    n_epochs = len(hist.epoch)
    tf.keras.backend.clear_session()
    return acc, n_epochs


# --------------------------------------------------------------------------
# Summary and plots
# --------------------------------------------------------------------------
def summarize(rows, outdir):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        ttest_rel = None

    df = pd.DataFrame(rows)
    order = list(df.sort_values("n_train")["size_label"].unique())
    wide = df.pivot_table(index=["size_label", "seed"], columns="eda", values="acc") * 100
    wide = wide.rename(columns={0: "base", 1: "eda"})
    wide["gain"] = wide["eda"] - wide["base"]

    def ms(v):
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        return v.mean(), sd

    lines = ["| Training size | Baseline (%) | + EDA (%) | Gain (points) | paired t-test p |",
             "|---|---|---|---|---|"]
    stats = {}
    for lab in order:
        w = wide.loc[lab]
        (bm, bs), (em, es), (gm, gs) = ms(w["base"]), ms(w["eda"]), ms(w["gain"])
        stats[lab] = dict(base=(bm, bs), eda=(em, es), gain=(gm, gs))
        if ttest_rel is not None and len(w) > 1:
            p = f"{ttest_rel(w['eda'], w['base']).pvalue:.3f}"
        else:
            p = "n/a"
        lines.append(f"| {lab} | {bm:.1f} ± {bs:.1f} | {em:.1f} ± {es:.1f} | "
                     f"{gm:+.1f} ± {gs:.1f} | {p} |")
    text = "\n".join(lines)
    with open(os.path.join(outdir, "table_results.md"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = np.arange(len(order))
    for key, name, color in [("base", "normal", "tab:blue"), ("eda", "EDA", "tab:orange")]:
        m = [stats[o][key][0] for o in order]
        s = [stats[o][key][1] for o in order]
        a1.errorbar(xs, m, yerr=s, marker="o", capsize=3, label=name, color=color)
        for i, o in enumerate(order):  # individual seeds
            a1.scatter([i] * len(wide.loc[o]), wide.loc[o][key], s=10, alpha=0.4, color=color)
    a1.set_xticks(xs)
    a1.set_xticklabels(order)
    a1.set_xlabel("training set size")
    a1.set_ylabel("test accuracy (%)")
    a1.set_title("Accuracy vs training size")
    a1.grid(alpha=0.3)
    a1.legend()

    g = [stats[o]["gain"][0] for o in order]
    gs_ = [stats[o]["gain"][1] for o in order]
    a2.bar(xs, g, yerr=gs_, capsize=3, color="tab:green", alpha=0.7)
    for i, o in enumerate(order):
        a2.scatter([i] * len(wide.loc[o]), wide.loc[o]["gain"], s=12, color="k", alpha=0.5)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(xs)
    a2.set_xticklabels(order)
    a2.set_xlabel("training set size")
    a2.set_ylabel("EDA gain (accuracy points)")
    a2.set_title("Gain from EDA (mean ± std, dots = seeds)")
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_accuracy_vs_size.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glove", required=True)
    ap.add_argument("--eda_dir", default="eda_nlp/code")
    ap.add_argument("--train_file", default=None, help="optional TSV (label<TAB>text)")
    ap.add_argument("--test_file", default=None, help="optional TSV (label<TAB>text)")
    ap.add_argument("--sizes", nargs="+", default=["500", "2000", "5000"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.02,
                    help="fraction of the TRAIN split used for validation")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--no_cache", action="store_true", help="ignore aug_cache.pkl")
    ap.add_argument("--outdir", default="eda_cnn_agnews_results")
    args = ap.parse_args()

    global EDA
    sys.path.insert(0, os.path.abspath(args.eda_dir))
    import eda as EDA_MODULE
    EDA = EDA_MODULE
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Data: official train/test, small validation carved from train ----
    (X_all, Y_all), (Xte, Yte) = load_ag_news(args.train_file, args.test_file)
    (X, Y), (Xva, Yva) = stratified_val_split(X_all, Y_all, args.val_frac, args.split_seed)
    n_classes = len(set(Y_all))
    pool_n = len(X)
    print(f"AG News: train pool {pool_n} / val {len(Xva)} / test {len(Xte)} "
          f"({n_classes} classes)")
    with open(os.path.join(args.outdir, "split_info.txt"), "w") as f:
        f.write(f"train pool: {pool_n}\nvalidation: {len(Xva)} (early stopping)\n"
                f"test (official): {len(Xte)}\nsplit_seed: {args.split_seed}\n"
                f"val_frac: {args.val_frac}\nsizes: {args.sizes}\nseeds: {args.seeds}\n"
                f"max_len: {args.max_len}\n")

    lens = np.array([len(s.split()) for s in X[:20000]])
    print(f"token length: median {int(np.median(lens))}, 95th pct "
          f"{int(np.percentile(lens, 95))}; "
          f"{(lens > args.max_len).mean() * 100:.1f}% truncated at max_len={args.max_len}")

    sizes, seen = [], set()
    for s in args.sizes:
        n, lab = (pool_n, "full") if s == "full" or int(s) >= pool_n else (int(s), s)
        if n not in seen:
            seen.add(n)
            sizes.append((n, lab))
    if any(lab == "full" for _, lab in sizes):
        print("WARNING: 'full' with EDA is very slow on AG News (hundreds of thousands of rows).")
    print(f"sizes {[lab for _, lab in sizes]}, {args.seeds} seeds")

    # ---- Phase 1: build all training subsets and EDA variants (cached) ----
    cache_path = os.path.join(args.outdir, "aug_cache.pkl")
    cache = {}
    if os.path.exists(cache_path) and not args.no_cache:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"loaded {len(cache)} cached augmentation sets")

    jobs = {}
    t0 = time.time()
    for seed in range(args.seeds):
        perm = list(range(pool_n))
        random.Random(seed).shuffle(perm)
        for n, lab in sizes:
            tr_s = [X[i] for i in perm[:n]]
            tr_y = [Y[i] for i in perm[:n]]
            alpha, n_aug = recommended(n)
            key = (seed, n, alpha, n_aug, args.split_seed, args.val_frac, pool_n)
            if key not in cache:
                random.seed(seed * 100003 + n)
                t1 = time.time()
                cache[key] = augment(tr_s, tr_y, alpha, n_aug)
                print(f"augmented seed {seed} size {lab}: +{len(cache[key][0])} rows "
                      f"({time.time() - t1:.0f}s)")
            jobs[(seed, n)] = (tr_s, tr_y, cache[key])
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"augmentation phase done in {time.time() - t0:.0f}s")

    # ---- Phase 2: one vocabulary / embedding matrix covering everything ----
    # Frozen pretrained vectors, so using val/test/augmented words here leaks no labels.
    words = set()
    for sents in [Xva, Xte] + [j[0] for j in jobs.values()] + [j[2][0] for j in jobs.values()]:
        for s in sents:
            words.update(s.split())
    t0 = time.time()
    glove = load_glove(args.glove, needed=words)
    print(f"GloVe: {len(glove)} of {len(words)} words found ({time.time() - t0:.0f}s)")
    vocab = build_vocab(words)
    E = embedding_matrix(vocab, glove)
    Xva_e, Xte_e = encode(Xva, vocab, args.max_len), encode(Xte, vocab, args.max_len)
    yva, yte = np.array(Yva), np.array(Yte)

    # ---- Phase 3: experiments ----
    rows = []
    fout = open(os.path.join(args.outdir, "results.csv"), "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=["dataset", "size_label", "n_train", "eda",
                                              "alpha", "n_aug", "seed", "train_rows",
                                              "epochs", "seconds", "acc"])
    writer.writeheader()
    for seed in range(args.seeds):
        for n, lab in sizes:
            tr_s, tr_y, (aug_s, aug_y) = jobs[(seed, n)]
            alpha, n_aug = recommended(n)
            for use_eda in (0, 1):
                s_in = tr_s + aug_s if use_eda else tr_s
                y_in = tr_y + aug_y if use_eda else tr_y
                t1 = time.time()
                acc, n_ep = train_eval(encode(s_in, vocab, args.max_len), np.array(y_in),
                                       Xva_e, yva, Xte_e, yte, n_classes, E,
                                       seed, args.epochs, args.batch)
                dt = time.time() - t1
                row = dict(dataset="ag_news", size_label=lab, n_train=n, eda=use_eda,
                           alpha=alpha if use_eda else "", n_aug=n_aug if use_eda else "",
                           seed=seed, train_rows=len(s_in), epochs=n_ep,
                           seconds=round(dt, 1), acc=round(acc, 5))
                rows.append(row)
                writer.writerow(row)
                fout.flush()
                print(f"seed {seed} size {lab:>5} {'EDA ' if use_eda else 'base'} "
                      f"rows {len(s_in):>6}  epochs {n_ep:>3}  {dt:5.0f}s  acc {acc:.4f}")
    fout.close()
    summarize(rows, args.outdir)


if __name__ == "__main__":
    main()