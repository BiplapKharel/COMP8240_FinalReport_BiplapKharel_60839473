#!/usr/bin/env python3
"""
EDA + CNN experiments on SST-2 (replication) and the new datasets (WNC, Christmas tweets).

Re-uses the CNN, augmentation wrapper and summary code from eda_cnn_agnews.py so
that all datasets (SST-2, AG News, WNC, tweets) are run with the SAME model
and the SAME EDA settings (Table 3 of Wei & Zou, 2019).

Modes
  split : fixed train / validation / test files (SST-2).
          For each size in --sizes and each seed, a random subset of the train
          file is drawn and a CNN is trained with and without EDA.
  cv    : stratified k-fold cross-validation for very small datasets (not used in the report).
          10% of each training fold is held out for early stopping.

Examples
  python run_eda_cnn.py split --name sst2 --glove glove.gz \
      --train data/sst2/train.txt --val data/sst2/dev.txt --test data/sst2/test.txt \
      --sizes 500 2000 5000 full --seeds 5 --max_len 60 --outdir results/sst2

  python run_eda_cnn.py cv --name mydata --glove glove.gz \
      --data data/mydata/train.txt --folds 5 --seeds 5 \
      --max_len 40 --outdir results/mydata
"""
import argparse
import csv
import gzip
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eda as EDA_MODULE          # noqa: E402  (original repo code, unchanged)
import eda_cnn_agnews as base     # noqa: E402  (shared CNN / augmentation / summary)

base.EDA = EDA_MODULE


def load_glove(path, needed):
    """Read GloVe vectors (plain or .gz) for the words in `needed` only."""
    opener = gzip.open if path.endswith(".gz") else open
    vecs = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            w, _, rest = line.partition(" ")
            if w in needed:
                vecs[w] = np.asarray(rest.split(), dtype="float32")
    return vecs


def build_embeddings(all_sents, glove_path, max_len):
    words = set()
    for s in all_sents:
        words.update(s.split())
    glove = load_glove(glove_path, words)
    print(f"GloVe coverage: {len(glove)}/{len(words)} words "
          f"({100 * len(glove) / max(1, len(words)):.1f}%)")
    vocab = base.build_vocab(words)
    return vocab, base.embedding_matrix(vocab, glove)


# ---------------------------------------------------------------------------
# "authors" protocol: mirrors the authors' experiment code (experiments/
# e_2_cnn_aug.py, methods.py, nlp_aug.py, e_config.py in jasonwei20/eda_nlp)
# rather than the paper's text:
#   * EDA rates alpha_sr=0.3, alpha_ri=0.2, alpha_rs=0.1, p_rd=0.15 (eda_4 defaults)
#   * n_aug = 32 for 500/2000 rows, 16 for 5000/full (SST-2 column of e_config)
#   * batch 1024; early stopping on val_loss, patience 3, NO best-weight restore
#   * validation = Keras validation_split=0.1 of the SHUFFLED augmented training
#     set (so augmented copies of training sentences can appear in validation)
# ---------------------------------------------------------------------------
AUTHOR_RATES = dict(alpha_sr=0.3, alpha_ri=0.2, alpha_rs=0.1, p_rd=0.15)


def author_n_aug(n):
    return 32 if n <= 2000 else 16


def author_augment(sents, labels, n_aug):
    out_s, out_y = [], []
    for s, y in zip(sents, labels):
        try:
            variants = EDA_MODULE.eda(s, num_aug=n_aug, **AUTHOR_RATES)
        except (ValueError, IndexError):
            continue
        for v in variants[:-1]:  # eda() appends the original last; caller adds it
            v = base.clean(v)
            if v:
                out_s.append(v)
                out_y.append(y)
    return out_s, out_y


def train_eval_authors(Xtr, ytr, Xte, yte, n_classes, E, seed, epochs):
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models
    tf.keras.utils.set_random_seed(seed)
    idx = np.random.RandomState(seed).permutation(len(ytr))
    Xtr, ytr = Xtr[idx], ytr[idx]
    inp = layers.Input(shape=(Xtr.shape[1],))
    emb = layers.Embedding(E.shape[0], E.shape[1], trainable=False)
    x = emb(inp)
    emb.set_weights([E])
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(20, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_initializer="random_normal")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(Xtr, ytr, validation_split=0.1, epochs=epochs, batch_size=1024,
                     callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3)],
                     verbose=0)
    acc = float(model.evaluate(Xte, yte, verbose=0)[1])
    tf.keras.backend.clear_session()
    return acc, len(hist.epoch)


def run_pair(name, lab, n, seed, tr_s, tr_y, aug, va, te, vocab, E, n_classes, args, writer, rows):
    """Train baseline and +EDA CNNs on the same subset; record both."""
    alpha, n_aug = base.recommended(n)
    if args.protocol == "authors":
        alpha, n_aug = "0.3/0.2/0.1/0.15", author_n_aug(n)
    Xva, yva = base.encode(va[0], vocab, args.max_len), np.array(va[1])
    Xte, yte = base.encode(te[0], vocab, args.max_len), np.array(te[1])
    for use_eda in (0, 1):
        s_in = tr_s + aug[0] if use_eda else tr_s
        y_in = tr_y + aug[1] if use_eda else tr_y
        t1 = time.time()
        if args.protocol == "authors":
            acc, n_ep = train_eval_authors(base.encode(s_in, vocab, args.max_len), np.array(y_in),
                                           Xte, yte, n_classes, E, seed, args.epochs)
        else:
            acc, n_ep = base.train_eval(base.encode(s_in, vocab, args.max_len), np.array(y_in),
                                        Xva, yva, Xte, yte, n_classes, E, seed,
                                        args.epochs, args.batch)
        row = dict(dataset=name, size_label=lab, n_train=n, eda=use_eda,
                   alpha=alpha if use_eda else "", n_aug=n_aug if use_eda else "",
                   seed=seed, train_rows=len(s_in), epochs=n_ep,
                   seconds=round(time.time() - t1, 1), acc=round(acc, 5))
        rows.append(row)
        writer.writerow(row)
        print(f"{name} seed {seed} {lab:>5} {'EDA ' if use_eda else 'base'} rows {len(s_in):>6} "
              f"ep {n_ep:>2} acc {acc:.4f}", flush=True)


def mode_split(args, writer, rows):
    X, Y = base.load_tsv(args.train)
    va, te = base.load_tsv(args.val), base.load_tsv(args.test)
    n_classes = len(set(Y))
    sizes = [(len(X), "full") if s == "full" else (int(s), s) for s in args.sizes]
    jobs = {}
    for seed in range(args.seeds):
        perm = list(range(len(X)))
        random.Random(seed).shuffle(perm)
        for n, lab in sizes:
            tr_s, tr_y = [X[i] for i in perm[:n]], [Y[i] for i in perm[:n]]
            random.seed(seed * 100003 + n)
            aug = (author_augment(tr_s, tr_y, author_n_aug(n)) if args.protocol == "authors"
                   else base.augment(tr_s, tr_y, *base.recommended(n)))
            jobs[(seed, n, lab)] = (tr_s, tr_y, aug)
    every = va[0] + te[0] + [s for j in jobs.values() for s in j[0] + j[2][0]]
    vocab, E = build_embeddings(every, args.glove, args.max_len)
    for (seed, n, lab), (tr_s, tr_y, aug) in jobs.items():
        run_pair(args.name, lab, n, seed, tr_s, tr_y, aug, va, te, vocab, E,
                 n_classes, args, writer, rows)


def mode_cv(args, writer, rows):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    X, Y = base.load_tsv(args.data)
    n_classes = len(set(Y))
    plan = []
    for seed in range(args.seeds):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(X, Y)):
            tr_s, tr_y = [X[i] for i in tr], [Y[i] for i in tr]
            tr_s, va_s, tr_y, va_y = train_test_split(tr_s, tr_y, test_size=0.1,
                                                      stratify=tr_y, random_state=seed)
            random.seed(seed * 1000 + k)
            aug = base.augment(tr_s, tr_y, *base.recommended(len(tr_s)))
            plan.append((seed, k, tr_s, tr_y, aug, (va_s, va_y),
                         ([X[i] for i in te], [Y[i] for i in te])))
    every = X + [s for p in plan for s in p[4][0]]
    vocab, E = build_embeddings(every, args.glove, args.max_len)
    for seed, k, tr_s, tr_y, aug, va, te in plan:
        # one "seed" in the summary = one (seed, fold) pair
        run_pair(args.name, f"{len(tr_s)}", len(tr_s), seed * 100 + k, tr_s, tr_y, aug,
                 va, te, vocab, E, n_classes, args, writer, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["split", "cv"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--glove", required=True)
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--test")
    ap.add_argument("--data")
    ap.add_argument("--sizes", nargs="+", default=["500", "2000", "5000", "full"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--protocol", choices=["clean", "authors"], default="clean",
                    help="clean = paper's Table 3 settings + held-out dev set (default); "
                         "authors = settings found in the authors' experiment code")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    with open(os.path.join(args.outdir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "size_label", "n_train", "eda",
                                               "alpha", "n_aug", "seed", "train_rows",
                                               "epochs", "seconds", "acc"])
        writer.writeheader()
        (mode_split if args.mode == "split" else mode_cv)(args, writer, rows)
    base.summarize(rows, args.outdir)


if __name__ == "__main__":
    main()
