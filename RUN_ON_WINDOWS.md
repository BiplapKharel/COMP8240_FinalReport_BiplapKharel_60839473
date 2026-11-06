# Running every experiment on your own Windows PC

`run_all.bat` runs every script in this repository in the right order and saves all
outputs under `results\`, with one log file per step in `results\logs\`.

## 1. One-time setup (about 10 minutes)

1. **Install Python 3.11** from <https://www.python.org/downloads/windows/>.
   On the first installer screen tick **"Add python.exe to PATH"**.
   (TensorFlow does not yet support every newer Python version on Windows; 3.11 is safe.)
2. Open **Command Prompt** (Start → type `cmd` → Enter) and go to the repo:
   ```
   cd /d "E:\macquarie\sem3\Application of Data Science\final\eda-replication-repo"
   ```
3. Install the packages into your normal Python (no virtual environment needed):
   ```
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   You do **not** need to download WordNet yourself: `run_all.bat` does it automatically
   the first time. (If you ever want to do it by hand, type `python` to open Python, then
   `import nltk`, then `nltk.download('wordnet')`, then `exit()`.)
   A `venv` folder is optional; if one exists, the script uses it, otherwise it uses `python`.
4. Open `run_all.bat` in Notepad and check the `GLOVE=` line points to your
   `glove.6B.300d.txt` (it is already set to the copy in your assignment1 folder).

## 2. Quick test first (about 15 minutes)

```
run_all.bat quick
```
This runs every step with 1 seed and small sizes and writes to `results_quick\`,
so nothing in `results\` is touched. If it ends with **ALL DONE**, the full run will work.
If a step fails, the screen names its log file in `results_quick\logs\`; open it to see why.

## 3. Full run (about 5–6 hours on a laptop)

Plug the laptop in, set it not to sleep (Settings → System → Power → Sleep: Never), then:
```
run_all.bat
```
Before overwriting anything, the script copies the current `results\` (the numbers in the
report) to `results_report_backup\`. Steps and rough times:

| Step | What | Output | Time |
|---|---|---|---|
| 07–08 | Download WNC pairs (57 MB) and build the bias dataset | `data\wnc\` | 1 min |
| 12 | WNC: baseline vs EDA | `results\wnc\` | 40 min |
| 13 | SST-2 replication (paper settings) | `results\sst2\` | 1.5 h |
| 14 | SST-2 with the authors' code settings | `results\sst2_authors\` | 1–1.5 h |
| 15 | AG News (downloads from Hugging Face) | `results\agnews\` | 1–2 h |
| 21–22 | Download the Christmas tweets (21 MB), extract them to a raw CSV, then preprocess it into the 5,000-tweet dataset | `data\tweets_raw\`, `data\tweets\` | 1 min |
| 23–24 | Check the emoji labels against reading (200 and 450 tweets) | `results\tweets\emoji_label_check_*.md` | seconds |
| 26 | The author's own labels on 100 tweets vs the first reading and the emoji labels | `results\tweets\human_check_100.md` | seconds |
| 25 | Christmas tweets: baseline vs EDA | `results\tweets\` | 20 min |
| 16–17 | Score the saved LLM-judge judgements | `results\judge*\judge_summary.md` | seconds |
| 20 | Summary figure | `results\fig_gain_by_size.pdf/.png` | seconds |

Each experiment folder gets `results.csv` (one row per run), `table_results.md`
(mean ± s.d., gain, paired t-test p) and `fig_accuracy_vs_size.png`.

## 4. Important: your numbers will differ slightly from the report

Neural-network training is not bit-for-bit reproducible across machines (different CPU,
TensorFlow build and Windows vs Linux), so expect each accuracy to move by a few tenths of
a point. The **direction** of every finding should hold. Either:
- keep the report as it is (its numbers are in `results_report_backup\`), and say in §3.1
  that you also re-ran everything on your own PC with similar results; or
- update the report's tables from your new `table_results.md` files.

The LLM-judge judgements are **re-scored, not re-generated** (re-generating needs an
Anthropic API key and would create different sentences).

## 5. Running one step by hand

Every line in `run_all.bat` is a normal command. Copy it into Command Prompt, replacing
`%PY%` with `python`, `%GLOVE%` with your GloVe path, `%OUT%` with
`results` and `%SEEDS%` with `5`.
