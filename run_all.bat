@echo off
REM ==========================================================================
REM  run_all.bat  -  run every experiment in this repository on Windows and
REM                  save all outputs under results\ (logs in results\logs\)
REM
REM  Usage (from this folder, in Command Prompt):
REM      run_all.bat            full run   (several hours)
REM      run_all.bat quick      smoke test (1 seed, small sizes, ~15 min)
REM
REM  Before the first run, set GLOVE below to your glove.6B.300d.txt file.
REM ==========================================================================
setlocal EnableExtensions

REM ---- 1. SETTINGS ---------------------------------------------------------
set "GLOVE=E:\macquarie\sem3\Application of Data Science\assignment1\eda_nlp-master\eda_nlp-master\glove.6B.300d.txt"
REM Uses venv\Scripts\python.exe if you made a virtual environment, otherwise your normal Python
if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else (set "PY=python")
set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

if /I "%MODE%"=="quick" (
    set "SEEDS=1"
    set "AUTH_SEEDS=1"
    set "SST_SIZES=500 2000"
    set "WNC_SIZES=500"
    set "AG_SIZES=500"
    set "TW_SIZES=500 full"
    set "OUT=results_quick"
) else (
    set "SEEDS=5"
    set "AUTH_SEEDS=3"
    set "SST_SIZES=500 2000 5000 full"
    set "WNC_SIZES=500 2000 5000"
    set "AG_SIZES=500 2000 5000"
    set "TW_SIZES=500 2000 full"
    set "OUT=results"
)

REM Python: UTF-8 everywhere (Windows default code page cannot write some symbols)
set "PYTHONUTF8=1"
set "TF_CPP_MIN_LOG_LEVEL=2"

REM ---- 2. CHECKS -------------------------------------------------------------
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11 from python.org and tick "Add python.exe to PATH".
    exit /b 1
)
%PY% -c "import tensorflow, nltk, sklearn, pandas, scipy, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Some Python packages are missing. Run:  python -m pip install -r requirements.txt
    exit /b 1
)
echo Checking WordNet ^(downloads once if missing^) ...
%PY% -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"
if not exist "%GLOVE%" (
    echo [ERROR] GloVe file not found: %GLOVE%
    echo         Edit the GLOVE line at the top of run_all.bat.
    exit /b 1
)
if not exist "%OUT%\logs" mkdir "%OUT%\logs"

REM Keep a copy of the results used in the report before overwriting them
if /I "%MODE%"=="full" if not exist results_report_backup (
    echo Backing up current results\ to results_report_backup\ ...
    xcopy results results_report_backup\ /E /I /Q /Y >nul
)

echo ============================================================
echo  Mode: %MODE%    Output folder: %OUT%    Started: %date% %time%
echo ============================================================

REM ---- 3. DATA PREPARATION (fast) -------------------------------------------
if not exist data\wnc_raw\biased.word.train.csv (
    echo.
    echo [07] Downloading WNC pairs ^(57 MB^) ...
    if not exist data\wnc_raw mkdir data\wnc_raw
    curl -L -f -o data\wnc_raw\biased.word.train.csv https://raw.githubusercontent.com/erickfm/bias-neutralization/main/data/biased.word.train.csv || goto :fail
)
call :step "08 WNC: build biased-vs-neutral dataset" ^
  %PY% code\build_wnc_dataset.py --csv data\wnc_raw\biased.word.train.csv --outdir data\wnc || goto :fail

REM ---- 4. EXPERIMENTS (slow) -------------------------------------------------
call :step "12 WNC: baseline vs EDA  (~40 min)" ^
  %PY% code\run_eda_cnn.py split --name wnc --glove "%GLOVE%" --train data\wnc\train.txt --val data\wnc\dev.txt --test data\wnc\test.txt --sizes %WNC_SIZES% --seeds %SEEDS% --max_len 50 --outdir %OUT%\wnc || goto :fail

call :step "13 SST-2 replication, paper settings  (~1.5 h)" ^
  %PY% code\run_eda_cnn.py split --name sst2 --glove "%GLOVE%" --train data\sst2\train.txt --val data\sst2\dev.txt --test data\sst2\test.txt --sizes %SST_SIZES% --seeds %SEEDS% --max_len 60 --outdir %OUT%\sst2 || goto :fail

call :step "14 SST-2, authors' code settings  (~1-1.5 h)" ^
  %PY% code\run_eda_cnn.py split --protocol authors --name sst2_authors --glove "%GLOVE%" --train data\sst2\train.txt --val data\sst2\dev.txt --test data\sst2\test.txt --sizes %WNC_SIZES% --seeds %AUTH_SEEDS% --max_len 50 --epochs 50 --outdir %OUT%\sst2_authors || goto :fail

call :step "15 AG News: baseline vs EDA (downloads from Hugging Face)  (~1-2 h)" ^
  %PY% code\eda_cnn_agnews.py --glove "%GLOVE%" --eda_dir code --sizes %AG_SIZES% --seeds %SEEDS% --outdir %OUT%\agnews || goto :fail

REM ---- Christmas tweets (emoji-labelled, from RUMgroup on GitHub) ---------
if not exist data\tweets_raw mkdir data\tweets_raw
if not exist data\tweets_raw\tweets.Christmas.sampled.RData (
    echo.
    echo [20a] Downloading Christmas 2014 tweets ^(8 MB^) ...
    curl -L -f -o data\tweets_raw\tweets.Christmas.sampled.RData https://raw.githubusercontent.com/RUMgroup/TweetingChristmas/master/tweets.Christmas.sampled.RData || goto :fail
)
if not exist data\tweets_raw\tweets.Christmas.RDS (
    echo [20a] Downloading Christmas 2017 tweets ^(13 MB^) ...
    curl -L -f -o data\tweets_raw\tweets.Christmas.RDS https://raw.githubusercontent.com/RUMgroup/TweetingChristmasv2/master/data/tweets.Christmas.RDS || goto :fail
)
call :step "21 Tweets: extract raw tweets to CSV (no cleaning)" ^
  %PY% code\extract_raw_tweets.py --rdata data\tweets_raw\tweets.Christmas.sampled.RData --rds data\tweets_raw\tweets.Christmas.RDS --out data\tweets_raw\christmas_tweets_raw.csv || goto :fail

call :step "22 Tweets: preprocess raw CSV into the 5,000-tweet dataset" ^
  %PY% code\preprocess_tweets.py --raw data\tweets_raw\christmas_tweets_raw.csv --n 5000 --n_test 1000 --n_dev 500 --outdir data\tweets || goto :fail

call :step "23 Tweets: emoji labels vs reading, 200 random tweets" ^
  %PY% code\check_emoji_labels.py --hand data\tweets\label_check_200.csv --out %OUT%\tweets\emoji_label_check_200.md || goto :fail

call :step "24 Tweets: emoji labels vs reading, 450 tweets" ^
  %PY% code\check_emoji_labels.py --hand data\tweets\hand_labels_450.csv --out %OUT%\tweets\emoji_label_check_450.md || goto :fail

call :step "26 Tweets: author's own labels vs first reading and emoji, 100 tweets" ^
  %PY% code\compare_human_reading.py --data data\tweets\human_check_100.csv --out %OUT%\tweets\human_check_100.md || goto :fail

call :step "25 Tweets: baseline vs EDA  (~20 min)" ^
  %PY% code\run_eda_cnn.py split --name tweets --glove "%GLOVE%" --train data\tweets\train.txt --val data\tweets\dev.txt --test data\tweets\test.txt --sizes %TW_SIZES% --seeds %SEEDS% --max_len 40 --outdir %OUT%\tweets || goto :fail

REM ---- 5. EVALUATION AND SUMMARY (fast) -------------------------------------
REM The LLM judgements were made once and are saved; they are re-SCORED here,
REM not re-generated (re-generating needs an API key and would change the items).
call :step "16 LLM judge: score SST-2 judgements" ^
  %PY% code\llm_judge.py score --key results\judge\judge_key.csv --judgements results\judge\judgements.csv --outdir %OUT%\judge || goto :fail

call :step "17 LLM judge: score WNC judgements" ^
  %PY% code\llm_judge.py score --key results\judge_wnc\judge_key.csv --judgements results\judge_wnc\judgements.csv --outdir %OUT%\judge_wnc || goto :fail

if /I "%MODE%"=="full" (
    call :step "20 Summary figure: EDA gain vs training size" ^
      %PY% code\plot_gains.py || goto :fail
)

echo.
echo ============================================================
echo  ALL DONE: %date% %time%
echo  Results: %OUT%\   Logs: %OUT%\logs\
echo ============================================================
exit /b 0

REM ---------------------------------------------------------------------------
:step
REM %1 = title, rest = command. Output goes to the screen summary and a log file.
set "TITLE=%~1"
shift
set "CMD="
:collect
if "%~1"=="" goto :run
set CMD=%CMD% %1
shift
goto :collect
:run
set "LOGNAME=%TITLE:~0,2%"
echo.
echo [%TITLE%]  started %time%
%CMD% > "%OUT%\logs\step_%LOGNAME%.log" 2>&1
if errorlevel 1 (
    echo     FAILED - see %OUT%\logs\step_%LOGNAME%.log
    exit /b 1
)
echo     done %time%   (log: %OUT%\logs\step_%LOGNAME%.log)
exit /b 0

:fail
echo.
echo ************************************************************
echo  Stopped because a step failed. Open the log named above,
echo  fix the problem, then run run_all.bat again.
echo ************************************************************
exit /b 1
