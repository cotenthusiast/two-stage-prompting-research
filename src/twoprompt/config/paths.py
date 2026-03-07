#src/twoprompt/config/paths.py

from pathlib import Path
from twoprompt.config.filenames import *

# Root ------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent

# Directories ----------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REVIEWS_DIR = DATA_DIR / "reviews"

RUNS_DIR = ROOT_DIR / "runs"
REPORTS_DIR = ROOT_DIR / "reports"
LOG_DIR = ROOT_DIR / "logs"

for directory in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    REVIEWS_DIR,
    RUNS_DIR,
    REPORTS_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Full file paths ------------------------------------
RAW_QUESTIONS_PATH = RAW_DIR / RAW_QUESTIONS_FILENAME
NORMALIZED_QUESTIONS_PATH = PROCESSED_DIR / NORMALIZED_QUESTIONS_FILENAME

BENCHMARK_SPLIT_PATH = SPLITS_DIR / BENCHMARK_SPLIT_FILENAME
FAITHFULNESS_SPLIT_PATH = SPLITS_DIR / FAITHFULNESS_SPLIT_FILENAME

HUMAN_REVIEW_PATH = REVIEWS_DIR / HUMAN_REVIEW_FILENAME