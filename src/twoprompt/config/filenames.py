from pathlib import Path

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

# Filenames ------------------------------------------
RAW_QUESTIONS_FILENAME = "mmlu_raw.csv"
NORMALIZED_QUESTIONS_FILENAME = "mmlu_normalized.csv"

BENCHMARK_SPLIT_FILENAME = "benchmark_split_ids.csv"
FAITHFULNESS_SPLIT_FILENAME = "faithfulness_split_ids.csv"

HUMAN_REVIEW_FILENAME = "faithfulness_human_review.csv"