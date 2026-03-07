# config/settings.py
# Paths, filenames, API keys, experiment constants

from pathlib import Path
from dotenv import load_dotenv
import os

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

# Full file paths ------------------------------------
RAW_QUESTIONS_PATH = RAW_DIR / RAW_QUESTIONS_FILENAME
NORMALIZED_QUESTIONS_PATH = PROCESSED_DIR / NORMALIZED_QUESTIONS_FILENAME

BENCHMARK_SPLIT_PATH = SPLITS_DIR / BENCHMARK_SPLIT_FILENAME
FAITHFULNESS_SPLIT_PATH = SPLITS_DIR / FAITHFULNESS_SPLIT_FILENAME

HUMAN_REVIEW_PATH = REVIEWS_DIR / HUMAN_REVIEW_FILENAME

# API / model settings -------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GEMINI_API_KEY is None:
    print("Gemini API key missing")
if OPENAI_API_KEY is None:
    print("OpenAI API key missing")
if GROQ_API_KEY is None:
    print("Groq API key missing")

SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 500
TIMEOUT = 30
MAX_RETRIES = 3

# Models ---------------------------------------------
GPT_FLASH_MODEL = "gpt-4.1-mini"
GEMINI_FLASH_MODEL = "gemini-2.0-flash"
LLAMA_MODEL = "llama-3.3-70b-versatile"

MODELS = [
    GPT_FLASH_MODEL,
    GEMINI_FLASH_MODEL,
    LLAMA_MODEL,
]

# MMLU ------------------------------------------------
MCQ_ANSWER_MAP = "ABCD"

SUBJECTS = [
    "high_school_physics",
    "college_mathematics",
    "anatomy",
    "college_chemistry",
    "computer_security",
    "medical_genetics",
    "college_biology",
    "clinical_knowledge",
    "high_school_psychology",
    "econometrics",
    "sociology",
    "philosophy",
    "high_school_world_history",
    "jurisprudence",
    "professional_law",
    "professional_medicine",
    "professional_accounting",
    "moral_scenarios",
    "nutrition",
    "global_facts",
]

NO_OF_SUBJECTS = len(SUBJECTS)

# Track A: robustness / accuracy ---------------------
ROBUSTNESS_TRACK_NAME = "robustness"
BENCHMARK_TOTAL_QUESTIONS = 1000
BENCHMARK_SUBJECTS = SUBJECTS

BASELINE_METHOD = "baseline"
TWOPROMPT_METHOD = "two_prompt"
PRIDE_METHOD = "pride"
TWOPROMPT_PRIDE_METHOD = "two_prompt_pride"

ROBUSTNESS_METHODS = [
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    PRIDE_METHOD,
    TWOPROMPT_PRIDE_METHOD,
]

# Track B: faithfulness ------------------------------
FAITHFULNESS_TRACK_NAME = "faithfulness"
FAITHFULNESS_SUBJECTS = SUBJECTS
FAITHFULNESS_QUESTIONS_PER_SUBJECT = 3
FAITHFULNESS_TOTAL_QUESTIONS = len(FAITHFULNESS_SUBJECTS) * FAITHFULNESS_QUESTIONS_PER_SUBJECT

ANSWER_MATCHING_METHOD = "answer_matching"

FAITHFULNESS_METHODS = [
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    ANSWER_MATCHING_METHOD,
]

# Human review labels --------------------------------
HUMAN_LABEL_CORRECT = "correct"
HUMAN_LABEL_INCORRECT = "incorrect"
HUMAN_LABEL_AMBIGUOUS = "ambiguous"

HUMAN_LABELS = [
    HUMAN_LABEL_CORRECT,
    HUMAN_LABEL_INCORRECT,
    HUMAN_LABEL_AMBIGUOUS,
]