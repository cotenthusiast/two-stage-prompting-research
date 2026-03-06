# config.settings
# Paths, Subject Lists, API Keys, Constants

from pathlib import Path
from dotenv import load_dotenv
import os

current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent.parent

# Logs ------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Eval ------------------------------------------------
EVAL_SCRIPT_DIR = BASE_DIR / "evaluation" 
EVAL_OUTPUT = BASE_DIR / "data" / "output"
EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

# LLMs ------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GEMINI_API_KEY == None:
    print("Gemini API key missing")
if OPENAI_API_KEY == None:
    print("Openai API key missing")
if GROQ_API_KEY == None:
    print("Groq API key missing")
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 500
TIMEOUT = 30
MAX_RETRIES = 3
LLMS_DIR = BASE_DIR / "models"

# MMLU ------------------------------------------------
QUESTIONS_PATH = BASE_DIR / "data" / "input" 
QUESTIONS_PATH.mkdir(parents=True, exist_ok=True)
QUESTIONS_PER_SUBJECT = 50
MCQ_ANSWER_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
    "global_facts"
]
NO_OF_SUBJECTS = len(SUBJECTS)
