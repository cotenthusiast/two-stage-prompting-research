# src/twoprompt/config/models.py

from dotenv import load_dotenv
import os

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
