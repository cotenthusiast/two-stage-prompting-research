#src/twoprompt/io/writers.py

from pathlib import Path
import pandas as pd
from datasets import load_dataset
from twoprompt.config.paths import RAW_QUESTIONS_PATH, NORMALIZED_QUESTIONS_PATH
from twoprompt.benchmarks.mmlu import build_normalized_dataframe

def write_raw_questions(raw_questions_path: Path = RAW_QUESTIONS_PATH) -> None:
    """
    Downloads the raw MMLU test split and saves it as a CSV file.

    Args:
        raw_questions_path: Full file path where the raw questions CSV
            should be written.
    """
    dataset = load_dataset("cais/mmlu", split = "test")
    df = dataset.to_pandas()
    df.to_csv(raw_questions_path, index=False)

def write_normalized_questions(
    raw_questions_path: Path = RAW_QUESTIONS_PATH,
    normalized_questions_path: Path = NORMALIZED_QUESTIONS_PATH,
) -> None:
    """
    Reads the raw MMLU CSV, converts it into the project's normalized
    schema, and saves the normalized result as a CSV file.

    Args:
        raw_questions_path: Full file path of the raw questions CSV.
        normalized_questions_path: Full file path where the normalized
            questions CSV should be written.
    """
    df = pd.read_csv(raw_questions_path)
    df_normalized = build_normalized_dataframe(df)
    df_normalized.to_csv(normalized_questions_path, index = False)