# src/twoprompt/io/readers.py

from pathlib import Path
from twoprompt.config.paths import RAW_DIR, PROCESSED_DIR
import pandas as pd

def read_raw_questions(filename: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame: 
    """
    Reads a csv file containing the raw MMLU questions

    Args:
        filename: filename of raw questions
        raw_dir: directory containing the file

    Returns:
        pandas dataframe containing the raw_questions
    """
    raw_location = raw_dir.joinpath(filename)
    df = pd.read_csv(raw_location)
    return df

def read_normalized_questions(filename : str, processed_dir: Path = PROCESSED_DIR)-> pd.DataFrame :
    """
    Reads a csv file containing the normalized MMLU questions

    Args:
        filename: filename of normalized questions
        processed_dir: directory containing the file

    Returns:
        pandas dataframe containing the normalized questions
    """
    processed_location = processed_dir.joinpath(filename)
    return pd.read_csv(processed_location)