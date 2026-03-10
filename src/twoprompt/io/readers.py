# src/twoprompt/io/readers.py
import json
from pathlib import Path
from typing import Any, Literal
from twoprompt.config.paths import RAW_DIR, PROCESSED_DIR, SPLITS_DIR
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

def read_split_ids(
    split_name: str,
    input_dir: Path = SPLITS_DIR,
    artifact_group: str = "",
) -> list[str]:
    """
    Read one split's question IDs from disk.

    Args:
        split_name: Logical split name, such as "robustness" or "review".
        input_dir: Root directory from which split artifacts should be read.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Ordered list of question IDs for the requested split.
    """
    filename = f"{split_name}_ids.json"
    split_ids_path = input_dir / artifact_group / filename

    with split_ids_path.open("r", encoding="utf-8") as f:
        split_ids = json.load(f)

    return split_ids


def read_split_metadata(
    split_name: str,
    input_dir: Path = SPLITS_DIR,
    artifact_group: str = "",
) -> dict[str, Any]:
    """
    Read one split's metadata dictionary from disk.

    Args:
        split_name: Logical split name, such as "robustness" or "review".
        input_dir: Root directory from which split artifacts should be read.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Metadata dictionary describing the requested split.
    """
    filename = f"{split_name}_metadata.json"
    split_metadata_path = input_dir / artifact_group / filename

    with split_metadata_path.open("r", encoding="utf-8") as f:
        split_metadata = json.load(f)

    return split_metadata


def read_group_splits(
    artifact_group: str,
    input_dir: Path = SPLITS_DIR,
) -> dict[str, dict[str, Any]]:
    """
    Read the full set of split artifacts for one artifact group from disk.

    Args:
        input_dir: Root directory from which split artifacts should be read.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Mapping from split name to its reconstructed artifact payload.
    """
    group_to_splits = {
        "benchmark": ["robustness"],
        "faithfulness": ["review"],
        "stronger_model": ["review"],
    }

    split_names = group_to_splits[artifact_group]

    data: dict[str, dict[str, Any]] = {}

    for split_name in split_names:
        data[split_name] = {
            "ids": read_split_ids(split_name, input_dir, artifact_group),
            "metadata": read_split_metadata(split_name, input_dir, artifact_group),
        }

    return data