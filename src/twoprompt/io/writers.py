# src/twoprompt/io/writers.py

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from twoprompt.benchmarks.mmlu import build_normalized_dataframe
from twoprompt.config.paths import NORMALIZED_QUESTIONS_PATH, RAW_QUESTIONS_PATH


def write_raw_questions(raw_questions_path: Path = RAW_QUESTIONS_PATH) -> None:
    """
    Downloads the raw MMLU test split and saves it as a CSV file.

    Args:
        raw_questions_path: Full file path where the raw questions CSV
            should be written.
    """
    dataset = load_dataset("cais/mmlu", split="test")
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
    df_normalized.to_csv(normalized_questions_path, index=False)


def write_split_ids(
    split_ids: list[str],
    split_name: str,
    output_dir: Path,
    artifact_group: str,
) -> None:
    """
    Write one split's question IDs to disk.

    Args:
        split_ids: Ordered list of question IDs belonging to a single split.
        split_name: Logical split name, such as "robustness" or "review".
        output_dir: Root directory under which split artifacts should be written.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Path to the written split-IDs artifact.

    Notes:
        This function is responsible only for persistence of the ID list for one
        split. It should not compute metadata or validate split correctness.
    """
    filename = split_name + "_ids.json"
    output_location = output_dir / artifact_group / filename
    output_location.parent.mkdir(parents=True, exist_ok=True)

    with open(output_location, "w", encoding="utf-8") as file:
        json.dump(split_ids, file, indent=2)


def write_split_metadata(
    split_metadata: dict[str, Any],
    split_name: str,
    output_dir: Path,
    artifact_group: str,
) -> None:
    """
    Write one split's metadata dictionary to disk.

    Args:
        split_metadata: Metadata describing a single split artifact.
        split_name: Logical split name, such as "robustness" or "review".
        output_dir: Root directory under which split artifacts should be written.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Path to the written metadata artifact.

    Notes:
        The metadata is expected to already be built before this function is called.
        This function only serializes and saves it.
    """
    filename = split_name + "_metadata.json"
    output_location = output_dir / artifact_group / filename
    output_location.parent.mkdir(parents=True, exist_ok=True)

    with open(output_location, "w", encoding="utf-8") as file:
        json.dump(split_metadata, file, indent=2)


def write_group_splits(
    split_artifacts: dict[str, dict[str, Any]],
    output_dir: Path,
    artifact_group: str,
) -> None:
    """
    Write the full set of split artifacts to disk.

    Args:
        split_artifacts: Mapping from split name to its full artifact payload.
            Each payload must contain:
                - "ids": list of question IDs
                - "metadata": metadata dictionary for that split
        output_dir: Root directory under which split artifacts should be written.
        artifact_group: Storage namespace that separates benchmark, faithfulness,
            and stronger-model artifacts.

    Returns:
        Mapping from logical artifact names to the paths that were written.

    Notes:
        This is the top-level split writer for the phase. It delegates to the
        single-split writer functions rather than duplicating file logic.
    """

    for split_name, split_artifact in split_artifacts.items():
        write_split_ids(
            split_artifact["ids"],
            split_name,
            output_dir,
            artifact_group,
        )
        write_split_metadata(
            split_artifact["metadata"],
            split_name,
            output_dir,
            artifact_group,
        )