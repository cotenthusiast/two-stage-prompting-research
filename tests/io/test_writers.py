import pandas as pd
import json
from twoprompt.io.writers import write_normalized_questions, write_split_ids,write_group_splits, write_split_metadata

def test_write_normalized_questions(sample_raw_dataframe, sample_normalized_dataframe, tmp_path):
    raw_csv_path = tmp_path / "raw.csv"
    normalized_csv_path = tmp_path / "normalized.csv"

    sample_raw_dataframe.to_csv(raw_csv_path, index=False)

    write_normalized_questions(
        raw_questions_path=raw_csv_path,
        normalized_questions_path=normalized_csv_path,
    )

    actual_df = pd.read_csv(normalized_csv_path)
    pd.testing.assert_frame_equal(actual_df, sample_normalized_dataframe)

def test_write_split_ids_writes_json_id_list(tmp_path):
    split_ids = ["q_001", "q_014", "q_203", "q_417"]
    write_split_ids(split_ids,"robustness", tmp_path, "benchmark")
    with open(tmp_path / "benchmark" / "robustness_ids.json", "r", encoding="utf-8") as f:
        actual = json.load(f)
    assert actual == split_ids


def test_write_split_metadata_writes_json_metadata_dict(tmp_path):
    split_ids = ["q_001", "q_014", "q_203", "q_417"]
    split_metadata = {
        "split_name": "robustness",
        "split_ids": split_ids,
        "subjects": ["anatomy", "economics"],
        "per_subject": 2,
        "seed": 42,
        "strategy": "balanced_subject_sample",
        "actual_size": 4,
        "actual_subject_counts": {
            "anatomy": 2,
            "economics": 2,
        },
        "eligible_pool_size": 120,
        "excluded_id_count": 10,
    }
    write_split_metadata(split_metadata,"robustness", tmp_path, "benchmark")
    with open(tmp_path / "benchmark" / "robustness_metadata.json", "r", encoding="utf-8") as f:
        actual = json.load(f)
    assert actual == split_metadata

def test_write_group_splits_writes_all_expected_split_artifacts(tmp_path):
    split_ids = ["q_001", "q_014", "q_203", "q_417"]
    split_metadata = {
        "split_name": "robustness",
        "split_ids": split_ids,
        "subjects": ["anatomy", "economics"],
        "per_subject": 2,
        "seed": 42,
        "strategy": "balanced_subject_sample",
        "actual_size": 4,
        "actual_subject_counts": {
            "anatomy": 2,
            "economics": 2,
        },
        "eligible_pool_size": 120,
        "excluded_id_count": 10,
    }

    split_artifacts = {
        "robustness": {
            "ids": split_ids,
            "metadata": split_metadata,
        }
    }

    write_group_splits(split_artifacts, tmp_path, "benchmark")

    with open(tmp_path / "benchmark" / "robustness_ids.json", "r", encoding="utf-8") as f:
        actual_ids = json.load(f)

    with open(tmp_path / "benchmark" / "robustness_metadata.json", "r", encoding="utf-8") as f:
        actual_metadata = json.load(f)

    assert actual_ids == split_ids
    assert actual_metadata == split_metadata