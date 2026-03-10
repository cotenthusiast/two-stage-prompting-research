import pandas as pd
import json
from twoprompt.io.readers import read_raw_questions, read_normalized_questions, read_split_ids, read_group_splits, read_split_metadata

def test_read_raw_questions(tmp_path, sample_raw_dataframe):
    raw_csv_path = tmp_path / "raw.csv"
    data = (
        "subject,question,choices,answer\n"
        'computer_security,Which protocol is primarily used to securely browse websites?,"[""FTP"", ""HTTP"", ""HTTPS"", ""SMTP""]",2\n'
        'high_school_physics,What is the SI unit of force?,"[""Joule"", ""Newton"", ""Watt"", ""Pascal""]",1'
    )
    raw_csv_path.write_text(data)
    pd.testing.assert_frame_equal(read_raw_questions("raw.csv",raw_dir=tmp_path),sample_raw_dataframe)

def test_read_normalized_questions(tmp_path, sample_normalized_dataframe):
    normalized_csv_path = tmp_path / "normalized.csv"
    data = (
        "question_id,subject,question_text,choice_a,choice_b,choice_c,choice_d,correct_option,correct_answer_text\n"
        "4865890d7f0efae8,computer_security,Which protocol is primarily used to securely browse websites?,FTP,HTTP,HTTPS,SMTP,C,HTTPS\n"
        "5e9876049bf053f9,high_school_physics,What is the SI unit of force?,Joule,Newton,Watt,Pascal,B,Newton"
    )
    normalized_csv_path.write_text(data)
    pd.testing.assert_frame_equal(read_normalized_questions("normalized.csv", processed_dir = tmp_path),sample_normalized_dataframe)

def test_read_split_ids_returns_id_list(tmp_path):
    split_ids = ["q_001", "q_014", "q_203", "q_417"]
    group_dir = tmp_path / "benchmark"
    group_dir.mkdir()

    with open(group_dir / "robustness_ids.json", "w", encoding="utf-8") as f:
        json.dump(split_ids, f)

    actual_ids = read_split_ids("robustness", tmp_path, "benchmark")
    assert actual_ids == split_ids

def test_read_split_metadata_returns_metadata_dict(tmp_path):
    split_metadata = {
        "split_name": "robustness",
        "split_ids": ["q_001", "q_014", "q_203", "q_417"],
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
    group_dir = tmp_path / "benchmark"
    group_dir.mkdir()

    with open(group_dir / "robustness_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_metadata, f)

    actual_ids = read_split_metadata("robustness", tmp_path, "benchmark")
    assert actual_ids == split_metadata

def test_read_group_splits_returns_nested_payload_for_benchmark(tmp_path):
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

    group_dir = tmp_path / "benchmark"
    group_dir.mkdir()

    with open(group_dir / "robustness_ids.json", "w", encoding="utf-8") as f:
        json.dump(split_ids, f)

    with open(group_dir / "robustness_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_metadata, f)

    actual_data = read_group_splits("benchmark", tmp_path)

    expected_data = {
        "robustness": {
            "ids": split_ids,
            "metadata": split_metadata,
        }
    }

    assert actual_data == expected_data

