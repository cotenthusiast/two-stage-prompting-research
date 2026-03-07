import pandas as pd
from twoprompt.benchmarks.mmlu import build_normalized_dataframe
from twoprompt.benchmarks.mmlu import normalize_row

def test_normalize_row(sample_raw_row):
    assert normalize_row(sample_raw_row) == {
    "question_id": "4865890d7f0efae8",
    "subject": "computer_security",
    "question_text": "Which protocol is primarily used to securely browse websites?",
    "choice_a": "FTP",
    "choice_b": "HTTP",
    "choice_c": "HTTPS",
    "choice_d": "SMTP",
    "correct_option": "C",
    "correct_answer_text": "HTTPS",
}

def test_build_normalize_dataframe(sample_raw_dataframe):
    expected_raw = [
    {
        "question_id": "4865890d7f0efae8",
        "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": "C",
        "correct_answer_text": "HTTPS",
    },
    {
        "question_id": "5e9876049bf053f9",
        "subject": "high_school_physics",
        "question_text": "What is the SI unit of force?",
        "choice_a": "Joule",
        "choice_b": "Newton",
        "choice_c": "Watt",
        "choice_d": "Pascal",
        "correct_option": "B",
        "correct_answer_text": "Newton",
        },
    ]
    df = pd.DataFrame(expected_raw)
    pd.testing.assert_frame_equal(build_normalized_dataframe(sample_raw_dataframe), df)