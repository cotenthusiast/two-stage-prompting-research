import pandas as pd
import pytest


@pytest.fixture
def sample_raw_row() -> dict[str, object]:
    return {
        "subject": "computer_security",
        "question": "Which protocol is primarily used to securely browse websites?",
        "choices": '["FTP", "HTTP", "HTTPS", "SMTP"]',
        "answer": 2,
    }


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "computer_security",
                "question": "Which protocol is primarily used to securely browse websites?",
                "choices": '["FTP", "HTTP", "HTTPS", "SMTP"]',
                "answer": 2,
            },
            {
                "subject": "high_school_physics",
                "question": "What is the SI unit of force?",
                "choices": '["Joule", "Newton", "Watt", "Pascal"]',
                "answer": 1,
            },
        ]
    )

@pytest.fixture
def sample_normalized_row() -> dict[str, object]:
    return {
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

@pytest.fixture
def sample_normalized_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
    )