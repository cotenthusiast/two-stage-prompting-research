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

from twoprompt.config.experiment import (
    MMLU_QUESTIONS_PER_SUBJECT,
    REVIEW_SUBJECTS,
    ROBUSTNESS_SUBJECTS,
)


def _make_normalized_rows(subjects: list[str], n_per_subject: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    options = ["A", "B", "C", "D"]

    for subject in subjects:
        for i in range(n_per_subject):
            correct_option = options[i % 4]
            rows.append(
                {
                    "question_id": f"{subject}__{i:02d}",
                    "subject": subject,
                    "question_text": f"{subject} question {i}",
                    "choice_a": f"{subject} choice A {i}",
                    "choice_b": f"{subject} choice B {i}",
                    "choice_c": f"{subject} choice C {i}",
                    "choice_d": f"{subject} choice D {i}",
                    "correct_option": correct_option,
                    "correct_answer_text": f"{subject} answer {correct_option} {i}",
                }
            )

    return rows


def _make_variable_count_rows(subject_counts: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    options = ["A", "B", "C", "D"]

    for subject, count in subject_counts.items():
        for i in range(count):
            correct_option = options[i % 4]
            rows.append(
                {
                    "question_id": f"{subject}__{i:02d}",
                    "subject": subject,
                    "question_text": f"{subject} question {i}",
                    "choice_a": f"{subject} choice A {i}",
                    "choice_b": f"{subject} choice B {i}",
                    "choice_c": f"{subject} choice C {i}",
                    "choice_d": f"{subject} choice D {i}",
                    "correct_option": correct_option,
                    "correct_answer_text": f"{subject} answer {correct_option} {i}",
                }
            )

    return rows


@pytest.fixture
def split_test_subjects() -> list[str]:
    return [
        "computer_security",
        "high_school_physics",
        "anatomy",
    ]


@pytest.fixture
def split_test_dataframe(split_test_subjects: list[str]) -> pd.DataFrame:
    return pd.DataFrame(_make_normalized_rows(split_test_subjects, 5))


@pytest.fixture
def split_test_exclude_ids() -> set[str]:
    return {
        "computer_security__00",
        "high_school_physics__00",
    }


@pytest.fixture
def split_test_valid_ids() -> list[str]:
    return [
        "computer_security__01",
        "computer_security__02",
        "high_school_physics__01",
        "anatomy__01",
    ]


@pytest.fixture
def split_test_duplicate_ids() -> list[str]:
    return [
        "computer_security__01",
        "computer_security__01",
        "high_school_physics__01",
    ]


@pytest.fixture
def split_test_unknown_ids() -> list[str]:
    return [
        "computer_security__01",
        "missing_question_id",
    ]


@pytest.fixture
def disjoint_split_map() -> dict[str, list[str]]:
    return {
        "robustness": [
            "computer_security__01",
            "computer_security__02",
            "high_school_physics__01",
        ],
        "review": [
            "anatomy__01",
            "anatomy__02",
            "high_school_physics__02",
        ],
    }


@pytest.fixture
def overlapping_split_map() -> dict[str, list[str]]:
    return {
        "robustness": [
            "computer_security__01",
            "computer_security__02",
            "high_school_physics__01",
        ],
        "review": [
            "anatomy__01",
            "computer_security__02",
            "high_school_physics__02",
        ],
    }


@pytest.fixture
def insufficient_split_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        _make_variable_count_rows(
            {
                "computer_security": 3,
                "high_school_physics": 1,
            }
        )
    )


@pytest.fixture
def full_default_split_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        _make_normalized_rows(ROBUSTNESS_SUBJECTS, MMLU_QUESTIONS_PER_SUBJECT)
    )


@pytest.fixture
def full_default_review_subjects() -> list[str]:
    return REVIEW_SUBJECTS


@pytest.fixture
def full_default_robustness_subjects() -> list[str]:
    return ROBUSTNESS_SUBJECTS