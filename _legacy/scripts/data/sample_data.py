# scripts/data/sample_data.py
# Downloads and samples 10 questions per subject from the MMLU dataset.

from __future__ import annotations
import pandas as pd
from datasets import load_dataset, Dataset

def load_subject(subject: str, split: str = "test") -> Dataset:
    """
    Loads the MMLU dataset for the given subject.

    Args:
        subject: Subject to load questions from.
        split: Dataset split to use (default: test).

    Returns:
        Dataset containing all questions for that subject.
    """
    return load_dataset("cais/mmlu", subject, split = split)

def main(subjects: list, path: str) -> None:
    """
    Samples 10 questions per subject and saves to CSV.

    Args:
        subjects: List of MMLU subject names.
        path: File path to save the output CSV.
    """
    rows = []
    for subject in subjects:
        temp_rows = load_subject(subject)
        for item in temp_rows.select(range(10)):
            rows.append(item)

    df = pd.DataFrame(rows)

    #saving pandas dataframe containing the questions to the specified path
    df.to_csv(path)

if __name__ == "__main__":
    subjects = ["anatomy", "high_school_physics","college_mathematics","computer_security","econometrics",
                "high_school_psychology","philosophy", "high_school_world_history", "medical_genetics", "jurisprudence"]
    path = "data/questions.csv"
    main(subjects, path)
