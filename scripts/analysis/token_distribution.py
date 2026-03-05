# scripts/analysis/token_distribution.py
# Compares the distribution of answer tokens (A/B/C/D) across the ground truth,
# baseline predictions, and two-stage predictions.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import (
    ANSWER_MAPPING,
    QUESTIONS_PATH,
    BASELINE_RESULTS_PATH,
    TWO_STAGE_RESULTS_PATH,
    TOKEN_DIST_PLOT_PATH,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses the questions, baseline, and two-stage results CSVs.

    Returns:
        Tuple of (questions, baseline, two_prompt) DataFrames with answer
        columns mapped to letters.
    """
    questions = pd.read_csv(QUESTIONS_PATH)
    baseline = pd.read_csv(BASELINE_RESULTS_PATH)
    two_prompt = pd.read_csv(TWO_STAGE_RESULTS_PATH)

    questions["answer"] = questions["answer"].map(ANSWER_MAPPING)
    baseline["actual"] = baseline["actual"].map(ANSWER_MAPPING)
    two_prompt["actual"] = two_prompt["actual"].map(ANSWER_MAPPING)

    return questions, baseline, two_prompt


def plot_token_distribution(questions: pd.DataFrame, baseline: pd.DataFrame,
                             two_prompt: pd.DataFrame) -> None:
    """
    Plots a grouped bar chart comparing token distributions across actual answers,
    baseline predictions, and two-stage predictions.

    Args:
        questions: Questions DataFrame with mapped answer column.
        baseline: Baseline results DataFrame with mapped actual column.
        two_prompt: Two-stage results DataFrame with mapped actual column.
    """
    tokens = ["A", "B", "C", "D"]
    x = np.arange(len(tokens))
    width = 0.25

    questions_count = questions["answer"].value_counts()
    baseline_count = baseline["predicted"].value_counts()
    two_prompt_count = two_prompt["predicted"].value_counts()

    print(questions_count)
    print(baseline_count)
    print(two_prompt_count)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, questions_count.reindex(tokens).values, width, label="Actual")
    ax.bar(x, baseline_count.reindex(tokens).values, width, label="Baseline")
    ax.bar(x + width, two_prompt_count.reindex(tokens).values, width, label="Two-Stage")

    ax.axhline(25, color="grey", linestyle="--", linewidth=0.8, label="Uniform (25%)")
    ax.set_xlabel("Answer Token")
    ax.set_ylabel("Count")
    ax.set_title("Token Distribution: Actual vs Baseline vs Two-Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(TOKEN_DIST_PLOT_PATH), exist_ok=True)
    plt.savefig(TOKEN_DIST_PLOT_PATH, dpi=150)
    print(f"Saved to {TOKEN_DIST_PLOT_PATH}")


def main() -> None:
    questions, baseline, two_prompt = load_data()
    plot_token_distribution(questions, baseline, two_prompt)


if __name__ == "__main__":
    main()
