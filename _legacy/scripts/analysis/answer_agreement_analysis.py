# scripts/analysis/answer_agreement_analysis.py
# Compares baseline and two-stage results on a per-question basis.
# Produces a pie chart (overall agreement) and a stacked bar chart (per-subject).

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import (
    ANSWER_MAPPING,
    AGREEMENT_CATEGORIES,
    AGREEMENT_COLOURS,
    QUESTIONS_PATH,
    BASELINE_RESULTS_PATH,
    TWO_STAGE_RESULTS_PATH,
    PIE_PLOT_PATH,
    BAR_PLOT_PATH,
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


def compute_agreement(baseline: pd.DataFrame, two_prompt: pd.DataFrame) -> tuple:
    """
    Computes per-question agreement categories between baseline and two-stage.

    Args:
        baseline: Baseline results DataFrame.
        two_prompt: Two-stage results DataFrame.

    Returns:
        Tuple of four boolean Series: (both_correct, both_wrong,
        two_stage_fixed, two_stage_regressed).
    """
    baseline["correct"] = baseline["predicted"] == baseline["actual"]
    two_prompt["correct"] = two_prompt["predicted"] == two_prompt["actual"]

    both_correct = baseline["correct"] & two_prompt["correct"]
    both_wrong = ~baseline["correct"] & ~two_prompt["correct"]
    two_stage_fixed = ~baseline["correct"] & two_prompt["correct"]
    two_stage_regressed = baseline["correct"] & ~two_prompt["correct"]

    return both_correct, both_wrong, two_stage_fixed, two_stage_regressed


def print_summary(counts: list, n: int) -> None:
    """
    Prints a formatted agreement summary to stdout.

    Args:
        counts: List of counts per agreement category.
        n: Total number of questions.
    """
    print("=" * 50)
    print("ANSWER AGREEMENT SUMMARY")
    print("=" * 50)
    for cat, count in zip(AGREEMENT_CATEGORIES, counts):
        print(f"  {cat:25s}: {count:3d} ({count / n * 100:.1f}%)")
    print(f"\n  Total: {n}")


def plot_pie(counts: list, n: int) -> None:
    """
    Saves a pie chart of overall agreement categories.

    Args:
        counts: List of counts per agreement category.
        n: Total number of questions (used in autopct labels).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        counts,
        labels=AGREEMENT_CATEGORIES,
        colors=AGREEMENT_COLOURS,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * n / 100))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title("Answer Agreement: Baseline vs Two-Stage", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(PIE_PLOT_PATH), exist_ok=True)
    plt.savefig(PIE_PLOT_PATH, dpi=150)
    print(f"\nSaved: {PIE_PLOT_PATH}")


def plot_stacked_bar(baseline: pd.DataFrame, both_correct, both_wrong,
                     two_stage_fixed, two_stage_regressed) -> None:
    """
    Saves a per-subject stacked horizontal bar chart of agreement categories.

    Args:
        baseline: Baseline results DataFrame (with subject column).
        both_correct, both_wrong, two_stage_fixed, two_stage_regressed:
            Boolean Series from compute_agreement().
    """
    baseline["category"] = "Both Wrong"
    baseline.loc[both_correct, "category"] = "Both Correct"
    baseline.loc[two_stage_fixed, "category"] = "Two-Stage Fixed"
    baseline.loc[two_stage_regressed, "category"] = "Two-Stage Regressed"

    subjects = sorted(baseline["subject"].unique())
    clean_names = [s.replace("_", " ").title() for s in subjects]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(subjects))
    left = np.zeros(len(subjects))

    for cat, colour in zip(AGREEMENT_CATEGORIES, AGREEMENT_COLOURS):
        vals = np.array([
            (baseline[baseline["subject"] == subject]["category"] == cat).sum()
            for subject in subjects
        ])
        ax.barh(y, vals, left=left, label=cat, color=colour)
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(left[j] + v / 2, j, str(v),
                        ha="center", va="center", fontsize=9, fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(clean_names)
    ax.set_xlabel("Number of Questions")
    ax.set_title("Per-Subject Answer Agreement: Baseline vs Two-Stage")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(BAR_PLOT_PATH, dpi=150)
    print(f"Saved: {BAR_PLOT_PATH}")


def main() -> None:
    questions, baseline, two_prompt = load_data()
    both_correct, both_wrong, two_stage_fixed, two_stage_regressed = compute_agreement(
        baseline, two_prompt
    )
    counts = [both_correct.sum(), both_wrong.sum(),
              two_stage_fixed.sum(), two_stage_regressed.sum()]
    n = len(baseline)

    print_summary(counts, n)
    plot_pie(counts, n)
    plot_stacked_bar(baseline, both_correct, both_wrong, two_stage_fixed, two_stage_regressed)


if __name__ == "__main__":
    main()
