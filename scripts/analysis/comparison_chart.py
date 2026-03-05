# scripts/analysis/comparison_chart.py
# Generates a back-to-back horizontal bar chart comparing baseline
# and two-stage accuracy per subject.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import (
    ANSWER_MAPPING,
    BASELINE_RESULTS_PATH,
    TWO_STAGE_RESULTS_PATH,
    COMPARISON_PLOT_PATH,
)


def clean_subject_name(name: str) -> str:
    """
    Converts underscore-separated subject names to title case.

    Args:
        name: Raw subject string (e.g. "high_school_physics").

    Returns:
        Cleaned string (e.g. "High School Physics").
    """
    return name.replace("_", " ").title()


def load_and_compute(path: str) -> pd.Series:
    """
    Loads a results CSV and computes per-subject accuracy.

    Args:
        path: Path to the results CSV.

    Returns:
        Series with subject as index and accuracy as values.
    """
    df = pd.read_csv(path)
    df["actual"] = df["actual"].map(ANSWER_MAPPING)
    return df.groupby("subject").apply(
        lambda g: (g["predicted"] == g["actual"]).sum() / len(g)
    )


def plot_comparison(baseline: pd.Series, two_stage: pd.Series) -> None:
    """
    Plots a back-to-back horizontal bar chart comparing baseline and
    two-stage accuracy per subject.

    Args:
        baseline: Per-subject accuracy for the baseline.
        two_stage: Per-subject accuracy for the two-stage approach.
    """
    subjects = [clean_subject_name(s) for s in baseline.index.tolist()]
    baseline_vals = baseline.values
    two_stage_vals = two_stage.values

    y = np.arange(len(subjects))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y, [-v for v in baseline_vals], label="Baseline", color="steelblue")
    ax.barh(y, two_stage_vals, label="Two-stage", color="darkorange")

    ax.set_yticks(y)
    ax.set_yticklabels(subjects)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%", "25%", "50%", "75%", "100%"])

    ax.set_xlabel("Accuracy")
    ax.set_title("Baseline vs Two-Stage Accuracy by Subject")
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(COMPARISON_PLOT_PATH), exist_ok=True)
    plt.savefig(COMPARISON_PLOT_PATH, dpi=150)
    print(f"Saved to {COMPARISON_PLOT_PATH}")


def main() -> None:
    baseline = load_and_compute(BASELINE_RESULTS_PATH)
    two_stage = load_and_compute(TWO_STAGE_RESULTS_PATH)

    # Align on shared subjects
    subjects = baseline.index.intersection(two_stage.index)
    baseline = baseline.loc[subjects]
    two_stage = two_stage.loc[subjects]

    plot_comparison(baseline, two_stage)


if __name__ == "__main__":
    main()
