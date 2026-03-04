# scripts/analysis/comparison_chart.py
# Generates a back-to-back horizontal bar chart comparing baseline
# and two-stage accuracy per subject.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASELINE_PATH = "results/baseline/baseline_results.csv"
TWO_STAGE_PATH = "results/two_stage/two_stage_results.csv"
PLOT_PATH = "results/plots/comparison.png"


def clean_subject_name(name: str) -> str:
    """
    Converts underscore-separated subject names to title case.
    Args:
        name: raw subject string (e.g. "high_school_physics")
    Returns:
        Cleaned string (e.g. "High School Physics")
    """
    return name.replace("_", " ").title()


def load_and_compute(path: str) -> pd.Series:
    """
    Loads a results CSV and computes per-subject accuracy.
    Args:
        path: path to the results CSV
    Returns:
        pd.Series with subject as index and accuracy as values
    """
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    df = pd.read_csv(path)
    df["actual"] = df["actual"].map(mapping)
    return df.groupby("subject").apply(
        lambda g: (g["predicted"] == g["actual"]).sum() / len(g)
    )


def plot_comparison(baseline: pd.Series, two_stage: pd.Series) -> None:
    """
    Plots a back-to-back horizontal bar chart comparing baseline and two-stage accuracy per subject.
    Args:
        baseline: per-subject accuracy for the baseline
        two_stage: per-subject accuracy for the two-stage approach
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
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved to {PLOT_PATH}")


def main():
    baseline = load_and_compute(BASELINE_PATH)
    two_stage = load_and_compute(TWO_STAGE_PATH)

    # Align on shared subjects
    subjects = baseline.index.intersection(two_stage.index)
    baseline = baseline.loc[subjects]
    two_stage = two_stage.loc[subjects]

    plot_comparison(baseline, two_stage)


if __name__ == "__main__":
    main()
