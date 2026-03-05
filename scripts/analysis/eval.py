# scripts/analysis/eval.py
# Evaluates a single experiment's results: overall accuracy, failure rate,
# and per-subject accuracy bar chart.

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.constants import ANSWER_MAPPING, PLOTS_DIR


def compute_metrics(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """
    Computes overall accuracy, failure rate, and per-subject accuracy.

    Args:
        df: Results DataFrame with actual, predicted, and subject columns.

    Returns:
        Tuple of (overall_accuracy, failure_rate, per_subject_accuracy).
    """
    df["actual"] = df["actual"].map(ANSWER_MAPPING)
    num_correct = np.sum(df["predicted"] == df["actual"])
    overall_accuracy = num_correct / df["actual"].count()
    per_subject_acc = df.groupby("subject").apply(
        lambda g: (g["predicted"] == g["actual"]).sum() / len(g)
    )
    failure_rate = df["predicted"].isna().sum() / len(df)
    return overall_accuracy, failure_rate, per_subject_acc


def plot_results(per_subject_acc: pd.Series, overall_accuracy: float, stem: str) -> None:
    """
    Plots a bar chart of per-subject accuracy with an overall accuracy line.

    Args:
        per_subject_acc: Series with subject as index and accuracy as values.
        overall_accuracy: Overall accuracy to draw as a horizontal reference line.
        stem: Used as the filename stem when saving the plot.
    """
    chart = per_subject_acc.plot(kind="bar")
    chart.set_xlabel("Subject")
    chart.set_ylabel("Accuracy")
    chart.set_title(f"{stem} accuracy by subject")
    chart.axhline(y=overall_accuracy, color='red', linestyle='--',
                  label=f"Overall: {overall_accuracy:.2%}")
    chart.legend()
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/{stem}_accuracy.png")
    print(f"Saved to {PLOTS_DIR}/{stem}_accuracy.png")


def main() -> None:
    """
    Reads a results CSV via argparse and calls compute_metrics() and plot_results().
    """
    parser = argparse.ArgumentParser(
        prog="evaluation script",
        description="Reads a results CSV and saves a per-subject accuracy plot.",
        epilog="Example: python eval.py results/baseline/baseline_results.csv"
    )
    parser.add_argument("filepath", help="Path to the results CSV file.")
    args = parser.parse_args()

    stem = Path(args.filepath).stem.replace("_results", "")
    df = pd.read_csv(args.filepath)
    accuracy, failure_rate, per_subject_acc = compute_metrics(df)
    plot_results(per_subject_acc, accuracy, stem)
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Failure rate: {failure_rate:.2%}")


if __name__ == "__main__":
    main()
