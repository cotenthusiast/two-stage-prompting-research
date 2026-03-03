# scripts.eval.py

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    prog = "evaluation script",
    description = "script that reads in a csv of results and saves a plot analysing the results",
    epilog= "thank you for using the script"
)

parser.add_argument("filepath")
args = parser.parse_args()

stem = Path(args.filepath).stem.replace("_results", "")

def compute_metrics(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """
    Computes overall accuracy, failure rate and accuracy per subject from a results df
    Args:
        df: the dataframe to be analysed
    Returns:
        A tuple containign the overall accuracy, failure rate and a pd series with the accuracy per subject
    """
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    df["actual"] = df["actual"].map(mapping)
    num_correct = np.sum(df["predicted"] == df["actual"])
    overall_accuracy = num_correct/df["actual"].count()
    per_subject_acc = df.groupby("subject").apply(
        lambda g: (g["predicted"] == g["actual"]).sum() / len(g)
        )
    failure_rate = df["predicted"].isna().sum() / len(df)
    return overall_accuracy, failure_rate, per_subject_acc

def plot_results(per_subject_acc: pd.Series, overall_accuracy: float, stem: str) -> None:
    """
    plots a bar chart of the accuracy per subject and plots a horizontal line at the overall accuracy
    Args:
        per_subject_acc: pd.Series with the accuracy per subject and the subject as index
        overall_accuracy: the overall accuracy of the results
        stem: used when saving the file
    """
    chart = per_subject_acc.plot(kind="bar")
    chart.set_xlabel("Subject")
    chart.set_ylabel("Accuracy")
    chart.set_title(f"{stem} accuracy by subject")
    chart.axhline(y=overall_accuracy, color='red', linestyle='--', label=f"Overall: {overall_accuracy:.2%}")
    chart.legend()
    plt.tight_layout()
    plt.savefig(f"results/{stem}_accuracy.png")

def main():
    """
    Imports csv as pd.DataFrame using argparse and calls compute_metrics() and plot_results()
    """
    df = pd.read_csv(args.filepath)
    accuracy, failure_rate, per_subject_acc = compute_metrics(df)
    plot_results(per_subject_acc, accuracy, stem)
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Failure rate: {failure_rate:.2%}")

if __name__ == "__main__":
    main()