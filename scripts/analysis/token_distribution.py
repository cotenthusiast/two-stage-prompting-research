# scripts/analysis/token_distribution.py
# Compares the distribution of answer tokens (A/B/C/D) across the ground truth,
# baseline predictions, and two-stage predictions.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

QUESTIONS_PATH = "data/questions.csv"
BASELINE_PATH = "results/baseline/baseline_results.csv"
TWO_STAGE_PATH = "results/two_stage/two_stage_results.csv"
PLOT_PATH = "results/plots/token_distribution.png"

questions = pd.read_csv(QUESTIONS_PATH)
baseline = pd.read_csv(BASELINE_PATH)
two_prompt = pd.read_csv(TWO_STAGE_PATH)

mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
questions["answer"] = questions["answer"].map(mapping)
baseline["actual"] = baseline["actual"].map(mapping)
two_prompt["actual"] = two_prompt["actual"].map(mapping)

questions_count = questions["answer"].value_counts()
baseline_count = baseline["predicted"].value_counts()
two_prompt_count = two_prompt["predicted"].value_counts()

print(questions_count)
print(baseline_count)
print(two_prompt_count)

tokens = ["A", "B", "C", "D"]
x = np.arange(len(tokens))
width = 0.25

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
plt.savefig(PLOT_PATH, dpi=150)
print(f"Saved to {PLOT_PATH}")
