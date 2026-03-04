import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

questions = pd.read_csv("data/questions.csv")
baseline = pd.read_csv("results/baseline_results.csv")
two_prompt = pd.read_csv("results/two_stage_results.csv")

mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
questions["answer"] = questions["answer"].map(mapping)
baseline["actual"] = baseline["actual"].map(mapping)
two_prompt["actual"] = two_prompt["actual"].map(mapping)

baseline["correct"] = baseline["predicted"] == baseline["actual"]
two_prompt["correct"] = two_prompt["predicted"] == two_prompt["actual"]

both_correct = baseline["correct"] & two_prompt["correct"]
both_wrong = ~baseline["correct"] & ~two_prompt["correct"]
two_stage_fixed = ~baseline["correct"] & two_prompt["correct"]
two_stage_regressed = baseline["correct"] & ~two_prompt["correct"]

# Print summary
categories = ["Both Correct", "Both Wrong", "Two-Stage Fixed", "Two-Stage Regressed"]
counts = [both_correct.sum(), both_wrong.sum(), two_stage_fixed.sum(), two_stage_regressed.sum()]
n = len(baseline)

print("=" * 50)
print("ANSWER AGREEMENT SUMMARY")
print("=" * 50)
for cat, count in zip(categories, counts):
    print(f"  {cat:25s}: {count:3d} ({count / n * 100:.1f}%)")
print(f"\n  Total: {n}")

# Pie chart
colours = ["#55A868", "#8C8C8C", "#4C72B0", "#C44E52"]

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(
    counts,
    labels=categories,
    colors=colours,
    autopct=lambda p: f"{p:.1f}%\n({int(round(p * n / 100))})",
    startangle=90,
    textprops={"fontsize": 11},
)
ax.set_title("Answer Agreement: Baseline vs Two-Stage", fontsize=13)
plt.tight_layout()
plt.savefig("results/agreement_overall.png", dpi=150)
print("\nSaved: results/agreement_overall.png")

# Per-subject stacked bar chart
baseline["category"] = "Both Wrong"
baseline.loc[both_correct, "category"] = "Both Correct"
baseline.loc[two_stage_fixed, "category"] = "Two-Stage Fixed"
baseline.loc[two_stage_regressed, "category"] = "Two-Stage Regressed"

subjects = sorted(baseline["subject"].unique())
clean_names = [s.replace("_", " ").title() for s in subjects]

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(subjects))
left = np.zeros(len(subjects))

for cat, colour in zip(categories, colours):
    vals = []
    for subject in subjects:
        sub = baseline[baseline["subject"] == subject]
        vals.append((sub["category"] == cat).sum())
    vals = np.array(vals)

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
plt.savefig("results/agreement_per_subject.png", dpi=150)
print("Saved: results/agreement_per_subject.png")