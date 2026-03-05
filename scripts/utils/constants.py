# scripts/utils/constants.py
# Shared constants used across experiment and analysis scripts.

# Answer mapping from integer labels to MCQ letters
ANSWER_MAPPING = {0: "A", 1: "B", 2: "C", 3: "D"}

# Agreement categories and their colours (used in pie + stacked bar charts)
AGREEMENT_CATEGORIES = ["Both Correct", "Both Wrong", "Two-Stage Fixed", "Two-Stage Regressed"]
AGREEMENT_COLOURS = ["#55A868", "#8C8C8C", "#4C72B0", "#C44E52"]

# Results paths
BASELINE_RESULTS_PATH = "results/baseline/baseline_results.csv"
TWO_STAGE_RESULTS_PATH = "results/two_stage/two_stage_results.csv"
QUESTIONS_PATH = "data/questions.csv"

# Plot output paths
PLOTS_DIR = "results/plots"
COMPARISON_PLOT_PATH = "results/plots/comparison.png"
PIE_PLOT_PATH = "results/plots/agreement_overall.png"
BAR_PLOT_PATH = "results/plots/agreement_per_subject.png"
TOKEN_DIST_PLOT_PATH = "results/plots/token_distribution.png"
