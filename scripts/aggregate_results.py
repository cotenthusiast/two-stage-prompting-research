"""Aggregate evaluation reports into paper-ready tables."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from twoprompt.config.experiment import BASELINE_METHOD
from twoprompt.config.paths import REPORTS_DIR

_ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = [
    "baseline",
    "two_prompt",
    "cyclic",
    "pride",
]

MODEL_ORDER = [
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "llama-3.1-8b-instant",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
]

MODEL_DISPLAY = {
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "llama-3.1-8b-instant": "Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen 2.5 7B Turbo",
}

METHOD_DISPLAY = {
    "baseline": "Baseline",
    "two_prompt": "Two-Stage",
    "cyclic": "Cyclic Perm.",
    "pride": "PriDe",
}


def load_report(report_dir: Path, filename: str) -> pd.DataFrame:
    path = report_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return pd.read_csv(path)


def _get_cell(df: pd.DataFrame, method: str, model: str, column: str):
    """Safely get one value from a method x model lookup."""
    match = df[(df["method"] == method) & (df["model"] == model)]
    if match.empty:
        return None
    return match.iloc[0][column]


# Main accuracy table


def build_main_accuracy_table(accuracy: pd.DataFrame) -> str:
    """Full accuracy table with total, scored, correct, both accuracy types."""
    lines = []
    lines.append(
        f"{'Method':<22} {'Model':<20} {'Total':>6} {'Scored':>7} "
        f"{'Correct':>8} {'E2E Acc':>8} {'Cond Acc':>9} {'API Fail':>9} {'Unscorable':>11}"
    )
    lines.append("-" * 110)

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            total = _get_cell(accuracy, method, model, "total")
            if total is None:
                lines.append(
                    f"{METHOD_DISPLAY.get(method, method):<22} "
                    f"{MODEL_DISPLAY.get(model, model):<20} {'—':>6}"
                )
                continue

            row = accuracy[(accuracy["method"] == method) & (accuracy["model"] == model)].iloc[0]
            lines.append(
                f"{METHOD_DISPLAY.get(method, method):<22} "
                f"{MODEL_DISPLAY.get(model, model):<20} "
                f"{int(row['total']):>6} "
                f"{int(row['scored']):>7} "
                f"{int(row['correct']):>8} "
                f"{row['end_to_end_accuracy'] * 100:>7.1f}% "
                f"{row['conditional_accuracy'] * 100:>8.1f}% "
                f"{int(row['api_failures']):>9} "
                f"{int(row['final_unscorable']):>11}"
            )
        lines.append("")

    return "\n".join(lines)


# Compact accuracy grid


def build_accuracy_grid(accuracy: pd.DataFrame, metric: str = "end_to_end_accuracy") -> str:
    """Method x model accuracy grid."""
    label = "End-to-End Accuracy" if metric == "end_to_end_accuracy" else "Conditional Accuracy"
    lines = [label]

    header = f"{'Method':<22}" + "".join(f"{MODEL_DISPLAY.get(m, m):>18}" for m in MODEL_ORDER)
    lines.append(header)
    lines.append("-" * len(header))

    for method in METHOD_ORDER:
        row = f"{METHOD_DISPLAY.get(method, method):<22}"
        for model in MODEL_ORDER:
            val = _get_cell(accuracy, method, model, metric)
            if val is None:
                row += f"{'—':>18}"
            else:
                row += f"{val * 100:>17.1f}%"
        lines.append(row)

    return "\n".join(lines)


# Delta table


def build_delta_table(accuracy: pd.DataFrame, metric: str = "end_to_end_accuracy") -> str:
    """Delta-from-baseline table in percentage points."""
    label = "E2E" if metric == "end_to_end_accuracy" else "Conditional"
    lines = [f"Delta from Baseline ({label})"]

    header = f"{'Method':<22}" + "".join(f"{MODEL_DISPLAY.get(m, m):>18}" for m in MODEL_ORDER)
    lines.append(header)
    lines.append("-" * len(header))

    baselines = {}
    for model in MODEL_ORDER:
        val = _get_cell(accuracy, BASELINE_METHOD, model, metric)
        baselines[model] = val * 100 if val is not None else None

    for method in METHOD_ORDER:
        if method == BASELINE_METHOD:
            continue

        row = f"{METHOD_DISPLAY.get(method, method):<22}"
        for model in MODEL_ORDER:
            val = _get_cell(accuracy, method, model, metric)
            if val is None or baselines[model] is None:
                row += f"{'—':>18}"
            else:
                delta = val * 100 - baselines[model]
                row += f"{delta:>+17.1f}pp"
        lines.append(row)

    return "\n".join(lines)


# Bias table


def build_bias_table(bias: pd.DataFrame) -> str:
    """Mean absolute deviation from ground-truth answer-position distribution."""
    lines = ["Positional Bias (Mean Absolute Deviation from Ground-Truth Distribution, pp)"]

    header = f"{'Method':<22}" + "".join(f"{MODEL_DISPLAY.get(m, m):>18}" for m in MODEL_ORDER)
    lines.append(header)
    lines.append("-" * len(header))

    for method in METHOD_ORDER:
        row = f"{METHOD_DISPLAY.get(method, method):<22}"
        for model in MODEL_ORDER:
            val = _get_cell(bias, method, model, "mean_abs_deviation")
            if val is None:
                row += f"{'—':>18}"
            else:
                row += f"{val:>16.2f}pp"
        lines.append(row)

    return "\n".join(lines)


# Overlap table


def build_overlap_table(overlap: pd.DataFrame) -> str:
    """Question-level overlap summary."""
    lines = ["Question-Level Overlap vs Baseline"]
    lines.append(
        f"{'Model':<22}{'Method':<22}{'N':>6}{'Both✓':>8}{'Both✗':>8}"
        f"{'BL only':>8}{'MT only':>8}{'Net':>8}"
    )
    lines.append("-" * 82)

    for model in MODEL_ORDER:
        model_rows = overlap[overlap["model"] == model]
        for method in METHOD_ORDER:
            if method == BASELINE_METHOD:
                continue
            match = model_rows[model_rows["method"] == method]
            if match.empty:
                continue
            r = match.iloc[0]
            lines.append(
                f"{MODEL_DISPLAY.get(model, model):<22}"
                f"{METHOD_DISPLAY.get(method, method):<22}"
                f"{int(r['n_compared']):>6}"
                f"{int(r['both_correct']):>8}"
                f"{int(r['both_wrong']):>8}"
                f"{int(r['baseline_only_correct']):>8}"
                f"{int(r['method_only_correct']):>8}"
                f"{int(r['net_effect']):>+8}"
            )

    return "\n".join(lines)


# Failure and unscorable table


def build_failure_table(accuracy: pd.DataFrame) -> str:
    """Provider failures, parse failures, and final unscorable counts per condition."""
    lines = ["Failures and Unscorables"]
    lines.append(
        f"{'Method':<22}{'Model':<20}{'Total':>7}{'API Fail':>9}"
        f"{'Parse Fail':>11}{'Scored':>8}{'Unscorable':>11}"
    )
    lines.append("-" * 88)

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            total = _get_cell(accuracy, method, model, "total")
            if total is None:
                continue

            r = accuracy[(accuracy["method"] == method) & (accuracy["model"] == model)].iloc[0]
            lines.append(
                f"{METHOD_DISPLAY.get(method, method):<22}"
                f"{MODEL_DISPLAY.get(model, model):<20}"
                f"{int(r['total']):>7}"
                f"{int(r['api_failures']):>9}"
                f"{int(r['parse_failures']):>11}"
                f"{int(r['scored']):>8}"
                f"{int(r['final_unscorable']):>11}"
            )

    return "\n".join(lines)


# Summary stats


def compute_summary_stats(accuracy: pd.DataFrame, bias: pd.DataFrame) -> str:
    """Cross-condition summary statistics."""
    lines = ["SUMMARY STATISTICS", "=" * 60]

    for method in METHOD_ORDER:
        method_acc = accuracy[accuracy["method"] == method]
        if method_acc.empty:
            continue

        mean_e2e = method_acc["end_to_end_accuracy"].mean() * 100
        mean_cond = method_acc["conditional_accuracy"].mean() * 100
        mean_api_fail = method_acc["api_failure_rate"].mean() * 100
        mean_unscorable = method_acc["final_unscorable_rate"].mean() * 100

        method_bias = bias[bias["method"] == method]
        mean_mad = method_bias["mean_abs_deviation"].mean() if not method_bias.empty else 0.0

        lines.append(f"\n{METHOD_DISPLAY.get(method, method)}:")
        lines.append(f"  Mean E2E accuracy:         {mean_e2e:.1f}%")
        lines.append(f"  Mean conditional accuracy: {mean_cond:.1f}%")
        lines.append(f"  Mean API failure rate:     {mean_api_fail:.1f}%")
        lines.append(f"  Mean final unscorable:     {mean_unscorable:.1f}%")
        lines.append(f"  Mean abs deviation:        {mean_mad:.2f}pp")

    total_failures = int(accuracy["api_failures"].sum())
    total_unscorable = int(accuracy["final_unscorable"].sum())
    total_requests = int(accuracy["total"].sum())

    lines.append(
        f"\nOverall API failures:   {total_failures}/{total_requests} "
        f"({total_failures / total_requests * 100:.1f}%)"
    )
    lines.append(
        f"Overall unscorable:     {total_unscorable}/{total_requests} "
        f"({total_unscorable / total_requests * 100:.1f}%)"
    )

    return "\n".join(lines)


# LaTeX tables


def build_latex_accuracy_table(accuracy: pd.DataFrame) -> str:
    """LaTeX table with both accuracy metrics."""
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{End-to-end and conditional accuracy (\\%) by method and model.}")
    lines.append("\\label{tab:accuracy}")
    lines.append("\\begin{tabular}{l" + "rr" * len(MODEL_ORDER) + "}")
    lines.append("\\toprule")

    header1 = " "
    for m in MODEL_ORDER:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{MODEL_DISPLAY.get(m, m)}}}"
    header1 += " \\\\"
    lines.append(header1)

    header2 = "Method"
    for _ in MODEL_ORDER:
        header2 += " & E2E & Cond."
    header2 += " \\\\"
    lines.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
    lines.append(header2)
    lines.append("\\midrule")

    for method in METHOD_ORDER:
        row = METHOD_DISPLAY.get(method, method)
        for model in MODEL_ORDER:
            e2e = _get_cell(accuracy, method, model, "end_to_end_accuracy")
            cond = _get_cell(accuracy, method, model, "conditional_accuracy")
            if e2e is None:
                row += " & --- & ---"
            else:
                row += f" & {e2e * 100:.1f} & {cond * 100:.1f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_latex_bias_table(bias: pd.DataFrame) -> str:
    """LaTeX table for positional bias."""
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Mean absolute deviation from ground-truth answer-position distribution (pp).}")
    lines.append("\\label{tab:bias}")
    lines.append("\\begin{tabular}{l" + "r" * len(MODEL_ORDER) + "}")
    lines.append("\\toprule")

    header = "Method"
    for m in MODEL_ORDER:
        header += f" & {MODEL_DISPLAY.get(m, m)}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for method in METHOD_ORDER:
        row = METHOD_DISPLAY.get(method, method)
        for model in MODEL_ORDER:
            val = _get_cell(bias, method, model, "mean_abs_deviation")
            if val is None:
                row += " & ---"
            else:
                row += f" & {val:.2f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_latex_failure_table(accuracy: pd.DataFrame) -> str:
    """LaTeX table for provider failures and final unscorables."""
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Provider failures, parse failures, and final unscorable questions per condition.}")
    lines.append("\\label{tab:failures}")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("Method & Model & Total & API Fail & Parse Fail & Unscorable & Scored \\\\")
    lines.append("\\midrule")

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            total = _get_cell(accuracy, method, model, "total")
            if total is None:
                continue

            r = accuracy[(accuracy["method"] == method) & (accuracy["model"] == model)].iloc[0]
            lines.append(
                f"{METHOD_DISPLAY.get(method, method)} & "
                f"{MODEL_DISPLAY.get(model, model)} & "
                f"{int(r['total'])} & "
                f"{int(r['api_failures'])} & "
                f"{int(r['parse_failures'])} & "
                f"{int(r['final_unscorable'])} & "
                f"{int(r['scored'])} \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation reports for paper.")
    parser.add_argument("run_id", help="Run ID (folder name under reports/)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: use built-in path constants)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark sub-folder within the run report dir (e.g. mmlu, arc_challenge)",
    )
    args = parser.parse_args()

    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text())
        reports_dir = _ROOT / cfg["paths"]["reports_dir"]
    else:
        reports_dir = REPORTS_DIR

    report_dir = reports_dir / args.run_id
    if args.benchmark:
        report_dir = report_dir / args.benchmark
    output_dir = report_dir / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[agg] Loading reports from {report_dir}...")

    accuracy = load_report(report_dir, "accuracy.csv")
    bias = load_report(report_dir, "positional_bias.csv")
    overlap = load_report(report_dir, "overlap.csv")

    print("\n" + "=" * 110)
    print("MAIN ACCURACY TABLE")
    print("=" * 110)
    main_table = build_main_accuracy_table(accuracy)
    print(main_table)

    print("\n" + "=" * 70)
    e2e_grid = build_accuracy_grid(accuracy, "end_to_end_accuracy")
    print(e2e_grid)

    print("\n" + "=" * 70)
    cond_grid = build_accuracy_grid(accuracy, "conditional_accuracy")
    print(cond_grid)

    print("\n" + "=" * 70)
    e2e_delta = build_delta_table(accuracy, "end_to_end_accuracy")
    print(e2e_delta)

    print("\n" + "=" * 70)
    cond_delta = build_delta_table(accuracy, "conditional_accuracy")
    print(cond_delta)

    print("\n" + "=" * 70)
    bias_table = build_bias_table(bias)
    print(bias_table)

    print("\n" + "=" * 82)
    overlap_table = build_overlap_table(overlap)
    print(overlap_table)

    print("\n" + "=" * 88)
    failure_table = build_failure_table(accuracy)
    print(failure_table)

    print("\n" + "=" * 60)
    summary = compute_summary_stats(accuracy, bias)
    print(summary)

    with open(output_dir / "tables.txt", "w", encoding="utf-8") as f:
        f.write("MAIN ACCURACY TABLE\n" + main_table + "\n\n")
        f.write("E2E ACCURACY GRID\n" + e2e_grid + "\n\n")
        f.write("CONDITIONAL ACCURACY GRID\n" + cond_grid + "\n\n")
        f.write("E2E DELTA FROM BASELINE\n" + e2e_delta + "\n\n")
        f.write("CONDITIONAL DELTA FROM BASELINE\n" + cond_delta + "\n\n")
        f.write("POSITIONAL BIAS\n" + bias_table + "\n\n")
        f.write("QUESTION-LEVEL OVERLAP\n" + overlap_table + "\n\n")
        f.write("FAILURES AND UNSCORABLES\n" + failure_table + "\n\n")
        f.write(summary + "\n")

    latex_acc = build_latex_accuracy_table(accuracy)
    latex_bias = build_latex_bias_table(bias)
    latex_fail = build_latex_failure_table(accuracy)

    with open(output_dir / "tables.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX tables\n\n")
        f.write(latex_acc + "\n\n")
        f.write(latex_bias + "\n\n")
        f.write(latex_fail + "\n")

    print(f"\n[complete] Paper tables saved to {output_dir}/")


if __name__ == "__main__":
    main()