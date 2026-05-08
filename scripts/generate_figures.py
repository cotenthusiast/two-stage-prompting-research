"""Generate paper-ready figures from evaluation CSVs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

from twoprompt.config.paths import REPORTS_DIR

_ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = ["baseline", "two_prompt", "cyclic", "pride"]
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
METHOD_COLORS = {
    "baseline": "#4C72B0",
    "two_prompt": "#DD8452",
    "cyclic": "#55A868",
    "pride": "#C44E52",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, figures_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = figures_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved {path}")
    plt.close(fig)


def _try_load(report_dir: Path, filename: str) -> pd.DataFrame | None:
    path = report_dir / filename
    if not path.exists():
        print(f"[warn] Missing {path}, skipping.")
        return None
    return pd.read_csv(path)


def _present_models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].unique()) if "model" in df.columns else set()
    return [m for m in MODEL_ORDER if m in present]


def _present_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].unique()) if "method" in df.columns else set()
    return [m for m in METHOD_ORDER if m in present]


def fig_accuracy(accuracy: pd.DataFrame, figures_dir: Path, metric: str) -> None:
    models = _present_models(accuracy)
    methods = _present_methods(accuracy)
    if not models or not methods:
        return

    ci_lo_col = f"{metric}_ci_lower"
    ci_hi_col = f"{metric}_ci_upper"

    n_models = len(models)
    n_methods = len(methods)
    width = 0.8 / n_methods
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.6), 4))

    for i, method in enumerate(methods):
        vals, errs_lo, errs_hi = [], [], []
        for model in models:
            row = accuracy[(accuracy["method"] == method) & (accuracy["model"] == model)]
            if row.empty:
                vals.append(np.nan)
                errs_lo.append(0)
                errs_hi.append(0)
            else:
                v = float(row.iloc[0][metric]) * 100
                lo = float(row.iloc[0][ci_lo_col]) * 100 if ci_lo_col in row.columns else v
                hi = float(row.iloc[0][ci_hi_col]) * 100 if ci_hi_col in row.columns else v
                vals.append(v)
                errs_lo.append(v - lo)
                errs_hi.append(hi - v)

        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            color=METHOD_COLORS.get(method, "#888888"),
            label=METHOD_DISPLAY.get(method, method),
            yerr=[errs_lo, errs_hi],
            capsize=2,
            error_kw={"elinewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    label = "End-to-End Accuracy" if metric == "end_to_end_accuracy" else "Conditional Accuracy"
    ax.set_title(label)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    _despine(ax)

    stem = "accuracy_e2e" if metric == "end_to_end_accuracy" else "accuracy_cond"
    _save(fig, figures_dir, stem)


def fig_positional_bias(bias: pd.DataFrame, figures_dir: Path) -> None:
    models = _present_models(bias)
    methods = _present_methods(bias)
    if not models or not methods:
        return

    n_models = len(models)
    n_methods = len(methods)
    width = 0.8 / n_methods
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.6), 4))

    for i, method in enumerate(methods):
        vals, errs_lo, errs_hi = [], [], []
        for model in models:
            row = bias[(bias["method"] == method) & (bias["model"] == model)]
            if row.empty:
                vals.append(np.nan)
                errs_lo.append(0)
                errs_hi.append(0)
            else:
                v = float(row.iloc[0]["mean_abs_deviation"])
                lo = float(row.iloc[0]["mean_abs_deviation_ci_lower"]) if "mean_abs_deviation_ci_lower" in row.columns else v
                hi = float(row.iloc[0]["mean_abs_deviation_ci_upper"]) if "mean_abs_deviation_ci_upper" in row.columns else v
                vals.append(v)
                errs_lo.append(v - lo)
                errs_hi.append(hi - v)

        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            color=METHOD_COLORS.get(method, "#888888"),
            label=METHOD_DISPLAY.get(method, method),
            yerr=[errs_lo, errs_hi],
            capsize=2,
            error_kw={"elinewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("Mean Absolute Deviation (pp)")
    ax.set_title("Positional Bias (MAD from Ground-Truth Distribution)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    _despine(ax)
    _save(fig, figures_dir, "positional_bias_mad")


def fig_answer_distribution(bias: pd.DataFrame, figures_dir: Path) -> None:
    options = ["A", "B", "C", "D"]
    models = _present_models(bias)
    methods = _present_methods(bias)
    if not models or not methods:
        return

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 3.5, 3.8), sharey=True)
    if n_models == 1:
        axes = [axes]

    all_handles = []
    all_labels = []

    gt_patch = mpatches.Patch(color="#AAAAAA", label="Ground Truth")
    all_handles.append(gt_patch)
    all_labels.append("Ground Truth")
    for method in methods:
        p = mpatches.Patch(color=METHOD_COLORS.get(method, "#888888"), label=METHOD_DISPLAY.get(method, method))
        all_handles.append(p)
        all_labels.append(METHOD_DISPLAY.get(method, method))

    bar_entities = ["gt"] + methods
    n_bars = len(bar_entities)
    width = 0.8 / n_bars
    x = np.arange(len(options))

    for ax, model in zip(axes, models):
        model_bias = bias[bias["model"] == model]

        gt_row = model_bias.iloc[0] if not model_bias.empty else None

        for j, entity in enumerate(bar_entities):
            vals = []
            for opt in options:
                if entity == "gt":
                    v = float(gt_row[f"gt_{opt}_pct"]) if gt_row is not None else 0.0
                else:
                    row = model_bias[model_bias["method"] == entity]
                    v = float(row.iloc[0][f"pred_{opt}_pct"]) if not row.empty else 0.0
                vals.append(v)

            color = "#AAAAAA" if entity == "gt" else METHOD_COLORS.get(entity, "#888888")
            offset = (j - n_bars / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(options)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        _despine(ax)

    axes[0].set_ylabel("Prediction Share (%)")
    fig.suptitle("Answer Distribution by Method", y=1.01)
    fig.legend(all_handles, all_labels, loc="lower center", ncol=len(all_handles),
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    _save(fig, figures_dir, "answer_distribution")


def fig_net_effect(overlap: pd.DataFrame, figures_dir: Path) -> None:
    if "model" not in overlap.columns:
        return
    models = [m for m in MODEL_ORDER if m in overlap["model"].unique()]
    methods = [m for m in METHOD_ORDER if m != "baseline" and m in overlap["method"].unique()]
    if not models or not methods:
        return

    n_models = len(models)
    n_methods = len(methods)
    width = 0.8 / n_methods
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.6), 4))

    for i, method in enumerate(methods):
        vals = []
        for model in models:
            row = overlap[(overlap["method"] == method) & (overlap["model"] == model)]
            vals.append(float(row.iloc[0]["net_effect"]) if not row.empty else 0.0)

        offset = (i - n_methods / 2 + 0.5) * width
        colors = [
            METHOD_COLORS.get(method, "#888888") if v >= 0
            else METHOD_COLORS.get(method, "#888888")
            for v in vals
        ]
        alphas = [1.0 if v >= 0 else 0.6 for v in vals]
        for k, (xk, v) in enumerate(zip(x + offset, vals)):
            ax.bar(
                xk, v,
                width=width * 0.9,
                color=METHOD_COLORS.get(method, "#888888"),
                alpha=alphas[k],
                label=METHOD_DISPLAY.get(method, method) if k == 0 else None,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("Net Effect (fixed − broken questions)")
    ax.set_title("Net Effect vs. Baseline")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    _despine(ax)
    _save(fig, figures_dir, "net_effect")


def fig_subject_heatmap(subject_acc: pd.DataFrame, figures_dir: Path) -> None:
    methods = [m for m in METHOD_ORDER if m != "baseline" and m in subject_acc["method"].unique()]
    if not methods or "baseline" not in subject_acc["method"].unique():
        return

    baseline_df = subject_acc[subject_acc["method"] == "baseline"].groupby("subject")["end_to_end_accuracy"].mean()
    subjects = sorted(baseline_df.index.tolist())

    rows = []
    for subject in subjects:
        row = {}
        for method in methods:
            method_df = subject_acc[subject_acc["method"] == method]
            method_mean = method_df[method_df["subject"] == subject]["end_to_end_accuracy"].mean()
            bl = baseline_df.get(subject, np.nan)
            row[method] = (method_mean - bl) if not np.isnan(method_mean) and not np.isnan(bl) else np.nan
        rows.append(row)

    delta_df = pd.DataFrame(rows, index=subjects)
    delta_df["mean_delta"] = delta_df.mean(axis=1)
    delta_df = delta_df.sort_values("mean_delta", ascending=False)

    if len(subjects) > 40:
        top20 = delta_df.head(20)
        bot20 = delta_df.tail(20)
        gap = pd.DataFrame([[np.nan] * len(delta_df.columns)], columns=delta_df.columns, index=[""])
        delta_df = pd.concat([top20, gap, bot20])

    display_df = delta_df[methods]
    vmax = float(np.nanmax(np.abs(display_df.values)))
    vmin = -vmax

    n_rows, n_cols = display_df.shape
    fig, ax = plt.subplots(figsize=(max(4, n_cols * 1.5), max(4, n_rows * 0.35)))

    mat = display_df.values.astype(float)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods], rotation=15, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(display_df.index, fontsize=7)
    ax.set_title("Per-Subject Accuracy Delta vs. Baseline (avg across models)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Δ Accuracy (pp)")

    fig.tight_layout()
    _save(fig, figures_dir, "subject_accuracy_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from evaluation CSVs.")
    parser.add_argument("run_id", help="Run ID (folder name under reports/)")
    parser.add_argument("--benchmark", default=None, help="Benchmark sub-folder (e.g. mmlu, arc_challenge)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text())
        reports_dir = _ROOT / cfg["paths"]["reports_dir"]
    else:
        reports_dir = REPORTS_DIR

    report_dir = reports_dir / args.run_id
    if args.benchmark:
        report_dir = report_dir / args.benchmark

    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    accuracy = _try_load(report_dir, "accuracy.csv")
    bias = _try_load(report_dir, "positional_bias.csv")
    overlap = _try_load(report_dir, "overlap.csv")
    subject_acc = _try_load(report_dir, "subject_accuracy.csv")

    if accuracy is not None:
        fig_accuracy(accuracy, figures_dir, "end_to_end_accuracy")
        fig_accuracy(accuracy, figures_dir, "conditional_accuracy")

    if bias is not None:
        fig_positional_bias(bias, figures_dir)
        fig_answer_distribution(bias, figures_dir)

    if overlap is not None:
        fig_net_effect(overlap, figures_dir)

    if subject_acc is not None:
        fig_subject_heatmap(subject_acc, figures_dir)

    print(f"\n[complete] Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
