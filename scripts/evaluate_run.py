"""Compute evaluation metrics for one experiment run."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from twoprompt.config.experiment import (
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    TWOPROMPT_CYCLIC_METHOD,
)
from twoprompt.config.paths import REPORTS_DIR, RUNS_DIR

OPTIONS = ["A", "B", "C", "D"]

METHOD_ORDER = [
    "baseline",
    "two_prompt",
    "cyclic",
    "two_prompt_cyclic",
]

MODEL_ORDER = [
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "llama-3.1-8b-instant",
]


def load_run(run_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSVs from a run directory."""
    frames = [pd.read_csv(f) for f in sorted(run_dir.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _apply_display_order(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent method/model ordering when those columns exist."""
    if "method" in df.columns:
        df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    if "model" in df.columns:
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


# Validation


def validate_run(df: pd.DataFrame, run_dir: Path) -> None:
    """Run basic integrity checks on the loaded data."""
    errors = []

    required_cols = [
        "question_id",
        "split_name",
        "method_name",
        "model_name",
        "correct_option",
        "model_status",
        "parsed_choice",
        "is_correct",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    splits = df["split_name"].dropna().unique()
    if len(splits) != 1:
        errors.append(f"Expected 1 split, found {len(splits)}: {splits}")

    dupes = df.groupby(["question_id", "method_name", "model_name"]).size()
    dupes = dupes[dupes > 1]
    if len(dupes) > 0:
        errors.append(f"Found {len(dupes)} duplicate (question, method, model) combinations")

    found_methods = set(df["method_name"].dropna().unique())
    found_models = set(df["model_name"].dropna().unique())
    print(f"[validate] Methods found: {sorted(found_methods)}")
    print(f"[validate] Models found: {sorted(found_models)}")

    counts = (
        df.groupby(["method_name", "model_name"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["method_name", "model_name"])
    )
    if not counts.empty:
        unique_counts = sorted(counts["n_rows"].unique())
        print(f"[validate] Present condition row counts: {unique_counts}")
        if len(unique_counts) > 1:
            print("[validate] WARNING: present conditions do not all have the same row count.")

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            subset = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if subset.empty:
                print(f"[validate] WARNING: missing condition {method}/{model}")

    if errors:
        for e in errors:
            print(f"[validate] ERROR: {e}")
        sys.exit(1)

    print("[validate] All checks passed.")


# Core accuracy


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy and failure accounting per method x model.

    Definitions:
    - end_to_end_accuracy = correct / total
    - conditional_accuracy = correct / scored
    - api_failures = rows where model_status == "failure"
    - parse_failures = among non-provider-failure rows, parsed_choice is missing
    - final_unscorable = rows where is_correct is missing
    """
    rows = []

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            group = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if group.empty:
                continue

            total = len(group)

            provider_failure_mask = group["model_status"].fillna("") == "failure"
            nonprovider_mask = ~provider_failure_mask

            api_failures = int(provider_failure_mask.sum())
            api_successes = int(nonprovider_mask.sum())

            parsed_total = int(group["parsed_choice"].notna().sum())
            parsed_nonprovider = int((nonprovider_mask & group["parsed_choice"].notna()).sum())
            parse_failures = int((nonprovider_mask & group["parsed_choice"].isna()).sum())

            scored = int(group["is_correct"].notna().sum())
            correct = int(group["is_correct"].eq(True).sum())
            final_unscorable = int(group["is_correct"].isna().sum())

            rows.append(
                {
                    "method": method,
                    "model": model,
                    "total": total,
                    "api_failures": api_failures,
                    "api_successes": api_successes,
                    "parsed_total": parsed_total,
                    "parsed_nonprovider": parsed_nonprovider,
                    "parse_failures": parse_failures,
                    "scored": scored,
                    "correct": correct,
                    "final_unscorable": final_unscorable,
                    "end_to_end_accuracy": correct / total if total > 0 else 0.0,
                    "conditional_accuracy": correct / scored if scored > 0 else 0.0,
                    "api_failure_rate": api_failures / total if total > 0 else 0.0,
                    "parse_success_rate_nonprovider": (
                        parsed_nonprovider / api_successes if api_successes > 0 else 0.0
                    ),
                    "final_unscorable_rate": final_unscorable / total if total > 0 else 0.0,
                }
            )

    result = pd.DataFrame(rows)
    result = _apply_display_order(result)
    return result.sort_values(["method", "model"]).reset_index(drop=True)


# Positional bias


def compute_positional_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Prediction distribution and deviation from ground truth per method x model."""
    rows = []

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            group = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if group.empty:
                continue

            scored = group[group["parsed_choice"].notna()]
            total_scored = len(scored)
            if total_scored == 0:
                continue

            gt_counts = group["correct_option"].value_counts()
            pred_counts = scored["parsed_choice"].value_counts()
            gt_total = len(group)

            deviations = []
            row = {"method": method, "model": model, "n_scored": total_scored}

            for opt in OPTIONS:
                gt_pct = gt_counts.get(opt, 0) / gt_total * 100
                pred_pct = pred_counts.get(opt, 0) / total_scored * 100
                deviation = pred_pct - gt_pct

                row[f"gt_{opt}"] = gt_counts.get(opt, 0)
                row[f"gt_{opt}_pct"] = gt_pct
                row[f"pred_{opt}"] = pred_counts.get(opt, 0)
                row[f"pred_{opt}_pct"] = pred_pct
                row[f"dev_{opt}"] = deviation
                deviations.append(abs(deviation))

            row["mean_abs_deviation"] = sum(deviations) / len(deviations)
            rows.append(row)

    result = pd.DataFrame(rows)
    result = _apply_display_order(result)
    return result.sort_values(["method", "model"]).reset_index(drop=True)


# Question-level overlap


def compute_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline vs each method at question level per model."""
    rows = []

    for model in MODEL_ORDER:
        model_group = df[df["model_name"] == model]
        baseline = model_group[model_group["method_name"] == BASELINE_METHOD]
        if baseline.empty:
            continue

        bl = baseline[["question_id", "is_correct", "parsed_choice"]].rename(
            columns={"is_correct": "bl_correct", "parsed_choice": "bl_choice"}
        )

        for method in METHOD_ORDER:
            if method == BASELINE_METHOD:
                continue

            method_df = model_group[model_group["method_name"] == method]
            if method_df.empty:
                continue

            mt = method_df[["question_id", "is_correct", "parsed_choice"]].rename(
                columns={"is_correct": "mt_correct", "parsed_choice": "mt_choice"}
            )

            merged = bl.merge(mt, on="question_id", how="inner")
            both_scored = merged[merged["bl_correct"].notna() & merged["mt_correct"].notna()]

            if both_scored.empty:
                continue

            both_correct = int(
                ((both_scored["bl_correct"] == True) & (both_scored["mt_correct"] == True)).sum()
            )
            both_wrong = int(
                ((both_scored["bl_correct"] == False) & (both_scored["mt_correct"] == False)).sum()
            )
            bl_only = int(
                ((both_scored["bl_correct"] == True) & (both_scored["mt_correct"] == False)).sum()
            )
            mt_only = int(
                ((both_scored["bl_correct"] == False) & (both_scored["mt_correct"] == True)).sum()
            )

            rows.append(
                {
                    "model": model,
                    "method": method,
                    "n_compared": int(len(both_scored)),
                    "both_correct": both_correct,
                    "both_wrong": both_wrong,
                    "baseline_only_correct": bl_only,
                    "method_only_correct": mt_only,
                    "net_effect": mt_only - bl_only,
                }
            )

    result = pd.DataFrame(rows)
    result = _apply_display_order(result)
    return result.sort_values(["model", "method"]).reset_index(drop=True)


# Choice shift analysis


def compute_choice_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how choices shift between baseline and each method."""
    rows = []

    for model in MODEL_ORDER:
        model_group = df[df["model_name"] == model]
        baseline = model_group[model_group["method_name"] == BASELINE_METHOD]
        if baseline.empty:
            continue

        bl = baseline[["question_id", "is_correct", "parsed_choice"]].rename(
            columns={"is_correct": "bl_correct", "parsed_choice": "bl_choice"}
        )

        for method in METHOD_ORDER:
            if method == BASELINE_METHOD:
                continue

            method_df = model_group[model_group["method_name"] == method]
            if method_df.empty:
                continue

            mt = method_df[["question_id", "is_correct", "parsed_choice"]].rename(
                columns={"is_correct": "mt_correct", "parsed_choice": "mt_choice"}
            )

            merged = bl.merge(mt, on="question_id", how="inner")

            broken = merged[(merged["bl_correct"] == True) & (merged["mt_correct"] == False)]
            for (from_c, to_c), count in broken.groupby(["bl_choice", "mt_choice"]).size().items():
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "direction": "broken",
                        "from_choice": from_c,
                        "to_choice": to_c,
                        "count": int(count),
                    }
                )

            fixed = merged[(merged["bl_correct"] == False) & (merged["mt_correct"] == True)]
            for (from_c, to_c), count in fixed.groupby(["bl_choice", "mt_choice"]).size().items():
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "direction": "fixed",
                        "from_choice": from_c,
                        "to_choice": to_c,
                        "count": int(count),
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = _apply_display_order(result)
        result = result.sort_values(["model", "method", "direction", "count"], ascending=[True, True, True, False])
    return result.reset_index(drop=True)


# Per-subject accuracy


def compute_subject_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per subject x method x model."""
    rows = []

    for (subject, method, model), group in df.groupby(["subject", "method_name", "model_name"]):
        total = len(group)
        scored = int(group["is_correct"].notna().sum())
        correct = int(group["is_correct"].eq(True).sum())

        rows.append(
            {
                "subject": subject,
                "method": method,
                "model": model,
                "total": total,
                "scored": scored,
                "correct": correct,
                "end_to_end_accuracy": correct / total if total > 0 else 0.0,
                "conditional_accuracy": correct / scored if scored > 0 else 0.0,
            }
        )

    result = pd.DataFrame(rows)
    result = _apply_display_order(result)
    return result.sort_values(["subject", "method", "model"]).reset_index(drop=True)


# Two-stage specific


def compute_two_stage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics specific to two-stage methods."""
    rows = []
    two_stage_methods = [TWOPROMPT_METHOD, TWOPROMPT_CYCLIC_METHOD]

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            if method not in two_stage_methods:
                continue

            group = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if group.empty:
                continue
            if "free_text_response" not in group.columns:
                continue

            total = len(group)
            has_free_text = int(group["free_text_response"].notna().sum())

            mean_ft_latency = None
            if "free_text_latency" in group.columns:
                ft_latencies = group["free_text_latency"].dropna()
                mean_ft_latency = ft_latencies.mean() if len(ft_latencies) > 0 else None

            rows.append(
                {
                    "method": method,
                    "model": model,
                    "total": total,
                    "free_text_available": has_free_text,
                    "free_text_rate": has_free_text / total if total > 0 else 0.0,
                    "mean_free_text_latency": mean_ft_latency,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = _apply_display_order(result)
        result = result.sort_values(["method", "model"]).reset_index(drop=True)
    return result


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one experiment run.")
    parser.add_argument("run_id", help="Run ID (folder name under runs/)")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    report_dir = REPORTS_DIR / args.run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Loading run {args.run_id}...")
    df = load_run(run_dir)
    print(f"[eval] {len(df)} total rows loaded")

    print("\n[eval] Validating run data...")
    validate_run(df, run_dir)

    print("\n[eval] Computing accuracy...")
    accuracy = compute_accuracy(df)
    accuracy.to_csv(report_dir / "accuracy.csv", index=False)
    print(
        accuracy[
            [
                "method",
                "model",
                "total",
                "correct",
                "scored",
                "final_unscorable",
                "end_to_end_accuracy",
                "conditional_accuracy",
                "api_failures",
                "parse_failures",
            ]
        ].to_string(index=False)
    )

    print("\n[eval] Computing positional bias...")
    bias = compute_positional_bias(df)
    bias.to_csv(report_dir / "positional_bias.csv", index=False)
    print(bias[["method", "model", "n_scored", "mean_abs_deviation"]].to_string(index=False))

    print("\n[eval] Computing question-level overlap...")
    overlap = compute_overlap(df)
    overlap.to_csv(report_dir / "overlap.csv", index=False)
    if not overlap.empty:
        print(overlap.to_string(index=False))

    print("\n[eval] Computing choice shifts...")
    shifts = compute_choice_shifts(df)
    shifts.to_csv(report_dir / "choice_shifts.csv", index=False)

    print("\n[eval] Computing per-subject accuracy...")
    subject_acc = compute_subject_accuracy(df)
    subject_acc.to_csv(report_dir / "subject_accuracy.csv", index=False)

    print("\n[eval] Computing two-stage metrics...")
    two_stage = compute_two_stage_metrics(df)
    if not two_stage.empty:
        two_stage.to_csv(report_dir / "two_stage_metrics.csv", index=False)
        print(two_stage.to_string(index=False))

    print(f"\n[complete] Reports saved to {report_dir}/")


if __name__ == "__main__":
    main()