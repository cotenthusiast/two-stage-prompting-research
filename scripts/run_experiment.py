"""Overnight experiment runner.

Usage:
    python -m scripts.run_experiment [options]

Options:
    --config PATH     Path to YAML config (default: config/default.yaml)
    --run-id ID       Explicit run ID; timestamp auto-generated if omitted
    --dry-run         Print preflight estimate and exit without running
    --no-cache        Disable response caching for this run
    --yes             Skip the preflight confirmation prompt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from twoprompt.clients.gemini_client import GeminiClient
from twoprompt.clients.groq_client import GroqClient
from twoprompt.clients.openai_client import OpenAIClient
from twoprompt.infra.cache import CachingClientWrapper, ResponseCache
from twoprompt.infra.checkpoint import CheckpointManager
from twoprompt.io.readers import read_normalized_questions, read_split_ids
from twoprompt.io.writers import write_run_results
from twoprompt.runners.direct_mcq import DirectMCQRunner
from twoprompt.runners.permutation import PermutationRunner
from twoprompt.runners.two_stage import TwoStageRunner
from twoprompt.runners.two_stage_permutation import TwoStagePermutationRunner

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METHOD_TO_RUNNER = {
    "baseline": DirectMCQRunner,
    "two_prompt": TwoStageRunner,
    "cyclic": PermutationRunner,
    "two_prompt_cyclic": TwoStagePermutationRunner,
}

# API calls per question for each method (used for preflight estimates).
_CALLS_PER_QUESTION = {
    "baseline": 1,
    "two_prompt": 2,       # stage 1 free-text + stage 2 matching
    "cyclic": 4,           # 4 cyclic permutations
    "two_prompt_cyclic": 5,  # 1 free-text + 4 permuted matching calls
}

# Approximate avg tokens per API call (very rough; for cost estimates only).
_AVG_INPUT_TOKENS = 200
_AVG_OUTPUT_TOKENS = 15

# Maps split name to the artifact_group subdirectory used by read_split_ids.
_SPLIT_TO_ARTIFACT_GROUP = {
    "robustness": "benchmark",
    "review": "faithfulness",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_paths(paths_cfg: dict) -> dict[str, Path]:
    return {key: ROOT / val for key, val in paths_cfg.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(benchmark: str, split: str, paths: dict[str, Path]) -> list[dict]:
    """Load normalized questions for a given benchmark and split."""
    if benchmark == "mmlu":
        df = read_normalized_questions(
            "mmlu_normalized.csv", paths["data_processed_dir"]
        )
        artifact_group = _SPLIT_TO_ARTIFACT_GROUP.get(split, split)
        split_ids = read_split_ids(split, paths["data_splits_dir"], artifact_group)
        df = df[df["question_id"].isin(split_ids)].drop_duplicates(subset="question_id")
        return df.to_dict(orient="records")
    raise ValueError(f"Unknown benchmark: {benchmark!r}")


def count_split_questions(benchmark: str, split: str, paths: dict[str, Path]) -> int:
    """Return question count without loading the full DataFrame."""
    if benchmark == "mmlu":
        artifact_group = _SPLIT_TO_ARTIFACT_GROUP.get(split, split)
        return len(read_split_ids(split, paths["data_splits_dir"], artifact_group))
    raise ValueError(f"Unknown benchmark: {benchmark!r}")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client(model_name: str, model_cfg: dict):
    """Construct a typed provider client from a model config dict."""
    provider = model_cfg["provider"]
    kwargs = dict(
        model_name=model_name,
        concurrency_limit=model_cfg.get("concurrency", 10),
        min_delay_seconds=model_cfg.get("min_delay_seconds", 0.0),
        max_retries=model_cfg.get("max_retries", 3),
        timeout=model_cfg.get("timeout", 30),
    )
    if provider == "openai":
        return OpenAIClient(**kwargs)
    if provider == "gemini":
        return GeminiClient(**kwargs)
    if provider == "groq":
        return GroqClient(**kwargs)
    raise ValueError(f"Unknown provider: {provider!r} for model {model_name!r}")


# ---------------------------------------------------------------------------
# Preflight estimation
# ---------------------------------------------------------------------------

def preflight_estimate(config: dict, paths: dict[str, Path]) -> None:
    """Print a call count, cost, and wall-clock estimate without running."""
    run_cfg = config["run"]
    models_cfg = config["models"]
    jobs = run_cfg["jobs"]

    print("\n── Preflight estimate ─────────────────────────────────────────")

    total_calls = 0
    total_cost = 0.0
    bottleneck_hours = 0.0

    for job in jobs:
        model_name = job["model"]
        methods = job["methods"]
        benchmark = job["benchmark"]
        split = job["split"]

        try:
            n_questions = count_split_questions(benchmark, split, paths)
        except Exception:
            n_questions = 1000  # fallback if data not present
            print(f"  (could not read split file for {benchmark}/{split} — using 1000)")

        model_cfg = models_cfg.get(model_name, {})
        pricing = model_cfg.get("pricing", {})
        input_price = pricing.get("input_per_1m", 0.0)
        output_price = pricing.get("output_per_1m", 0.0)

        job_calls = sum(
            _CALLS_PER_QUESTION.get(m, 1) * n_questions for m in methods
        )
        job_cost = job_calls * (
            (_AVG_INPUT_TOKENS * input_price + _AVG_OUTPUT_TOKENS * output_price) / 1_000_000
        )

        min_delay = model_cfg.get("min_delay_seconds", 0.0)
        concurrency = model_cfg.get("concurrency", 10)
        if min_delay > 0:
            rate = min(concurrency, 1.0 / min_delay)
        else:
            rate = concurrency / 2.0  # assume ~2s avg latency
        job_hours = (job_calls / rate) / 3600

        total_calls += job_calls
        total_cost += job_cost
        bottleneck_hours = max(bottleneck_hours, job_hours)

        cost_str = f"~${job_cost:.2f}" if job_cost > 0 else "free"
        print(
            f"  {model_name:<30}  {job_calls:>6} calls  {cost_str:>8}  ~{job_hours:.1f}h"
        )

    print(f"  {'─'*60}")
    total_cost_str = f"~${total_cost:.2f}" if total_cost > 0 else "$0.00"
    print(f"  {'Total':<30}  {total_calls:>6} calls  {total_cost_str:>8}")
    print(f"  Wall-clock (models run in parallel): ~{bottleneck_hours:.1f}h")
    print(f"  Cache: {'enabled' if run_cfg.get('cache_enabled', True) else 'disabled'}")
    print(f"  Checkpoint every: {run_cfg.get('checkpoint_every_n', 50)} questions")
    print("───────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

async def run_single_method(
    client,
    model_name: str,
    method: str,
    benchmark: str,
    split: str,
    questions: list[dict],
    run_id: str,
    run_cfg: dict,
    paths: dict[str, Path],
) -> dict:
    """Run one (model, method, benchmark, split) job with checkpointing.

    Returns a summary dict with counts for the final log.
    """
    runner_cls = _METHOD_TO_RUNNER[method]
    runner = runner_cls(
        client=client,
        method_name=method,
        split_name=split,
        prompt_version=run_cfg.get("prompt_version", "v1"),
        prompts_dir=paths["prompts_dir"],
        run_id=run_id,
        temperature=run_cfg.get("temperature", 0.0),
        max_tokens=run_cfg.get("max_tokens", 500),
        seed=run_cfg.get("seed", 42),
    )

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=paths["checkpoints_dir"],
        run_id=run_id,
        condition=method,
        model=model_name,
        benchmark=benchmark,
    )

    state = checkpoint_mgr.load()
    completed_ids: list[str] = state["completed_ids"] if state else []
    accumulated: list[dict] = state["results"] if state else []
    started_at: str = (
        state["started_at"] if state else datetime.now(timezone.utc).isoformat()
    )

    completed_set = set(completed_ids)
    remaining = [q for q in questions if q["question_id"] not in completed_set]

    tag = f"[{model_name}] {method}"

    if not remaining:
        logger.info("%s  already complete (%d questions) — writing CSV", tag, len(accumulated))
    else:
        logger.info(
            "%s  starting: %d remaining / %d total",
            tag,
            len(remaining),
            len(questions),
        )

        n = run_cfg.get("checkpoint_every_n", 50)
        failed = 0

        for batch_start in range(0, len(remaining), n):
            batch = remaining[batch_start : batch_start + n]
            try:
                batch_results = await runner.run_many(batch)
            except Exception as exc:
                logger.error("%s  batch failed: %s — continuing", tag, exc)
                failed += len(batch)
                continue

            accumulated.extend(batch_results)
            completed_ids.extend(r["question_id"] for r in batch_results)
            checkpoint_mgr.save(completed_ids, accumulated, started_at)

            done = len(accumulated)
            logger.info("%s  %d/%d questions complete", tag, done, len(questions))

    output_path = write_run_results(
        results=accumulated,
        output_dir=paths["runs_dir"] / run_id,
        run_id=run_id,
        method_name=method,
        model_name=model_name,
    )
    checkpoint_mgr.delete()
    logger.info("%s  done → %s", tag, output_path)

    n_success = sum(1 for r in accumulated if r.get("model_status") == "success")
    return {
        "model": model_name,
        "method": method,
        "benchmark": benchmark,
        "total": len(accumulated),
        "success": n_success,
        "output": str(output_path),
    }


async def run_model_jobs(
    model_name: str,
    model_cfg: dict,
    jobs_for_model: list[dict],
    questions_cache: dict[tuple, list[dict]],
    run_id: str,
    run_cfg: dict,
    paths: dict[str, Path],
    use_cache: bool,
) -> list[dict]:
    """Run all jobs for one model sequentially, sharing a single client."""
    try:
        client = build_client(model_name, model_cfg)
    except Exception as exc:
        logger.error("[%s] Failed to build client: %s — skipping all jobs", model_name, exc)
        return []

    if use_cache:
        cache = ResponseCache(paths["cache_dir"])
        client = CachingClientWrapper(client, cache)

    summaries = []
    for job in jobs_for_model:
        for method in job["methods"]:
            key = (job["benchmark"], job["split"])
            questions = questions_cache[key]
            try:
                summary = await run_single_method(
                    client=client,
                    model_name=model_name,
                    method=method,
                    benchmark=job["benchmark"],
                    split=job["split"],
                    questions=questions,
                    run_id=run_id,
                    run_cfg=run_cfg,
                    paths=paths,
                )
                summaries.append(summary)
            except Exception as exc:
                logger.error(
                    "[%s] %s failed: %s — continuing with remaining methods",
                    model_name,
                    method,
                    exc,
                )
                summaries.append(
                    {"model": model_name, "method": method, "error": str(exc)}
                )

    return summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overnight experiment runner for two-prompt research."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run ID string; auto-generates a timestamp if omitted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print preflight estimate and exit without making any API calls",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching for this run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the preflight confirmation prompt (for automated runs)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config["paths"])
    run_cfg = config["run"]
    models_cfg = config["models"]
    jobs = run_cfg["jobs"]

    # Always show preflight estimate.
    preflight_estimate(config, paths)

    if args.dry_run:
        print("--dry-run specified. Exiting.")
        return

    if not args.yes:
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    use_cache = run_cfg.get("cache_enabled", True) and not args.no_cache

    logger.info("Run ID: %s", run_id)
    logger.info("Cache: %s", "enabled" if use_cache else "disabled")
    logger.info("Config: %s", args.config)

    # Pre-load questions for each unique (benchmark, split) combination.
    questions_cache: dict[tuple, list[dict]] = {}
    for job in jobs:
        key = (job["benchmark"], job["split"])
        if key not in questions_cache:
            logger.info("Loading questions: benchmark=%s split=%s", *key)
            questions_cache[key] = load_questions(job["benchmark"], job["split"], paths)
            logger.info("  %d questions loaded", len(questions_cache[key]))

    # Group jobs by model so each model gets one coroutine.
    jobs_by_model: dict[str, list[dict]] = {}
    for job in jobs:
        jobs_by_model.setdefault(job["model"], []).append(job)

    # Launch one coroutine per model; they run concurrently.
    logger.info("Launching %d model(s) concurrently...", len(jobs_by_model))
    model_coroutines = [
        run_model_jobs(
            model_name=model_name,
            model_cfg=models_cfg[model_name],
            jobs_for_model=model_jobs,
            questions_cache=questions_cache,
            run_id=run_id,
            run_cfg=run_cfg,
            paths=paths,
            use_cache=use_cache,
        )
        for model_name, model_jobs in jobs_by_model.items()
    ]
    all_summaries: list[list[dict]] = await asyncio.gather(*model_coroutines)

    # Print final summary.
    flat = [s for per_model in all_summaries for s in per_model]
    print("\n── Run complete ───────────────────────────────────────────────")
    print(f"  Run ID: {run_id}")
    print(f"  Results in: {paths['runs_dir'] / run_id}/")
    print()
    for s in flat:
        if "error" in s:
            print(f"  FAILED  {s['model']:<30}  {s['method']:<22}  {s['error']}")
        else:
            pct = f"{100 * s['success'] / s['total']:.0f}%" if s["total"] else "—"
            print(
                f"  OK      {s['model']:<30}  {s['method']:<22}"
                f"  {s['success']}/{s['total']} success ({pct})"
            )
    print("───────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    asyncio.run(main())
