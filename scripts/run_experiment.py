"""Run the core experiment: 4 methods × 3 models on the robustness split."""

import asyncio
from datetime import datetime, timezone

from twoprompt.clients.gemini_client import GeminiClient
from twoprompt.clients.groq_client import GroqClient
from twoprompt.clients.openai_client import OpenAIClient
from twoprompt.config.experiment import (
    BASELINE_METHOD,
    CYCLIC_METHOD,
    ROBUSTNESS_TRACK_NAME,
    TWOPROMPT_CYCLIC_METHOD,
    TWOPROMPT_METHOD,
)
from twoprompt.config.models import (
    GEMINI_CORE_MODEL,
    GROQ_CORE_MODEL,
    OPENAI_CORE_MODEL,
)
from twoprompt.config.paths import (
    NORMALIZED_QUESTIONS_FILENAME,
    PROCESSED_DIR,
    RUNS_DIR,
    SPLITS_DIR,
)
from twoprompt.io.readers import read_normalized_questions, read_split_ids
from twoprompt.io.writers import write_run_results
from twoprompt.runners.direct_mcq import DirectMCQRunner
from twoprompt.runners.permutation import PermutationRunner
from twoprompt.runners.two_stage import TwoStageRunner
from twoprompt.runners.two_stage_permutation import TwoStagePermutationRunner

PROMPT_VERSION = "v1"

ALL_METHODS = [BASELINE_METHOD, TWOPROMPT_METHOD, CYCLIC_METHOD, TWOPROMPT_CYCLIC_METHOD]
GEMINI_METHODS = [BASELINE_METHOD, TWOPROMPT_METHOD, CYCLIC_METHOD]  #combined tomorrow


def load_split_questions() -> list[dict]:
    """Load normalized questions filtered to the robustness split."""
    df = read_normalized_questions(NORMALIZED_QUESTIONS_FILENAME, PROCESSED_DIR)
    split_ids = read_split_ids(
        ROBUSTNESS_TRACK_NAME, SPLITS_DIR, "benchmark",
    )
    df = df[df["question_id"].isin(split_ids)]
    df = df.drop_duplicates(subset="question_id")
    return df.to_dict(orient="records")


def build_runners_for_model(
    client, split_name: str, run_id: str, methods: list[str],
) -> list[tuple[str, object]]:
    """Create runners for specified methods for one model client."""
    shared = dict(
        client=client,
        split_name=split_name,
        prompt_version=PROMPT_VERSION,
        run_id=run_id,
    )
    method_to_runner = {
        BASELINE_METHOD: DirectMCQRunner,
        TWOPROMPT_METHOD: TwoStageRunner,
        CYCLIC_METHOD: PermutationRunner,
        TWOPROMPT_CYCLIC_METHOD: TwoStagePermutationRunner,
    }
    return [
        (method, method_to_runner[method](method_name=method, **shared))
        for method in methods
    ]


async def run_model(
    client, model_name: str, questions: list[dict], run_id: str, methods: list[str],
):
    """Run specified methods for a single model, sequentially."""
    runners = build_runners_for_model(client, ROBUSTNESS_TRACK_NAME, run_id, methods)

    for method_name, runner in runners:
        print(f"  [{model_name}] Running {method_name} ({len(questions)} questions)...")
        results = await runner.run_many(questions)

        output_path = write_run_results(
            results=results,
            output_dir=RUNS_DIR / run_id,
            run_id=run_id,
            method_name=method_name,
            model_name=model_name,
        )
        print(f"  [{model_name}] {method_name} done -> {output_path}")


async def main():
    """Run experiment conditions across models."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    questions = load_split_questions()

    print(f"[start] Run ID: {run_id}")
    print(f"[start] {len(questions)} questions loaded from robustness split")

    clients = [
        (OpenAIClient(OPENAI_CORE_MODEL), OPENAI_CORE_MODEL, ALL_METHODS),
        (GeminiClient(GEMINI_CORE_MODEL, concurrency_limit=2), GEMINI_CORE_MODEL, GEMINI_METHODS),
        (GroqClient(GROQ_CORE_MODEL, concurrency_limit=2), GROQ_CORE_MODEL, ALL_METHODS),
    ]

    print("[run] Launching models concurrently...")
    await asyncio.gather(*[
        run_model(client, model_name, questions, run_id, methods)
        for client, model_name, methods in clients
    ])

    print(f"[complete] Results in {RUNS_DIR / run_id}/")


if __name__ == "__main__":
    asyncio.run(main())