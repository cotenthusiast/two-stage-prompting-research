from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from twoprompt.clients.types import (
    ModelResponse,
    ProviderConfigurationError,
    RequestMetadata,
    UsageInfo,
    SUCCESS_STATUS,
)
from twoprompt.runners.pride import PriDeRunner

from tests.runners.conftest import MockClient


REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _together_response(metadata: RequestMetadata, raw_text: str) -> ModelResponse:
    return ModelResponse(
        provider="together",
        model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        status=SUCCESS_STATUS,
        latency_seconds=0.1,
        metadata=metadata,
        raw_text=raw_text,
        finish_reason="stop",
        usage=UsageInfo(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        error=None,
        timestamp_utc=None,
        logprobs=[
            {
                "token": raw_text.strip()[:1],
                "logprob": -0.01,
                "top_logprobs": [
                    {"token": lt, "logprob": -2.0 - j * 0.1}
                    for j, lt in enumerate(("A", "B", "C", "D"))
                    if lt != raw_text.strip()[:1]
                ],
            },
        ],
    )


class TestPriDeRunnerIntegration:
    @pytest.fixture
    def pride_meta_template(self, runner_metadata: RequestMetadata) -> RequestMetadata:
        return RequestMetadata(
            question_id=runner_metadata.question_id,
            split_name=runner_metadata.split_name,
            method_name="pride",
            subject=runner_metadata.subject,
            run_id=runner_metadata.run_id,
            prompt_version=runner_metadata.prompt_version,
            perturbation_name=None,
            sample_index=runner_metadata.sample_index,
        )

    def test_raises_on_non_together_provider(self, tmp_path: Path):
        client = MockClient(responses=[], provider="openai", model_name="gpt-5-mini")
        with pytest.raises(ProviderConfigurationError):
            PriDeRunner(
                client=client,
                method_name="pride",
                split_name="robustness",
                prompt_version="v1",
                prompts_dir=_PROMPTS_DIR,
                run_id="r1",
                calibration_n=0,
                calibration_seed=0,
                calibration_benchmark="mmlu",
                calibration_runs_dir=tmp_path,
                calibration_questions=[],
            )

    def test_call_counts_separate_calibration_then_eq8_inference(
            self, runner_question_row, pride_meta_template, tmp_path: Path,
    ):
        """K=1 calibration question (separate from eval): 4 cyclic + 1 direct = 5 calls.

        All eval rows must use eq8_transfer mode.
        """
        cal_question = {
            **runner_question_row,
            "question_id": "cal_qid",
            "correct_option": "B",
        }
        eval_question = {
            **runner_question_row,
            "question_id": "eval_qid",
            "correct_option": "A",
        }
        n_calls = 4 + 1  # 4 cyclic rollouts for calibration + 1 direct for eval
        reps = [_together_response(pride_meta_template, "B\n")] * n_calls
        client = MockClient(
            responses=reps,
            provider="together",
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        )

        runner = PriDeRunner(
            client=client,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_integration",
            calibration_n=1,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[cal_question],
        )

        async def _drive():
            return await runner.run_many([eval_question])

        rows = asyncio.run(_drive())
        assert len(rows) == 1
        assert rows[0]["pride_inference_mode"] == "eq8_transfer"
        assert rows[0]["model_status"] == "success"
        assert len(client.requests_received) == n_calls
        assert all(r.request_logprobs for r in client.requests_received)

    def test_empty_logprobs_skips_debiasing(
            self, runner_question_row, pride_meta_template, tmp_path: Path, caplog,
    ):
        """Empty logprobs: debiasing is skipped, warning logged, adjusted_choice is None."""
        resp = ModelResponse(
            provider="together",
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
            status=SUCCESS_STATUS,
            latency_seconds=0.1,
            metadata=pride_meta_template,
            raw_text="A",
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=1, total_tokens=11),
            error=None,
            timestamp_utc=None,
            logprobs=[],
        )
        client = MockClient(
            responses=[resp],
            provider="together",
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        )
        runner = PriDeRunner(
            client=client,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_empty_lp",
            calibration_n=0,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[],
        )

        with caplog.at_level(logging.WARNING, logger="twoprompt.runners.pride"):
            rows = asyncio.run(runner.run_many([runner_question_row]))

        assert rows[0]["pride_adjusted_choice"] is None
        assert "empty logprobs" in caplog.text

    def test_no_calibration_questions_uses_uniform_prior_one_call_per_eval(
            self, runner_question_row, pride_meta_template, tmp_path: Path,
    ):
        """Empty calibration pool → uniform prior, only 1 direct call per eval question."""
        reps = [_together_response(pride_meta_template, "C\n")]
        client = MockClient(
            responses=reps,
            provider="together",
            model_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        )
        runner = PriDeRunner(
            client=client,
            method_name="pride",
            split_name="robustness",
            prompt_version="v1",
            prompts_dir=_PROMPTS_DIR,
            run_id="pride_no_cal",
            calibration_n=50,
            calibration_seed=0,
            calibration_benchmark="mmlu",
            calibration_runs_dir=tmp_path,
            calibration_questions=[],  # no calibration data → uniform prior
        )

        async def _drive():
            return await runner.run_many([runner_question_row])

        rows = asyncio.run(_drive())
        assert len(rows) == 1
        assert rows[0]["pride_inference_mode"] == "eq8_transfer"
        assert len(client.requests_received) == 1  # no calibration calls
        import json
        prior = json.loads(rows[0]["peprior_json"])
        for v in prior.values():
            assert abs(v - 0.25) < 1e-6, "Prior should be uniform when no calibration data"
