# tests/runners/test_two_stage_permutation.py

from pathlib import Path

import pytest

from twoprompt.runners.two_stage_permutation import TwoStagePermutationRunner
from twoprompt.runners.permutation import PermutationRunner
from twoprompt.clients.types import ModelResponse, RequestMetadata
from twoprompt.scoring.types import SCORE_CORRECT, SCORE_INCORRECT
from twoprompt.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import MockClient, _make_success_response, _make_failure_response

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(client):
    return TwoStagePermutationRunner(
        client=client,
        method_name="two_prompt_pride",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestTwoStagePermutationRunnerRunOne:
    """Tests for TwoStagePermutationRunner.run_one execution flow."""

    @pytest.mark.asyncio
    async def test_correct_combined(self, runner_question_row, runner_metadata):
        """Free text correct, all permutations match — should score correct."""
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = PermutationRunner._generate_permutations(canonical)

        responses = [_make_success_response("HTTPS", runner_metadata)]
        for perm in perms:
            for letter, text in perm.items():
                if text == "HTTPS":
                    responses.append(_make_success_response(letter, runner_metadata))
                    break

        result = await _make_runner(MockClient(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    @pytest.mark.asyncio
    async def test_stage_one_failure_returns_early(
            self, runner_question_row, runner_metadata
    ):
        """If free-text call fails, should return immediately."""
        client = MockClient(responses=[_make_failure_response(runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert len(client.requests_received) == 1

    @pytest.mark.asyncio
    async def test_all_permutations_fail(self, runner_question_row, runner_metadata):
        """Free text succeeds but all 4 matching calls fail — no valid vote."""
        responses = [_make_success_response("HTTPS", runner_metadata)]
        responses.extend([_make_failure_response(runner_metadata) for _ in range(4)])
        result = await _make_runner(MockClient(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["free_text_response"] == "HTTPS"

    @pytest.mark.asyncio
    async def test_majority_wins(self, runner_question_row, runner_metadata):
        """Three permutations correct, one wrong — majority should win."""
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        perms = PermutationRunner._generate_permutations(canonical)

        responses = [_make_success_response("HTTPS", runner_metadata)]
        for i, perm in enumerate(perms):
            if i == 0:
                responses.append(_make_success_response("A", runner_metadata))
            else:
                for letter, text in perm.items():
                    if text == "HTTPS":
                        responses.append(_make_success_response(letter, runner_metadata))
                        break

        result = await _make_runner(MockClient(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    @pytest.mark.asyncio
    async def test_makes_five_api_calls(self, runner_question_row, runner_metadata):
        """Should fire 5 requests — 1 free text + 4 permutations."""
        responses = [_make_success_response("HTTPS", runner_metadata)]
        responses.extend([_make_success_response("C", runner_metadata) for _ in range(4)])
        client = MockClient(responses=responses)
        await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert len(client.requests_received) == 5

    @pytest.mark.asyncio
    async def test_free_text_response_preserved(
            self, runner_question_row, runner_metadata
    ):
        """The intermediate free-text response should be in the result row."""
        responses = [_make_success_response("HTTPS", runner_metadata)]
        responses.extend([_make_success_response("C", runner_metadata) for _ in range(4)])
        result = await _make_runner(MockClient(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["free_text_response"] == "HTTPS"
        assert result["free_text_prompt"] is not None
        assert result["free_text_latency"] is not None

    @pytest.mark.asyncio
    async def test_result_row_metadata(self, runner_question_row, runner_metadata):
        """Result row should carry trace metadata."""
        responses = [_make_success_response("HTTPS", runner_metadata)]
        responses.extend([_make_success_response("C", runner_metadata) for _ in range(4)])
        result = await _make_runner(MockClient(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "two_prompt_pride"
        assert result["split_name"] == "robustness"

    @pytest.mark.asyncio
    async def test_incorrect_combined(self, runner_question_row, runner_metadata):
        """All permutations agree on wrong answer — should score incorrect."""
        responses = [_make_success_response("FTP", runner_metadata)]
        responses.extend([_make_success_response("A", runner_metadata) for _ in range(4)])
        result = await _make_runner(MockClient(responses=responses)).run_one(
            runner_question_row, sample_index=0
        )

        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT
