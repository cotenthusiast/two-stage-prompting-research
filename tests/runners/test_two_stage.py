# tests/runners/test_two_stage.py

from pathlib import Path

import pytest

from twoprompt.runners.two_stage import TwoStageRunner
from twoprompt.clients.types import ModelResponse, RequestMetadata
from twoprompt.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE
from twoprompt.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import MockClient, _make_success_response, _make_failure_response

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(client):
    return TwoStageRunner(
        client=client,
        method_name="two_prompt",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestTwoStageRunnerRunOne:
    """Tests for TwoStageRunner.run_one execution flow."""

    @pytest.mark.asyncio
    async def test_correct_two_stage(self, runner_question_row, runner_metadata):
        """Free text returns 'HTTPS', matching returns 'C' — should score correct."""
        client = MockClient(responses=[
            _make_success_response("HTTPS", runner_metadata),
            _make_success_response("C", runner_metadata),
        ])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    @pytest.mark.asyncio
    async def test_incorrect_two_stage(self, runner_question_row, runner_metadata):
        """Free text returns 'FTP', matching returns 'A' — should score incorrect."""
        client = MockClient(responses=[
            _make_success_response("FTP", runner_metadata),
            _make_success_response("A", runner_metadata),
        ])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    @pytest.mark.asyncio
    async def test_stage_one_failure_returns_early(
            self, runner_question_row, runner_metadata
    ):
        """If stage 1 fails, should return immediately with no parse or score."""
        client = MockClient(responses=[_make_failure_response(runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None
        assert len(client.requests_received) == 1

    @pytest.mark.asyncio
    async def test_stage_two_failure(self, runner_question_row, runner_metadata):
        """Stage 1 succeeds but stage 2 fails — parse and score should be None."""
        client = MockClient(responses=[
            _make_success_response("HTTPS", runner_metadata),
            _make_failure_response(runner_metadata),
        ])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["free_text_response"] == "HTTPS"

    @pytest.mark.asyncio
    async def test_free_text_response_preserved(
            self, runner_question_row, runner_metadata
    ):
        """The intermediate free-text response should be saved in the result row."""
        client = MockClient(responses=[
            _make_success_response("HTTPS", runner_metadata),
            _make_success_response("C", runner_metadata),
        ])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["free_text_response"] == "HTTPS"
        assert result["free_text_prompt"] is not None
        assert result["free_text_latency"] is not None

    @pytest.mark.asyncio
    async def test_makes_two_api_calls(self, runner_question_row, runner_metadata):
        """Should fire exactly 2 requests — free text then matching."""
        client = MockClient(responses=[
            _make_success_response("HTTPS", runner_metadata),
            _make_success_response("C", runner_metadata),
        ])
        await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert len(client.requests_received) == 2

    @pytest.mark.asyncio
    async def test_matching_prompt_contains_free_text(
            self, runner_question_row, runner_metadata
    ):
        """The option-matching prompt should include the free-text answer."""
        client = MockClient(responses=[
            _make_success_response("HTTPS", runner_metadata),
            _make_success_response("C", runner_metadata),
        ])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert "HTTPS" in result["prompt"]

    @pytest.mark.asyncio
    async def test_result_row_metadata(self, runner_question_row, runner_metadata):
        """Result row should carry trace metadata."""
        client = MockClient(responses=[
            _make_success_response("HTTPS", runner_metadata),
            _make_success_response("C", runner_metadata),
        ])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "two_prompt"
        assert result["split_name"] == "robustness"

    @pytest.mark.asyncio
    async def test_unparseable_matching_response(
            self, runner_question_row, runner_metadata
    ):
        """Stage 2 returns gibberish — should be unscorable."""
        client = MockClient(responses=[
            _make_success_response("HTTPS", runner_metadata),
            _make_success_response("I think it might be one of those", runner_metadata),
        ])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE
        assert result["free_text_response"] == "HTTPS"
