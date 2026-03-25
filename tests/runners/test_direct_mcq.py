# tests/runners/test_direct_mcq.py

import pytest

from twoprompt.runners.direct_mcq import DirectMCQRunner
from twoprompt.clients.types import ModelResponse, RequestMetadata
from twoprompt.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE
from twoprompt.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import MockClient, _make_success_response, _make_failure_response


class TestDirectMCQRunnerRunOne:
    """Tests for DirectMCQRunner.run_one execution flow."""

    @pytest.mark.asyncio
    async def test_correct_answer(
            self,
            runner_question_row,
            runner_metadata,
    ):
        """Model returns the correct letter — should parse and score correct."""
        response = _make_success_response("C", runner_metadata)
        client = MockClient(responses=[response])
        runner = DirectMCQRunner(
            client=client,
            method_name="baseline",
            split_name="robustness",
            prompt_version="v1",
            run_id="test_run_001",
        )

        result = await runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT
        assert result["parse_status"] == PARSE_OK

    @pytest.mark.asyncio
    async def test_incorrect_answer(
            self,
            runner_question_row,
            runner_metadata,
    ):
        """Model returns a wrong letter — should parse and score incorrect."""
        response = _make_success_response("A", runner_metadata)
        client = MockClient(responses=[response])
        runner = DirectMCQRunner(
            client=client,
            method_name="baseline",
            split_name="robustness",
            prompt_version="v1",
            run_id="test_run_001",
        )

        result = await runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    @pytest.mark.asyncio
    async def test_failed_response(
            self,
            runner_question_row,
            runner_metadata,
    ):
        """Model call fails — parsed and score fields should be None."""
        response = _make_failure_response(runner_metadata)
        client = MockClient(responses=[response])
        runner = DirectMCQRunner(
            client=client,
            method_name="baseline",
            split_name="robustness",
            prompt_version="v1",
            run_id="test_run_001",
        )

        result = await runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None
        assert result["error_type"] == "ProviderTimeoutError"

    @pytest.mark.asyncio
    async def test_unparseable_response(
            self,
            runner_question_row,
            runner_metadata,
    ):
        """Model returns gibberish — should parse as missing, score unscorable."""
        response = _make_success_response("I'm not sure about this question", runner_metadata)
        client = MockClient(responses=[response])
        runner = DirectMCQRunner(
            client=client,
            method_name="baseline",
            split_name="robustness",
            prompt_version="v1",
            run_id="test_run_001",
        )

        result = await runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    @pytest.mark.asyncio
    async def test_result_row_metadata(
            self,
            runner_question_row,
            runner_metadata,
    ):
        """Result row should carry all trace metadata correctly."""
        response = _make_success_response("C", runner_metadata)
        client = MockClient(responses=[response])
        runner = DirectMCQRunner(
            client=client,
            method_name="baseline",
            split_name="robustness",
            prompt_version="v1",
            run_id="test_run_001",
        )

        result = await runner.run_one(runner_question_row, sample_index=3)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "baseline"
        assert result["split_name"] == "robustness"
        assert result["subject"] == "computer_security"
        assert result["provider"] == "openai"
        assert result["model_name"] == "gpt-5-mini"
        assert result["sample_index"] == 3

    @pytest.mark.asyncio
    async def test_prompt_contains_question_and_options(
            self,
            runner_question_row,
            runner_metadata,
    ):
        """The prompt sent to the model should include the question and all options."""
        response = _make_success_response("C", runner_metadata)
        client = MockClient(responses=[response])
        runner = DirectMCQRunner(
            client=client,
            method_name="baseline",
            split_name="robustness",
            prompt_version="v1",
            run_id="test_run_001",
        )

        result = await runner.run_one(runner_question_row, sample_index=0)

        assert "securely browse websites" in result["prompt"]
        assert "FTP" in result["prompt"]
        assert "HTTP" in result["prompt"]
        assert "HTTPS" in result["prompt"]
        assert "SMTP" in result["prompt"]

    @pytest.mark.asyncio
    async def test_lowercase_answer_parsed(
            self,
            runner_question_row,
            runner_metadata,
    ):
        """Model returns lowercase letter — should still parse correctly."""
        response = _make_success_response("c", runner_metadata)
        client = MockClient(responses=[response])
        runner = DirectMCQRunner(
            client=client,
            method_name="baseline",
            split_name="robustness",
            prompt_version="v1",
            run_id="test_run_001",
        )

        result = await runner.run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True


class TestDirectMCQRunnerBuildPrompt:
    """Tests for DirectMCQRunner._build_prompt."""

    def test_prompt_format(self, runner_question_row):
        """Prompt should be a non-empty string containing the question."""
        prompt = DirectMCQRunner._build_prompt(runner_question_row)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert runner_question_row["question_text"] in prompt

    def test_prompt_contains_all_options(self, runner_question_row):
        """Prompt should include all four option texts."""
        prompt = DirectMCQRunner._build_prompt(runner_question_row)

        assert "FTP" in prompt
        assert "HTTP" in prompt
        assert "HTTPS" in prompt
        assert "SMTP" in prompt