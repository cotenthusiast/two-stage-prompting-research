# src/twoprompt/runners/two_stage.py

from typing import Any

from twoprompt.pipeline.prompt_builder import (
    build_free_text_prompt,
    build_option_matching_prompt,
)
from twoprompt.runners.base import ExperimentRunner


class TwoStageRunner(ExperimentRunner):
    """Runner for the two-stage prompting condition.

    Stage one elicits a free-text answer without exposing options.
    Stage two asks the model to match that free-text answer to one
    of the four canonical options. The final letter from stage two
    is parsed and scored.

    The intermediate free-text response is preserved in the result
    row for downstream answer-matching evaluation.
    """

    async def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through the two-stage pipeline.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary containing trace, model output,
            parse, and score fields, plus the intermediate free-text
            response.
        """
        # Stage 1: free-text response
        free_text_prompt = build_free_text_prompt(
            question=question_row["question_text"],
        )
        free_text_request = self._build_model_request(
            question_row, free_text_prompt, sample_index,
        )
        free_text_response = await self.client.generate(free_text_request)

        # If stage 1 fails, return early
        if not free_text_response.is_success():
            return self._build_result_row(
                question_row=question_row,
                prompt=free_text_prompt,
                model_request=free_text_request,
                model_response=free_text_response,
                parsed_result=None,
                score_result=None,
            )

        free_text_answer = free_text_response.raw_text

        # Stage 2: option matching using the free-text answer
        matching_prompt = build_option_matching_prompt(
            question=question_row["question_text"],
            free_text=free_text_answer,
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )
        matching_request = self._build_model_request(
            question_row, matching_prompt, sample_index,
        )
        matching_response = await self.client.generate(matching_request)

        # Parse and score the stage 2 response
        parsed_result = None
        score_result = None

        if matching_response.is_success():
            parsed_result, score_result = self._parse_and_score(
                raw_text=matching_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        result = self._build_result_row(
            question_row=question_row,
            prompt=matching_prompt,
            model_request=matching_request,
            model_response=matching_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

        # Preserve the intermediate free-text response for answer matching
        result["free_text_prompt"] = free_text_prompt
        result["free_text_response"] = free_text_answer
        result["free_text_latency"] = free_text_response.latency_seconds

        return result
