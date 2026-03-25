# src/twoprompt/runners/direct_mcq.py

from typing import Any

from twoprompt.runners.base import ExperimentRunner
from twoprompt.pipeline.prompt_builder import build_direct_mcq_prompt


class DirectMCQRunner(ExperimentRunner):
    """Runner for the direct MCQ baseline condition.

    Presents the model with a standard multiple-choice question and
    expects a single letter response. One prompt, one API call per
    question.
    """

    async def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through the direct MCQ baseline.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary containing trace, model output,
            parse, and score fields.
        """
        prompt = self._build_prompt(question_row)
        model_request = self._build_model_request(question_row, prompt, sample_index)
        model_response = await self.client.generate(model_request)

        parsed_result = None
        score_result = None

        if model_response.is_success():
            parsed_result, score_result = self._parse_and_score(
                raw_text=model_response.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            model_request=model_request,
            model_response=model_response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

    @staticmethod
    def _build_prompt(question_row: Any) -> str:
        """Build a direct multiple-choice prompt from a question row.

        Args:
            question_row: Normalized question record.

        Returns:
            Fully formatted direct MCQ prompt string.
        """
        return build_direct_mcq_prompt(
            question=question_row["question_text"],
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )