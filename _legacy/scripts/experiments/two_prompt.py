# scripts/experiments/two_prompt.py
# Runs the two-stage prompting experiment.
# Stage 1: Free-text reasoning without answer choices.
# Stage 2: Match reasoning to MCQ options.
# Supports multiple models via --model flag (gemini, gpt, llama).

# stdlib
import os
import time
import re
import ast
import argparse

# third-party
import pandas as pd

# local
from utils.pid import make_pid
from utils.model_client import call_model, SUPPORTED_MODELS
from utils.constants import TWO_STAGE_RESULTS_PATH

MAX_API_CALLS = 250

def format_prompt1(question: str) -> str:
    """
    Formats the first prompt: asks the model to reason through the question
    without seeing the answer choices.

    Args:
        question: The MMLU question string.

    Returns:
        Formatted stage 1 prompt string.
    """
    return f"Answer the following question, explaining your reasoning step by step:\n{question}"


def format_prompt2(question: str, free_text_answer: str, choices: list) -> str:
    """
    Formats the second prompt: presents the model's prior reasoning alongside
    the answer choices to select the best match.

    Args:
        question: The original MMLU question string.
        free_text_answer: The reasoning generated in stage 1.
        choices: List of four answer choice strings.

    Returns:
        Formatted stage 2 prompt string.
    """
    return (
        f"Question: {question}\n\n"
        f"Reasoning: {free_text_answer}\n\n"
        f"Which of the following options best matches the reasoning above?\n\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        f"Respond with only the letter A, B, C, or D."
    )


def extract_answer(response: str) -> str:
    """
    Extracts the answer letter (A-D) from a model response.

    Args:
        response: The model's raw response string.

    Returns:
        The extracted letter in uppercase, or None if not found.
    """
    match = re.search(r'(?i)Answer\s*:\s*([ABCD])', response)
    if not match:
        match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1).upper()
    return None


def main(questions: pd.DataFrame, model: str, path: str = TWO_STAGE_RESULTS_PATH) -> None:
    """
    Runs the two-stage prompting experiment. For each question, calls the model
    twice: once for free-text reasoning, once for MCQ matching. Saves results
    to a CSV including the stage 1 reasoning for inspection.

    Args:
        questions: DataFrame containing MMLU questions.
        model: Model backend to use (gemini, gpt, or llama).
        path: Output CSV path for results.
    """
    results = []
    api_call_count = 0

    os.makedirs(os.path.dirname(path), exist_ok=True)

    for _, row in questions.iterrows():
        if api_call_count >= MAX_API_CALLS:
            print(f"Reached MAX_API_CALLS limit of {MAX_API_CALLS}. Stopping.")
            break
        try:
            choices = ast.literal_eval(row["choices"])
            qid = make_qid(row["subject"], row["question"], choices)
            prompt1 = format_prompt1(row["question"])
            stage1_response = call_model(prompt1, model)
            api_call_count += 1
            prompt2 = format_prompt2(row["question"], stage1_response, choices)
            answer = extract_answer(call_model(prompt2, model))
            api_call_count += 1
            results.append([qid, row["answer"], answer, row["subject"], model, stage1_response])            print(f"[{api_call_count}/{MAX_API_CALLS}] {[row['answer'], answer]}")
            time.sleep(1)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    pd.DataFrame(
        results, columns=["qid", "actual", "predicted", "subject", "model", "stage1_response"]
    ).to_csv(path, index=False)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the two-stage MCQ experiment.")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="gemini",
                        help="Model to use for inference (default: gemini)")
    parser.add_argument("--output", default=TWO_STAGE_RESULTS_PATH,
                        help="Path to save results CSV")
    args = parser.parse_args()

    df = pd.read_csv("data/questions.csv")
    main(df, model=args.model, path=args.output)
