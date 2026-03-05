# scripts/experiments/baseline.py
# Runs the standard MCQ baseline experiment.
# Supports multiple models via --model flag (gemini, gpt, llama).

import re
import os
import sys
import pandas as pd
import argparse
import time
import ast
import hashlib
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_client import call_model, SUPPORTED_MODELS
from utils.constants import BASELINE_RESULTS_PATH

MAX_API_CALLS = 150

def make_qid(subject: str, question: str, choices: list) -> str:
    payload = {"subject": subject, "question": question, "choices": choices}
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def format_mcq_prompt(question: str, choices: list) -> str:
    """
    Formats a question and its choices into a standard MCQ prompt.

    Args:
        question: The MCQ question string.
        choices: List of four answer choice strings.

    Returns:
        Formatted prompt string ready to send to the model.
    """
    letters = ["A", "B", "C", "D"]
    formatted = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    return (
        f"Answer the following multiple choice question.\n\n"
        f"Question: {question}\n\n"
        f"{formatted}\n\n"
        f"Respond with only a single character: A, B, C, or D. No other text."
    )

def extract_answer(response: str) -> str:
    """
    Extracts the answer letter (A-D) from a model response.

    Args:
        response: The model's raw response string.

    Returns:
        The extracted letter in uppercase, or None if not found.
    """
    if response is None:
        return None
    response = response.upper().strip()
    if len(response) == 0:
        return None
    if len(response) == 1:
        return response if response in "ABCD" else None
    if len(response) == 2 and response[1] in ".):":
        return response[0] if response[0] in "ABCD" else None
    return None
        

def main(questions: pd.DataFrame, model: str, path: str = BASELINE_RESULTS_PATH) -> None:
    """
    Runs the baseline MCQ experiment. For each question, sends a formatted
    prompt to the model and records the predicted vs actual answer.

    Args:
        questions: DataFrame containing MMLU questions.
        model: Model backend to use (gemini, gpt, or llama).
        path: Output CSV path for results.
    """
    results = []
    api_call_count = 0

    os.makedirs(os.path.dirname(path), exist_ok=True)

    for _, row in questions.iterrows():
        qid = None
        raw_response = None
        if api_call_count >= MAX_API_CALLS:
            print(f"Reached MAX_API_CALLS limit of {MAX_API_CALLS}. Stopping.")
            break
        try:
            choices = ast.literal_eval(row["choices"])
            qid = make_qid(row["subject"], row["question"], choices)
            prompt = format_mcq_prompt(row["question"], choices)
            api_call_count += 1
            raw_response = call_model(prompt, model)
            answer = extract_answer(raw_response)
            results.append([qid, row["answer"], answer, row["subject"], raw_response, model])
            print(f"[{api_call_count}/{MAX_API_CALLS}] {[row['answer'], answer]}")
            time.sleep(1)
        except Exception as e:
            print(f"Error processing row: {e}")
            results.append([qid, row["answer"], None, row["subject"], str(e),  model])
            continue

    pd.DataFrame(results, columns=["qid", "actual", "predicted", "subject", "raw_response","model"]).to_csv(path, index=False)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the MCQ baseline experiment.")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="gemini",
                        help="Model to use for inference (default: gemini)")
    parser.add_argument("--output", default=BASELINE_RESULTS_PATH,
                        help="Path to save results CSV")
    args = parser.parse_args()

    df = pd.read_csv("data/questions.csv")
    main(df, model=args.model, path=args.output)
