# scripts/experiments/two_prompt.py
# Runs the two-stage prompting experiment using Gemini.
# Stage 1: Free-text reasoning without answer choices.
# Stage 2: Match reasoning to MCQ options.

import re
import os
import pandas as pd
from google import genai  
from dotenv import load_dotenv
import time
import ast

MAX_API_CALLS = 250

def format_prompt1(question: str) -> str:
    """
    Formats the first prompt, asking the model to reason through the question
    without seeing the answer choices.
    Args:
        question: the MMLU question string
    Returns:
        The formatted prompt string
    """
    return f"Answer the following question, explaining your reasoning step by step:\n{question}"

def format_prompt2(question: str, free_text_answer: str, choices: list) -> str:
    """
    Formats the second prompt, presenting the model's prior reasoning
    alongside the answer choices to select the best match.
    Args:
        question: the original MMLU question string
        free_text_answer: the reasoning generated in stage 1
        choices: list of four answer choice strings
    Returns:
        The formatted prompt string
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
    Reads the response from Gemini and returns the extracted letter.
    It looks for 'A', 'B', 'C', or 'D' in the response.
    Args:
        response: gemini's response to the MCQ question
    Returns:
        The letter found or None if nothing found
    """
    match = re.search(r'(?i)Answer\s*:\s*([ABCD])', response)
    if not match:
        match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1).upper()
    return None

def call_gemini(client: genai.Client, prompt: str) -> str:
    """
    Takes the question with choices and passes it into Gemini and returns Gemini's reponse
    Args:
        prompt: the string thatll be passed to Gemini API
        client: the defined Gemini client
    Returns:
        Gemini's Answer
    """
    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text

def main(client: genai.Client, questions: pd.DataFrame, path: str = "results/two_stage/two_stage_results.csv") -> None:
    """
    Runs the two-stage prompting experiment. For each question, calls Gemini
    twice: once for free-text reasoning, once for MCQ matching. Saves results
    to a CSV including the stage 1 reasoning for inspection.
    Args:
        client: the defined Gemini client
        questions: pandas dataframe containing the MMLU questions
        path: path to save the results (default is "results/two_stage/two_stage_results.csv")
    """
    results = []
    api_call_count = 0

    # Ensure results directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for _, row in questions.iterrows():
        if api_call_count >= MAX_API_CALLS:
            print(f"Reached MAX_API_CALLS limit of {MAX_API_CALLS}. Stopping.")
            break
        try:
            choices = ast.literal_eval(row["choices"])
            prompt1 = format_prompt1(row["question"])
            response_prompt_1 = call_gemini(client, prompt1)
            api_call_count += 1
            prompt2 = format_prompt2(row["question"], response_prompt_1, choices)
            answer = call_gemini(client, prompt2)
            answer = extract_answer(answer)
            api_call_count += 1
            results.append([row["answer"], answer, row["subject"], response_prompt_1])           
            print(f"[{api_call_count}/{MAX_API_CALLS}] {[row['answer'], answer]}")
            time.sleep(1)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    pd.DataFrame(results, columns=["actual", "predicted", "subject", "stage1_response"]).to_csv(path, index=False)

if __name__ == "__main__":
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df = pd.read_csv("data/questions.csv")
    main(client, df)
