# scripts/experiments/baseline.py
# Runs the standard MCQ baseline experiment using Gemini.

import re
import os
import pandas as pd
from google import genai  
from dotenv import load_dotenv
import time
import ast

MAX_API_CALLS = 150

def format_mcq_prompt(question: str, choices: str) -> str:
    """
    Used to format the questions and choices to pass onto Gemini.
    Args:
        question: the MCQ question
        choices: the MCQ choices for that question
    Returns:
        Formatted string containing the qustion and answers after.
    """
    letters = ["A", "B", "C", "D"]
    formatted = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    return (
        f"Answer the following multiple choice question.\n\n"
        f"Question: {question}\n\n"
        f"{formatted}\n\nAnswer:"
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
    letters = ["A", "B", "C", "D"]
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

def main(client: genai.Client, questions: pd.DataFrame, path: str = "results/baseline/baseline_results.csv") -> None:
    """
    Main method that formats the questions and choices, passes the prompt
    to Gemini, extracts the token from the answer and stores the original
    answer alongside Gemini's answer in a csv file
    Args:
        client: The defined Gemini client
        questions: pandas dataframe containing the MMLU questions
        path: path to save the results (default is "results/baseline/baseline_results.csv")
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
            temp_choices = ast.literal_eval(row["choices"])
            prompt = format_mcq_prompt(row["question"], temp_choices)
            answer = extract_answer(call_gemini(client, prompt))
            api_call_count += 1
            results.append([row["answer"], answer, row["subject"]])
            print(f"[{api_call_count}/{MAX_API_CALLS}] {[row['answer'], answer]}")
            time.sleep(1)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    pd.DataFrame(results, columns=["actual", "predicted", "subject"]).to_csv(path, index=False)

if __name__ == "__main__":
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    df = pd.read_csv("data/questions.csv")
    main(client, df)
