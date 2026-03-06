# core.extractor
# ABCD extraction logic

from __future__ import annotations
import re

def extract_answer(response: str) -> str:
    """
    Extracts the answer from the given LLMs response. Only extracts if length == 1 or length == 2 and the second character is 
    within a set of allowed punctuation. Otherwise, it returns None

    Args:
        response: The LLMs response that contain's the answer
    """
    if response is None:
        return None
    response = response.strip().upper()
    if len(response) == 0:
        return None
    elif len(response) == 1:
        return response if response in "ABCD" else None
    elif len(response) == 2:
        if response[1] in ".:)":
            response = response[0]
            return response if response in "ABCD" else None
    else:
        return _regex_search(response)

def _regex_search(response:str) -> str:
    """
    Internal method used in case the LLM response isnt 1 or 2 characters only. 
    Uses regex search to intelligently search through the string for the answer
    """
    match = re.findall(r"\b([A-D])\b", response, re.IGNORECASE)
    return None if not match else match[-1].upper()