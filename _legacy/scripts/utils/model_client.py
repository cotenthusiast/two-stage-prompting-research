# scripts/utils/model_client.py
# Unified model call interface.
# Add new model handlers here — baseline.py and two_prompt.py stay untouched.

import os
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_MODELS = ["gemini", "gpt", "llama"]


def call_model(prompt: str, model: str) -> str:
    """
    Routes a prompt to the correct model backend and returns the response text.

    Args:
        prompt: The prompt string to send to the model.
        model: One of "gemini", "gpt", or "llama".

    Returns:
        The model's response as a plain string.

    Raises:
        ValueError: If an unsupported model name is passed.
    """
    if model == "gemini":
        return _call_gemini(prompt)
    elif model == "gpt":
        return _call_gpt(prompt)
    elif model == "llama":
        return _call_llama(prompt)
    else:
        raise ValueError(f"Unsupported model '{model}'. Choose from: {SUPPORTED_MODELS}")


def _call_gemini(prompt: str) -> str:
    """
    Calls Gemini 2.5 Flash via the google-genai SDK.

    Args:
        prompt: The prompt string to send.

    Returns:
        Gemini's response text.
    """
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text


def _call_gpt(prompt: str) -> str:
    """
    Calls GPT-4o-mini via the OpenAI SDK.

    Args:
        prompt: The prompt string to send.

    Returns:
        GPT's response text.
    """
    # pip install openai
    # Add OPENAI_API_KEY to your .env file
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def _call_llama(prompt: str) -> str:
    """
    Calls Llama via the Groq API (groq SDK).
    Uses llama-3.3-70b-versatile by default.

    Args:
        prompt: The prompt string to send.

    Returns:
        Llama's response text.
    """
    # pip install groq
    # Add GROQ_API_KEY to your .env file
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
