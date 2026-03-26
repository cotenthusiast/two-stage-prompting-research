"""Verify that all provider API keys are configured and clients can connect."""

import asyncio

from twoprompt.clients.gemini_client import GeminiClient
from twoprompt.clients.groq_client import GroqClient
from twoprompt.clients.openai_client import OpenAIClient
from twoprompt.clients.types import ModelRequest, RequestMetadata
from twoprompt.config.models import (
    GEMINI_CORE_MODEL,
    GROQ_CORE_MODEL,
    OPENAI_CORE_MODEL,
    validate_api_keys,
)

SMOKE_PROMPT = "What is 2 + 2?"

SMOKE_METADATA = RequestMetadata(
    question_id="smoke",
    split_name="smoke",
    method_name="smoke",
    subject="smoke",
    run_id="smoke",
    prompt_version="smoke",
    perturbation_name=None,
    sample_index=0,
)


async def smoke_test_client(client, provider: str, model: str) -> bool:
    """Send one trivial request and report pass/fail."""
    request = ModelRequest(
        provider=provider,
        model_name=model,
        payload=SMOKE_PROMPT,
        metadata=SMOKE_METADATA,
    )
    response = await client.generate(request)

    if response.is_success():
        print(f"[pass] {provider}/{model} -> {response.raw_text.strip()[:80]}")
        return True
    else:
        print(f"[FAIL] {provider}/{model} -> {response.error.error_type}: {response.error.message[:120]}")
        return False


async def main():
    validate_api_keys()

    client_configs = [
        (OpenAIClient, "openai", OPENAI_CORE_MODEL),
        (GeminiClient, "gemini", GEMINI_CORE_MODEL),
        (GroqClient, "groq", GROQ_CORE_MODEL),
    ]

    results = []
    for client_cls, provider, model in client_configs:
        print(f"[test] {provider}/{model}...")
        try:
            client = client_cls(model)
        except Exception as exc:
            print(f"[FAIL] {provider}/{model} -> client init failed: {exc}")
            results.append((provider, False))
            continue

        passed = await smoke_test_client(client, provider, model)
        results.append((provider, passed))

    print()
    passed_count = sum(1 for _, p in results if p)
    print(f"[summary] {passed_count}/{len(results)} providers connected successfully.")


if __name__ == "__main__":
    asyncio.run(main())