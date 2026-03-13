import pytest

from twoprompt.clients.types import (
    RequestValidationError,
    ResponseValidationError,
)

# PYTHONPATH=src pytest tests/clients/test_types.py -v


def test_request_metadata_validate_passes_for_valid_metadata(valid_metadata):
    valid_metadata.validate()


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("question_id", ""),
        ("split_name", ""),
        ("method_name", ""),
        ("subject", ""),
        ("run_id", ""),
        ("prompt_version", ""),
    ],
)
def test_request_metadata_validate_fails_for_invalid_required_strings(
    valid_metadata,
    field_name,
    bad_value,
):
    setattr(valid_metadata, field_name, bad_value)

    with pytest.raises(RequestValidationError):
        valid_metadata.validate()


@pytest.mark.parametrize(
    "perturbation_name,sample_index",
    [
        ("", 0),
        (123, 0),
        ("original", -1),
        ("original", "zero"),
    ],
)
def test_request_metadata_validate_fails_for_invalid_optional_or_index_fields(
    valid_metadata,
    perturbation_name,
    sample_index,
):
    setattr(valid_metadata, "perturbation_name", perturbation_name)
    setattr(valid_metadata, "sample_index", sample_index)

    with pytest.raises(RequestValidationError):
        valid_metadata.validate()


def test_model_request_validate_passes_for_valid_request(valid_request):
    valid_request.validate()


@pytest.mark.parametrize(
    "provider,model_name",
    [
        ("bad_provider", "gpt-5-mini"),
        ("openai", "gemini-2.5-flash"),
        ("gemini", "gpt-5-mini"),
        ("groq", "gpt-5-mini"),
    ],
)
def test_model_request_validate_fails_for_invalid_provider_or_model_pair(
    valid_request,
    provider,
    model_name,
):
    valid_request.model_name = model_name
    valid_request.provider = provider

    with pytest.raises(RequestValidationError):
        valid_request.validate()


@pytest.mark.parametrize(
    "payload",
    [
        "",
        123,
        None,
    ],
)
def test_model_request_validate_fails_for_invalid_payload(
    valid_request,
    payload,
):
    valid_request.payload = payload

    with pytest.raises(RequestValidationError):
        valid_request.validate()


@pytest.mark.parametrize(
    "temperature,max_tokens,seed",
    [
        ("hot", 500, 42),
        (-0.1, 500, 42),
        (2.1, 500, 42),
        (0.0, 0, 42),
        (0.0, -1, 42),
        (0.0, 1.5, 42),
        (0.0, 500, "seed"),
    ],
)
def test_model_request_validate_fails_for_invalid_generation_parameters(
    valid_request,
    temperature,
    max_tokens,
    seed,
):
    valid_request.temperature = temperature
    valid_request.max_tokens = max_tokens
    valid_request.seed = seed

    with pytest.raises(RequestValidationError):
        valid_request.validate()


def test_model_request_validate_fails_for_invalid_metadata(valid_request):
    valid_request.metadata = {}

    with pytest.raises(RequestValidationError):
        valid_request.validate()


def test_model_response_validate_passes_for_valid_success_response(successful_response):
    successful_response.validate()


def test_model_response_validate_passes_for_valid_failure_response(failed_response):
    failed_response.validate()


@pytest.mark.parametrize(
    "status,latency_seconds,metadata",
    [
        ("bad_status", 0.2, "valid"),
        ("success", -1.0, "valid"),
        ("success", "fast", "valid"),
        ("success", 0.2, object()),
    ],
)
def test_model_response_validate_fails_for_invalid_status_latency_or_metadata(
    successful_response,
    valid_metadata,
    status,
    latency_seconds,
    metadata,
):
    successful_response.status = status
    successful_response.latency_seconds = latency_seconds
    if metadata != "valid":
        successful_response.metadata = metadata
    else:
        successful_response.metadata = valid_metadata

    with pytest.raises(ResponseValidationError):
        successful_response.validate()


def test_model_response_validate_fails_for_broken_success_invariants(successful_response):
    successful_response.raw_text = ""

    with pytest.raises(ResponseValidationError):
        successful_response.validate()


def test_model_response_validate_fails_for_broken_failure_invariants(failed_response):
    failed_response.error = None

    with pytest.raises(ResponseValidationError):
        failed_response.validate()


@pytest.mark.parametrize(
    "usage,finish_reason,timestamp_utc",
    [
        (object(), "stop", "2026-03-13T21:00:00Z"),
        (None, "", "2026-03-13T21:00:00Z"),
        (None, 123, "2026-03-13T21:00:00Z"),
        (None, "stop", ""),
        (None, "stop", 123),
    ],
)
def test_model_response_validate_fails_for_invalid_optional_fields(
    successful_response,
    usage,
    finish_reason,
    timestamp_utc,
):
    successful_response.timestamp_utc = timestamp_utc
    successful_response.usage = usage
    successful_response.finish_reason = finish_reason

    with pytest.raises(ResponseValidationError):
        successful_response.validate()


def test_model_response_is_success_returns_true_for_success(successful_response):
    assert successful_response.is_success()


def test_model_response_is_success_returns_false_for_failure(failed_response):
    assert not failed_response.is_success()