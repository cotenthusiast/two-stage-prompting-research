import pytest

from twoprompt.clients.base import BaseClient
from twoprompt.clients.types import (
    ErrorInfo,
    FAILURE_STATUS,
    ModelResponse,
    ProviderTimeoutError,
    SUCCESS_STATUS,
    ValidationError,
    ProviderConfigurationError, ModelRequest, RequestMetadata,

)


class DummyClient(BaseClient):
    def __init__(
        self,
        provider: str,
        model_name: str,
        behavior: str = "success",
        timeout: int = 30,
        concurrency_limit: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            provider=provider,
            model_name=model_name,
            timeout=timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
        )
        self.behavior = behavior

    async def _generate_provider_response(self, request):
        if self.behavior == "success":
            return ModelResponse(
                provider=request.provider,
                model_name=request.model_name,
                status=SUCCESS_STATUS,
                latency_seconds=0.0,
                metadata=request.metadata,
                raw_text="B",
                finish_reason="stop",
                usage=None,
                error=None,
                timestamp_utc=None,
            )

        if self.behavior == "timeout":
            raise ProviderTimeoutError("Request timed out.")

        if self.behavior == "generic_error":
            raise RuntimeError("Unexpected provider failure.")

        if self.behavior == "invalid_response":
            return ModelResponse(
                provider=request.provider,
                model_name=request.model_name,
                status=SUCCESS_STATUS,
                latency_seconds=0.0,
                metadata=request.metadata,
                raw_text="",
                finish_reason="stop",
                usage=None,
                error=None,
                timestamp_utc=None,
            )

        if self.behavior == "failure_response":
            return ModelResponse(
                provider=request.provider,
                model_name=request.model_name,
                status=FAILURE_STATUS,
                latency_seconds=0.0,
                metadata=request.metadata,
                raw_text=None,
                finish_reason=None,
                usage=None,
                error=ErrorInfo(
                    error_type="ProviderCallError",
                    message="Forced failure response.",
                    retryable=True,
                    stage="provider_call",
                ),
                timestamp_utc=None,
            )

        raise RuntimeError(f"Unknown dummy behavior: {self.behavior}")


@pytest.fixture
def dummy_success_client() -> DummyClient:
    return DummyClient(
        provider="openai",
        model_name="gpt-5-mini",
        behavior="success",
    )


@pytest.fixture
def dummy_timeout_client() -> DummyClient:
    return DummyClient(
        provider="openai",
        model_name="gpt-5-mini",
        behavior="timeout",
    )


@pytest.fixture
def dummy_generic_error_client() -> DummyClient:
    return DummyClient(
        provider="openai",
        model_name="gpt-5-mini",
        behavior="generic_error",
    )


@pytest.fixture
def dummy_invalid_response_client() -> DummyClient:
    return DummyClient(
        provider="openai",
        model_name="gpt-5-mini",
        behavior="invalid_response",
    )


@pytest.fixture
def dummy_failure_response_client() -> DummyClient:
    return DummyClient(
        provider="openai",
        model_name="gpt-5-mini",
        behavior="failure_response",
    )


@pytest.fixture
def dummy_mismatch_client() -> DummyClient:
    return DummyClient(
        provider="gemini",
        model_name="gemini-2.5-flash",
        behavior="success",
    )


@pytest.mark.asyncio
async def test_generate_returns_valid_success_response_for_valid_request(
    dummy_success_client,
    valid_request,
):
    response = await dummy_success_client.generate(valid_request)

    assert response.is_success()
    assert response.error is None
    assert response.model_name == valid_request.model_name
    assert response.provider == valid_request.provider
    assert response.latency_seconds >= 0
    response.metadata.validate()


@pytest.mark.asyncio
async def test_generate_returns_failed_response_for_provider_model_mismatch(
    dummy_mismatch_client,
    valid_request,
):
    response = await dummy_mismatch_client.generate(valid_request)

    assert not response.is_success()
    assert response.status == FAILURE_STATUS
    assert response.error is not None
    assert response.error.error_type == "ProviderConfigurationError"
    assert response.error.stage == "request_compatibility"
    assert response.error.retryable is False
    assert response.latency_seconds >= 0
    response.metadata.validate()


@pytest.mark.asyncio
async def test_generate_returns_failed_response_for_provider_timeout(
    dummy_timeout_client,
    valid_request,
):
    response = await dummy_timeout_client.generate(valid_request)

    assert not response.is_success()
    assert response.status == FAILURE_STATUS
    assert response.error is not None
    assert response.error.error_type == "ProviderTimeoutError"
    assert response.error.stage == "provider_call"
    assert response.error.retryable is True
    assert response.latency_seconds >= 0
    response.metadata.validate()


@pytest.mark.asyncio
async def test_generate_returns_failed_response_for_unknown_generic_exception(
    dummy_generic_error_client,
    valid_request,
):
    response = await dummy_generic_error_client.generate(valid_request)

    assert not response.is_success()
    assert response.status == FAILURE_STATUS
    assert response.error is not None
    assert response.error.error_type == "RuntimeError"
    assert response.error.stage == "provider_call"
    assert response.error.retryable is False
    assert response.latency_seconds >= 0
    response.metadata.validate()



@pytest.mark.asyncio
async def test_generate_returns_failed_response_for_invalid_provider_response(
    dummy_invalid_response_client,
    valid_request,
):
    response = await dummy_invalid_response_client.generate(valid_request)
    assert not response.is_success()
    assert response.status == FAILURE_STATUS
    assert response.error is not None
    assert response.latency_seconds >= 0
    response.metadata.validate()
    assert response.error.error_type == "ResponseValidationError"
    assert response.error.stage == "response_validation"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_generate_batch_returns_one_response_per_request(
    dummy_success_client,
    valid_request,
):
    requests = [
        ModelRequest(
            provider=valid_request.provider,
            model_name=valid_request.model_name,
            payload="Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
            metadata=RequestMetadata(
                question_id="q_001",
                split_name=valid_request.metadata.split_name,
                method_name=valid_request.metadata.method_name,
                subject=valid_request.metadata.subject,
                run_id=valid_request.metadata.run_id,
                prompt_version=valid_request.metadata.prompt_version,
                perturbation_name=valid_request.metadata.perturbation_name,
                sample_index=0,
            ),
            temperature=valid_request.temperature,
            max_tokens=valid_request.max_tokens,
            seed=valid_request.seed,
        ),
        ModelRequest(
            provider=valid_request.provider,
            model_name=valid_request.model_name,
            payload="Question: What is 3 + 3?\nA. 5\nB. 6\nC. 7\nD. 8",
            metadata=RequestMetadata(
                question_id="q_002",
                split_name=valid_request.metadata.split_name,
                method_name=valid_request.metadata.method_name,
                subject=valid_request.metadata.subject,
                run_id=valid_request.metadata.run_id,
                prompt_version=valid_request.metadata.prompt_version,
                perturbation_name=valid_request.metadata.perturbation_name,
                sample_index=1,
            ),
            temperature=valid_request.temperature,
            max_tokens=valid_request.max_tokens,
            seed=valid_request.seed,
        ),
        ModelRequest(
            provider=valid_request.provider,
            model_name=valid_request.model_name,
            payload="Question: What is 4 + 4?\nA. 7\nB. 8\nC. 9\nD. 10",
            metadata=RequestMetadata(
                question_id="q_003",
                split_name=valid_request.metadata.split_name,
                method_name=valid_request.metadata.method_name,
                subject=valid_request.metadata.subject,
                run_id=valid_request.metadata.run_id,
                prompt_version=valid_request.metadata.prompt_version,
                perturbation_name=valid_request.metadata.perturbation_name,
                sample_index=2,
            ),
            temperature=valid_request.temperature,
            max_tokens=valid_request.max_tokens,
            seed=valid_request.seed,
        ),
    ]
    responses = await dummy_success_client.generate_batch(requests)
    assert len(responses) == len(requests)
    for response in responses:
        assert response.is_success()
        assert response.error is None
        assert response.latency_seconds >= 0
        response.metadata.validate()




@pytest.mark.asyncio
async def test_generate_batch_preserves_input_order(
    dummy_success_client,
    valid_request,
):
    requests = [
        ModelRequest(
            provider=valid_request.provider,
            model_name=valid_request.model_name,
            payload="Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
            metadata=RequestMetadata(
                question_id="q_001",
                split_name=valid_request.metadata.split_name,
                method_name=valid_request.metadata.method_name,
                subject=valid_request.metadata.subject,
                run_id=valid_request.metadata.run_id,
                prompt_version=valid_request.metadata.prompt_version,
                perturbation_name=valid_request.metadata.perturbation_name,
                sample_index=0,
            ),
            temperature=valid_request.temperature,
            max_tokens=valid_request.max_tokens,
            seed=valid_request.seed,
        ),
        ModelRequest(
            provider=valid_request.provider,
            model_name=valid_request.model_name,
            payload="Question: What is 3 + 3?\nA. 5\nB. 6\nC. 7\nD. 8",
            metadata=RequestMetadata(
                question_id="q_002",
                split_name=valid_request.metadata.split_name,
                method_name=valid_request.metadata.method_name,
                subject=valid_request.metadata.subject,
                run_id=valid_request.metadata.run_id,
                prompt_version=valid_request.metadata.prompt_version,
                perturbation_name=valid_request.metadata.perturbation_name,
                sample_index=1,
            ),
            temperature=valid_request.temperature,
            max_tokens=valid_request.max_tokens,
            seed=valid_request.seed,
        ),
        ModelRequest(
            provider=valid_request.provider,
            model_name=valid_request.model_name,
            payload="Question: What is 4 + 4?\nA. 7\nB. 8\nC. 9\nD. 10",
            metadata=RequestMetadata(
                question_id="q_003",
                split_name=valid_request.metadata.split_name,
                method_name=valid_request.metadata.method_name,
                subject=valid_request.metadata.subject,
                run_id=valid_request.metadata.run_id,
                prompt_version=valid_request.metadata.prompt_version,
                perturbation_name=valid_request.metadata.perturbation_name,
                sample_index=2,
            ),
            temperature=valid_request.temperature,
            max_tokens=valid_request.max_tokens,
            seed=valid_request.seed,
        ),
    ]
    responses = await dummy_success_client.generate_batch(requests)
    input_question_ids = [request.metadata.question_id for request in requests]
    output_question_ids = [response.metadata.question_id for response in responses]

    assert output_question_ids == input_question_ids

def test_build_failure_response_returns_valid_failure_model_response(
    dummy_success_client,
    valid_request,
):
    error = ErrorInfo(
        error_type="ProviderTimeoutError",
        message="Request timed out.",
        retryable=True,
        stage="provider_call",
    )

    response = dummy_success_client._build_failure_response(
        request=valid_request,
        error=error,
        latency_seconds=0.5,
    )

    assert response.status == FAILURE_STATUS
    assert not response.is_success()
    assert response.error is error
    assert response.provider == valid_request.provider
    assert response.model_name == valid_request.model_name
    assert response.metadata == valid_request.metadata
    assert response.raw_text is None
    assert response.finish_reason is None
    assert response.usage is None
    assert response.latency_seconds == 0.5
    response.validate()


@pytest.mark.parametrize(
    "exc,stage,expected_error_type,expected_retryable",
    [
        (ValidationError("bad request"), "request_validation", "ValidationError", False),
        (ProviderTimeoutError("timed out"), "provider_call", "ProviderTimeoutError", True),
        (
            ProviderConfigurationError("mismatch"),
            "request_compatibility",
            "ProviderConfigurationError",
            False,
        ),
        (RuntimeError("boom"), "provider_call", "RuntimeError", False),
        (ValueError("bad value"), "response_validation", "ValueError", False),
    ],
)
def test_normalize_exception_maps_exceptions_correctly(
    dummy_success_client,
    exc,
    stage,
    expected_error_type,
    expected_retryable,
):
    error = dummy_success_client._normalize_exception(exc, stage)

    assert error.error_type == expected_error_type
    assert error.message == str(exc)
    assert error.retryable is expected_retryable
    assert error.stage == stage

@pytest.mark.parametrize(
    "exc,stage,expected_error_type,expected_retryable",
    [
        (RuntimeError("boom"), "provider_call", "RuntimeError", False),
        (ValueError("bad value"), "response_validation", "ValueError", False),
    ],
)
def test_normalize_exception_falls_back_for_unknown_exceptions(
    dummy_success_client,
    exc,
    stage,
    expected_error_type,
    expected_retryable,
):
    error = dummy_success_client._normalize_exception(exc, stage)

    assert error.error_type == expected_error_type
    assert error.message == str(exc)
    assert error.retryable is expected_retryable
    assert error.stage == stage