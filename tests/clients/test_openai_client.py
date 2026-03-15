import httpx
import pytest
import openai

from types import SimpleNamespace
from unittest.mock import AsyncMock

from twoprompt.clients.openai_client import OpenAIClient
from twoprompt.clients.types import (
    ModelRequest,
    RequestMetadata,
    ProviderResponseError,
    ProviderCallError,
    ProviderConfigurationError,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def openai_client() -> OpenAIClient:
    return OpenAIClient(
        model_name="gpt-4o-mini",
        api_key="test-key",
    )


@pytest.fixture
def request_metadata() -> RequestMetadata:
    return RequestMetadata(
        question_id="q1",
        split_name="robustness",
        method_name="direct",
        subject="anatomy",
        run_id="run_001",
        prompt_version="v1",
        perturbation_name=None,
        sample_index=0,
    )


@pytest.fixture
def model_request(request_metadata: RequestMetadata) -> ModelRequest:
    return ModelRequest(
        provider="openai",
        model_name="gpt-4o-mini",
        payload="test prompt",
        temperature=0.2,
        max_tokens=128,
        metadata=request_metadata,
    )


@pytest.fixture
def make_openai_response():
    def _make_response(
        *,
        text: str | None = "hello",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        total_tokens: int = 15,
        include_usage: bool = True,
    ):
        usage = None
        if include_usage:
            usage = SimpleNamespace(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        return SimpleNamespace(
            output_text=text,
            usage=usage,
        )

    return _make_response


@pytest.fixture
def mock_create(openai_client: OpenAIClient) -> AsyncMock:
    mock = AsyncMock()
    openai_client.client.responses.create = mock
    return mock


async def test_generate_provider_response_returns_model_response_on_success(
    openai_client: OpenAIClient,
    model_request: ModelRequest,
    make_openai_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_openai_response()
    mock_create.return_value = fake_response

    response = await openai_client._generate_provider_response(model_request)

    assert response.raw_text == "hello"
    assert response.provider == "openai"
    assert response.model_name == model_request.model_name
    assert response.finish_reason is None
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


async def test_generate_provider_response_raises_provider_response_error_when_text_is_none(
    openai_client: OpenAIClient,
    model_request: ModelRequest,
    make_openai_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_openai_response(text=None)
    mock_create.return_value = fake_response

    with pytest.raises(ProviderResponseError):
        await openai_client._generate_provider_response(model_request)


async def test_generate_provider_response_raises_provider_response_error_when_text_is_whitespace(
    openai_client: OpenAIClient,
    model_request: ModelRequest,
    make_openai_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_openai_response(text="   ")
    mock_create.return_value = fake_response

    with pytest.raises(ProviderResponseError):
        await openai_client._generate_provider_response(model_request)


async def test_generate_provider_response_allows_missing_usage(
    openai_client: OpenAIClient,
    model_request: ModelRequest,
    make_openai_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_openai_response(include_usage=False)
    mock_create.return_value = fake_response

    response = await openai_client._generate_provider_response(model_request)

    assert response.raw_text == "hello"
    assert response.usage is None
    assert response.finish_reason is None


async def test_generate_provider_response_raises_provider_configuration_error_for_404(
    openai_client: OpenAIClient,
    model_request: ModelRequest,
    mock_create: AsyncMock,
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    http_response = httpx.Response(404, request=request)

    mock_create.side_effect = openai.NotFoundError(
        "not found",
        response=http_response,
        body={"message": "not found"},
    )

    with pytest.raises(ProviderConfigurationError):
        await openai_client._generate_provider_response(model_request)


async def test_generate_provider_response_raises_provider_call_error_for_500(
    openai_client: OpenAIClient,
    model_request: ModelRequest,
    mock_create: AsyncMock,
) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    http_response = httpx.Response(500, request=request)

    mock_create.side_effect = openai.InternalServerError(
        "server error",
        response=http_response,
        body={"message": "server error"},
    )

    with pytest.raises(ProviderCallError):
        await openai_client._generate_provider_response(model_request)