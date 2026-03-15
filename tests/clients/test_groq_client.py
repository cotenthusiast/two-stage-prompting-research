import httpx
import pytest
import groq

from types import SimpleNamespace
from unittest.mock import AsyncMock

from twoprompt.clients.groq_client import GroqClient
from twoprompt.clients.types import (
    ModelRequest,
    RequestMetadata,
    ProviderResponseError,
    ProviderCallError,
    ProviderConfigurationError,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def groq_client() -> GroqClient:
    return GroqClient(
        model_name="llama-3.3-70b-versatile",
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
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        payload="test prompt",
        temperature=0.2,
        max_tokens=128,
        metadata=request_metadata,
    )


@pytest.fixture
def make_groq_response():
    def _make_response(
        *,
        text: str | None = "hello",
        finish_reason: str | None = "stop",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        total_tokens: int = 15,
        include_choices: bool = True,
        include_usage: bool = True,
    ):
        choices = None
        if include_choices:
            choices = [
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason=finish_reason,
                )
            ]

        usage = None
        if include_usage:
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        return SimpleNamespace(
            choices=choices,
            usage=usage,
        )

    return _make_response


@pytest.fixture
def mock_create(groq_client: GroqClient) -> AsyncMock:
    mock = AsyncMock()
    groq_client.client.chat.completions.create = mock
    return mock


async def test_generate_provider_response_returns_model_response_on_success(
    groq_client: GroqClient,
    model_request: ModelRequest,
    make_groq_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_groq_response()
    mock_create.return_value = fake_response

    response = await groq_client._generate_provider_response(model_request)

    assert response.raw_text == "hello"
    assert response.provider == "groq"
    assert response.model_name == model_request.model_name
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


async def test_generate_provider_response_raises_provider_response_error_when_text_is_none(
    groq_client: GroqClient,
    model_request: ModelRequest,
    make_groq_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_groq_response(text=None)
    mock_create.return_value = fake_response

    with pytest.raises(ProviderResponseError):
        await groq_client._generate_provider_response(model_request)


async def test_generate_provider_response_raises_provider_response_error_when_text_is_whitespace(
    groq_client: GroqClient,
    model_request: ModelRequest,
    make_groq_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_groq_response(text="   ")
    mock_create.return_value = fake_response

    with pytest.raises(ProviderResponseError):
        await groq_client._generate_provider_response(model_request)


async def test_generate_provider_response_allows_missing_usage(
    groq_client: GroqClient,
    model_request: ModelRequest,
    make_groq_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_groq_response(include_usage=False)
    mock_create.return_value = fake_response

    response = await groq_client._generate_provider_response(model_request)

    assert response.raw_text == "hello"
    assert response.finish_reason == "stop"
    assert response.usage is None


async def test_generate_provider_response_raises_provider_response_error_when_choices_are_missing(
    groq_client: GroqClient,
    model_request: ModelRequest,
    make_groq_response,
    mock_create: AsyncMock,
) -> None:
    fake_response = make_groq_response(include_choices=False)
    mock_create.return_value = fake_response

    with pytest.raises(ProviderResponseError):
        await groq_client._generate_provider_response(model_request)


async def test_generate_provider_response_raises_provider_configuration_error_for_404(
    groq_client: GroqClient,
    model_request: ModelRequest,
    mock_create: AsyncMock,
) -> None:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    http_response = httpx.Response(404, request=request)

    mock_create.side_effect = groq.NotFoundError(
        "not found",
        response=http_response,
        body={"error": {"message": "not found"}},
    )

    with pytest.raises(ProviderConfigurationError):
        await groq_client._generate_provider_response(model_request)


async def test_generate_provider_response_raises_provider_call_error_for_500(
    groq_client: GroqClient,
    model_request: ModelRequest,
    mock_create: AsyncMock,
) -> None:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    http_response = httpx.Response(500, request=request)

    mock_create.side_effect = groq.InternalServerError(
        "server error",
        response=http_response,
        body={"error": {"message": "server error"}},
    )

    with pytest.raises(ProviderCallError):
        await groq_client._generate_provider_response(model_request)