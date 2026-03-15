import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from google.genai import errors

from twoprompt.clients.gemini_client import GeminiClient
from twoprompt.clients.types import (
    ModelRequest,
    RequestMetadata,
    ProviderResponseError,
    ProviderCallError,
    ProviderConfigurationError,
)

@pytest.fixture
def gemini_client() -> GeminiClient:
    return GeminiClient(model_name="gemini-2.0-flash")


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
        provider="gemini",
        model_name="gemini-2.0-flash",
        payload="test prompt",
        temperature=0.2,
        max_tokens=128,
        metadata=request_metadata,
    )

@pytest.fixture
def make_gemini_response():
    def _make_response(
        *,
        text="hello",
        finish_reason="STOP",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        include_candidates=True,
        include_usage=True,
    ):
        candidates = None
        if include_candidates:
            candidates = [
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name=finish_reason)
                )
            ]

        usage_metadata = None
        if include_usage:
            usage_metadata = SimpleNamespace(
                prompt_token_count=prompt_tokens,
                candidates_token_count=completion_tokens,
                total_token_count=total_tokens,
            )

        return SimpleNamespace(
            text=text,
            candidates=candidates,
            usage_metadata=usage_metadata,
        )

    return _make_response


@pytest.fixture
def mock_generate_content(gemini_client: GeminiClient) -> AsyncMock:
    mock = AsyncMock()
    gemini_client.client.aio.models.generate_content = mock
    return mock

pytestmark = pytest.mark.asyncio

async def test_generate_provider_response_returns_model_response_on_success(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    make_gemini_response,
    mock_generate_content: AsyncMock,
) -> None:
    fake_response = make_gemini_response()
    mock_generate_content.return_value = fake_response

    response = await gemini_client._generate_provider_response(model_request)

    assert response.raw_text == "hello"
    assert response.provider == "gemini"
    assert response.model_name == model_request.model_name


async def test_generate_provider_response_raises_provider_response_error_when_text_is_none(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    make_gemini_response,
    mock_generate_content: AsyncMock,
) -> None:
    fake_response = make_gemini_response()
    fake_response.text = None
    mock_generate_content.return_value = fake_response

    with pytest.raises(ProviderResponseError):
        await gemini_client._generate_provider_response(model_request)


async def test_generate_provider_response_raises_provider_response_error_when_text_is_whitespace(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    make_gemini_response,
    mock_generate_content: AsyncMock,
) -> None:
    fake_response = make_gemini_response()
    fake_response.text = " "
    mock_generate_content.return_value = fake_response

    with pytest.raises(ProviderResponseError):
        await gemini_client._generate_provider_response(model_request)


async def test_generate_provider_response_allows_missing_candidates(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    make_gemini_response,
    mock_generate_content: AsyncMock,
) -> None:
    fake_response = make_gemini_response()
    fake_response.candidates = None
    mock_generate_content.return_value = fake_response
    response = await gemini_client._generate_provider_response(model_request)

    assert response.raw_text == "hello"
    assert response.provider == "gemini"
    assert response.finish_reason is None



async def test_generate_provider_response_allows_missing_usage_metadata(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    make_gemini_response,
    mock_generate_content: AsyncMock,
) -> None:
    fake_response = make_gemini_response()
    fake_response.usage_metadata = None
    mock_generate_content.return_value = fake_response
    response = await gemini_client._generate_provider_response(model_request)

    assert response.raw_text == "hello"
    assert response.provider == "gemini"
    assert response.usage is None
    assert response.finish_reason == "STOP"


async def test_generate_provider_response_falls_back_to_total_minus_prompt_for_completion_tokens(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    make_gemini_response,
    mock_generate_content: AsyncMock,
) -> None:
    fake_response = make_gemini_response()

    fake_response.usage_metadata.candidates_token_count = None
    fake_response.usage_metadata.response_token_count = None
    fake_response.usage_metadata.prompt_token_count = 10
    fake_response.usage_metadata.total_token_count = 15

    mock_generate_content.return_value = fake_response

    response = await gemini_client._generate_provider_response(model_request)

    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15



async def test_generate_provider_response_raises_provider_configuration_error_for_404(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    mock_generate_content: AsyncMock,
) -> None:

    mock_generate_content.side_effect = errors.ClientError(
        404,
        {"message": "not found"},
        None,
    )

    with pytest.raises(ProviderConfigurationError):
        await gemini_client._generate_provider_response(model_request)


async def test_generate_provider_response_raises_provider_call_error_for_500(
    gemini_client: GeminiClient,
    model_request: ModelRequest,
    mock_generate_content: AsyncMock,
) -> None:
    mock_generate_content.side_effect = errors.ServerError(
        500,
        {"message": "server error"},
        None,
    )

    with pytest.raises(ProviderCallError):
        await gemini_client._generate_provider_response(model_request)
