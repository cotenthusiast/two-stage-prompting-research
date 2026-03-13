from __future__ import annotations


import asyncio
import time
from abc import ABC, abstractmethod


from twoprompt.clients.types import (
    ValidationError,
    ProviderCallError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConfigurationError,
    FAILURE_STATUS,
    ErrorInfo,
    ModelRequest,
    ModelResponse,
)


from twoprompt.config.models import MAX_RETRIES, TIMEOUT


class BaseClient(ABC):
    """Abstract base client for provider-backed model calls.

    This class defines the shared async interface and shared client behavior
    used by all provider-specific clients. It is responsible for validating
    request/client compatibility, enforcing bounded concurrency, and
    standardizing the single-request and batch-request APIs.

    Concrete subclasses must implement the provider-specific raw call method.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        timeout: int = TIMEOUT,
        concurrency_limit: int = 10,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """Initialize shared client configuration.

        Args:
            provider: Provider name associated with this client.
            model_name: Exact model name associated with this client.
            timeout: Request timeout in seconds.
            concurrency_limit: Maximum number of in-flight requests allowed
                for this client at once.
            max_retries: Maximum number of retry attempts for provider calls.
        """
        self.provider = provider
        self.model_name = model_name
        self.timeout = timeout
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute one standardized model request.

        This method should validate the request, verify that it matches the
        current client, enforce shared execution behavior, and return one
        standardized response object.

        Args:
            request: Standardized model request to execute.

        Returns:
            A standardized model response.
        """
        stage = "request_validation"
        start_time = time.perf_counter()

        try:
            request.validate()

            stage = "request_compatibility"
            self._validate_request_compatibility(request)

            stage = "provider_call"
            async with self.semaphore:
                response = await self._generate_provider_response(request)

            response.latency_seconds = time.perf_counter() - start_time

            stage = "response_validation"
            response.validate()

            return response

        except Exception as exc:
            error = self._normalize_exception(exc, stage)
            latency_seconds = time.perf_counter() - start_time
            return self._build_failure_response(
                request=request,
                error=error,
                latency_seconds=latency_seconds,
            )



    async def generate_batch(
        self,
        requests: list[ModelRequest],
    ) -> list[ModelResponse]:
        """Execute multiple standardized model requests concurrently.

        This method should run many requests through the shared async path
        while respecting the client's concurrency limit.

        Args:
            requests: Standardized model requests to execute.

        Returns:
            A list of standardized model responses, one per input request.
        """
        coroutines = [self.generate(request) for request in requests]
        responses = await asyncio.gather(*coroutines)
        return list(responses)

    @abstractmethod
    async def _generate_provider_response(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Execute the provider-specific raw model call.

        Subclasses must implement this method to translate a standardized
        request into the provider's API format, perform the actual request,
        and return a standardized successful response or raise a provider-side
        exception.

        Args:
            request: Validated model request for this provider/model pair.

        Returns:
            A standardized model response.
        """
        pass

    def _validate_request_compatibility(self, request: ModelRequest) -> None:
        """Validate that the request matches this client.

        This method should confirm that the incoming request is intended for
        the same provider and model configured on this client.

        Args:
            request: Standardized model request to check.
        """
        if request.provider != self.provider:
            raise ProviderConfigurationError(f"Mismatch between request provider ({request.provider} and client provider ({self.provider}")

        if request.model_name != self.model_name:
            raise ProviderConfigurationError(f"Mismatch between request model ({request.model_name} and client model ({self.model_name}")

    def _build_failure_response(
        self,
        request: ModelRequest,
        error: ErrorInfo,
        latency_seconds: float,
    ) -> ModelResponse:
        """Build a standardized failed response object.

        Args:
            request: Original request that failed.
            error: Standardized error information describing the failure.
            latency_seconds: Observed execution latency before failure.

        Returns:
            A failed standardized model response.
        """
        response = ModelResponse(
            provider=request.provider,
            model_name=request.model_name,
            status=FAILURE_STATUS,
            latency_seconds=latency_seconds,
            metadata=request.metadata,
            error=error,
            raw_text=None,
            finish_reason=None,
            usage=None,
            timestamp_utc=None
        )
        response.validate()
        return response

    def _normalize_exception(
            self,
            exc: Exception,
            stage: str,
    ) -> ErrorInfo:
        """Convert a raw exception into standardized error information.

        Args:
            exc: Raw exception raised during request execution.
            stage: Logical execution stage where the failure occurred.

        Returns:
            Standardized error information.
        """
        if isinstance(exc, ValidationError):
            return ErrorInfo(
                type(exc).__name__,
                str(exc),
                False,
                stage,
            )

        if isinstance(exc, ProviderRateLimitError):
            return ErrorInfo(
                type(exc).__name__,
                str(exc),
                True,
                stage,
            )

        if isinstance(exc, ProviderTimeoutError):
            return ErrorInfo(
                type(exc).__name__,
                str(exc),
                True,
                stage,
            )

        if isinstance(exc, ProviderConfigurationError):
            return ErrorInfo(
                type(exc).__name__,
                str(exc),
                False,
                stage,
            )

        if isinstance(exc, ProviderCallError):
            return ErrorInfo(
                type(exc).__name__,
                str(exc),
                True,
                stage,
            )

        return ErrorInfo(
            type(exc).__name__,
            str(exc) if str(exc) else "Unexpected exception with no message.",
            False,
            stage,
        )