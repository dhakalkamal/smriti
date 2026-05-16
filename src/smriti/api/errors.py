from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from smriti.assistant import (
    AssistantGenerationFailedError,
    AssistantGenerationUnavailableError,
    InvalidAssistantRequestError,
)
from smriti.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)
from smriti.memory import (
    ConversationNotFoundError,
    EmbeddingModelNotFoundError,
    InvalidMemoryRequestError,
    InvalidProvenanceTargetError,
    InvalidRetrievalRequestError,
    MemoryAccessDeniedError,
    MemoryServiceError,
    ScopeNotFoundError,
    VectorDimensionError,
)


def register_error_handlers(app: FastAPI) -> None:
    """Register API error mappings for thin route adapters."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_400_BAD_REQUEST, "Invalid request")

    @app.exception_handler(InvalidAssistantRequestError)
    async def invalid_assistant_request_handler(
        request: Request,
        exc: InvalidAssistantRequestError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_400_BAD_REQUEST, "Invalid assistant request")

    @app.exception_handler(AssistantGenerationUnavailableError)
    async def assistant_generation_unavailable_handler(
        request: Request,
        exc: AssistantGenerationUnavailableError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Local assistant generation unavailable",
        )

    @app.exception_handler(AssistantGenerationFailedError)
    async def assistant_generation_failed_handler(
        request: Request,
        exc: AssistantGenerationFailedError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Assistant generation failed",
        )

    @app.exception_handler(InvalidMemoryRequestError)
    async def invalid_memory_request_handler(
        request: Request,
        exc: InvalidMemoryRequestError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_400_BAD_REQUEST, "Invalid memory request")

    @app.exception_handler(InvalidRetrievalRequestError)
    async def invalid_retrieval_request_handler(
        request: Request,
        exc: InvalidRetrievalRequestError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_400_BAD_REQUEST, "Invalid retrieval request")

    @app.exception_handler(InvalidProvenanceTargetError)
    async def invalid_provenance_target_handler(
        request: Request,
        exc: InvalidProvenanceTargetError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_400_BAD_REQUEST, "Invalid provenance target")

    @app.exception_handler(MemoryAccessDeniedError)
    async def memory_access_denied_handler(
        request: Request,
        exc: MemoryAccessDeniedError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_403_FORBIDDEN, "Memory resource access denied")

    @app.exception_handler(ScopeNotFoundError)
    async def scope_not_found_handler(
        request: Request,
        exc: ScopeNotFoundError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_404_NOT_FOUND, "Scope not found")

    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(
        request: Request,
        exc: ConversationNotFoundError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_404_NOT_FOUND, "Conversation not found")

    @app.exception_handler(EmbeddingConnectionError)
    async def embedding_connection_error_handler(
        request: Request,
        exc: EmbeddingConnectionError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_503_SERVICE_UNAVAILABLE, "Local embedder unavailable")

    @app.exception_handler(EmbeddingTimeoutError)
    async def embedding_timeout_error_handler(
        request: Request,
        exc: EmbeddingTimeoutError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_503_SERVICE_UNAVAILABLE, "Local embedder timed out")

    @app.exception_handler(EmbeddingConfigurationError)
    async def embedding_configuration_error_handler(
        request: Request,
        exc: EmbeddingConfigurationError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Embedder configuration error")

    @app.exception_handler(EmbeddingResponseError)
    async def embedding_response_error_handler(
        request: Request,
        exc: EmbeddingResponseError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_503_SERVICE_UNAVAILABLE, "Invalid embedder response")

    @app.exception_handler(EmbeddingModelNotFoundError)
    async def embedding_model_not_found_handler(
        request: Request,
        exc: EmbeddingModelNotFoundError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Embedding model not configured")

    @app.exception_handler(VectorDimensionError)
    async def vector_dimension_error_handler(
        request: Request,
        exc: VectorDimensionError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Embedding dimension mismatch")

    @app.exception_handler(MemoryServiceError)
    async def memory_service_error_handler(
        request: Request,
        exc: MemoryServiceError,
    ) -> JSONResponse:
        _ = (request, exc)
        return _json_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Memory service error")


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})
