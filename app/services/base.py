"""
Base utilities and shared components for SELVE services.
Provides common error handling, validation, logging, and client management.
"""
import os
import hashlib
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

T = TypeVar("T")


# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Centralized configuration with validation and defaults."""

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = 1536

    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")  # For cloud deployments
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "selve_knowledge")

    # YouTube Transcript API (youtube-transcript.io)
    YOUTUBE_TRANSCRIPT_API_TOKEN: str = os.getenv("YOUTUBE_TRANSCRIPT_API_TOKEN", "")
    YOUTUBE_LIVE_FETCH_ENABLED: bool = os.getenv("YOUTUBE_LIVE_FETCH_ENABLED", "false").lower() == "true"

    # Timeouts
    RAG_TIMEOUT_SECONDS: float = float(os.getenv("RAG_TIMEOUT_SECONDS", "5.0"))
    MEMORY_TIMEOUT_SECONDS: float = float(os.getenv("MEMORY_TIMEOUT_SECONDS", "3.0"))
    EMBEDDING_TIMEOUT_SECONDS: float = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "10.0"))

    # Limits
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "10000"))
    MAX_CHUNKS_PER_QUERY: int = int(os.getenv("MAX_CHUNKS_PER_QUERY", "10"))
    MAX_EMBEDDING_BATCH_SIZE: int = int(os.getenv("MAX_EMBEDDING_BATCH_SIZE", "100"))

    # Costs (per 1M tokens)
    EMBEDDING_COST_PER_1M_TOKENS: float = 0.020

    @classmethod
    def validate(cls) -> List[str]:
        """Validate required configuration. Returns list of missing/invalid configs."""
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        return errors


# =============================================================================
# RESULT TYPES
# =============================================================================


class ResultStatus(Enum):
    """Standard result statuses."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"


@dataclass
class Result(Generic[T]):
    """
    Standard result wrapper for all service operations.
    Provides consistent error handling and metadata.
    """

    status: ResultStatus
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None

    @property
    def is_success(self) -> bool:
        return self.status == ResultStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        return self.status in (
            ResultStatus.ERROR,
            ResultStatus.TIMEOUT,
            ResultStatus.VALIDATION_ERROR,
        )

    @classmethod
    def success(cls, data: T, **metadata) -> "Result[T]":
        return cls(status=ResultStatus.SUCCESS, data=data, metadata=metadata)

    @classmethod
    def failure(
        cls,
        error: str,
        error_code: Optional[str] = None,
        status: ResultStatus = ResultStatus.ERROR,
    ) -> "Result[T]":
        return cls(status=status, error=error, error_code=error_code)

    @classmethod
    def timeout(cls, operation: str) -> "Result[T]":
        return cls(
            status=ResultStatus.TIMEOUT,
            error=f"Operation timed out: {operation}",
            error_code="TIMEOUT",
        )

    @classmethod
    def validation_error(cls, message: str) -> "Result[T]":
        return cls(
            status=ResultStatus.VALIDATION_ERROR,
            error=message,
            error_code="VALIDATION_ERROR",
        )


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ServiceError(Exception):
    """Base exception for service errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SERVICE_ERROR"
        self.details = details or {}


class ValidationError(ServiceError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, error_code="VALIDATION_ERROR", details={"field": field})
        self.field = field


class ConfigurationError(ServiceError):
    """Raised when service configuration is invalid."""

    def __init__(self, message: str, missing_configs: Optional[List[str]] = None):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={"missing_configs": missing_configs or []},
        )


class ExternalServiceError(ServiceError):
    """Raised when an external service (OpenAI, Qdrant) fails."""

    def __init__(self, service: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"{service} error: {message}",
            error_code=f"{service.upper()}_ERROR",
            details={"original_error": str(original_error) if original_error else None},
        )
        self.service = service
        self.original_error = original_error


# =============================================================================
# VALIDATION
# =============================================================================


class Validator:
    """Input validation utilities."""

    @staticmethod
    def validate_string(
        value: Any,
        field_name: str,
        min_length: int = 1,
        max_length: int = 10000,
        allow_none: bool = False,
    ) -> Optional[str]:
        """Validate and sanitize a string input."""
        if value is None:
            if allow_none:
                return None
            raise ValidationError(f"{field_name} is required", field=field_name)

        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name)

        # Strip whitespace
        value = value.strip()

        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field=field_name,
            )

        if len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be at most {max_length} characters",
                field=field_name,
            )

        return value

    @staticmethod
    def validate_positive_int(
        value: Any,
        field_name: str,
        min_value: int = 1,
        max_value: int = 1000,
        default: Optional[int] = None,
    ) -> int:
        """Validate a positive integer."""
        if value is None:
            if default is not None:
                return default
            raise ValidationError(f"{field_name} is required", field=field_name)

        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be an integer", field=field_name)

        if int_value < min_value or int_value > max_value:
            raise ValidationError(
                f"{field_name} must be between {min_value} and {max_value}",
                field=field_name,
            )

        return int_value

    @staticmethod
    def validate_float_range(
        value: Any,
        field_name: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        default: Optional[float] = None,
    ) -> float:
        """Validate a float within a range."""
        if value is None:
            if default is not None:
                return default
            raise ValidationError(f"{field_name} is required", field=field_name)

        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be a number", field=field_name)

        if float_value < min_value or float_value > max_value:
            raise ValidationError(
                f"{field_name} must be between {min_value} and {max_value}",
                field=field_name,
            )

        return float_value

    @staticmethod
    def validate_user_id(user_id: Any, field_name: str = "user_id") -> str:
        """Validate a user ID (CUID or Clerk ID format)."""
        if not user_id:
            raise ValidationError(f"{field_name} is required", field=field_name)

        if not isinstance(user_id, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name)

        user_id = user_id.strip()

        # Basic format validation (alphanumeric, underscores, hyphens)
        if not all(c.isalnum() or c in "_-" for c in user_id):
            raise ValidationError(
                f"{field_name} contains invalid characters",
                field=field_name,
            )

        if len(user_id) < 5 or len(user_id) > 100:
            raise ValidationError(
                f"{field_name} must be between 5 and 100 characters",
                field=field_name,
            )

        return user_id


# =============================================================================
# UTILITIES
# =============================================================================


def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash for content deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def string_to_point_id(string_id: str) -> int:
    """
    Convert a string ID to an integer point ID for Qdrant.
    Uses first 8 bytes of SHA-256 hash.
    """
    hash_bytes = hashlib.sha256(string_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big")


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Safely truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_json_parse(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback."""
    import json

    try:
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return default


# =============================================================================
# DECORATORS
# =============================================================================


def with_timeout(timeout_seconds: float, operation_name: str = "operation"):
    """
    Decorator to add timeout to async functions.
    Returns Result.timeout() on timeout instead of raising.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger = logging.getLogger(func.__module__)
                logger.warning(f"Timeout in {operation_name} after {timeout_seconds}s")
                return Result.timeout(operation_name)

        return wrapper

    return decorator


def with_error_handling(logger_name: Optional[str] = None):
    """
    Decorator for consistent error handling.
    Catches exceptions and returns Result objects.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Result:
            logger = logging.getLogger(logger_name or func.__module__)
            try:
                return await func(*args, **kwargs)
            except ValidationError as e:
                logger.warning(f"Validation error in {func.__name__}: {e.message}")
                return Result.validation_error(e.message)
            except ExternalServiceError as e:
                logger.error(f"External service error in {func.__name__}: {e.message}")
                return Result.failure(e.message, error_code=e.error_code)
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}")
                return Result.failure(str(e), error_code="UNEXPECTED_ERROR")

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Result:
            logger = logging.getLogger(logger_name or func.__module__)
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                logger.warning(f"Validation error in {func.__name__}: {e.message}")
                return Result.validation_error(e.message)
            except ExternalServiceError as e:
                logger.error(f"External service error in {func.__name__}: {e.message}")
                return Result.failure(e.message, error_code=e.error_code)
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}")
                return Result.failure(str(e), error_code="UNEXPECTED_ERROR")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# CLIENT MANAGEMENT
# =============================================================================


class ClientManager:
    """
    Manages external service clients with lazy initialization,
    connection pooling, and health checks.
    """

    _openai_client: Optional[Any] = None
    _qdrant_client: Optional[Any] = None
    _initialized: bool = False

    @classmethod
    def get_openai_client(cls) -> Any:
        """Get or create OpenAI client."""
        if cls._openai_client is None:
            if not Config.OPENAI_API_KEY:
                raise ConfigurationError(
                    "OpenAI API key not configured",
                    missing_configs=["OPENAI_API_KEY"],
                )
            from openai import OpenAI

            cls._openai_client = OpenAI(
                api_key=Config.OPENAI_API_KEY,
                timeout=Config.EMBEDDING_TIMEOUT_SECONDS,
            )
        return cls._openai_client

    @classmethod
    def get_qdrant_client(cls) -> Any:
        """Get or create Qdrant client."""
        if cls._qdrant_client is None:
            from qdrant_client import QdrantClient

            if Config.QDRANT_URL:
                cls._qdrant_client = QdrantClient(
                    url=Config.QDRANT_URL,
                    api_key=Config.QDRANT_API_KEY if Config.QDRANT_API_KEY else None,
                )
            else:
                cls._qdrant_client = QdrantClient(
                    host=Config.QDRANT_HOST,
                    port=Config.QDRANT_PORT,
                )
        return cls._qdrant_client

    @classmethod
    def health_check(cls) -> Dict[str, bool]:
        """Check health of all external services."""
        health = {"openai": False, "qdrant": False}

        try:
            client = cls.get_openai_client()
            client.models.list()
            health["openai"] = True
        except Exception:
            pass

        try:
            client = cls.get_qdrant_client()
            client.get_collections()
            health["qdrant"] = True
        except Exception:
            pass

        return health

    @classmethod
    def reset(cls) -> None:
        """Reset all clients (useful for testing)."""
        cls._openai_client = None
        cls._qdrant_client = None
        cls._initialized = False


# =============================================================================
# BASE SERVICE
# =============================================================================


class BaseService(ABC):
    """
    Abstract base class for all SELVE services.
    Provides common functionality and enforces consistent patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate service-specific configuration. Override in subclasses."""
        pass

    @property
    def openai(self) -> Any:
        """Lazy-loaded OpenAI client."""
        return ClientManager.get_openai_client()

    @property
    def qdrant(self) -> Any:
        """Lazy-loaded Qdrant client."""
        return ClientManager.get_qdrant_client()
