"""
Rate limiting and circuit breaker infrastructure.

Features:
- Redis-based distributed rate limiting with sliding window
- In-memory fallback for single instance or Redis failures
- Circuit breaker for external service resilience
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from .core import CrawlerConfig

logger = logging.getLogger(__name__)


# =============================================================================
# RATE LIMITING
# =============================================================================


class RateLimiter(ABC):
    """Abstract rate limiter interface."""

    @abstractmethod
    async def acquire(self, key: str) -> bool:
        """Try to acquire a rate limit slot."""

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""

    async def close(self) -> None:  # pragma: no cover - default no-op
        """Cleanup resources."""


class RedisRateLimiter(RateLimiter):
    """
    Redis-based distributed rate limiter using sliding window.

    Uses Redis sorted sets for efficient sliding window implementation.
    Falls back to in-memory limiting on Redis failure.
    """

    def __init__(
        self,
        redis_url: str,
        requests_per_minute: int,
        prefix: str = "ratelimit",
    ):
        self.redis_url = redis_url
        self.requests_per_minute = requests_per_minute
        self.prefix = prefix
        self.window_seconds = 60
        self._redis: Optional[Any] = None
        self._fallback = InMemoryRateLimiter(requests_per_minute)
        self._connection_failed = False

    async def _get_redis(self):
        """Get Redis connection with lazy initialization."""
        if self._connection_failed:
            return None

        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = await aioredis.from_url(
                    self.redis_url,
                    socket_timeout=CrawlerConfig.REDIS_TIMEOUT,
                    socket_connect_timeout=CrawlerConfig.REDIS_TIMEOUT,
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("Redis rate limiter connected")
            except Exception as e:  # pragma: no cover - runtime safeguard
                logger.warning(f"Redis connection failed: {e}")
                self._connection_failed = True
                self._redis = None
                return None

        return self._redis

    async def acquire(self, key: str) -> bool:
        """Try to acquire a rate limit slot using sliding window."""
        redis = await self._get_redis()
        if redis is None:
            return await self._fallback.acquire(key)

        full_key = f"{self.prefix}:{key}"
        now = time.time()
        window_start = now - self.window_seconds

        try:
            async with redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(full_key, 0, window_start)
                pipe.zcard(full_key)
                member = f"{now}:{id(asyncio.current_task())}"
                pipe.zadd(full_key, {member: now})
                pipe.expire(full_key, self.window_seconds + 1)

                results = await pipe.execute()
                current_count = results[1]

                if current_count >= self.requests_per_minute:
                    await redis.zrem(full_key, member)
                    logger.debug(f"Rate limit exceeded for {key}")
                    return False

                return True

        except Exception as e:  # pragma: no cover - runtime safeguard
            logger.warning(f"Redis error, using fallback: {e}")
            return await self._fallback.acquire(key)

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        redis = await self._get_redis()
        if redis:
            try:
                await redis.delete(f"{self.prefix}:{key}")
            except Exception:
                pass
        await self._fallback.reset(key)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None


class InMemoryRateLimiter(RateLimiter):
    """
    In-memory rate limiter using sliding window.

    Thread-safe using asyncio locks.
    """

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str) -> bool:
        """Try to acquire a rate limit slot."""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            if key not in self._requests:
                self._requests[key] = []

            self._requests[key] = [t for t in self._requests[key] if t > window_start]

            if len(self._requests[key]) >= self.requests_per_minute:
                return False

            self._requests[key].append(now)
            return True

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            self._requests.pop(key, None)


def create_rate_limiter(
    requests_per_minute: int,
    prefix: str,
) -> RateLimiter:
    """Factory function to create appropriate rate limiter."""
    if CrawlerConfig.REDIS_ENABLED:
        return RedisRateLimiter(
            redis_url=CrawlerConfig.REDIS_URL,
            requests_per_minute=requests_per_minute,
            prefix=prefix,
        )
    return InMemoryRateLimiter(requests_per_minute)


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""


class CircuitBreaker:
    """
    Circuit breaker for external service resilience.

    Prevents cascading failures by failing fast when a service is down.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = CrawlerConfig.CB_THRESHOLD,
        recovery_timeout: int = CrawlerConfig.CB_TIMEOUT,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    async def can_execute(self) -> bool:
        """Check if request can proceed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        logger.info(f"Circuit breaker {self.name} entering half-open state")
                        return True
                return False

            return True

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker {self.name} recovered, closing")
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} re-opened after test failure")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} opened after {self._failure_count} failures"
                )

    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(f"Circuit breaker {self.name} manually reset")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure": self._last_failure_time,
        }


def circuit_protected(circuit_breaker: CircuitBreaker):
    """
    Decorator for circuit breaker protection.

    Usage:
        @circuit_protected(my_circuit_breaker)
        async def fetch_data():
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not await circuit_breaker.can_execute():
                raise CircuitBreakerOpen(
                    f"Circuit breaker {circuit_breaker.name} is open"
                )

            try:
                result = await func(*args, **kwargs)
                await circuit_breaker.record_success()
                return result
            except CircuitBreakerOpen:
                raise
            except Exception:
                await circuit_breaker.record_failure()
                raise

        return wrapper

    return decorator
