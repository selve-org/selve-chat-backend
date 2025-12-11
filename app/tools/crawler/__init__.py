"""
Content Crawler Package

Production-grade content crawling for personality-related external content.

Features:
- YouTube transcript extraction
- Reddit discussion search
- Web page fetching
- SSRF protection
- Redis-based rate limiting
- Circuit breaker resilience
- Pydantic validation
- BeautifulSoup HTML parsing

Usage:
    from app.tools.crawler import invoke_tool, list_tools, shutdown

    # Get available tools
    tools = list_tools()

    # Invoke a tool
    result = await invoke_tool("youtube_transcript", video_url="https://...")

    # Cleanup on shutdown
    await shutdown()

With context manager:
    async with crawler_context():
        result = await invoke_tool("reddit_search", query="personality types")
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, ClassVar, Dict, Final, List, Optional, Union

from .core import (
    ContentSource,
    ContentValidator,
    CrawledContent,
    CrawlerConfig,
    DomainAllowlist,
    RedditRequest,
    SSRFProtection,
    WebpageRequest,
    YouTubeRequest,
)
from .http import HTMLParser, HTTPClientManager
from .resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    InMemoryRateLimiter,
    RateLimiter,
    RedisRateLimiter,
    create_rate_limiter,
)
from .tools import RedditTool, WebPageTool, YouTubeTool

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL REGISTRY
# =============================================================================


class ToolRegistry:
    """
    Singleton registry for crawler tools.

    Manages tool lifecycle and provides clean access.
    """

    _youtube: ClassVar[Optional[YouTubeTool]] = None
    _reddit: ClassVar[Optional[RedditTool]] = None
    _webpage: ClassVar[Optional[WebPageTool]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _initialized: ClassVar[bool] = False

    @classmethod
    async def youtube(cls) -> YouTubeTool:
        """Get YouTube tool instance."""
        async with cls._lock:
            if cls._youtube is None:
                cls._youtube = YouTubeTool()
            return cls._youtube

    @classmethod
    async def reddit(cls) -> RedditTool:
        """Get Reddit tool instance."""
        async with cls._lock:
            if cls._reddit is None:
                cls._reddit = RedditTool()
            return cls._reddit

    @classmethod
    async def webpage(cls) -> WebPageTool:
        """Get WebPage tool instance."""
        async with cls._lock:
            if cls._webpage is None:
                cls._webpage = WebPageTool()
            return cls._webpage

    @classmethod
    async def shutdown(cls) -> None:
        """Shutdown all tools and cleanup resources."""
        async with cls._lock:
            if cls._youtube:
                await cls._youtube.close()
                cls._youtube = None

            if cls._reddit:
                await cls._reddit.close()
                cls._reddit = None

            if cls._webpage:
                await cls._webpage.close()
                cls._webpage = None

            cls._initialized = False

        # Close HTTP clients
        await HTTPClientManager.close_all()
        logger.info("Crawler tools shutdown complete")

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get status of all tools."""
        return {
            "youtube": {
                "initialized": cls._youtube is not None,
                "circuit_breaker": cls._youtube.circuit_breaker.get_status() if cls._youtube else None,
            },
            "reddit": {
                "initialized": cls._reddit is not None,
                "circuit_breaker": cls._reddit.circuit_breaker.get_status() if cls._reddit else None,
            },
            "webpage": {
                "initialized": cls._webpage is not None,
                "circuit_breaker": cls._webpage.circuit_breaker.get_status() if cls._webpage else None,
            },
            "http_clients": HTTPClientManager.get_status(),
        }


# =============================================================================
# PUBLIC API
# =============================================================================


TOOL_DEFINITIONS: Final[List[Dict[str, Any]]] = [
    {
        "name": "youtube_transcript",
        "description": "Extract transcript from YouTube video about personality topics",
        "parameters": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "YouTube video URL (youtube.com or youtu.be)",
                }
            },
            "required": ["video_url"],
        },
    },
    {
        "name": "reddit_search",
        "description": "Search personality-related Reddit discussions (psychology, MBTI, self-improvement)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "webpage_fetch",
        "description": "Fetch content from trusted psychology/personality websites",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (must be from allowed domain)",
                }
            },
            "required": ["url"],
        },
    },
]


def list_tools() -> List[Dict[str, Any]]:
    """Get all available tool definitions for LLM function calling."""
    return TOOL_DEFINITIONS


async def invoke_tool(
    name: str,
    **kwargs: Any,
) -> Union[CrawledContent, List[CrawledContent]]:
    """
    Invoke a crawler tool by name.

    Args:
        name: Tool name (youtube_transcript, reddit_search, webpage_fetch)
        **kwargs: Tool-specific arguments

    Returns:
        CrawledContent or list of CrawledContent

    Raises:
        ValueError: Unknown tool name
        pydantic.ValidationError: Invalid arguments
    """
    if name == "youtube_transcript":
        request = YouTubeRequest(**kwargs)
        tool = await ToolRegistry.youtube()
        return await tool.get_transcript(request)

    if name == "reddit_search":
        request = RedditRequest(**kwargs)
        tool = await ToolRegistry.reddit()
        return await tool.search(request)

    if name == "webpage_fetch":
        request = WebpageRequest(**kwargs)
        tool = await ToolRegistry.webpage()
        return await tool.fetch(request)

    raise ValueError(f"Unknown tool: {name}")


async def shutdown() -> None:
    """
    Shutdown crawler tools and cleanup all resources.

    Call this on application exit.
    """
    await ToolRegistry.shutdown()


def get_status() -> Dict[str, Any]:
    """Get status of all crawler components."""
    return ToolRegistry.get_status()


@asynccontextmanager
async def crawler_context() -> AsyncGenerator[None, None]:
    """
    Context manager for crawler lifecycle.

    Automatically cleans up resources on exit.

    Usage:
        async with crawler_context():
            result = await invoke_tool("youtube_transcript", video_url="...")
    """
    try:
        yield
    finally:
        await shutdown()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Core types
    "ContentSource",
    "CrawledContent",
    "CrawlerConfig",
    # Request types
    "YouTubeRequest",
    "RedditRequest",
    "WebpageRequest",
    # Security
    "SSRFProtection",
    "DomainAllowlist",
    "ContentValidator",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
    "RateLimiter",
    "RedisRateLimiter",
    "InMemoryRateLimiter",
    # HTTP
    "HTTPClientManager",
    "HTMLParser",
    # Tools
    "YouTubeTool",
    "RedditTool",
    "WebPageTool",
    "ToolRegistry",
    # Public API
    "list_tools",
    "invoke_tool",
    "shutdown",
    "get_status",
    "crawler_context",
    "TOOL_DEFINITIONS",
]
