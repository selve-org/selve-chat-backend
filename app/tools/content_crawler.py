"""
Adapter for the production-grade crawler tools.

Provides a stable import path for existing code while delegating to the
modular crawler package under `app.tools.crawler`.
"""

from __future__ import annotations

from typing import Any, List, Union

from .crawler import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    ContentSource,
    ContentValidator,
    CrawledContent,
    CrawlerConfig,
    DomainAllowlist,
    HTMLParser,
    HTTPClientManager,
    InMemoryRateLimiter,
    RateLimiter,
    RedditRequest,
    RedisRateLimiter,
    SSRFProtection,
    TOOL_DEFINITIONS,
    ToolRegistry,
    WebPageTool,
    WebpageRequest,
    YouTubeRequest,
    YouTubeTool,
    crawler_context,
    get_status,
    invoke_tool,
    list_tools,
    shutdown,
)

CONTENT_TOOLS = TOOL_DEFINITIONS


async def get_tool(name: str):
    """Return a tool instance by name for backwards compatibility."""
    if name == "youtube_transcript":
        return await ToolRegistry.youtube()
    if name == "reddit_search":
        return await ToolRegistry.reddit()
    if name == "webpage_fetch":
        return await ToolRegistry.webpage()
    raise ValueError(f"Unknown tool: {name}")


async def invoke(name: str, **kwargs: Any) -> Union[CrawledContent, List[CrawledContent]]:
    """Alias to `invoke_tool` for older call sites."""
    return await invoke_tool(name, **kwargs)


cleanup = shutdown

__all__ = [
    "ContentValidator",
    "CrawledContent",
    "YouTubeTool",
    "RedditTool",
    "WebPageTool",
    "CONTENT_TOOLS",
    "get_tool",
    "list_tools",
    "invoke_tool",
    "invoke",
    "cleanup",
    "TOOL_DEFINITIONS",
    "ToolRegistry",
    "crawler_context",
    "shutdown",
    "get_status",
    "ContentSource",
    "CrawlerConfig",
    "SSRFProtection",
    "DomainAllowlist",
    "RateLimiter",
    "RedisRateLimiter",
    "InMemoryRateLimiter",
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
    "HTTPClientManager",
    "HTMLParser",
    "YouTubeRequest",
    "RedditRequest",
    "WebpageRequest",
]
