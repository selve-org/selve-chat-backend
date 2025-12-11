"""
HTTP client management and HTML parsing utilities.

Features:
- Managed HTTP client lifecycle with connection pooling
- BeautifulSoup-based HTML parsing with lxml
- Text extraction with boilerplate removal
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import ClassVar, Dict, Final, Optional, Set

import httpx
from bs4 import BeautifulSoup, Comment

from .core import CrawlerConfig

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP CLIENT MANAGER
# =============================================================================


class HTTPClientManager:
    """
    Manages HTTP client lifecycle with proper resource cleanup.

    Features:
    - Named client instances for different services
    - Connection pooling and keepalive
    - Thread-safe client creation
    - Graceful shutdown
    """

    _clients: ClassVar[Dict[str, httpx.AsyncClient]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get_client(
        cls,
        name: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.AsyncClient:
        """
        Get or create a named HTTP client.

        Args:
            name: Client identifier (e.g., "youtube", "reddit")
            headers: Custom headers for this client
            timeout: Request timeout (defaults to config)

        Returns:
            Configured httpx.AsyncClient
        """
        async with cls._lock:
            if name not in cls._clients or cls._clients[name].is_closed:
                cls._clients[name] = httpx.AsyncClient(
                    timeout=timeout or CrawlerConfig.HTTP_TIMEOUT,
                    follow_redirects=True,
                    max_redirects=CrawlerConfig.MAX_REDIRECTS,
                    headers=headers or {},
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=30,
                    ),
                )
                logger.debug(f"Created HTTP client: {name}")

            return cls._clients[name]

    @classmethod
    async def close(cls, name: str) -> None:
        """Close a specific client."""
        async with cls._lock:
            if name in cls._clients:
                client = cls._clients[name]
                if not client.is_closed:
                    await client.aclose()
                    logger.debug(f"Closed HTTP client: {name}")
                del cls._clients[name]

    @classmethod
    async def close_all(cls) -> None:
        """Close all HTTP clients."""
        async with cls._lock:
            for name, client in list(cls._clients.items()):
                if not client.is_closed:
                    await client.aclose()
                    logger.debug(f"Closed HTTP client: {name}")
            cls._clients.clear()

    @classmethod
    def get_status(cls) -> Dict[str, bool]:
        """Get status of all clients."""
        return {
            name: not client.is_closed
            for name, client in cls._clients.items()
        }


# =============================================================================
# HTML PARSER
# =============================================================================


class HTMLParser:
    """
    Production HTML parser using BeautifulSoup with lxml.

    Features:
    - Robust parsing of malformed HTML
    - Script/style/comment removal
    - Boilerplate removal (nav, header, footer)
    - Title extraction with fallbacks
    - Whitespace normalization
    """

    REMOVE_TAGS: Final[Set[str]] = frozenset({
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "video",
        "audio",
        "map",
        "object",
        "embed",
        "form",
        "input",
        "button",
    })

    BOILERPLATE_TAGS: Final[Set[str]] = frozenset({
        "nav",
        "header",
        "footer",
        "aside",
        "menu",
        "menubar",
        "toolbar",
        "sidebar",
    })

    BOILERPLATE_PATTERNS: Final[tuple] = (
        re.compile(r"(nav|menu|sidebar|footer|header|banner|ad|comment)", re.I),
    )

    @classmethod
    def extract_text(
        cls,
        html: str,
        remove_boilerplate: bool = True,
    ) -> str:
        """
        Extract clean text from HTML.

        Args:
            html: Raw HTML string
            remove_boilerplate: Remove navigation/header/footer etc.

        Returns:
            Cleaned text content
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")

            for tag in cls.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
                comment.extract()

            if remove_boilerplate:
                for tag in cls.BOILERPLATE_TAGS:
                    for element in soup.find_all(tag):
                        element.decompose()

                for pattern in cls.BOILERPLATE_PATTERNS:
                    for element in soup.find_all(
                        class_=pattern
                    ) + soup.find_all(id=pattern):
                        element.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        except Exception as e:  # pragma: no cover - defensive fallback
            logger.warning(f"BeautifulSoup parsing failed: {e}, using fallback")
            return cls._fallback_extract(html)

    @classmethod
    def extract_title(cls, html: str) -> str:
        """
        Extract page title with fallbacks.

        Tries in order:
        1. <title> tag
        2. og:title meta tag
        3. twitter:title meta tag
        4. First <h1> tag
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")

            title_tag = soup.find("title")
            if title_tag and title_tag.string:
                return cls._clean_title(title_tag.string)

            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                return cls._clean_title(og_title["content"])

            twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
            if twitter_title and twitter_title.get("content"):
                return cls._clean_title(twitter_title["content"])

            h1 = soup.find("h1")
            if h1:
                return cls._clean_title(h1.get_text())

            return ""

        except Exception as e:  # pragma: no cover - defensive fallback
            logger.debug(f"Title extraction failed: {e}")
            return ""

    @classmethod
    def extract_meta_description(cls, html: str) -> str:
        """Extract meta description."""
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "lxml")

            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                return meta_desc["content"].strip()[:500]

            og_desc = soup.find("meta", property="og:description")
            if og_desc and og_desc.get("content"):
                return og_desc["content"].strip()[:500]

            return ""

        except Exception:
            return ""

    @classmethod
    def _clean_title(cls, title: str) -> str:
        """Clean and truncate title."""
        title = title.strip()
        title = re.sub(r"\s+", " ", title)
        return title[:500]

    @classmethod
    def _fallback_extract(cls, html: str) -> str:
        """Fallback regex-based text extraction."""
        html = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html = re.sub(
            r"<!--.*?-->",
            "",
            html,
            flags=re.DOTALL,
        )

        text = re.sub(r"<[^>]+>", " ", html)
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = re.sub(r"\s+", " ", text)

        return text.strip()
