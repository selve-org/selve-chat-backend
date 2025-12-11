"""
Content crawler tools for YouTube, Reddit, and web pages.

Each tool includes:
- Input validation via Pydantic
- SSRF protection
- Rate limiting
- Circuit breaker resilience
- Content validation
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Final, List, Optional, Pattern, Tuple

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
from .resilience import CircuitBreaker, RateLimiter, create_rate_limiter

logger = logging.getLogger(__name__)


# =============================================================================
# YOUTUBE TOOL
# =============================================================================


class YouTubeTool:
    """
    YouTube transcript extraction.

    Security:
    - Only accepts youtube.com/youtu.be URLs
    - Video ID validation
    - Rate limited
    - Circuit breaker protection
    """

    VIDEO_ID_PATTERN: Final[Pattern] = re.compile(r"^[0-9A-Za-z_-]{11}$")

    EXTRACTION_PATTERNS: Final[Tuple[Pattern, ...]] = (
        re.compile(r"[?&]v=([0-9A-Za-z_-]{11})"),
        re.compile(r"youtu\.be/([0-9A-Za-z_-]{11})"),
        re.compile(r"embed/([0-9A-Za-z_-]{11})"),
        re.compile(r"shorts/([0-9A-Za-z_-]{11})"),
    )

    def __init__(self):
        self.rate_limiter: RateLimiter = create_rate_limiter(
            CrawlerConfig.RATE_LIMIT_YOUTUBE,
            "rl:youtube",
        )
        self.circuit_breaker = CircuitBreaker(name="youtube")
        self._transcript_api = None

    def _get_transcript_api(self):
        """Lazy load YouTube transcript API."""
        if self._transcript_api is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi

                self._transcript_api = YouTubeTranscriptApi
            except ImportError as exc:  # pragma: no cover - runtime import guard
                raise ImportError(
                    "Install youtube-transcript-api: pip install youtube-transcript-api"
                ) from exc
        return self._transcript_api

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract and validate video ID from URL."""
        for pattern in self.EXTRACTION_PATTERNS:
            match = pattern.search(url)
            if match:
                video_id = match.group(1)
                if self.VIDEO_ID_PATTERN.match(video_id):
                    return video_id
        return None

    async def get_transcript(self, request: YouTubeRequest) -> CrawledContent:
        """
        Extract transcript from YouTube video.

        Args:
            request: Validated YouTube request

        Returns:
            CrawledContent with transcript or error
        """
        url = request.video_url

        is_safe, error = SSRFProtection.is_safe_url(url)
        if not is_safe:
            return CrawledContent.create_error(
                ContentSource.YOUTUBE, url, error or "Unsafe URL"
            )

        video_id = self._extract_video_id(url)
        if not video_id:
            return CrawledContent.create_error(
                ContentSource.YOUTUBE, url, "Could not extract video ID"
            )

        if not await self.rate_limiter.acquire("youtube"):
            return CrawledContent.create_error(
                ContentSource.YOUTUBE, url, "Rate limit exceeded"
            )

        if not await self.circuit_breaker.can_execute():
            return CrawledContent.create_error(
                ContentSource.YOUTUBE, url, "Service temporarily unavailable"
            )

        try:
            api = self._get_transcript_api()
            loop = asyncio.get_event_loop()
            transcript = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: api.get_transcript(video_id)),
                timeout=CrawlerConfig.YOUTUBE_TIMEOUT,
            )

            await self.circuit_breaker.record_success()

            transcript = transcript[: CrawlerConfig.MAX_YOUTUBE_SEGMENTS]
            full_text = " ".join(t.get("text", "") for t in transcript)
            full_text = full_text[: CrawlerConfig.MAX_CONTENT_LENGTH]

            is_valid, score, reason = ContentValidator.validate(full_text)

            return CrawledContent(
                source=ContentSource.YOUTUBE,
                url=url,
                title=f"YouTube: {video_id}",
                content=full_text,
                relevance_score=score,
                is_valid=is_valid,
                metadata={
                    "video_id": video_id,
                    "segments": len(transcript),
                },
                error=reason if not is_valid else None,
            )

        except asyncio.TimeoutError:
            await self.circuit_breaker.record_failure()
            return CrawledContent.create_error(
                ContentSource.YOUTUBE, url, "Request timed out"
            )
        except Exception as exc:  # pragma: no cover - runtime errors only
            await self.circuit_breaker.record_failure()
            logger.error(f"YouTube transcript error: {exc}")
            return CrawledContent.create_error(
                ContentSource.YOUTUBE, url, str(exc)[:200]
            )

    async def close(self) -> None:
        """Cleanup resources."""
        await self.rate_limiter.close()


# =============================================================================
# REDDIT TOOL
# =============================================================================


class RedditTool:
    """
    Reddit search with subreddit allowlist.

    Security:
    - Only searches allowlisted subreddits
    - Query sanitization
    - NSFW filtering
    - Rate limited
    """

    def __init__(self):
        self.rate_limiter: RateLimiter = create_rate_limiter(
            CrawlerConfig.RATE_LIMIT_REDDIT,
            "rl:reddit",
        )
        self.circuit_breaker = CircuitBreaker(name="reddit")

    async def search(self, request: RedditRequest) -> List[CrawledContent]:
        """
        Search Reddit for personality content.

        Args:
            request: Validated Reddit request

        Returns:
            List of CrawledContent results
        """
        if request.subreddits:
            subreddits = DomainAllowlist.filter_subreddits(request.subreddits)
        else:
            subreddits = ["psychology", "selfimprovement", "mbti"]

        if not subreddits:
            logger.warning("No valid subreddits in request")
            return []

        if not await self.circuit_breaker.can_execute():
            logger.warning("Reddit circuit breaker open")
            return []

        results: List[CrawledContent] = []
        client = await HTTPClientManager.get_client(
            "reddit",
            headers={
                "User-Agent": "SELVE-Bot/2.0 (Personality Research; +https://selve.app/bot)",
            },
        )

        for subreddit in subreddits[:3]:
            if not await self.rate_limiter.acquire(f"reddit:{subreddit}"):
                logger.debug(f"Rate limit hit for r/{subreddit}")
                continue

            try:
                response = await client.get(
                    f"https://www.reddit.com/r/{subreddit}/search.json",
                    params={
                        "q": request.query,
                        "limit": request.limit,
                        "restrict_sr": "true",
                        "sort": "relevance",
                        "t": "year",
                    },
                )

                if response.status_code == 429:
                    logger.warning("Reddit rate limit (429) hit")
                    await self.circuit_breaker.record_failure()
                    break

                if response.status_code != 200:
                    logger.warning(f"Reddit API error: {response.status_code}")
                    continue

                await self.circuit_breaker.record_success()

                data = response.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    post_data = post.get("data", {})

                    if post_data.get("over_18"):
                        continue

                    title = post_data.get("title", "")[:200]
                    selftext = post_data.get("selftext", "")[:2000]
                    content = f"{title}\n\n{selftext}".strip()

                    is_valid, score, _ = ContentValidator.validate(content)

                    if is_valid:
                        permalink = post_data.get("permalink", "")
                        results.append(
                            CrawledContent(
                                source=ContentSource.REDDIT,
                                url=f"https://reddit.com{permalink}" if permalink else "https://reddit.com",
                                title=title,
                                content=content,
                                relevance_score=score,
                                is_valid=True,
                                metadata={
                                    "subreddit": subreddit,
                                    "score": post_data.get("score", 0),
                                    "comments": post_data.get("num_comments", 0),
                                    "author": post_data.get("author", "[deleted]"),
                                },
                            )
                        )

            except asyncio.TimeoutError:
                logger.warning(f"Timeout searching r/{subreddit}")
                await self.circuit_breaker.record_failure()
            except Exception as exc:  # pragma: no cover - runtime errors only
                logger.error(f"Reddit error for r/{subreddit}: {exc}")
                await self.circuit_breaker.record_failure()

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[: request.limit]

    async def close(self) -> None:
        """Cleanup resources."""
        await self.rate_limiter.close()


# =============================================================================
# WEBPAGE TOOL
# =============================================================================


class WebPageTool:
    """
    Web page fetching with domain allowlist.

    Security:
    - SSRF protection
    - Domain allowlist enforcement
    - Content type validation
    - Size limits
    - Rate limited
    """

    def __init__(self):
        self.rate_limiter: RateLimiter = create_rate_limiter(
            CrawlerConfig.RATE_LIMIT_WEBPAGE,
            "rl:webpage",
        )
        self.circuit_breaker = CircuitBreaker(name="webpage")

    async def fetch(self, request: WebpageRequest) -> CrawledContent:
        """
        Fetch and parse a webpage.

        Args:
            request: Validated webpage request

        Returns:
            CrawledContent with parsed content or error
        """
        url = request.url

        is_safe, error = SSRFProtection.is_safe_url(url)
        if not is_safe:
            return CrawledContent.create_error(
                ContentSource.WEBPAGE, url, error or "Unsafe URL"
            )

        is_allowed, error = DomainAllowlist.is_webpage_allowed(url)
        if not is_allowed:
            return CrawledContent.create_error(
                ContentSource.WEBPAGE, url, error or "Domain not allowed"
            )

        if not await self.rate_limiter.acquire("webpage"):
            return CrawledContent.create_error(
                ContentSource.WEBPAGE, url, "Rate limit exceeded"
            )

        if not await self.circuit_breaker.can_execute():
            return CrawledContent.create_error(
                ContentSource.WEBPAGE, url, "Service temporarily unavailable"
            )

        client = await HTTPClientManager.get_client(
            "webpage",
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SELVE-Bot/2.0; +https://selve.app/bot)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        try:
            response = await client.get(url)

            if response.status_code != 200:
                return CrawledContent.create_error(
                    ContentSource.WEBPAGE, url, f"HTTP {response.status_code}"
                )

            await self.circuit_breaker.record_success()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                return CrawledContent.create_error(
                    ContentSource.WEBPAGE, url, f"Not HTML: {content_type[:50]}"
                )

            if len(response.content) > CrawlerConfig.MAX_RESPONSE_SIZE:
                return CrawledContent.create_error(
                    ContentSource.WEBPAGE, url, "Response too large"
                )

            html = response.text
            text = HTMLParser.extract_text(html)
            title = HTMLParser.extract_title(html)

            text = text[: CrawlerConfig.MAX_CONTENT_LENGTH]

            is_valid, score, reason = ContentValidator.validate(text)

            return CrawledContent(
                source=ContentSource.WEBPAGE,
                url=str(response.url),
                title=title or url,
                content=text,
                relevance_score=score,
                is_valid=is_valid,
                metadata={
                    "content_type": content_type[:100],
                    "content_length": len(text),
                    "original_url": url,
                },
                error=reason if not is_valid else None,
            )

        except asyncio.TimeoutError:
            await self.circuit_breaker.record_failure()
            return CrawledContent.create_error(
                ContentSource.WEBPAGE, url, "Request timed out"
            )
        except Exception as exc:  # pragma: no cover - runtime errors only
            await self.circuit_breaker.record_failure()
            logger.error(f"Webpage fetch error: {exc}")
            return CrawledContent.create_error(
                ContentSource.WEBPAGE, url, str(exc)[:200]
            )

    async def close(self) -> None:
        """Cleanup resources."""
        await self.rate_limiter.close()
