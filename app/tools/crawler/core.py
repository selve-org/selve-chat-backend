"""
Crawler configuration, types, and core utilities.
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Pattern, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class CrawlerConfig:
    """Centralized configuration from environment variables."""

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"

    # Timeouts
    HTTP_TIMEOUT: float = float(os.getenv("CRAWLER_HTTP_TIMEOUT", "15"))
    YOUTUBE_TIMEOUT: float = float(os.getenv("CRAWLER_YOUTUBE_TIMEOUT", "30"))
    REDIS_TIMEOUT: float = float(os.getenv("CRAWLER_REDIS_TIMEOUT", "1"))

    # Content limits
    MAX_CONTENT_LENGTH: int = int(os.getenv("CRAWLER_MAX_CONTENT", "10000"))
    MAX_RESPONSE_SIZE: int = int(os.getenv("CRAWLER_MAX_RESPONSE", "5000000"))
    MIN_CONTENT_LENGTH: int = int(os.getenv("CRAWLER_MIN_CONTENT", "50"))
    MAX_YOUTUBE_SEGMENTS: int = int(os.getenv("CRAWLER_MAX_YT_SEGMENTS", "500"))
    MAX_REDDIT_RESULTS: int = int(os.getenv("CRAWLER_MAX_REDDIT", "10"))

    # Rate limits (per minute)
    RATE_LIMIT_YOUTUBE: int = int(os.getenv("CRAWLER_RATE_YOUTUBE", "20"))
    RATE_LIMIT_REDDIT: int = int(os.getenv("CRAWLER_RATE_REDDIT", "10"))
    RATE_LIMIT_WEBPAGE: int = int(os.getenv("CRAWLER_RATE_WEBPAGE", "30"))

    # Security
    MAX_REDIRECTS: int = int(os.getenv("CRAWLER_MAX_REDIRECTS", "3"))

    # Circuit breaker
    CB_THRESHOLD: int = int(os.getenv("CRAWLER_CB_THRESHOLD", "5"))
    CB_TIMEOUT: int = int(os.getenv("CRAWLER_CB_TIMEOUT", "60"))


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class ContentSource(str, Enum):
    """Content source types."""

    YOUTUBE = "youtube"
    REDDIT = "reddit"
    WEBPAGE = "webpage"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class CrawledContent(BaseModel):
    """Immutable crawled content with validation."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    source: ContentSource
    url: str = Field(..., min_length=1, max_length=2048)
    title: str = Field(default="", max_length=500)
    content: str = Field(default="", max_length=50000)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_valid: bool = False
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None, max_length=500)

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @property
    def content_preview(self) -> str:
        return self.content[:200] + "..." if len(self.content) > 200 else self.content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "url": self.url,
            "title": self.title,
            "content_preview": self.content_preview,
            "content_length": len(self.content),
            "relevance_score": self.relevance_score,
            "is_valid": self.is_valid,
            "crawled_at": self.crawled_at.isoformat(),
            "metadata": self.metadata,
            "error": self.error,
        }

    @classmethod
    def create_error(cls, source: ContentSource, url: str, error: str) -> "CrawledContent":
        return cls(
            source=source,
            url=url or "unknown://error",
            error=error[:500],
        )


class YouTubeRequest(BaseModel):
    """Validated YouTube request."""

    model_config = ConfigDict(str_strip_whitespace=True)
    video_url: str = Field(..., min_length=10, max_length=200)

    @field_validator("video_url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        from urllib.parse import urlparse

        hostname = (urlparse(v).hostname or "").lower()
        if hostname not in {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}:
            raise ValueError(f"Not a YouTube URL: {hostname}")
        return v


class RedditRequest(BaseModel):
    """Validated Reddit request."""

    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(default=5, ge=1, le=10)
    subreddits: Optional[List[str]] = None

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return re.sub(r"[^\w\s\-.,!?']", "", v).strip()


class WebpageRequest(BaseModel):
    """Validated webpage request."""

    model_config = ConfigDict(str_strip_whitespace=True)
    url: str = Field(..., min_length=10, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


# =============================================================================
# SSRF PROTECTION
# =============================================================================


class SSRFProtection:
    """Comprehensive SSRF protection."""

    BLOCKED_HOSTS: Final[Set[str]] = frozenset({
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "[::1]",
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.azure.com",
        "169.254.170.2",
        "kubernetes.default",
    })

    BLOCKED_SUFFIXES: Final[Tuple[str, ...]] = (
        ".local",
        ".localhost",
        ".internal",
        ".localdomain",
        ".intranet",
        ".corp",
        ".lan",
    )

    PRIVATE_IP_PATTERNS: Final[Tuple[Pattern, ...]] = (
        re.compile(r"^127\."),
        re.compile(r"^10\."),
        re.compile(r"^172\.(1[6-9]|2[0-9]|3[01])\."),
        re.compile(r"^192\.168\."),
        re.compile(r"^169\.254\."),
        re.compile(r"^0\."),
        re.compile(r"^224\."),
    )

    IPV4_PATTERN: Final[Pattern] = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")

    @classmethod
    def is_safe_url(cls, url: str) -> Tuple[bool, Optional[str]]:
        if not url:
            return False, "Empty URL"

        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except Exception:
            return False, "Invalid URL format"

        if parsed.scheme.lower() not in ("http", "https"):
            return False, f"Disallowed protocol: {parsed.scheme}"

        if parsed.username or parsed.password:
            return False, "URLs with credentials not allowed"

        hostname = parsed.hostname
        if not hostname:
            return False, "URL must have a hostname"

        hostname_lower = hostname.lower()

        if hostname_lower in cls.BLOCKED_HOSTS:
            return False, f"Blocked host: {hostname}"

        for suffix in cls.BLOCKED_SUFFIXES:
            if hostname_lower.endswith(suffix):
                return False, f"Blocked domain suffix: {suffix}"

        ip_match = cls.IPV4_PATTERN.match(hostname)
        if ip_match:
            octets = [int(g) for g in ip_match.groups()]
            if any(o > 255 for o in octets):
                return False, "Invalid IP address"
            for pattern in cls.PRIVATE_IP_PATTERNS:
                if pattern.match(hostname):
                    return False, f"Private IP not allowed: {hostname}"
            return False, "Direct IP access not allowed"

        if ".." in parsed.path:
            return False, "Path traversal not allowed"

        if parsed.port and parsed.port not in (80, 443, 8080, 8443):
            return False, f"Non-standard port not allowed: {parsed.port}"

        return True, None


# =============================================================================
# DOMAIN ALLOWLISTS
# =============================================================================


class DomainAllowlist:
    """Curated domain allowlists."""

    WEBPAGE_DOMAINS: Final[Set[str]] = frozenset({
        "psychologytoday.com",
        "verywellmind.com",
        "positivepsychology.com",
        "apa.org",
        "bps.org.uk",
        "16personalities.com",
        "truity.com",
        "mindtools.com",
        "jamesclear.com",
        "markmanson.net",
        "scholar.google.com",
        "ncbi.nlm.nih.gov",
    })

    REDDIT_SUBREDDITS: Final[Set[str]] = frozenset({
        "psychology",
        "emotionalintelligence",
        "selfimprovement",
        "decidingtobebetter",
        "getdisciplined",
        "mbti",
        "intj",
        "intp",
        "entj",
        "entp",
        "infj",
        "infp",
        "enfj",
        "enfp",
        "istj",
        "isfj",
        "estj",
        "esfj",
        "istp",
        "isfp",
        "estp",
        "esfp",
        "enneagram",
    })

    @classmethod
    def is_webpage_allowed(cls, url: str) -> Tuple[bool, Optional[str]]:
        from urllib.parse import urlparse

        hostname = (urlparse(url).hostname or "").lower()
        for domain in cls.WEBPAGE_DOMAINS:
            if hostname == domain or hostname.endswith(f".{domain}"):
                return True, None
        return False, f"Domain not in allowlist: {hostname}"

    @classmethod
    def filter_subreddits(cls, subreddits: List[str]) -> List[str]:
        return [s for s in subreddits if s.lower() in cls.REDDIT_SUBREDDITS]


# =============================================================================
# CONTENT VALIDATOR
# =============================================================================


class ContentValidator:
    """Content validation for personality relevance."""

    PRIMARY_KEYWORDS: Final[Set[str]] = frozenset({
        "personality",
        "temperament",
        "introvert",
        "extrovert",
        "mbti",
        "big five",
        "enneagram",
        "emotional intelligence",
        "self-awareness",
        "lumen",
        "aether",
        "orpheus",
        "vara",
        "chronos",
        "kael",
        "orin",
        "lyra",
    })

    SECONDARY_KEYWORDS: Final[Set[str]] = frozenset({
        "psychology",
        "self-improvement",
        "personal growth",
        "relationships",
        "communication style",
        "empathy",
        "mindfulness",
        "resilience",
        "leadership",
        "stress management",
        "motivation",
        "values",
    })

    BLOCKED_KEYWORDS: Final[Set[str]] = frozenset({
        "javascript",
        "python programming",
        "api endpoint",
        "suicide method",
        "self-harm",
        "drug dosage",
        "weapon",
        "explosive",
        "click here",
        "buy now",
        "limited time",
        "casino",
    })

    @classmethod
    def validate(cls, content: str) -> Tuple[bool, float, Optional[str]]:
        if not content:
            return False, 0.0, "Empty content"

        content = content.strip()
        if len(content) < CrawlerConfig.MIN_CONTENT_LENGTH:
            return False, 0.0, f"Content too short ({len(content)} chars)"

        content_lower = content.lower()

        for keyword in cls.BLOCKED_KEYWORDS:
            if keyword in content_lower:
                return False, 0.0, f"Blocked content: {keyword}"

        score = sum(2 for kw in cls.PRIMARY_KEYWORDS if kw in content_lower)
        score += sum(1 for kw in cls.SECONDARY_KEYWORDS if kw in content_lower)
        relevance = min(score / 10, 1.0)

        if relevance < 0.2:
            return False, relevance, "Insufficient personality relevance"

        return True, round(relevance, 3), None
