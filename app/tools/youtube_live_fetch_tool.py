"""
YouTube Live Fetch Tool - Agentic On-Demand Transcript Retrieval

This tool fetches NEW YouTube transcripts on-demand and "initiates" them into
the SELVE knowledge base through validation and ingestion.

SELVE Validation Philosophy:
"Just as a cult initiates a new member into their group, they cleanse, purify,
and now the new member becomes part of the cult. Any content we fetch from
outside must be validated and initiated into SELVE."

Flow:
1. Fetch transcript from youtube-transcript.io API
2. VALIDATE ("cleanse & purify") using ContentValidationService
3. INGEST approved content into Qdrant
4. Return results for immediate use in chat response
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from app.services.base import Config, Result, ResultStatus
from app.services.content_ingestion_service import ContentIngestionService
from app.services.content_validation_service import (
    ContentValidationService,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class YouTubeTranscriptResult:
    """Result from fetching and validating a YouTube transcript."""

    video_id: str
    title: str
    channel: str
    transcript_text: str
    validation_status: str
    validation_scores: Optional[Dict[str, int]] = None
    ingested: bool = False
    chunks_created: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "validation_status": self.validation_status,
            "validation_scores": self.validation_scores,
            "ingested": self.ingested,
            "chunks_created": self.chunks_created,
            "error": self.error,
        }


# =============================================================================
# YOUTUBE LIVE FETCH TOOL
# =============================================================================


class YouTubeLiveFetchTool:
    """
    Agentic tool for fetching YouTube transcripts on-demand.

    Implements the SELVE "initiation" philosophy:
    1. Fetch raw content from external source (YouTube)
    2. Cleanse & purify through SELVE validation
    3. Ingest approved content into knowledge base
    4. Return validated content for immediate use
    """

    API_URL = "https://www.youtube-transcript.io/api/transcripts"
    YOUTUBE_URL_PATTERNS = [
        r"(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be\/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})",
    ]

    def __init__(self):
        """Initialize YouTube Live Fetch Tool."""
        self.api_token = Config.YOUTUBE_TRANSCRIPT_API_TOKEN
        self.enabled = Config.YOUTUBE_LIVE_FETCH_ENABLED
        self.logger = logging.getLogger(self.__class__.__name__)

        # Lazy-loaded services
        self._validation_service = None
        self._ingestion_service = None

    @property
    def validation_service(self) -> ContentValidationService:
        """Lazy-load validation service."""
        if self._validation_service is None:
            self._validation_service = ContentValidationService()
        return self._validation_service

    @property
    def ingestion_service(self) -> ContentIngestionService:
        """Lazy-load ingestion service."""
        if self._ingestion_service is None:
            self._ingestion_service = ContentIngestionService()
        return self._ingestion_service

    def extract_video_id(self, url_or_id: str) -> Optional[str]:
        """
        Extract YouTube video ID from URL or return as-is if already an ID.

        Args:
            url_or_id: YouTube URL or video ID

        Returns:
            11-character video ID or None if invalid
        """
        # Try each URL pattern
        for pattern in self.YOUTUBE_URL_PATTERNS:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)

        # If no pattern matched, check if it's already a video ID
        if len(url_or_id) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
            return url_or_id

        return None

    def _fetch_transcript_from_api(self, video_id: str) -> Dict[str, Any]:
        """
        Fetch transcript from youtube-transcript.io API.

        Args:
            video_id: YouTube video ID

        Returns:
            API response dict

        Raises:
            Exception if API call fails
        """
        if not self.api_token:
            raise ValueError("YOUTUBE_TRANSCRIPT_API_TOKEN not configured in .env")

        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Basic {self.api_token}",
                "Content-Type": "application/json",
            },
            json={"ids": [video_id]},
            timeout=30,
        )

        if response.status_code != 200:
            raise Exception(
                f"YouTube Transcript API failed with status {response.status_code}: {response.text}"
            )

        data = response.json()

        # Handle different response formats
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        elif isinstance(data, dict) and "transcripts" in data:
            transcripts = data["transcripts"]
            return transcripts[0] if transcripts else {}
        elif isinstance(data, dict):
            return data
        else:
            raise Exception(f"Unexpected API response format: {type(data)}")

    def _extract_transcript_text(self, video_data: Dict[str, Any]) -> str:
        """
        Extract clean transcript text from API response.

        Args:
            video_data: Raw API response for video

        Returns:
            Clean transcript text
        """
        # Try tracks array format (English only)
        tracks = video_data.get("tracks", [])
        if tracks:
            for track in tracks:
                lang = track.get("language", "").lower()
                if lang in ["english", "en"] or "english" in lang:
                    segments = track.get("transcript", [])
                    text_parts = []
                    for segment in segments:
                        if isinstance(segment, dict):
                            text = segment.get("text", "").replace("\n", " ")
                            text_parts.append(text)
                    return " ".join(text_parts)

        # Fallback: old transcript format
        transcript = video_data.get("transcript", [])
        if isinstance(transcript, list):
            text_parts = []
            for segment in transcript:
                if isinstance(segment, dict):
                    text_parts.append(segment.get("text", ""))
                else:
                    text_parts.append(str(segment))
            return " ".join(text_parts)
        elif isinstance(transcript, str):
            return transcript

        return ""

    def _get_video_metadata(self, video_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract metadata from video data."""
        title = video_data.get("title", "Untitled Video")

        # Get channel from microformat if available
        channel = ""
        microformat = video_data.get("microformat", {})
        if microformat:
            renderer = microformat.get("playerMicroformatRenderer", {})
            channel = renderer.get("ownerChannelName", "Unknown Channel")
        else:
            channel = video_data.get("channel", "Unknown Channel")

        return {"title": title, "channel": channel}

    async def fetch_and_validate(
        self, url_or_id: str, auto_ingest: bool = True
    ) -> Result[YouTubeTranscriptResult]:
        """
        Fetch YouTube transcript, validate it through SELVE, and optionally ingest.

        This is the main "initiation" process:
        1. FETCH: Get raw content from YouTube
        2. CLEANSE: Validate against SELVE framework
        3. PURIFY: Only approved content proceeds
        4. INITIATE: Ingest into knowledge base

        Args:
            url_or_id: YouTube URL or video ID
            auto_ingest: If True, automatically ingest approved content

        Returns:
            Result containing YouTubeTranscriptResult
        """
        # Check if feature is enabled
        if not self.enabled:
            return Result.failure(
                "YouTube Live Fetch is disabled. Set YOUTUBE_LIVE_FETCH_ENABLED=true in .env",
                error_code="FEATURE_DISABLED",
            )

        # Extract video ID
        video_id = self.extract_video_id(url_or_id)
        if not video_id:
            return Result.validation_error(
                f"Invalid YouTube URL or video ID: {url_or_id}"
            )

        self.logger.info(f"📺 Fetching YouTube transcript for video ID: {video_id}")

        try:
            # Step 1: FETCH transcript from API
            video_data = self._fetch_transcript_from_api(video_id)
            transcript_text = self._extract_transcript_text(video_data)

            if not transcript_text:
                return Result.failure(
                    "No transcript found for this video (may not have captions)",
                    error_code="NO_TRANSCRIPT",
                )

            metadata = self._get_video_metadata(video_data)
            title = metadata["title"]
            channel = metadata["channel"]

            self.logger.info(
                f"✅ Fetched transcript: '{title}' by {channel} ({len(transcript_text)} chars)"
            )

            # Step 2: CLEANSE & PURIFY - Validate through SELVE
            content_hash = hashlib.sha256(
                (video_id + transcript_text).encode()
            ).hexdigest()

            self.logger.info(
                f"🔍 Initiating SELVE validation (cleanse & purify)..."
            )

            validation_result = await self.validation_service.validate_content(
                content=transcript_text[:10000],  # Validate first 10k chars
                source=f"youtube:{channel}",
                content_hash=content_hash,
            )

            validation_status = validation_result.get("status", "error")
            validation_scores = validation_result.get("scores", {})

            self.logger.info(
                f"📊 Validation result: {validation_status} | "
                f"SELVE: {validation_scores.get('selve_aligned', 0)}/10, "
                f"Accuracy: {validation_scores.get('factually_accurate', 0)}/10, "
                f"Tone: {validation_scores.get('appropriate_tone', 0)}/10"
            )

            result = YouTubeTranscriptResult(
                video_id=video_id,
                title=title,
                channel=channel,
                transcript_text=transcript_text,
                validation_status=validation_status,
                validation_scores=validation_scores,
            )

            # Step 3: INITIATE - Ingest if approved
            if validation_status == "approved" and auto_ingest:
                self.logger.info(
                    f"✨ Content APPROVED - Initiating into SELVE knowledge base..."
                )

                # Prepare markdown content
                markdown_content = f"""# {title}

**Video ID:** {video_id}
**YouTube URL:** https://www.youtube.com/watch?v={video_id}
**Channel:** {channel}
**Source:** YouTube Transcript (Auto-fetched)

---

## Transcript

{transcript_text}
"""

                # Ingest into Qdrant
                ingestion_result = await self.ingestion_service.ingest_content(
                    content=markdown_content,
                    title=title,
                    source="youtube",
                    collection_name="selve_knowledge",
                    metadata={
                        "video_id": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "channel": channel,
                        "validation_status": validation_status,
                        "auto_fetched": True,
                    },
                )

                result.ingested = True
                result.chunks_created = ingestion_result.chunks_created

                self.logger.info(
                    f"🎉 Successfully initiated into SELVE! Created {result.chunks_created} chunks"
                )

            elif validation_status == "needs_revision":
                self.logger.warning(
                    f"⚠️ Content needs revision - NOT ingested. "
                    f"Suggestions: {validation_result.get('suggestions', [])}"
                )
                result.error = "Content needs revision before ingestion"

            elif validation_status == "rejected":
                self.logger.warning(
                    f"❌ Content REJECTED by SELVE validation - NOT ingested. "
                    f"Suggestions: {validation_result.get('suggestions', [])}"
                )
                result.error = "Content rejected by SELVE validation"

            return Result.success(result)

        except Exception as e:
            self.logger.error(f"Failed to fetch/validate YouTube transcript: {e}")
            return Result.failure(str(e), error_code="FETCH_ERROR")


# =============================================================================
# Convenience Functions for Thinking Engine
# =============================================================================


async def fetch_youtube_transcript(
    url_or_id: str, auto_ingest: bool = True
) -> Dict[str, Any]:
    """
    Fetch, validate, and optionally ingest a YouTube transcript.

    Use this tool when the user asks about a specific YouTube video
    or when you need fresh psychology/behavior content from YouTube.

    The tool implements SELVE "initiation":
    - Fetches raw transcript
    - Validates against SELVE framework (cleanse & purify)
    - Ingests approved content into knowledge base

    Args:
        url_or_id: YouTube URL or video ID
        auto_ingest: Automatically ingest if validation passes

    Returns:
        Dict with transcript, validation status, and ingestion results
    """
    tool = YouTubeLiveFetchTool()
    result = await tool.fetch_and_validate(url_or_id, auto_ingest=auto_ingest)

    if result.is_success:
        return result.data.to_dict()
    else:
        return {
            "error": result.error,
            "error_code": result.error_code,
        }
