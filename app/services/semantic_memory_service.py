"""
Semantic Memory Service - Extract long-term behavioral patterns.

Semantic memories are aggregated patterns extracted from multiple
episodic memories over time, providing long-term user understanding.

Security & Robustness:
- User isolation in all queries
- Safe JSON parsing with fallbacks
- LLM output validation
- Rate limiting awareness
- Proper error propagation
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import (
    BaseService,
    Result,
    Validator,
    safe_json_parse,
    with_error_handling,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


class ConfidenceLevel(Enum):
    """Confidence levels for semantic memories."""

    HYPOTHESIS = "HYPOTHESIS"  # < 0.5 confidence
    OBSERVED = "OBSERVED"  # 0.5 - 0.7 confidence
    CONFIRMED = "CONFIRMED"  # > 0.7 confidence


@dataclass
class SemanticPattern:
    """Extracted behavioral pattern."""

    recurring_themes: List[str] = field(default_factory=list)
    behavioral_patterns: List[str] = field(default_factory=list)
    interest_areas: List[str] = field(default_factory=list)
    communication_style: str = "conversational"
    growth_areas: List[str] = field(default_factory=list)
    confidence: float = 0.0
    episodes_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recurring_themes": self.recurring_themes,
            "behavioral_patterns": self.behavioral_patterns,
            "interest_areas": self.interest_areas,
            "communication_style": self.communication_style,
            "growth_areas": self.growth_areas,
            "confidence": self.confidence,
            "episodes_analyzed": self.episodes_analyzed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticPattern":
        """Create from dictionary."""
        return cls(
            recurring_themes=data.get("recurring_themes", []),
            behavioral_patterns=data.get("behavioral_patterns", []),
            interest_areas=data.get("interest_areas", []),
            communication_style=data.get("communication_style", "conversational"),
            growth_areas=data.get("growth_areas", []),
            confidence=data.get("confidence", 0.0),
            episodes_analyzed=data.get("episodes_analyzed", 0),
        )


@dataclass
class SemanticMemory:
    """A semantic memory record."""

    id: str
    user_id: str
    pattern: SemanticPattern
    confidence_level: ConfidenceLevel
    created_at: datetime
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "confidence": self.confidence_level.value,
            "created_at": self.created_at.isoformat(),
            **self.pattern.to_dict(),
        }


# =============================================================================
# SEMANTIC MEMORY SERVICE
# =============================================================================


class SemanticMemoryService(BaseService):
    """
    Service for extracting and managing semantic memories.

    Semantic memories are long-term behavioral patterns extracted
    from multiple episodic memories over time.

    Examples:
    - "User consistently explores LUMEN dimension topics"
    - "User prefers practical advice over theoretical"
    - "User tends to ask follow-up questions about relationships"

    Features:
    - Pattern extraction from episodic memories
    - Confidence scoring and tracking
    - Memory lifecycle management
    - Context formatting for LLM
    """

    # Extraction settings
    MIN_EPISODES_FOR_EXTRACTION = 3
    EXTRACTION_INTERVAL_EPISODES = 5
    MIN_CONFIDENCE_THRESHOLD = 0.5

    # LLM settings
    EXTRACTION_TEMPERATURE = 0.3
    EXTRACTION_MAX_TOKENS = 600

    def __init__(self, llm_service=None, db=None):
        """
        Initialize semantic memory service.

        Args:
            llm_service: LLM service for pattern extraction
            db: Database client for persistence
        """
        super().__init__()
        self._llm_service = llm_service
        self._db = db
        self._extraction_prompt = self._build_extraction_prompt()

    @property
    def llm_service(self):
        """Lazy-loaded LLM service."""
        if self._llm_service is None:
            from .llm_service import LLMService

            self._llm_service = LLMService()
        return self._llm_service

    @property
    def db(self):
        """Lazy-loaded database client."""
        if self._db is None:
            from app.db import db

            self._db = db
        return self._db

    def _build_extraction_prompt(self) -> str:
        """Build the semantic memory extraction prompt."""
        return """You are a behavioral pattern analyst for the SELVE personality chatbot.

Analyze the provided episodic memories (conversation summaries) and extract long-term patterns:

1. **Recurring Themes**: Topics the user consistently discusses or returns to
2. **Behavioral Patterns**: How the user approaches conversations (analytical, emotional, practical, etc.)
3. **Interest Areas**: Specific SELVE dimensions or life areas of sustained interest
4. **Communication Style**: How the user prefers to receive information
5. **Growth Areas**: Topics where the user is actively seeking development

IMPORTANT RULES:
- Only extract patterns that appear in at least 2 different memories
- Be specific and actionable, not vague
- Focus on patterns that would help personalize future conversations
- Set confidence based on pattern strength:
  - 0.5-0.7 = emerging pattern (2-3 occurrences)
  - 0.7-0.9 = established pattern (4+ occurrences)
  - 0.9-1.0 = strong pattern (consistent across majority)

Format your response as JSON ONLY (no markdown, no explanation):
{
    "recurring_themes": ["theme 1", "theme 2"],
    "behavioral_patterns": ["pattern 1", "pattern 2"],
    "interest_areas": ["interest 1", "interest 2"],
    "communication_preferences": "detailed|concise|conversational|analytical",
    "growth_areas": ["area 1", "area 2"],
    "confidence": 0.0
}"""

    def _validate_extraction_response(
        self,
        data: Dict[str, Any],
    ) -> Optional[SemanticPattern]:
        """
        Validate and sanitize LLM extraction response.

        Args:
            data: Parsed JSON response

        Returns:
            SemanticPattern if valid, None if invalid
        """
        if not isinstance(data, dict):
            return None

        # Extract with defaults and type validation
        def safe_list(value: Any, max_items: int = 10) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(item)[:200] for item in value[:max_items] if item]

        def safe_string(value: Any, default: str, max_length: int = 100) -> str:
            if not isinstance(value, str):
                return default
            return value[:max_length]

        def safe_float(value: Any, default: float, min_val: float, max_val: float) -> float:
            try:
                f = float(value)
                return max(min_val, min(max_val, f))
            except (TypeError, ValueError):
                return default

        return SemanticPattern(
            recurring_themes=safe_list(data.get("recurring_themes")),
            behavioral_patterns=safe_list(data.get("behavioral_patterns")),
            interest_areas=safe_list(data.get("interest_areas")),
            communication_style=safe_string(
                data.get("communication_preferences"),
                "conversational",
            ),
            growth_areas=safe_list(data.get("growth_areas")),
            confidence=safe_float(data.get("confidence"), 0.5, 0.0, 1.0),
        )

    async def extract_semantic_patterns(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Result[SemanticPattern]:
        """
        Extract semantic patterns from episodic memories.

        Args:
            episodes: List of episodic memory dicts with title, summary, etc.

        Returns:
            Result containing SemanticPattern
        """
        if len(episodes) < self.MIN_EPISODES_FOR_EXTRACTION:
            return Result.validation_error(
                f"Need at least {self.MIN_EPISODES_FOR_EXTRACTION} episodes, "
                f"got {len(episodes)}"
            )

        try:
            # Build episode summaries for LLM
            episode_summaries = []
            for ep in episodes:
                summary = {
                    "title": ep.get("title", "")[:200],
                    "summary": ep.get("summary", "")[:500],
                    "key_insights": ep.get("keyInsights", ep.get("key_insights", []))[:5],
                    "emotional_state": ep.get("emotionalState", ep.get("emotional_state")),
                    "date": ep.get("date", ""),
                }
                episode_summaries.append(summary)

            # Format for LLM
            episodes_text = json.dumps(episode_summaries, indent=2)

            messages = [
                {"role": "system", "content": self._extraction_prompt},
                {
                    "role": "user",
                    "content": f"Extract semantic patterns from these {len(episodes)} "
                    f"conversation summaries:\n\n{episodes_text}",
                },
            ]

            # Call LLM
            result = self.llm_service.generate_response(
                messages=messages,
                temperature=self.EXTRACTION_TEMPERATURE,
                max_tokens=self.EXTRACTION_MAX_TOKENS,
            )

            # Parse response
            content = result.get("content", "")
            data = safe_json_parse(content)

            if data is None:
                self.logger.warning(f"Failed to parse LLM response: {content[:200]}")
                return Result.failure(
                    "Failed to parse extraction response",
                    error_code="PARSE_ERROR",
                )

            # Validate and create pattern
            pattern = self._validate_extraction_response(data)
            if pattern is None:
                return Result.failure(
                    "Invalid extraction response structure",
                    error_code="VALIDATION_ERROR",
                )

            pattern.episodes_analyzed = len(episodes)

            return Result.success(
                pattern,
                cost=result.get("cost", 0),
            )

        except Exception as e:
            self.logger.error(f"Semantic extraction failed: {e}")
            return Result.failure(str(e), error_code="EXTRACTION_ERROR")

    async def get_user_episodes(
        self,
        clerk_user_id: str,
        limit: int = 50,
    ) -> Result[List[Dict[str, Any]]]:
        """
        Fetch user's episodic memories from database.

        Args:
            clerk_user_id: Clerk user ID
            limit: Maximum episodes to fetch

        Returns:
            Result containing list of episode dicts
        """
        try:
            clerk_user_id = Validator.validate_user_id(clerk_user_id, "clerk_user_id")
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            episodes = await self.db.episodicmemory.find_many(
                where={
                    "session": {
                        "clerkUserId": clerk_user_id,
                    }
                },
                order={"spanEnd": "desc"},
                take=limit,
                include={"session": True},
            )

            return Result.success([
                {
                    "title": ep.title,
                    "summary": ep.summary,
                    "keyInsights": ep.keyInsights or [],
                    "emotionalState": ep.emotionalState,
                    "date": ep.spanEnd.strftime("%Y-%m-%d") if ep.spanEnd else "",
                }
                for ep in episodes
            ])

        except Exception as e:
            self.logger.error(f"Failed to fetch episodes: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    async def extract_and_save(
        self,
        clerk_user_id: str,
        user_id: str,
    ) -> Result[str]:
        """
        Extract semantic patterns and save to database.

        Args:
            clerk_user_id: Clerk user ID (for fetching episodes)
            user_id: Internal user ID (for storing semantic memory)

        Returns:
            Result containing semantic memory ID
        """
        try:
            clerk_user_id = Validator.validate_user_id(clerk_user_id, "clerk_user_id")
            user_id = Validator.validate_user_id(user_id, "user_id")
        except Exception as e:
            return Result.validation_error(str(e))

        # Fetch episodes
        episodes_result = await self.get_user_episodes(clerk_user_id)
        if episodes_result.is_error:
            return Result.failure(
                episodes_result.error or "Failed to fetch episodes",
                error_code=episodes_result.error_code,
            )

        episodes = episodes_result.data or []
        if len(episodes) < self.MIN_EPISODES_FOR_EXTRACTION:
            return Result.failure(
                f"Insufficient episodes: {len(episodes)}/{self.MIN_EPISODES_FOR_EXTRACTION}",
                error_code="INSUFFICIENT_DATA",
            )

        # Extract patterns
        extraction_result = await self.extract_semantic_patterns(episodes)
        if extraction_result.is_error:
            return Result.failure(
                extraction_result.error or "Extraction failed",
                error_code=extraction_result.error_code,
            )

        pattern = extraction_result.data

        # Check confidence threshold
        if pattern.confidence < self.MIN_CONFIDENCE_THRESHOLD:
            self.logger.info(
                f"Pattern confidence too low: {pattern.confidence:.2f}"
            )
            return Result.failure(
                f"Confidence below threshold: {pattern.confidence:.2f}",
                error_code="LOW_CONFIDENCE",
            )

        try:
            # Deactivate existing semantic memory
            existing = await self.db.semanticmemory.find_first(
                where={
                    "userId": user_id,
                    "category": "aggregate_patterns",
                    "isActive": True,
                }
            )

            if existing:
                await self.db.semanticmemory.update(
                    where={"id": existing.id},
                    data={"isActive": False},
                )

            # Determine confidence level
            if pattern.confidence >= 0.7:
                confidence_level = ConfidenceLevel.CONFIRMED
            elif pattern.confidence >= 0.5:
                confidence_level = ConfidenceLevel.OBSERVED
            else:
                confidence_level = ConfidenceLevel.HYPOTHESIS

            # Create new semantic memory
            now = datetime.utcnow()
            semantic_memory = await self.db.semanticmemory.create(
                data={
                    "userId": user_id,
                    "category": "aggregate_patterns",
                    "content": json.dumps(pattern.to_dict()),
                    "confidence": confidence_level.value,
                    "evidenceCount": pattern.episodes_analyzed,
                    "firstObservedAt": now,
                    "lastObservedAt": now,
                    "sourceMessageIds": [],
                    "isActive": True,
                }
            )

            self.logger.info(
                f"Created semantic memory {semantic_memory.id} "
                f"(confidence: {pattern.confidence:.2f})"
            )

            return Result.success(
                semantic_memory.id,
                confidence=pattern.confidence,
                episodes_analyzed=pattern.episodes_analyzed,
            )

        except Exception as e:
            self.logger.error(f"Failed to save semantic memory: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    async def get_user_semantic_memory(
        self,
        user_id: Optional[str] = None,
        clerk_user_id: Optional[str] = None,
    ) -> Result[Optional[SemanticPattern]]:
        """
        Get current semantic memory for a user.

        Args:
            user_id: Internal user ID (preferred)
            clerk_user_id: Clerk user ID (fallback)

        Returns:
            Result containing SemanticPattern or None
        """
        if not user_id and not clerk_user_id:
            return Result.validation_error("Either user_id or clerk_user_id required")

        try:
            memory = None

            if user_id:
                memory = await self.db.semanticmemory.find_first(
                    where={
                        "userId": user_id,
                        "category": "aggregate_patterns",
                        "isActive": True,
                    }
                )
            elif clerk_user_id:
                # Get user IDs from sessions
                sessions = await self.db.chatsession.find_many(
                    where={"clerkUserId": clerk_user_id}
                )
                user_ids = list({s.userId for s in sessions if s.userId})

                if user_ids:
                    memory = await self.db.semanticmemory.find_first(
                        where={
                            "userId": {"in": user_ids},
                            "category": "aggregate_patterns",
                            "isActive": True,
                        }
                    )

            if not memory:
                return Result.success(None)

            # Parse stored content
            content_data = safe_json_parse(memory.content, {})
            pattern = SemanticPattern.from_dict(content_data)
            pattern.episodes_analyzed = memory.evidenceCount or 0

            return Result.success(pattern)

        except Exception as e:
            self.logger.error(f"Failed to get semantic memory: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    async def should_extract(
        self,
        clerk_user_id: str,
    ) -> bool:
        """
        Check if it's time to extract semantic memory.

        Triggers extraction when sufficient episodes exist and extraction
        hasn't been done recently (or periodically every N episodes).

        Args:
            clerk_user_id: Clerk user ID

        Returns:
            True if extraction should run
        """
        try:
            episodes = await self.db.episodicmemory.find_many(
                where={
                    "session": {
                        "clerkUserId": clerk_user_id,
                    }
                }
            )

            count = len(episodes)
            
            # Trigger if we have minimum episodes and haven't extracted recently
            # OR trigger every N episodes
            if count >= self.MIN_EPISODES_FOR_EXTRACTION:
                # Check if we've extracted in the last 24 hours
                from datetime import datetime, timedelta
                cutoff = datetime.utcnow() - timedelta(hours=24)
                
                recent_extractions = await self.db.semanticmemory.find_many(
                    where={
                        "userId": episodes[0].userId if episodes else None,
                        "createdAt": {"gte": cutoff}
                    }
                ) if episodes else []
                
                # Extract if no recent extraction OR on extraction interval
                return (
                    len(recent_extractions) == 0 
                    or count % self.EXTRACTION_INTERVAL_EPISODES == 0
                )
            
            return False

        except Exception as e:
            self.logger.error(f"Failed to check extraction trigger: {e}")
            return False

    def format_for_context(
        self,
        pattern: SemanticPattern,
    ) -> str:
        """
        Format semantic memory as context string for LLM.

        Args:
            pattern: SemanticPattern to format

        Returns:
            Formatted context string
        """
        if not pattern:
            return ""

        parts = [
            "=== LONG-TERM USER PATTERNS (Semantic Memory) ===",
            "",
            "Based on analysis of past conversations:",
        ]

        if pattern.recurring_themes:
            themes = ", ".join(pattern.recurring_themes[:3])
            parts.append(f"• Recurring Themes: {themes}")

        if pattern.behavioral_patterns:
            behaviors = ", ".join(pattern.behavioral_patterns[:3])
            parts.append(f"• Behavioral Patterns: {behaviors}")

        if pattern.interest_areas:
            interests = ", ".join(pattern.interest_areas[:3])
            parts.append(f"• Key Interests: {interests}")

        if pattern.communication_style:
            parts.append(f"• Communication Style: {pattern.communication_style}")

        if pattern.growth_areas:
            growth = ", ".join(pattern.growth_areas[:3])
            parts.append(f"• Growth Areas: {growth}")

        parts.append("")
        confidence_pct = int(pattern.confidence * 100)
        parts.append(
            f"(Confidence: {confidence_pct}%, "
            f"based on {pattern.episodes_analyzed} conversations)"
        )
        parts.append("===")

        return "\n".join(parts)