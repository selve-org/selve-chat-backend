"""
Conversation State Service - Track topics, emotional tone, and conversation flow.

Tracks:
- Active topics being discussed
- Emotional tone of conversation
- Unresolved questions
- Conversation intent patterns

Security & Robustness:
- Safe JSON parsing of LLM responses
- Input validation
- Graceful fallbacks
- Session isolation
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
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


class EmotionalTone(Enum):
    """Recognized emotional tones."""

    CURIOUS = "curious"
    CONCERNED = "concerned"
    EXCITED = "excited"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    HOPEFUL = "hopeful"
    ANXIOUS = "anxious"
    REFLECTIVE = "reflective"


class ConversationIntent(Enum):
    """Recognized conversation intents."""

    SEEKING_INFORMATION = "seeking_information"
    EXPLORING_PERSONALITY = "exploring_personality"
    REQUESTING_ADVICE = "requesting_advice"
    PROCESSING_EMOTIONS = "processing_emotions"
    BUILDING_UNDERSTANDING = "building_understanding"
    TAKING_ACTION = "taking_action"


@dataclass
class ConversationState:
    """Current state of a conversation."""

    topics: List[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    unresolved: List[str] = field(default_factory=list)
    intent: str = "seeking_information"
    analyzed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topics": self.topics,
            "emotional_tone": self.emotional_tone,
            "unresolved": self.unresolved,
            "intent": self.intent,
            "analyzed_at": self.analyzed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        return cls(
            topics=data.get("topics", []),
            emotional_tone=data.get("emotional_tone", "neutral"),
            unresolved=data.get("unresolved", []),
            intent=data.get("intent", "seeking_information"),
            analyzed_at=data.get("analyzed_at"),
        )

    @classmethod
    def default(cls) -> "ConversationState":
        return cls(
            topics=[],
            emotional_tone="neutral",
            unresolved=[],
            intent="seeking_information",
            analyzed_at=datetime.utcnow().isoformat(),
        )


# =============================================================================
# CONVERSATION STATE SERVICE
# =============================================================================


class ConversationStateService(BaseService):
    """
    Service for tracking conversation state and dynamics.

    Analyzes conversations to extract:
    - Active topics being discussed
    - User's emotional tone
    - Unresolved questions or topics
    - Conversation intent

    This enables:
    - Better follow-up questions
    - Appropriate emotional responses
    - Topic continuity across messages
    """

    # LLM settings
    ANALYSIS_TEMPERATURE = 0.3
    ANALYSIS_MAX_TOKENS = 300

    # Limits
    MAX_TOPICS = 10
    MAX_HISTORY_MESSAGES = 4

    def __init__(self, llm_service=None, db=None):
        """
        Initialize conversation state service.

        Args:
            llm_service: LLM service for analysis
            db: Database client
        """
        super().__init__()
        self._llm_service = llm_service
        self._db = db
        self._analysis_prompt = self._build_analysis_prompt()

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

    def _build_analysis_prompt(self) -> str:
        """Build the conversation analysis prompt."""
        return """You are a conversation analyst for the SELVE personality chatbot.

Analyze the most recent exchange and extract:
1. Topics discussed (main subjects of conversation)
2. Emotional tone (user's emotional state)
3. Unresolved questions or topics that need follow-up
4. Conversation intent (what the user is trying to achieve)

Valid emotional tones: curious, concerned, excited, neutral, frustrated, hopeful, anxious, reflective

Valid intents: seeking_information, exploring_personality, requesting_advice, processing_emotions, building_understanding, taking_action

Format your response as JSON ONLY (no markdown):
{
    "topics": ["topic 1", "topic 2"],
    "emotional_tone": "neutral",
    "unresolved": ["unresolved item 1"],
    "intent": "seeking_information"
}

Be concise and accurate. Maximum 3 topics, 2 unresolved items."""

    def _validate_analysis_response(
        self,
        data: Dict[str, Any],
    ) -> ConversationState:
        """
        Validate and sanitize LLM analysis response.

        Args:
            data: Parsed JSON response

        Returns:
            ConversationState (always returns valid state)
        """
        if not isinstance(data, dict):
            return ConversationState.default()

        # Extract and validate topics
        topics = data.get("topics", [])
        if isinstance(topics, list):
            topics = [str(t)[:100] for t in topics[:5] if t]
        else:
            topics = []

        # Extract and validate emotional tone
        emotional_tone = str(data.get("emotional_tone", "neutral")).lower()
        valid_tones = {e.value for e in EmotionalTone}
        if emotional_tone not in valid_tones:
            emotional_tone = "neutral"

        # Extract and validate unresolved items
        unresolved = data.get("unresolved", [])
        if isinstance(unresolved, list):
            unresolved = [str(u)[:200] for u in unresolved[:3] if u]
        else:
            unresolved = []

        # Extract and validate intent
        intent = str(data.get("intent", "seeking_information")).lower()
        valid_intents = {i.value for i in ConversationIntent}
        if intent not in valid_intents:
            intent = "seeking_information"

        return ConversationState(
            topics=topics,
            emotional_tone=emotional_tone,
            unresolved=unresolved,
            intent=intent,
            analyzed_at=datetime.utcnow().isoformat(),
        )

    async def analyze_conversation_state(
        self,
        user_message: str,
        assistant_response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Result[ConversationState]:
        """
        Analyze conversation state from recent exchange.

        Args:
            user_message: Latest user message
            assistant_response: Latest assistant response
            conversation_history: Previous messages (optional)

        Returns:
            Result containing ConversationState
        """
        # Validate inputs
        try:
            user_message = Validator.validate_string(
                user_message,
                "user_message",
                min_length=1,
                max_length=5000,
            )
            assistant_response = Validator.validate_string(
                assistant_response,
                "assistant_response",
                min_length=1,
                max_length=10000,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            # Build analysis text
            analysis_text = f"User: {user_message}\nAssistant: {assistant_response}"

            # Add recent history for context
            if conversation_history:
                recent = conversation_history[-self.MAX_HISTORY_MESSAGES :]
                history_text = "\n".join(
                    f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')[:500]}"
                    for msg in recent
                    if isinstance(msg, dict)
                )
                analysis_text = f"Recent history:\n{history_text}\n\nLatest exchange:\n{analysis_text}"

            messages = [
                {"role": "system", "content": self._analysis_prompt},
                {"role": "user", "content": f"Analyze this conversation:\n\n{analysis_text}"},
            ]

            # Call LLM
            result = self.llm_service.generate_response(
                messages=messages,
                temperature=self.ANALYSIS_TEMPERATURE,
                max_tokens=self.ANALYSIS_MAX_TOKENS,
            )

            # Parse and validate response
            content = result.get("content", "")
            data = safe_json_parse(content)

            if data is None:
                self.logger.warning(f"Failed to parse analysis response: {content[:200]}")
                return Result.success(ConversationState.default())

            state = self._validate_analysis_response(data)
            return Result.success(state, cost=result.get("cost", 0))

        except Exception as e:
            self.logger.error(f"Conversation analysis failed: {e}")
            return Result.success(ConversationState.default())

    async def update_session_state(
        self,
        session_id: str,
        state: ConversationState,
    ) -> Result[bool]:
        """
        Update session with conversation state.

        Args:
            session_id: Chat session ID
            state: Conversation state to store

        Returns:
            Result indicating success
        """
        try:
            session_id = Validator.validate_string(
                session_id,
                "session_id",
                min_length=5,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            # Get current session
            session = await self.db.chatsession.find_unique(
                where={"id": session_id}
            )

            if not session:
                return Result.failure("Session not found", error_code="NOT_FOUND")

            # Merge topics (keep unique, limit count)
            current_metadata = session.metadata or {}
            current_topics = current_metadata.get("topics", [])
            new_topics = list(set(current_topics + state.topics))[-self.MAX_TOPICS :]

            # Update session metadata
            await self.db.chatsession.update(
                where={"id": session_id},
                data={
                    "metadata": {
                        "topics": new_topics,
                        "last_emotional_tone": state.emotional_tone,
                        "unresolved": state.unresolved,
                        "last_intent": state.intent,
                        "state_updated_at": state.analyzed_at,
                    }
                },
            )

            return Result.success(True)

        except Exception as e:
            self.logger.error(f"Failed to update session state: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    async def get_session_state(
        self,
        session_id: str,
    ) -> Result[Optional[ConversationState]]:
        """
        Get current session state.

        Args:
            session_id: Chat session ID

        Returns:
            Result containing ConversationState or None
        """
        try:
            session_id = Validator.validate_string(
                session_id,
                "session_id",
                min_length=5,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            session = await self.db.chatsession.find_unique(
                where={"id": session_id}
            )

            if not session or not session.metadata:
                return Result.success(None)

            metadata = session.metadata
            state = ConversationState(
                topics=metadata.get("topics", []),
                emotional_tone=metadata.get("last_emotional_tone", "neutral"),
                unresolved=metadata.get("unresolved", []),
                intent=metadata.get("last_intent", "seeking_information"),
                analyzed_at=metadata.get("state_updated_at"),
            )

            return Result.success(state)

        except Exception as e:
            self.logger.error(f"Failed to get session state: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    def format_state_for_context(
        self,
        state: ConversationState,
    ) -> str:
        """
        Format session state as context string for LLM.

        Args:
            state: Conversation state

        Returns:
            Formatted context string (empty if no useful state)
        """
        if not state or not state.topics:
            return ""

        parts = [
            "=== CURRENT CONVERSATION STATE ===",
            "",
            f"Active Topics: {', '.join(state.topics)}",
            f"Emotional Tone: {state.emotional_tone}",
            f"User Intent: {state.intent}",
        ]

        if state.unresolved:
            parts.append(f"Unresolved Topics: {', '.join(state.unresolved)}")

        parts.extend([
            "",
            "Use this context to maintain conversation flow and address unresolved topics.",
            "===",
        ])

        return "\n".join(parts)

    async def analyze_and_update(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Result[ConversationState]:
        """
        Convenience method to analyze and update session state in one call.

        Args:
            session_id: Chat session ID
            user_message: Latest user message
            assistant_response: Latest assistant response
            conversation_history: Previous messages

        Returns:
            Result containing updated ConversationState
        """
        # Analyze state
        analysis_result = await self.analyze_conversation_state(
            user_message=user_message,
            assistant_response=assistant_response,
            conversation_history=conversation_history,
        )

        if analysis_result.is_error:
            return analysis_result

        state = analysis_result.data

        # Update session
        update_result = await self.update_session_state(
            session_id=session_id,
            state=state,
        )

        if update_result.is_error:
            self.logger.warning(
                f"Failed to update session state: {update_result.error}"
            )
            # Return state anyway - analysis succeeded

        return Result.success(state)