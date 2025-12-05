"""
Conversation State Service - Track topics, emotional tone, and conversation flow
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.db import db
from .llm_service import LLMService
import json


class ConversationStateService:
    """
    Service for tracking conversation state and dynamics

    Tracks:
    - Active topics being discussed
    - Emotional tone of conversation
    - Unresolved questions
    - Conversation flow patterns
    """

    def __init__(self):
        """Initialize with LLM for state analysis"""
        self.llm_service = LLMService()
        self.analysis_prompt = self._load_analysis_prompt()

    def _load_analysis_prompt(self) -> str:
        """Load conversation analysis prompt"""
        return """You are a conversation analyst for the SELVE chatbot.

Analyze the most recent exchange and extract:
1. Topics discussed (main subjects of conversation)
2. Emotional tone (user's emotional state)
3. Unresolved questions or topics that need follow-up
4. Conversation intent (what the user is trying to achieve)

Format your response as JSON:
{
    "topics": ["topic 1", "topic 2", ...],
    "emotional_tone": "curious|concerned|excited|neutral|frustrated|...",
    "unresolved": ["unresolved item 1", ...],
    "intent": "seeking_information|exploring_personality|requesting_advice|..."
}

Be concise and accurate."""

    async def analyze_conversation_state(
        self,
        user_message: str,
        assistant_response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze conversation state from recent exchange

        Args:
            user_message: Latest user message
            assistant_response: Latest assistant response
            conversation_history: Previous messages (optional, for context)

        Returns:
            {
                "topics": [...],
                "emotional_tone": "...",
                "unresolved": [...],
                "intent": "..."
            }
        """
        try:
            # Build analysis context
            analysis_text = f"User: {user_message}\nAssistant: {assistant_response}"

            # Add recent history for context (last 2 exchanges)
            if conversation_history and len(conversation_history) > 0:
                recent_history = conversation_history[-4:]  # Last 2 exchanges (4 messages)
                history_text = "\n".join([
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in recent_history
                ])
                analysis_text = f"Recent history:\n{history_text}\n\nLatest exchange:\n{analysis_text}"

            messages = [
                {"role": "system", "content": self.analysis_prompt},
                {"role": "user", "content": f"Analyze this conversation:\n\n{analysis_text}"}
            ]

            # Use LLM to analyze (with lower temp for consistency)
            result = self.llm_service.generate_response(
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )

            # Parse JSON response
            analysis = json.loads(result["content"])

            return {
                "topics": analysis.get("topics", []),
                "emotional_tone": analysis.get("emotional_tone", "neutral"),
                "unresolved": analysis.get("unresolved", []),
                "intent": analysis.get("intent", "seeking_information"),
                "analyzed_at": datetime.utcnow().isoformat()
            }

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse conversation analysis: {e}")
            return self._default_state()
        except Exception as e:
            print(f"❌ Conversation analysis error: {e}")
            return self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        """Return default state if analysis fails"""
        return {
            "topics": [],
            "emotional_tone": "neutral",
            "unresolved": [],
            "intent": "seeking_information",
            "analyzed_at": datetime.utcnow().isoformat()
        }

    async def update_session_state(
        self,
        session_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """
        Update session with conversation state

        Args:
            session_id: Chat session ID
            state: Conversation state dict

        Returns:
            True if successful
        """
        try:
            # Get current session
            session = await db.chatsession.find_unique(
                where={"id": session_id}
            )

            if not session:
                return False

            # Merge topics (keep unique)
            current_topics = session.metadata.get("topics", []) if session.metadata else []
            new_topics = list(set(current_topics + state["topics"]))

            # Update session metadata
            await db.chatsession.update(
                where={"id": session_id},
                data={
                    "metadata": {
                        "topics": new_topics[-10:],  # Keep last 10 topics
                        "last_emotional_tone": state["emotional_tone"],
                        "unresolved": state["unresolved"],
                        "last_intent": state["intent"],
                        "state_updated_at": state["analyzed_at"]
                    }
                }
            )

            return True

        except Exception as e:
            print(f"❌ Failed to update session state: {e}")
            return False

    async def get_session_state(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current session state

        Args:
            session_id: Chat session ID

        Returns:
            Session state dict or None
        """
        try:
            session = await db.chatsession.find_unique(
                where={"id": session_id}
            )

            if not session or not session.metadata:
                return None

            return {
                "topics": session.metadata.get("topics", []),
                "last_emotional_tone": session.metadata.get("last_emotional_tone", "neutral"),
                "unresolved": session.metadata.get("unresolved", []),
                "last_intent": session.metadata.get("last_intent", "seeking_information"),
                "state_updated_at": session.metadata.get("state_updated_at")
            }

        except Exception as e:
            print(f"❌ Failed to get session state: {e}")
            return None

    def format_state_for_context(
        self,
        state: Dict[str, Any]
    ) -> str:
        """
        Format session state as context string for LLM

        Args:
            state: Session state dict

        Returns:
            Formatted context string
        """
        if not state or not state.get("topics"):
            return ""

        context_parts = [
            "=== CURRENT CONVERSATION STATE ===",
            "",
            f"Active Topics: {', '.join(state['topics'])}",
            f"Emotional Tone: {state['last_emotional_tone']}",
            f"User Intent: {state['last_intent']}"
        ]

        if state.get("unresolved"):
            context_parts.append(f"Unresolved Topics: {', '.join(state['unresolved'])}")

        context_parts.append("")
        context_parts.append("Use this context to maintain conversation flow and address unresolved topics.")
        context_parts.append("===")

        return "\n".join(context_parts)
