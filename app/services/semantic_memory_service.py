"""
Semantic Memory Service - Extract long-term behavioral patterns
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.db import db
from .llm_service import LLMService
import json


class SemanticMemoryService:
    """
    Service for extracting and managing semantic memories

    Semantic memories are long-term behavioral patterns and insights
    extracted from multiple episodic memories over time.

    Examples:
    - "User is consistently curious about LUMEN dimension"
    - "User tends to ask follow-up questions about relationships"
    - "User prefers practical advice over theoretical explanations"
    """

    def __init__(self):
        """Initialize with LLM for pattern extraction"""
        self.llm_service = LLMService()
        self.extraction_prompt = self._load_extraction_prompt()

    def _load_extraction_prompt(self) -> str:
        """Load semantic memory extraction prompt"""
        return """You are a behavioral pattern analyst for the SELVE chatbot.

Analyze the provided episodic memories (conversation summaries) and extract long-term patterns:

1. **Recurring Themes**: Topics the user consistently discusses or returns to
2. **Behavioral Patterns**: How the user approaches conversations (analytical, emotional, practical, etc.)
3. **Interest Areas**: Specific SELVE dimensions or life areas of sustained interest
4. **Communication Style**: How the user prefers to receive information
5. **Growth Areas**: Topics where the user is actively seeking development

Format your response as JSON:
{
    "recurring_themes": ["theme 1", "theme 2", ...],
    "behavioral_patterns": ["pattern 1", "pattern 2", ...],
    "interest_areas": ["interest 1", "interest 2", ...],
    "communication_preferences": "detailed|concise|conversational|...",
    "growth_areas": ["area 1", "area 2", ...],
    "confidence": 0.0-1.0
}

Only extract patterns that appear in at least 2 different memories.
Set confidence based on pattern strength (0.5-0.7 = emerging, 0.7-0.9 = established, 0.9-1.0 = strong)."""

    async def extract_semantic_memories(
        self,
        clerk_user_id: str,
        min_episodes: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Extract semantic memories from user's episodic memories

        Args:
            clerk_user_id: Clerk user ID
            min_episodes: Minimum number of episodes needed for extraction

        Returns:
            Semantic memory dict or None if insufficient data
        """
        try:
            # Get all user's episodic memories
            episodes = await db.episodicmemory.find_many(
                where={
                    "session": {
                        "clerkUserId": clerk_user_id
                    }
                },
                order={"spanEnd": "desc"},
                include={"session": True}
            )

            if len(episodes) < min_episodes:
                print(f"⚠️ Insufficient episodic memories for semantic extraction: {len(episodes)}/{min_episodes}")
                return None

            # Build context from episodes
            episode_summaries = []
            for ep in episodes:
                summary = {
                    "title": ep.title,
                    "summary": ep.summary,
                    "key_insights": ep.keyInsights,
                    "emotional_state": ep.emotionalState,
                    "date": ep.spanEnd.strftime("%Y-%m-%d")
                }
                episode_summaries.append(summary)

            # Format for LLM analysis
            episodes_text = json.dumps(episode_summaries, indent=2)

            messages = [
                {"role": "system", "content": self.extraction_prompt},
                {"role": "user", "content": f"Extract semantic patterns from these {len(episodes)} conversation summaries:\n\n{episodes_text}"}
            ]

            # Use LLM to extract patterns
            result = self.llm_service.generate_response(
                messages=messages,
                temperature=0.3,  # Lower temp for consistent analysis
                max_tokens=600
            )

            # Parse JSON response
            semantic_data = json.loads(result["content"])

            # Add metadata
            semantic_data["extraction_cost"] = result["cost"]
            semantic_data["episodes_analyzed"] = len(episodes)
            semantic_data["extracted_at"] = datetime.utcnow().isoformat()

            return semantic_data

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse semantic memory JSON: {e}")
            return None
        except Exception as e:
            print(f"❌ Semantic memory extraction error: {e}")
            return None

    async def create_or_update_semantic_memory(
        self,
        clerk_user_id: str,
        user_id: str
    ) -> Optional[str]:
        """
        Create or update semantic memory for a user

        Args:
            clerk_user_id: Clerk user ID
            user_id: Internal user ID

        Returns:
            Semantic memory ID or None if failed
        """
        try:
            # Extract semantic patterns
            semantic_data = await self.extract_semantic_memories(clerk_user_id)

            if not semantic_data:
                return None

            # Check confidence threshold
            confidence = semantic_data.get("confidence", 0.0)
            if confidence < 0.5:
                print(f"⚠️ Semantic memory confidence too low: {confidence}")
                return None

            # Check if semantic memory already exists for this category
            existing = await db.semanticmemory.find_first(
                where={
                    "userId": user_id,
                    "category": "aggregate_patterns",
                    "isActive": True
                }
            )

            if existing:
                # Mark old memory as inactive
                await db.semanticmemory.update(
                    where={"id": existing.id},
                    data={"isActive": False}
                )

            # Serialize patterns as JSON content
            content_data = {
                "recurring_themes": semantic_data.get("recurring_themes", []),
                "behavioral_patterns": semantic_data.get("behavioral_patterns", []),
                "interest_areas": semantic_data.get("interest_areas", []),
                "communication_style": semantic_data.get("communication_preferences", "conversational"),
                "growth_areas": semantic_data.get("growth_areas", []),
                "episodes_analyzed": semantic_data.get("episodes_analyzed", 0)
            }

            # Create new semantic memory using actual schema fields
            now = datetime.utcnow()
            semantic_memory = await db.semanticmemory.create(
                data={
                    "userId": user_id,
                    "category": "aggregate_patterns",
                    "content": json.dumps(content_data),
                    "confidence": "CONFIRMED" if confidence >= 0.7 else "OBSERVED" if confidence >= 0.5 else "HYPOTHESIS",
                    "evidenceCount": semantic_data.get("episodes_analyzed", 1),
                    "firstObservedAt": now,
                    "lastObservedAt": now,
                    "sourceMessageIds": [],
                    "isActive": True
                }
            )

            print(f"✅ Semantic memory created/updated: {semantic_memory.id} (confidence: {confidence})")
            return semantic_memory.id

        except Exception as e:
            print(f"❌ Failed to create semantic memory: {e}")
            return None

    async def get_user_semantic_memory(
        self,
        user_id: Optional[str] = None,
        clerk_user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get current semantic memory for a user

        Args:
            user_id: Internal user ID (optional)
            clerk_user_id: Clerk user ID (optional)

        Returns:
            Semantic memory dict or None
        """
        try:
            # Build where clause based on available ID
            memory = None
            if user_id:
                memory = await db.semanticmemory.find_first(
                    where={
                        "userId": user_id,
                        "category": "aggregate_patterns",
                        "isActive": True
                    }
                )
            elif clerk_user_id:
                # Query via session relationship - first get user IDs from sessions
                sessions = await db.chatsession.find_many(
                    where={"clerkUserId": clerk_user_id}
                )
                user_ids = [s.userId for s in sessions if s.userId]
                
                if not user_ids:
                    return None
                
                memory = await db.semanticmemory.find_first(
                    where={
                        "userId": {"in": user_ids},
                        "category": "aggregate_patterns",
                        "isActive": True
                    }
                )
            
            if not memory:
                return None

            # Parse content JSON
            try:
                content_data = json.loads(memory.content) if memory.content else {}
            except json.JSONDecodeError:
                content_data = {}

            return {
                "id": memory.id,
                "recurring_themes": content_data.get("recurring_themes", []),
                "behavioral_patterns": content_data.get("behavioral_patterns", []),
                "interest_areas": content_data.get("interest_areas", []),
                "communication_style": content_data.get("communication_style", "conversational"),
                "growth_areas": content_data.get("growth_areas", []),
                "confidence": memory.confidence,
                "episodes_analyzed": content_data.get("episodes_analyzed", memory.evidenceCount),
                "created_at": memory.createdAt.isoformat()
            }

        except Exception as e:
            print(f"❌ Failed to get semantic memory: {e}")
            return None

    def format_semantic_memory_for_context(
        self,
        semantic_memory: Dict[str, Any]
    ) -> str:
        """
        Format semantic memory as context string for LLM

        Args:
            semantic_memory: Semantic memory dict

        Returns:
            Formatted context string
        """
        if not semantic_memory:
            return ""

        context_parts = [
            "=== LONG-TERM USER PATTERNS (Semantic Memory) ===",
            "",
            "Based on analysis of past conversations:"
        ]

        if semantic_memory.get("recurring_themes"):
            context_parts.append(f"• Recurring Themes: {', '.join(semantic_memory['recurring_themes'][:3])}")

        if semantic_memory.get("behavioral_patterns"):
            context_parts.append(f"• Behavioral Patterns: {', '.join(semantic_memory['behavioral_patterns'][:3])}")

        if semantic_memory.get("interest_areas"):
            context_parts.append(f"• Key Interests: {', '.join(semantic_memory['interest_areas'][:3])}")

        if semantic_memory.get("communication_style"):
            context_parts.append(f"• Communication Style: {semantic_memory['communication_style']}")

        if semantic_memory.get("growth_areas"):
            context_parts.append(f"• Growth Areas: {', '.join(semantic_memory['growth_areas'][:3])}")

        context_parts.append("")
        context_parts.append(f"(Confidence: {int(semantic_memory.get('confidence', 0) * 100)}%, based on {semantic_memory.get('episodes_analyzed', 0)} conversations)")
        context_parts.append("===")
        context_parts.append("")

        return "\n".join(context_parts)

    async def should_extract_semantic_memory(
        self,
        clerk_user_id: str
    ) -> bool:
        """
        Check if it's time to extract semantic memory

        Args:
            clerk_user_id: Clerk user ID

        Returns:
            True if extraction should be triggered
        """
        try:
            # Get current semantic memory
            # Note: Need to get user_id from clerk_user_id first
            # For now, check episode count directly

            episodes = await db.episodicmemory.find_many(
                where={
                    "session": {
                        "clerkUserId": clerk_user_id
                    }
                }
            )

            # Extract semantic memory every 5 episodes
            return len(episodes) >= 5 and len(episodes) % 5 == 0

        except Exception as e:
            print(f"❌ Error checking semantic extraction trigger: {e}")
            return False
