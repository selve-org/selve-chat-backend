"""
Compression Service - Automatic conversation compression at 70% context window
Preserves conversation narrative while reducing token count
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.db import db
from .llm_service import LLMService
from .memory_search_service import MemorySearchService


class CompressionService:
    """
    Service for compressing long conversations into episodic memories

    Compression Strategy (Friend 1 approach from plan):
    - Trigger at 70% of context window (70,000 tokens for Claude Haiku)
    - Compress oldest messages into episodic memory
    - Keep most recent 30% of conversation uncompressed
    - Extract key insights, topics, and emotional states
    - Store in database and embed for semantic search
    """

    # Context window sizes (in tokens)
    CONTEXT_WINDOWS = {
        "claude-3-5-haiku-20241022": 100000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-opus-4-20250514": 200000,
        "gpt-4o-mini": 128000,
        "gpt-5-nano": 128000,
    }

    # Compression threshold (70% of context window)
    COMPRESSION_THRESHOLD = 0.7

    # Keep last 30% uncompressed
    KEEP_RECENT_RATIO = 0.3

    def __init__(self):
        """Initialize compression service with LLM for summarization"""
        # Use GPT-4o-mini for compression (cheaper and good at summarization)
        self.llm_service = LLMService()
        self.memory_search_service = MemorySearchService()
        self.compression_prompt = self._load_compression_prompt()

    def _load_compression_prompt(self) -> str:
        """Load the compression system prompt"""
        return """You are a conversation summarization assistant for the SELVE chatbot.

Your task is to compress a segment of conversation into a concise episodic memory while preserving:
1. Key insights about the user's personality or situation
2. Topics discussed and questions asked
3. Emotional state and tone of the conversation
4. Any unresolved topics or follow-up items
5. Behavioral patterns observed

Format your response as JSON with these fields:
{
    "title": "Brief title summarizing this conversation segment",
    "summary": "2-3 paragraph summary of the conversation",
    "key_insights": ["insight 1", "insight 2", ...],
    "topics_discussed": ["topic 1", "topic 2", ...],
    "emotional_state": "user's emotional state (e.g., 'curious', 'concerned', 'excited')",
    "unresolved_topics": ["topic 1", "topic 2", ...],
    "dimensions_referenced": ["DIMENSION1", "DIMENSION2", ...]
}

Be concise but preserve important details. Focus on what matters for future conversations."""

    def needs_compression(
        self,
        total_tokens: int,
        model: str = "claude-3-5-haiku-20241022"
    ) -> bool:
        """
        Check if conversation needs compression

        Args:
            total_tokens: Current total tokens in conversation
            model: LLM model being used

        Returns:
            True if compression needed (>70% capacity)
        """
        max_tokens = self.CONTEXT_WINDOWS.get(model, 100000)
        threshold = max_tokens * self.COMPRESSION_THRESHOLD

        return total_tokens >= threshold

    async def compress_conversation(
        self,
        session_id: str,
        user_id: str,
        clerk_user_id: str
    ) -> Dict[str, Any]:
        """
        Compress a conversation session

        Args:
            session_id: Chat session ID
            user_id: Internal user ID
            clerk_user_id: Clerk authentication ID

        Returns:
            {
                "compressed": bool,
                "episodic_memory_id": str,
                "messages_compressed": int,
                "tokens_saved": int,
                "summary": str
            }
        """
        # Get session with messages
        session = await db.chatsession.find_unique(
            where={"id": session_id},
            include={"messages": {"orderBy": {"createdAt": "asc"}}}
        )

        if not session or not session.messages:
            return {
                "compressed": False,
                "error": "Session not found or has no messages"
            }

        total_tokens = session.totalTokens
        messages = session.messages

        # Calculate how many messages to compress (oldest 70%)
        keep_count = int(len(messages) * self.KEEP_RECENT_RATIO)
        compress_count = len(messages) - keep_count

        if compress_count <= 0:
            return {
                "compressed": False,
                "error": "Not enough messages to compress"
            }

        # Get messages to compress and messages to keep
        messages_to_compress = messages[:compress_count]
        messages_to_keep = messages[compress_count:]

        # Build conversation text for compression
        conversation_text = self._build_conversation_text(messages_to_compress)

        # Generate compression summary using LLM
        compression_result = await self._generate_compression(conversation_text)

        if not compression_result:
            return {
                "compressed": False,
                "error": "Compression generation failed"
            }

        # Calculate tokens saved
        compressed_tokens = sum(msg.tokenCount for msg in messages_to_compress)
        tokens_saved = compressed_tokens - compression_result.get("summary_tokens", 0)

        # Create episodic memory in database
        span_start = messages_to_compress[0].createdAt
        span_end = messages_to_compress[-1].createdAt

        episodic_memory = await db.episodicmemory.create(
            data={
                "userId": user_id,
                "sessionId": session_id,
                "title": compression_result["title"],
                "summary": compression_result["summary"],
                "keyInsights": compression_result["key_insights"],
                "unresolvedTopics": compression_result.get("unresolved_topics", []),
                "emotionalState": compression_result.get("emotional_state"),
                "sourceMessageIds": [msg.id for msg in messages_to_compress],
                "compressionModel": self.llm_service.model,
                "compressionCost": compression_result.get("cost", 0),
                "spanStart": span_start,
                "spanEnd": span_end,
                "embedded": False  # Will be embedded in next phase
            }
        )

        # Mark compressed messages as compressed
        for msg in messages_to_compress:
            await db.chatmessage.update(
                where={"id": msg.id},
                data={
                    "isCompressed": True,
                    "compressedAt": datetime.utcnow()
                }
            )

        # Update session compression count
        await db.chatsession.update(
            where={"id": session_id},
            data={"compressionCount": {"increment": 1}}
        )

        # Embed the memory for vector search (async, non-blocking)
        try:
            await self.memory_search_service.embed_memory(episodic_memory.id)
        except Exception as e:
            print(f"⚠️ Failed to embed memory {episodic_memory.id}: {e}")

        return {
            "compressed": True,
            "episodic_memory_id": episodic_memory.id,
            "messages_compressed": compress_count,
            "messages_kept": keep_count,
            "tokens_saved": tokens_saved,
            "compression_ratio": round(tokens_saved / compressed_tokens * 100, 1),
            "summary": compression_result["summary"],
            "title": compression_result["title"]
        }

    def _build_conversation_text(self, messages: List[Any]) -> str:
        """Build conversation text from messages for compression"""
        lines = []

        for msg in messages:
            role = msg.role.upper()
            content = msg.content
            timestamp = msg.createdAt.strftime("%Y-%m-%d %H:%M")

            lines.append(f"[{timestamp}] {role}: {content}")

        return "\n".join(lines)

    async def _generate_compression(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """
        Generate compression summary using LLM

        Args:
            conversation_text: Full conversation text to compress

        Returns:
            Dict with compression results or None if failed
        """
        try:
            messages = [
                {"role": "system", "content": self.compression_prompt},
                {"role": "user", "content": f"Compress this conversation:\n\n{conversation_text}"}
            ]

            # Use LLM to generate compression
            result = self.llm_service.generate_response(
                messages=messages,
                temperature=0.5,  # Lower temperature for consistent summaries
                max_tokens=1000
            )

            # Parse JSON response
            import json
            compression_data = json.loads(result["content"])

            # Add token and cost info
            compression_data["summary_tokens"] = result["usage"]["output_tokens"]
            compression_data["cost"] = result["cost"]

            return compression_data

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse compression JSON: {e}")
            return None
        except Exception as e:
            print(f"❌ Compression generation error: {e}")
            return None

    async def get_session_memories(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all episodic memories for a session

        Args:
            session_id: Chat session ID

        Returns:
            List of episodic memories
        """
        memories = await db.episodicmemory.find_many(
            where={"sessionId": session_id},
            order={"spanStart": "asc"}
        )

        return [
            {
                "id": mem.id,
                "title": mem.title,
                "summary": mem.summary,
                "key_insights": mem.keyInsights,
                "topics": mem.unresolvedTopics,
                "emotional_state": mem.emotionalState,
                "span_start": mem.spanStart.isoformat(),
                "span_end": mem.spanEnd.isoformat()
            }
            for mem in memories
        ]

    async def should_trigger_compression(
        self,
        session_id: str
    ) -> bool:
        """
        Check if a session should trigger compression

        Args:
            session_id: Chat session ID

        Returns:
            True if compression should be triggered
        """
        session = await db.chatsession.find_unique(
            where={"id": session_id}
        )

        if not session:
            return False

        return self.needs_compression(session.totalTokens)

    async def get_user_memories(
        self,
        clerk_user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent episodic memories for a user (across all sessions)

        Args:
            clerk_user_id: Clerk user ID
            limit: Maximum number of memories to retrieve

        Returns:
            List of episodic memories with summaries
        """
        # First get user's internal ID
        # Note: This assumes a User table exists. If not, we query by clerkUserId directly
        memories = await db.episodicmemory.find_many(
            where={
                "session": {
                    "clerkUserId": clerk_user_id
                }
            },
            order={"spanEnd": "desc"},
            take=limit,
            include={"session": True}
        )

        return [
            {
                "id": mem.id,
                "title": mem.title,
                "summary": mem.summary,
                "key_insights": mem.keyInsights,
                "emotional_state": mem.emotionalState,
                "span_start": mem.spanStart.isoformat(),
                "span_end": mem.spanEnd.isoformat(),
                "session_title": mem.session.title if mem.session else None
            }
            for mem in memories
        ]

    def format_memories_for_context(
        self,
        memories: List[Dict[str, Any]]
    ) -> str:
        """
        Format episodic memories as context string for LLM

        Args:
            memories: List of episodic memories

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        context_parts = [
            "=== CONVERSATION HISTORY (Episodic Memories) ===",
            "",
            "You have access to compressed summaries from previous conversations:",
            ""
        ]

        for i, mem in enumerate(memories, 1):
            context_parts.append(f"Memory {i}: {mem['title']}")
            context_parts.append(f"Date: {mem['span_end'][:10]}")
            context_parts.append(f"Summary: {mem['summary']}")

            if mem.get("key_insights"):
                context_parts.append(f"Key Insights: {', '.join(mem['key_insights'][:3])}")

            context_parts.append("")

        context_parts.append("Use these memories to maintain conversation continuity and recall past discussions.")
        context_parts.append("===")
        context_parts.append("")

        return "\n".join(context_parts)
