"""
Redis Service - Working Memory Cache
Stores temporary conversation context before compression
"""
import os
import json
import redis
from typing import List, Dict, Any, Optional
from datetime import timedelta


class RedisService:
    """
    Redis-based working memory for chat conversations

    Caches:
    - Recent conversation messages (before DB write)
    - User context (SELVE scores, preferences)
    - Conversation state (topics discussed, emotional tone)
    """

    def __init__(self):
        """Initialize Redis connection"""
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))

        self.client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True  # Auto-decode bytes to strings
        )

        # Default TTL for working memory (1 hour)
        self.default_ttl = timedelta(hours=1)

    def ping(self) -> bool:
        """Check if Redis is connected"""
        try:
            return self.client.ping()
        except Exception:
            return False

    # ========================================================================
    # Conversation Context Caching
    # ========================================================================

    def cache_conversation_context(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Cache recent conversation messages for quick retrieval

        Args:
            session_id: Chat session ID
            messages: List of messages [{"role": "user", "content": "..."}]
            ttl: Time to live (default: 1 hour)

        Returns:
            True if cached successfully
        """
        try:
            key = f"conversation:{session_id}"
            value = json.dumps(messages)
            ttl = ttl or self.default_ttl

            self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            print(f"❌ Redis cache error: {e}")
            return False

    def get_conversation_context(
        self,
        session_id: str
    ) -> Optional[List[Dict[str, str]]]:
        """Retrieve cached conversation context"""
        try:
            key = f"conversation:{session_id}"
            cached = self.client.get(key)

            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            print(f"❌ Redis get error: {e}")
            return None

    def append_message_to_cache(
        self,
        session_id: str,
        message: Dict[str, str]
    ) -> bool:
        """Append a single message to cached conversation"""
        try:
            messages = self.get_conversation_context(session_id) or []
            messages.append(message)
            return self.cache_conversation_context(session_id, messages)
        except Exception:
            return False

    # ========================================================================
    # User Context Caching
    # ========================================================================

    def cache_user_context(
        self,
        user_id: str,
        context: Dict[str, Any],
        ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Cache user context (SELVE scores, preferences, demographics)

        Args:
            user_id: User ID
            context: User context data
            ttl: Time to live (default: 1 hour)
        """
        try:
            key = f"user_context:{user_id}"
            value = json.dumps(context)
            ttl = ttl or self.default_ttl

            self.client.setex(key, ttl, value)
            return True
        except Exception:
            return False

    def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached user context"""
        try:
            key = f"user_context:{user_id}"
            cached = self.client.get(key)

            if cached:
                return json.loads(cached)
            return None
        except Exception:
            return None

    # ========================================================================
    # Conversation State Tracking
    # ========================================================================

    def track_conversation_state(
        self,
        session_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """
        Track conversation state (topics, emotional tone, unresolved questions)

        State structure:
        {
            "topics_discussed": ["stress", "relationships"],
            "emotional_tone": "concerned",
            "unresolved_questions": ["How to manage stress?"],
            "dimensions_referenced": ["AETHER", "ORPHEUS"]
        }
        """
        try:
            key = f"conversation_state:{session_id}"
            value = json.dumps(state)

            self.client.setex(key, self.default_ttl, value)
            return True
        except Exception:
            return False

    def get_conversation_state(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve conversation state"""
        try:
            key = f"conversation_state:{session_id}"
            cached = self.client.get(key)

            if cached:
                return json.loads(cached)
            return None
        except Exception:
            return None

    # ========================================================================
    # Compression Triggers
    # ========================================================================

    def check_compression_needed(
        self,
        session_id: str,
        current_tokens: int,
        max_tokens: int = 100000  # Claude Haiku context window
    ) -> bool:
        """
        Check if conversation needs compression (70% threshold)

        Args:
            session_id: Session ID
            current_tokens: Current total tokens in conversation
            max_tokens: Maximum context window size

        Returns:
            True if compression needed (>70% capacity)
        """
        threshold = max_tokens * 0.7
        return current_tokens >= threshold

    def mark_compression_needed(self, session_id: str) -> bool:
        """Flag a session for compression"""
        try:
            key = f"needs_compression:{session_id}"
            self.client.setex(key, timedelta(minutes=5), "1")
            return True
        except Exception:
            return False

    def is_compression_flagged(self, session_id: str) -> bool:
        """Check if session is flagged for compression"""
        try:
            key = f"needs_compression:{session_id}"
            return self.client.exists(key) > 0
        except Exception:
            return False

    def clear_compression_flag(self, session_id: str) -> bool:
        """Clear compression flag after compression is done"""
        try:
            key = f"needs_compression:{session_id}"
            self.client.delete(key)
            return True
        except Exception:
            return False

    # ========================================================================
    # Cache Management
    # ========================================================================

    def clear_session_cache(self, session_id: str) -> bool:
        """Clear all cached data for a session"""
        try:
            keys = [
                f"conversation:{session_id}",
                f"conversation_state:{session_id}",
                f"needs_compression:{session_id}"
            ]
            self.client.delete(*keys)
            return True
        except Exception:
            return False

    def clear_user_cache(self, user_id: str) -> bool:
        """Clear all cached data for a user"""
        try:
            key = f"user_context:{user_id}"
            self.client.delete(key)
            return True
        except Exception:
            return False
