"""
Memory Search Service - Vector search through episodic memories.
Uses Qdrant for semantic similarity matching.

Security & Robustness:
- User isolation (memories filtered by user_id)
- Input validation and sanitization
- Proper async/sync boundary handling
- Rate limiting ready (via Config)
- No sensitive data in logs
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .base import (
    BaseService,
    Config,
    ExternalServiceError,
    Result,
    Validator,
    string_to_point_id,
    with_error_handling,
    with_timeout,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class MemorySearchResult:
    """A single memory search result."""

    memory_id: str
    title: str
    summary: str
    relevance_score: float
    key_insights: List[str]
    emotional_state: Optional[str] = None
    unresolved_topics: List[str] = None
    span_start: Optional[str] = None
    span_end: Optional[str] = None

    def __post_init__(self):
        if self.unresolved_topics is None:
            self.unresolved_topics = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "title": self.title,
            "summary": self.summary,
            "relevance_score": self.relevance_score,
            "key_insights": self.key_insights,
            "emotional_state": self.emotional_state,
            "unresolved_topics": self.unresolved_topics,
            "span_start": self.span_start,
            "span_end": self.span_end,
        }


# =============================================================================
# MEMORY SEARCH SERVICE
# =============================================================================


class MemorySearchService(BaseService):
    """
    Service for vector search through episodic memories.

    Features:
    - User-scoped memory search (security isolation)
    - Semantic similarity search via Qdrant
    - Memory embedding and indexing
    - Related memory discovery
    - Lazy collection initialization

    Security:
    - All searches are scoped to user_id (mandatory filter)
    - Input validation on all public methods
    - No cross-user data leakage
    """

    COLLECTION_NAME = "episodic_memories"
    DEFAULT_TOP_K = 5
    DEFAULT_SCORE_THRESHOLD = 0.5
    MAX_TOP_K = 20

    def __init__(self):
        """Initialize memory search service."""
        super().__init__()
        self._collection_initialized = False

    def _ensure_collection_exists(self) -> bool:
        """
        Create the episodic memories collection if it doesn't exist.

        Returns:
            True if collection exists or was created
        """
        if self._collection_initialized:
            return True

        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.COLLECTION_NAME not in collection_names:
                self.qdrant.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=Config.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE,
                    ),
                )
                self.logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME}")

            self._collection_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to ensure collection exists: {e}")
            return False

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ExternalServiceError: If OpenAI call fails
        """
        try:
            response = self.openai.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise ExternalServiceError("OpenAI", "Failed to generate embedding", e)

    def _build_embedding_text(
        self,
        title: str,
        summary: str,
        key_insights: Optional[List[str]] = None,
    ) -> str:
        """
        Build text for embedding from memory components.

        Args:
            title: Memory title
            summary: Memory summary
            key_insights: Optional list of key insights

        Returns:
            Combined text for embedding
        """
        parts = [title, "", summary]

        if key_insights:
            parts.extend(["", "Key Insights:"])
            parts.extend(f"- {insight}" for insight in key_insights)

        return "\n".join(parts)

    async def embed_memory(
        self,
        memory_id: str,
        user_id: str,
        session_id: str,
        title: str,
        summary: str,
        key_insights: Optional[List[str]] = None,
        emotional_state: Optional[str] = None,
        unresolved_topics: Optional[List[str]] = None,
        span_start: Optional[datetime] = None,
        span_end: Optional[datetime] = None,
    ) -> Result[bool]:
        """
        Embed an episodic memory and store in Qdrant.

        This method is called after creating an episodic memory in the database.
        It generates an embedding and stores it for semantic search.

        Args:
            memory_id: Unique memory identifier
            user_id: User ID (for filtering)
            session_id: Chat session ID
            title: Memory title
            summary: Memory summary
            key_insights: Key insights from conversation
            emotional_state: User's emotional state
            unresolved_topics: Topics to follow up on
            span_start: Memory time span start
            span_end: Memory time span end

        Returns:
            Result indicating success or failure
        """
        # Validate required inputs
        try:
            memory_id = Validator.validate_string(memory_id, "memory_id", min_length=5)
            user_id = Validator.validate_user_id(user_id, "user_id")
            title = Validator.validate_string(title, "title", max_length=500)
            summary = Validator.validate_string(summary, "summary", max_length=5000)
        except Exception as e:
            return Result.validation_error(str(e))

        # Ensure collection exists
        if not self._ensure_collection_exists():
            return Result.failure(
                "Failed to initialize memory collection",
                error_code="COLLECTION_ERROR",
            )

        try:
            # Build embedding text
            embedding_text = self._build_embedding_text(title, summary, key_insights)

            # Generate embedding
            embedding_vector = self._generate_embedding(embedding_text)

            # Convert memory ID to point ID
            point_id = string_to_point_id(memory_id)

            # Build payload
            payload = {
                "memory_id": memory_id,
                "user_id": user_id,  # CRITICAL: Used for user isolation
                "session_id": session_id,
                "title": title,
                "summary": summary,
                "key_insights": key_insights or [],
                "emotional_state": emotional_state,
                "unresolved_topics": unresolved_topics or [],
                "text": embedding_text,
                "embedded_at": datetime.utcnow().isoformat(),
            }

            # Add optional datetime fields
            if span_start:
                payload["span_start"] = span_start.isoformat()
            if span_end:
                payload["span_end"] = span_end.isoformat()

            # Upsert to Qdrant
            point = PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload=payload,
            )

            self.qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point],
            )

            self.logger.info(f"Embedded memory {memory_id[:8]}... for user {user_id[:8]}...")
            return Result.success(True)

        except ExternalServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to embed memory: {e}")
            return Result.failure(str(e), error_code="EMBEDDING_ERROR")

    async def search_memories(
        self,
        query: str,
        user_id: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> Result[List[MemorySearchResult]]:
        """
        Search episodic memories using vector similarity.

        SECURITY: All searches are filtered by user_id to prevent
        cross-user data access.

        Args:
            query: Search query
            user_id: User ID (REQUIRED for security)
            top_k: Number of results (1-20)
            score_threshold: Minimum similarity score (0-1)

        Returns:
            Result containing list of matching memories
        """
        # Validate inputs - user_id is REQUIRED for security
        try:
            query = Validator.validate_string(
                query,
                "query",
                min_length=1,
                max_length=Config.MAX_QUERY_LENGTH,
            )
            user_id = Validator.validate_user_id(user_id, "user_id")
            top_k = Validator.validate_positive_int(
                top_k,
                "top_k",
                min_value=1,
                max_value=self.MAX_TOP_K,
                default=self.DEFAULT_TOP_K,
            )
            score_threshold = Validator.validate_float_range(
                score_threshold,
                "score_threshold",
                min_value=0.0,
                max_value=1.0,
                default=self.DEFAULT_SCORE_THRESHOLD,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        # Ensure collection exists
        if not self._ensure_collection_exists():
            return Result.success([])  # Empty result if no collection

        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # SECURITY: Always filter by user_id
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            )

            # Search in Qdrant
            results = self.qdrant.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            )

            # Convert to MemorySearchResult objects
            memories: List[MemorySearchResult] = []
            for result in results:
                if result.score >= score_threshold:
                    payload = result.payload or {}
                    memory = MemorySearchResult(
                        memory_id=payload.get("memory_id", ""),
                        title=payload.get("title", ""),
                        summary=payload.get("summary", ""),
                        relevance_score=round(result.score, 3),
                        key_insights=payload.get("key_insights", []),
                        emotional_state=payload.get("emotional_state"),
                        unresolved_topics=payload.get("unresolved_topics", []),
                        span_start=payload.get("span_start"),
                        span_end=payload.get("span_end"),
                    )
                    memories.append(memory)

            self.logger.debug(
                f"Found {len(memories)} memories for user {user_id[:8]}..."
            )
            return Result.success(memories)

        except ExternalServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return Result.failure(str(e), error_code="SEARCH_ERROR")

    async def get_related_memories(
        self,
        memory_id: str,
        user_id: str,
        top_k: int = 3,
    ) -> Result[List[MemorySearchResult]]:
        """
        Find memories similar to a given memory.

        Args:
            memory_id: Memory ID to find similar memories for
            user_id: User ID (REQUIRED for security)
            top_k: Number of similar memories to return

        Returns:
            Result containing list of similar memories
        """
        # Validate inputs
        try:
            memory_id = Validator.validate_string(memory_id, "memory_id", min_length=5)
            user_id = Validator.validate_user_id(user_id, "user_id")
            top_k = Validator.validate_positive_int(
                top_k,
                "top_k",
                min_value=1,
                max_value=10,
                default=3,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        if not self._ensure_collection_exists():
            return Result.success([])

        try:
            # Get the memory's point from Qdrant
            point_id = string_to_point_id(memory_id)
            memory_points = self.qdrant.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[point_id],
                with_vectors=True,
            )

            if not memory_points:
                return Result.success([])

            # SECURITY: Verify the memory belongs to this user
            point_user_id = memory_points[0].payload.get("user_id")
            if point_user_id != user_id:
                self.logger.warning(
                    f"User {user_id[:8]}... attempted to access memory "
                    f"belonging to {point_user_id[:8] if point_user_id else 'unknown'}..."
                )
                return Result.failure(
                    "Memory not found",
                    error_code="NOT_FOUND",
                )

            # Search for similar memories (excluding the original)
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            )

            results = self.qdrant.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=memory_points[0].vector,
                limit=top_k + 1,  # +1 to account for self
                query_filter=search_filter,
            )

            # Filter out the original memory
            memories: List[MemorySearchResult] = []
            for result in results:
                payload = result.payload or {}
                if payload.get("memory_id") != memory_id:
                    memory = MemorySearchResult(
                        memory_id=payload.get("memory_id", ""),
                        title=payload.get("title", ""),
                        summary=payload.get("summary", ""),
                        relevance_score=round(result.score, 3),
                        key_insights=payload.get("key_insights", []),
                    )
                    memories.append(memory)

            return Result.success(memories[:top_k])

        except Exception as e:
            self.logger.error(f"Failed to find related memories: {e}")
            return Result.failure(str(e), error_code="SEARCH_ERROR")

    async def delete_memory(
        self,
        memory_id: str,
        user_id: str,
    ) -> Result[bool]:
        """
        Delete a memory from the vector store.

        Args:
            memory_id: Memory ID to delete
            user_id: User ID (for verification)

        Returns:
            Result indicating success
        """
        try:
            memory_id = Validator.validate_string(memory_id, "memory_id", min_length=5)
            user_id = Validator.validate_user_id(user_id, "user_id")
        except Exception as e:
            return Result.validation_error(str(e))

        if not self._ensure_collection_exists():
            return Result.success(True)  # Nothing to delete

        try:
            point_id = string_to_point_id(memory_id)

            # Verify ownership before deletion
            points = self.qdrant.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[point_id],
            )

            if points:
                point_user_id = points[0].payload.get("user_id")
                if point_user_id != user_id:
                    return Result.failure(
                        "Memory not found",
                        error_code="NOT_FOUND",
                    )

            # Delete the point
            self.qdrant.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[point_id],
            )

            self.logger.info(f"Deleted memory {memory_id[:8]}...")
            return Result.success(True)

        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return Result.failure(str(e), error_code="DELETE_ERROR")

    def format_search_results_for_context(
        self,
        memories: List[MemorySearchResult],
        max_memories: int = 3,
    ) -> str:
        """
        Format search results as context string for LLM.

        Args:
            memories: List of memory search results
            max_memories: Maximum memories to include

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        memories = memories[:max_memories]

        parts = [
            "=== RELEVANT PAST CONVERSATIONS ===",
            "",
            "Memories from previous conversations:",
            "",
        ]

        for i, mem in enumerate(memories, 1):
            parts.append(
                f"{i}. {mem.title} (relevance: {mem.relevance_score:.0%})"
            )
            parts.append(f"   {mem.summary}")

            if mem.key_insights:
                insights = ", ".join(mem.key_insights[:2])
                parts.append(f"   Key insights: {insights}")

            parts.append("")

        parts.append(
            "Use these memories to provide continuity and recall relevant past discussions."
        )
        parts.append("===")

        return "\n".join(parts)