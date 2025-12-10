"""
RAG Service for SELVE Chatbot
Retrieves relevant context from Qdrant vector database.

Security & Robustness:
- Input validation and sanitization
- Query length limits to prevent abuse
- Proper error handling with Result types
- No sensitive data in logs
- Connection pooling via ClientManager
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, VectorParams

from .base import (
    BaseService,
    Config,
    ExternalServiceError,
    Result,
    Validator,
    with_error_handling,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class RetrievedChunk:
    """A single retrieved context chunk with metadata."""

    content: str
    title: str
    score: float
    source: str
    dimension: str = ""
    section: str = ""
    chunk_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "text": self.content,  # Backward compatibility
            "title": self.title,
            "score": self.score,
            "source": self.source,
            "dimension": self.dimension,
            "section": self.section,
        }


@dataclass
class RAGResult:
    """Result of a RAG retrieval operation."""

    context: str
    chunks: List[RetrievedChunk]
    retrieved_count: int
    query_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context": self.context,
            "chunks": [c.to_dict() for c in self.chunks],
            "retrieved_count": self.retrieved_count,
        }


# =============================================================================
# RAG SERVICE
# =============================================================================


class RAGService(BaseService):
    """
    Service for RAG (Retrieval-Augmented Generation).

    Retrieves relevant context from Qdrant vector database based on
    semantic similarity to user queries.

    Features:
    - Input validation and sanitization
    - Configurable top-k and score thresholds
    - Source filtering
    - Proper error handling
    - No direct Langfuse instrumentation (handled at ChatService level)
    """

    # Default configuration
    DEFAULT_TOP_K: int = 3
    DEFAULT_SCORE_THRESHOLD: float = 0.3
    MIN_SCORE_THRESHOLD: float = 0.0
    MAX_SCORE_THRESHOLD: float = 1.0
    MAX_TOP_K: int = 10

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize RAG service.

        Args:
            collection_name: Qdrant collection name (defaults to config)
            embedding_model: OpenAI embedding model (defaults to config)
        """
        super().__init__()
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self._collection_verified = False

    def _validate_config(self) -> None:
        """Validate RAG-specific configuration."""
        if not Config.OPENAI_API_KEY:
            self.logger.warning("OPENAI_API_KEY not set - embedding generation will fail")

    def _ensure_collection_exists(self) -> bool:
        """
        Verify collection exists in Qdrant.

        Returns:
            True if collection exists or was created, False on error
        """
        if self._collection_verified:
            return True

        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.logger.warning(
                    f"Collection '{self.collection_name}' not found. "
                    "Content ingestion may be required."
                )
                # Don't auto-create - let ingestion service handle it
                return False

            self._collection_verified = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to verify collection: {e}")
            return False

    def generate_embedding(self, text: str) -> Result[List[float]]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed (will be validated and truncated if needed)

        Returns:
            Result containing embedding vector or error
        """
        # Validate input
        try:
            text = Validator.validate_string(
                text,
                "text",
                min_length=1,
                max_length=Config.MAX_QUERY_LENGTH,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            response = self.openai.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            embedding = response.data[0].embedding
            return Result.success(
                embedding,
                tokens=response.usage.total_tokens,
                model=self.embedding_model,
            )

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise ExternalServiceError("OpenAI", "Failed to generate embedding", e)

    def retrieve_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        source_filter: Optional[str] = None,
    ) -> Result[List[RetrievedChunk]]:
        """
        Retrieve relevant context from Qdrant.

        Args:
            query: User's query text
            top_k: Number of top results (1-10, default 3)
            score_threshold: Minimum similarity score (0-1, default 0.3)
            source_filter: Optional source to filter by

        Returns:
            Result containing list of RetrievedChunk objects
        """
        # Validate inputs
        try:
            query = Validator.validate_string(
                query,
                "query",
                min_length=1,
                max_length=Config.MAX_QUERY_LENGTH,
            )
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
                min_value=self.MIN_SCORE_THRESHOLD,
                max_value=self.MAX_SCORE_THRESHOLD,
                default=self.DEFAULT_SCORE_THRESHOLD,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        # Ensure collection exists
        if not self._ensure_collection_exists():
            return Result.success([])  # Return empty rather than error

        # Generate query embedding
        embedding_result = self.generate_embedding(query)
        if embedding_result.is_error:
            return Result.failure(
                embedding_result.error or "Embedding generation failed",
                error_code=embedding_result.error_code,
            )

        query_embedding = embedding_result.data

        try:
            # Build optional filter
            search_filter = None
            if source_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_filter),
                        )
                    ]
                )

            # Search in Qdrant
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            )

            # Filter by score threshold and convert to RetrievedChunk objects
            chunks: List[RetrievedChunk] = []
            for result in results:
                if result.score >= score_threshold:
                    payload = result.payload or {}

                    # Extract text content (handle different payload structures)
                    text_content = (
                        payload.get("text")
                        or payload.get("content")
                        or ""
                    )

                    chunk = RetrievedChunk(
                        content=text_content,
                        title=payload.get("title", ""),
                        score=round(result.score, 4),
                        source=payload.get("source", "knowledge_base"),
                        dimension=payload.get("dimension_name", ""),
                        section=payload.get("section", ""),
                        chunk_index=payload.get("chunk_index"),
                    )
                    chunks.append(chunk)

            self.logger.debug(
                f"Retrieved {len(chunks)} chunks for query "
                f"(top_k={top_k}, threshold={score_threshold})"
            )

            return Result.success(
                chunks,
                query_tokens=embedding_result.metadata.get("tokens"),
            )

        except Exception as e:
            self.logger.error(f"Qdrant search failed: {e}")
            raise ExternalServiceError("Qdrant", "Search operation failed", e)

    def format_context_for_prompt(
        self,
        chunks: List[RetrievedChunk],
        include_scores: bool = True,
        max_chunks: int = 5,
    ) -> str:
        """
        Format retrieved chunks into a prompt string.

        Args:
            chunks: List of retrieved chunks
            include_scores: Whether to include relevance scores
            max_chunks: Maximum chunks to include

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        # Limit chunks
        chunks = chunks[:max_chunks]

        parts = ["Relevant SELVE Framework Context:", ""]

        for i, chunk in enumerate(chunks, 1):
            if include_scores:
                parts.append(f"[{i}] {chunk.title} (relevance: {chunk.score:.2f})")
            else:
                parts.append(f"[{i}] {chunk.title}")
            parts.append(chunk.content)
            parts.append("")

        return "\n".join(parts).strip()

    def get_context_for_query(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main method to get formatted context for a query.

        This is the primary interface for the ChatService.

        Args:
            query: User's query
            top_k: Number of results
            score_threshold: Minimum score
            source_filter: Optional source filter

        Returns:
            Dictionary with context, chunks, and metadata
        """
        result = self.retrieve_context(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            source_filter=source_filter,
        )

        if result.is_error:
            self.logger.warning(f"Context retrieval failed: {result.error}")
            return {
                "context": "",
                "chunks": [],
                "retrieved_count": 0,
                "error": result.error,
            }

        chunks = result.data or []
        formatted_context = self.format_context_for_prompt(chunks)

        return {
            "context": formatted_context,
            "chunks": [c.to_dict() for c in chunks],
            "retrieved_count": len(chunks),
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of RAG service dependencies.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "collection_exists": False,
            "openai_connected": False,
            "qdrant_connected": False,
        }

        # Check Qdrant
        try:
            collections = self.qdrant.get_collections().collections
            health["qdrant_connected"] = True
            health["collection_exists"] = any(
                c.name == self.collection_name for c in collections
            )
        except Exception as e:
            health["status"] = "degraded"
            health["qdrant_error"] = str(e)

        # Check OpenAI (simple test)
        try:
            self.openai.models.list()
            health["openai_connected"] = True
        except Exception as e:
            health["status"] = "degraded"
            health["openai_error"] = str(e)

        if not health["qdrant_connected"] or not health["openai_connected"]:
            health["status"] = "unhealthy"

        return health