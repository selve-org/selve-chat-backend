"""
Content Ingestion Service - Ingest, chunk, embed, and store SELVE content.

Handles the full pipeline from raw content to searchable vectors:
1. Content validation and sanitization
2. Semantic chunking with overlap
3. Embedding generation (batched)
4. Qdrant storage with metadata
5. Deduplication

Security & Robustness:
- Content size limits to prevent abuse
- Batch processing limits
- Deduplication via content hashing
- Input validation on all methods
- Cost tracking and limits
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client.models import Distance, PointStruct, VectorParams

from .base import (
    BaseService,
    Config,
    ExternalServiceError,
    Result,
    Validator,
    generate_content_hash,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class ChunkingConfig:
    """Configuration for content chunking."""

    max_chunk_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_tokens: int = 50
    chars_per_token: int = 4  # Approximate


@dataclass
class IngestionResult:
    """Result of a single content ingestion."""

    success: bool
    content_hash: str
    chunks_created: int = 0
    embedding_cost: float = 0.0
    validation_status: str = "not_validated"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ingested": self.success,
            "content_hash": self.content_hash,
            "chunks_created": self.chunks_created,
            "embedding_cost": self.embedding_cost,
            "validation_status": self.validation_status,
            "error": self.error,
        }


@dataclass
class BatchIngestionResult:
    """Result of batch content ingestion."""

    total: int
    successful: int
    failed: int
    total_cost: float
    results: List[IngestionResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "total_cost": self.total_cost,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# CONTENT INGESTION SERVICE
# =============================================================================


class ContentIngestionService(BaseService):
    """
    Service for ingesting content into the RAG knowledge base.

    Pipeline:
    1. Validate and sanitize content
    2. Check for duplicates via content hash
    3. Chunk content with overlap for context continuity
    4. Generate embeddings (batched for efficiency)
    5. Store in Qdrant with metadata
    6. Optionally validate against SELVE framework

    Security:
    - Maximum content size limits
    - Maximum batch size limits
    - Rate limiting ready
    - Input sanitization
    """

    # Limits
    MAX_CONTENT_LENGTH = 100_000  # 100KB
    MAX_BATCH_SIZE = 50
    MAX_CHUNKS_PER_CONTENT = 200

    def __init__(
        self,
        collection_name: Optional[str] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        validation_service=None,
    ):
        """
        Initialize content ingestion service.

        Args:
            collection_name: Qdrant collection name
            chunking_config: Chunking configuration
            validation_service: Optional content validation service
        """
        super().__init__()
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        self.chunking_config = chunking_config or ChunkingConfig()
        self._validation_service = validation_service
        self._ensure_collection()

    @property
    def validation_service(self):
        """Lazy-loaded validation service."""
        if self._validation_service is None:
            try:
                from .content_validation_service import ContentValidationService

                self._validation_service = ContentValidationService()
            except ImportError:
                self.logger.warning("ContentValidationService not available")
        return self._validation_service

    def _ensure_collection(self) -> None:
        """Ensure Qdrant collection exists with proper configuration."""
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=Config.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE,
                    ),
                )
                self.logger.info(f"Created collection: {self.collection_name}")
            else:
                self.logger.debug(f"Collection exists: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to ensure collection: {e}")
            raise

    def chunk_content(
        self,
        content: str,
        config: Optional[ChunkingConfig] = None,
    ) -> List[str]:
        """
        Chunk content into smaller pieces with overlap.

        Uses sentence-based chunking to maintain semantic coherence.

        Args:
            content: Full content text
            config: Optional chunking configuration

        Returns:
            List of content chunks
        """
        config = config or self.chunking_config

        # Validate content
        if not content or not content.strip():
            return []

        # Split into sentences (simple heuristic)
        sentences = self._split_sentences(content)
        if not sentences:
            return []

        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = len(sentence) // config.chars_per_token

            # Check if adding this sentence exceeds chunk size
            if (
                current_length + sentence_tokens > config.max_chunk_tokens
                and current_chunk
            ):
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_count = min(2, len(current_chunk))
                current_chunk = current_chunk[-overlap_count:] + [sentence]
                current_length = sum(
                    len(s) // config.chars_per_token for s in current_chunk
                )
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens

        # Add final chunk if it meets minimum size
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) // config.chars_per_token >= config.min_chunk_tokens:
                chunks.append(chunk_text)

        # Enforce chunk limit
        if len(chunks) > self.MAX_CHUNKS_PER_CONTENT:
            self.logger.warning(
                f"Content produced {len(chunks)} chunks, "
                f"limiting to {self.MAX_CHUNKS_PER_CONTENT}"
            )
            chunks = chunks[: self.MAX_CHUNKS_PER_CONTENT]

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Simple heuristic that handles common cases.
        For production, consider using spaCy or NLTK.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Normalize whitespace
        text = " ".join(text.split())

        # Split on sentence boundaries
        sentences = []
        current = []

        for char in text:
            current.append(char)
            if char in ".!?" and len(current) > 1:
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

        # Add remaining text
        if current:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def generate_embeddings(
        self,
        texts: List[str],
    ) -> Result[tuple[List[List[float]], float]]:
        """
        Generate embeddings for a list of texts.

        Batches requests for efficiency.

        Args:
            texts: List of text chunks to embed

        Returns:
            Result containing (embeddings, cost)
        """
        if not texts:
            return Result.success(([], 0.0))

        # Validate batch size
        if len(texts) > Config.MAX_EMBEDDING_BATCH_SIZE:
            return Result.validation_error(
                f"Batch size {len(texts)} exceeds limit {Config.MAX_EMBEDDING_BATCH_SIZE}"
            )

        try:
            response = self.openai.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]

            # Calculate cost
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1_000_000) * Config.EMBEDDING_COST_PER_1M_TOKENS

            return Result.success((embeddings, cost))

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise ExternalServiceError("OpenAI", "Failed to generate embeddings", e)

    def _check_duplicate(self, content_hash: str) -> bool:
        """
        Check if content with this hash already exists.

        Args:
            content_hash: SHA-256 hash of content

        Returns:
            True if duplicate exists
        """
        try:
            # Search for existing content with same hash
            results = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "content_hash",
                            "match": {"value": content_hash},
                        }
                    ]
                },
                limit=1,
            )

            return len(results[0]) > 0

        except Exception:
            # On error, proceed with ingestion (dedup is best-effort)
            return False

    async def ingest_content(
        self,
        content: str,
        source: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        validate: bool = True,
        skip_duplicate_check: bool = False,
    ) -> IngestionResult:
        """
        Ingest content into the knowledge base.

        Args:
            content: Full content text
            source: Content source (e.g., "selve_framework", "blog_post")
            content_type: Type of content (e.g., "article", "dimension_description")
            metadata: Additional metadata (title, author, etc.)
            validate: Whether to validate against SELVE framework
            skip_duplicate_check: Skip deduplication (for reimports)

        Returns:
            IngestionResult with details
        """
        # Validate inputs
        try:
            content = Validator.validate_string(
                content,
                "content",
                min_length=10,
                max_length=self.MAX_CONTENT_LENGTH,
            )
            source = Validator.validate_string(
                source,
                "source",
                min_length=1,
                max_length=100,
            )
            content_type = Validator.validate_string(
                content_type,
                "content_type",
                min_length=1,
                max_length=50,
            )
        except Exception as e:
            return IngestionResult(
                success=False,
                content_hash="",
                error=str(e),
            )

        # Generate content hash
        content_hash = generate_content_hash(content)

        # Check for duplicates
        if not skip_duplicate_check and self._check_duplicate(content_hash):
            self.logger.info(f"Duplicate content detected: {content_hash[:16]}...")
            return IngestionResult(
                success=False,
                content_hash=content_hash,
                error="Content already exists (duplicate)",
            )

        try:
            # Chunk content
            chunks = self.chunk_content(content)
            if not chunks:
                return IngestionResult(
                    success=False,
                    content_hash=content_hash,
                    error="Content produced no valid chunks",
                )

            self.logger.debug(f"Chunked content into {len(chunks)} pieces")

            # Generate embeddings
            embedding_result = self.generate_embeddings(chunks)
            if embedding_result.is_error:
                return IngestionResult(
                    success=False,
                    content_hash=content_hash,
                    error=embedding_result.error,
                )

            embeddings, embedding_cost = embedding_result.data

            # Prepare metadata
            base_metadata = metadata or {}
            base_metadata.update({
                "source": source,
                "content_type": content_type,
                "content_hash": content_hash,
                "ingested_at": datetime.utcnow().isoformat(),
            })

            # Create points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate deterministic point ID
                point_id_str = f"{content_hash}_{i}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id_str))

                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "text": chunk,
                })

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=chunk_metadata,
                    )
                )

            # Upsert to Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            self.logger.info(
                f"Ingested {len(points)} chunks "
                f"(hash: {content_hash[:16]}..., cost: ${embedding_cost:.6f})"
            )

            # Validate if requested
            validation_status = "not_validated"
            if validate and self.validation_service:
                try:
                    validation_result = await self.validation_service.validate_content(
                        content=content,
                        source=source,
                        content_hash=content_hash,
                    )
                    validation_status = validation_result.get("status", "error")
                except Exception as e:
                    self.logger.warning(f"Validation failed: {e}")
                    validation_status = "validation_error"

            return IngestionResult(
                success=True,
                content_hash=content_hash,
                chunks_created=len(chunks),
                embedding_cost=embedding_cost,
                validation_status=validation_status,
            )

        except ExternalServiceError:
            raise
        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}")
            return IngestionResult(
                success=False,
                content_hash=content_hash,
                error=str(e),
            )

    async def ingest_batch(
        self,
        contents: List[Dict[str, Any]],
    ) -> BatchIngestionResult:
        """
        Ingest multiple content items in batch.

        Args:
            contents: List of content dicts with keys:
                - content: str
                - source: str
                - content_type: str
                - metadata: Optional[Dict]
                - validate: Optional[bool]

        Returns:
            BatchIngestionResult with aggregate stats
        """
        # Validate batch size
        if len(contents) > self.MAX_BATCH_SIZE:
            return BatchIngestionResult(
                total=len(contents),
                successful=0,
                failed=len(contents),
                total_cost=0.0,
                results=[
                    IngestionResult(
                        success=False,
                        content_hash="",
                        error=f"Batch size {len(contents)} exceeds limit {self.MAX_BATCH_SIZE}",
                    )
                ],
            )

        results = []
        total_cost = 0.0
        successful = 0
        failed = 0

        for item in contents:
            result = await self.ingest_content(
                content=item.get("content", ""),
                source=item.get("source", ""),
                content_type=item.get("content_type", ""),
                metadata=item.get("metadata"),
                validate=item.get("validate", True),
            )

            results.append(result)

            if result.success:
                successful += 1
                total_cost += result.embedding_cost
            else:
                failed += 1

        return BatchIngestionResult(
            total=len(contents),
            successful=successful,
            failed=failed,
            total_cost=total_cost,
            results=results,
        )

    async def delete_by_hash(
        self,
        content_hash: str,
    ) -> Result[int]:
        """
        Delete all chunks associated with a content hash.

        Args:
            content_hash: Content hash to delete

        Returns:
            Result containing number of deleted points
        """
        try:
            content_hash = Validator.validate_string(
                content_hash,
                "content_hash",
                min_length=64,
                max_length=64,
            )
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            # Find all points with this hash
            results, _ = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "content_hash",
                            "match": {"value": content_hash},
                        }
                    ]
                },
                limit=1000,
            )

            if not results:
                return Result.success(0)

            point_ids = [point.id for point in results]

            # Delete points
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )

            self.logger.info(
                f"Deleted {len(point_ids)} chunks for hash {content_hash[:16]}..."
            )
            return Result.success(len(point_ids))

        except Exception as e:
            self.logger.error(f"Delete failed: {e}")
            return Result.failure(str(e), error_code="DELETE_ERROR")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Collection statistics
        """
        try:
            info = self.qdrant.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
                "segments_count": len(info.segments or []),
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}