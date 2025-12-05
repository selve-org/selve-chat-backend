"""
Content Ingestion Service - Ingest, chunk, embed, and store SELVE content
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.db import db
from .content_validation_service import ContentValidationService


class ContentIngestionService:
    """
    Service for ingesting content into the RAG knowledge base

    Handles:
    - Content chunking (semantic, paragraph-based)
    - Embedding generation (OpenAI text-embedding-3-small)
    - Qdrant storage with metadata
    - Content validation tracking
    - Deduplication
    """

    # Embedding model configuration
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536
    EMBEDDING_COST_PER_1M_TOKENS = 0.020  # $0.02 per 1M tokens

    # Chunking configuration
    MAX_CHUNK_SIZE = 512  # tokens
    OVERLAP_SIZE = 50     # tokens for context continuity

    def __init__(self):
        """Initialize with OpenAI, Qdrant, and validation services"""
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection_name = os.getenv("QDRANT_COLLECTION", "selve_knowledge")

        self.validation_service = ContentValidationService()

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure Qdrant collection exists with proper configuration"""
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created Qdrant collection: {self.collection_name}")
            else:
                print(f"✅ Qdrant collection exists: {self.collection_name}")

        except Exception as e:
            print(f"❌ Failed to ensure collection: {e}")
            raise

    def chunk_content(
        self,
        content: str,
        chunk_size: int = MAX_CHUNK_SIZE,
        overlap: int = OVERLAP_SIZE
    ) -> List[str]:
        """
        Chunk content into smaller pieces with overlap

        Args:
            content: Full content text
            chunk_size: Maximum chunk size in tokens (approximate)
            overlap: Overlap size between chunks in tokens

        Returns:
            List of content chunks
        """
        # Simple sentence-based chunking (could be improved with semantic chunking)
        sentences = content.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Rough token estimation (1 token ≈ 4 characters)
            sentence_tokens = len(sentence) // 4

            if current_length + sentence_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) // 4 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)

        return chunks

    def generate_embeddings(
        self,
        texts: List[str]
    ) -> tuple[List[List[float]], float]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text chunks to embed

        Returns:
            Tuple of (embeddings list, cost in USD)
        """
        try:
            response = self.openai.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]

            # Calculate cost
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1_000_000) * self.EMBEDDING_COST_PER_1M_TOKENS

            return embeddings, cost

        except Exception as e:
            print(f"❌ Failed to generate embeddings: {e}")
            raise

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()

    async def ingest_content(
        self,
        content: str,
        source: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest content into the knowledge base

        Args:
            content: Full content text
            source: Content source (e.g., "blog_post", "selve_framework", "tim_lahaye_book")
            content_type: Type of content (e.g., "article", "dimension_description", "chapter")
            metadata: Additional metadata (title, author, date, etc.)
            validate: Whether to validate content against SELVE framework

        Returns:
            {
                "ingested": bool,
                "chunks_created": int,
                "embedding_cost": float,
                "validation_status": str,
                "content_hash": str
            }
        """
        try:
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(content)

            # TODO: Check if already ingested once ContentIngestionLog model exists
            # For now, check Qdrant directly for duplicate detection
            # existing_sync = await db.contentingestionlog.find_first(
            #     where={"contentHash": content_hash}
            # )
            # if existing_sync:
            #     print(f"⚠️ Content already ingested: {content_hash[:8]}")
            #     return {
            #         "ingested": False,
            #         "error": "Content already exists",
            #         "content_hash": content_hash
            #     }

            # Chunk content
            chunks = self.chunk_content(content)
            print(f"📄 Chunked content into {len(chunks)} pieces")

            # Generate embeddings
            embeddings, embedding_cost = self.generate_embeddings(chunks)
            print(f"✅ Generated {len(embeddings)} embeddings (cost: ${embedding_cost:.6f})")

            # Prepare metadata
            base_metadata = metadata or {}
            base_metadata.update({
                "source": source,
                "content_type": content_type,
                "content_hash": content_hash,
                "ingested_at": datetime.utcnow().isoformat()
            })

            # Store in Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate deterministic UUID from content hash and chunk index
                point_id_str = f"{content_hash}_{i}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id_str))

                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "text": chunk
                })

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=chunk_metadata
                    )
                )

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✅ Stored {len(points)} chunks in Qdrant")

            # Note: Skipping database sync record for now
            # (ChatbotKnowledgeSync model designed for user data, not content ingestion)
            # TODO: Create ContentIngestionLog model in Prisma schema
            sync_record = None

            # Validate content if requested
            validation_status = "not_validated"
            validation_result = None
            if validate:
                validation_result = await self.validation_service.validate_content(
                    content=content,
                    source=source,
                    content_hash=content_hash
                )
                validation_status = validation_result.get("status", "error")

            return {
                "ingested": True,
                "chunks_created": len(chunks),
                "embedding_cost": embedding_cost,
                "validation_status": validation_status,
                "validation_result": validation_result,
                "content_hash": content_hash,
                "sync_record_id": None  # Will add once ContentIngestionLog model exists
            }

        except Exception as e:
            print(f"❌ Content ingestion failed: {e}")
            return {
                "ingested": False,
                "error": str(e)
            }

    async def ingest_batch(
        self,
        contents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest multiple content items in batch

        Args:
            contents: List of content dicts with keys: content, source, content_type, metadata

        Returns:
            {
                "total": int,
                "successful": int,
                "failed": int,
                "total_cost": float,
                "results": [...]
            }
        """
        results = []
        total_cost = 0.0
        successful = 0
        failed = 0

        for item in contents:
            result = await self.ingest_content(
                content=item["content"],
                source=item["source"],
                content_type=item["content_type"],
                metadata=item.get("metadata"),
                validate=item.get("validate", True)
            )

            results.append(result)

            if result.get("ingested"):
                successful += 1
                total_cost += result.get("embedding_cost", 0.0)
            else:
                failed += 1

        return {
            "total": len(contents),
            "successful": successful,
            "failed": failed,
            "total_cost": total_cost,
            "results": results
        }

    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ingested content

        Returns:
            {
                "total_syncs": int,
                "total_chunks": int,
                "total_cost": float,
                "by_source": {...},
                "recent_syncs": [...]
            }
        """
        try:
            # TODO: Get stats from ContentIngestionLog once model exists
            # For now, return empty stats (Qdrant ingestion still works)
            return {
                "total_syncs": 0,
                "total_chunks": 0,
                "total_cost": 0.0,
                "by_source": {},
                "recent_syncs": []
            }

        except Exception as e:
            print(f"❌ Failed to get ingestion stats: {e}")
            return {}
