"""
Memory Search Service - Vector search through episodic memories
Uses Qdrant for semantic similarity matching
"""
import os
import hashlib
from typing import List, Dict, Any, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.db import db


class MemorySearchService:
    """Service for vector search through episodic memories"""

    def __init__(self):
        """Initialize OpenAI and Qdrant clients"""
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.collection_name = "episodic_memories"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimension = 1536  # text-embedding-3-small dimension
        self._collection_initialized = False

    def _ensure_collection_exists(self):
        """Create the episodic memories collection if it doesn't exist (lazy initialization)"""
        if self._collection_initialized:
            return

        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created Qdrant collection: {self.collection_name}")

            self._collection_initialized = True
        except Exception as e:
            print(f"⚠️ Error ensuring collection exists: {e}")
            # Don't set initialized flag so it retries next time

    def _memory_id_to_point_id(self, memory_id: str) -> int:
        """Convert memory ID (CUID) to integer point ID for Qdrant"""
        # Use hash to convert string ID to integer
        # Take first 8 bytes of SHA256 hash and convert to int
        hash_bytes = hashlib.sha256(memory_id.encode()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big')

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    async def embed_memory(self, memory_id: str) -> bool:
        """
        Embed an episodic memory and store in Qdrant

        Args:
            memory_id: Episodic memory ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure Qdrant collection exists
            self._ensure_collection_exists()

            # Fetch memory from database
            memory = await db.episodicmemory.find_unique(where={"id": memory_id})

            if not memory:
                print(f"⚠️ Memory {memory_id} not found")
                return False

            # Create embedding text from title + summary + key insights
            embedding_text = f"{memory.title}\n\n{memory.summary}"
            if memory.keyInsights:
                embedding_text += "\n\nKey Insights:\n" + "\n".join(memory.keyInsights)

            # Generate embedding
            embedding_vector = self.generate_embedding(embedding_text)

            # Convert memory ID to integer point ID for Qdrant
            point_id = self._memory_id_to_point_id(memory_id)

            # Store in Qdrant
            point = PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload={
                    "memory_id": memory_id,
                    "user_id": memory.userId,
                    "session_id": memory.sessionId,
                    "title": memory.title,
                    "summary": memory.summary,
                    "key_insights": memory.keyInsights or [],
                    "emotional_state": memory.emotionalState,
                    "unresolved_topics": memory.unresolvedTopics or [],
                    "span_start": memory.spanStart.isoformat() if memory.spanStart else None,
                    "span_end": memory.spanEnd.isoformat() if memory.spanEnd else None,
                    "text": embedding_text  # Full text for reference
                }
            )

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            # Mark memory as embedded in database
            await db.episodicmemory.update(
                where={"id": memory_id},
                data={"embedded": True}
            )

            print(f"✅ Embedded memory {memory_id}: {memory.title}")
            return True

        except Exception as e:
            print(f"❌ Error embedding memory {memory_id}: {e}")
            return False

    async def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        clerk_user_id: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memories using vector similarity

        Args:
            query: Search query
            user_id: Optional internal user ID to filter by
            clerk_user_id: Optional Clerk user ID to filter by
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of matching memories with metadata
        """
        try:
            # Ensure Qdrant collection exists
            self._ensure_collection_exists()

            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Build filter for user-specific search
            search_filter = None
            if user_id:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )

            # Search in Qdrant
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter
            )

            # Format results
            memories = []
            for result in results:
                if result.score >= score_threshold:
                    memories.append({
                        "memory_id": result.payload.get("memory_id"),
                        "title": result.payload.get("title"),
                        "summary": result.payload.get("summary"),
                        "key_insights": result.payload.get("key_insights", []),
                        "emotional_state": result.payload.get("emotional_state"),
                        "unresolved_topics": result.payload.get("unresolved_topics", []),
                        "span_start": result.payload.get("span_start"),
                        "span_end": result.payload.get("span_end"),
                        "relevance_score": round(result.score, 3)
                    })

            return memories

        except Exception as e:
            print(f"❌ Error searching memories: {e}")
            return []

    async def get_related_memories(
        self,
        memory_id: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find memories similar to a given memory

        Args:
            memory_id: Memory ID to find similar memories for
            top_k: Number of similar memories to return

        Returns:
            List of similar memories
        """
        try:
            # Ensure Qdrant collection exists
            self._ensure_collection_exists()

            # Convert memory ID to point ID
            point_id = self._memory_id_to_point_id(memory_id)

            # Get the memory's embedding from Qdrant
            memory_point = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )

            if not memory_point:
                return []

            # Use the memory's vector to search for similar memories
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=memory_point[0].vector,
                limit=top_k + 1  # +1 because it will include itself
            )

            # Filter out the original memory and format results
            memories = []
            for result in results:
                if result.payload.get("memory_id") != memory_id:
                    memories.append({
                        "memory_id": result.payload.get("memory_id"),
                        "title": result.payload.get("title"),
                        "summary": result.payload.get("summary"),
                        "relevance_score": round(result.score, 3)
                    })

            return memories[:top_k]

        except Exception as e:
            print(f"❌ Error finding related memories: {e}")
            return []

    def format_search_results_for_context(
        self,
        memories: List[Dict[str, Any]]
    ) -> str:
        """
        Format search results as context string for LLM

        Args:
            memories: List of memory search results

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        context_parts = [
            "=== RELEVANT PAST CONVERSATIONS ===",
            "",
            "I found these relevant memories from your previous conversations:",
            ""
        ]

        for i, mem in enumerate(memories, 1):
            context_parts.append(f"{i}. {mem['title']} (relevance: {mem['relevance_score']:.0%})")
            context_parts.append(f"   {mem['summary']}")

            if mem.get('key_insights'):
                context_parts.append(f"   Key insights: {', '.join(mem['key_insights'][:2])}")

            context_parts.append("")

        context_parts.append("Use these memories to provide continuity and recall relevant past discussions.")
        context_parts.append("===")
        context_parts.append("")

        return "\n".join(context_parts)
