"""
Integration tests for compression and memory search features
"""
import pytest
from app.services.compression_service import CompressionService
from app.services.memory_search_service import MemorySearchService
from app.db import db


class TestCompressionService:
    """Test compression service functionality"""

    @pytest.fixture
    def compression_service(self):
        return CompressionService()

    @pytest.mark.asyncio
    async def test_needs_compression_threshold(self, compression_service):
        """Test compression threshold detection"""
        # Below threshold - should not trigger
        assert not compression_service.needs_compression(
            total_tokens=50000,
            model="claude-3-5-haiku-20241022"
        )

        # At threshold - should trigger
        assert compression_service.needs_compression(
            total_tokens=70000,
            model="claude-3-5-haiku-20241022"
        )

        # Above threshold - should trigger
        assert compression_service.needs_compression(
            total_tokens=85000,
            model="claude-3-5-haiku-20241022"
        )

    @pytest.mark.asyncio
    async def test_should_trigger_compression(self, compression_service, test_session):
        """Test session-based compression trigger"""
        session = test_session["session"]

        # Session has 75,000 tokens (above 70% threshold)
        should_compress = await compression_service.should_trigger_compression(session.id)
        assert should_compress is True

    @pytest.mark.asyncio
    async def test_compress_conversation(self, compression_service, test_session):
        """Test full compression workflow"""
        session = test_session["session"]
        messages = test_session["messages"]

        # Compress conversation
        result = await compression_service.compress_conversation(
            session_id=session.id,
            user_id="test_user_123",
            clerk_user_id="clerk_test_123"
        )

        # Verify compression succeeded
        assert result["compressed"] is True
        assert result["episodic_memory_id"] is not None
        assert result["messages_compressed"] > 0
        assert result["messages_kept"] > 0
        assert result["tokens_saved"] > 0
        assert result["compression_ratio"] > 0
        assert "title" in result
        assert "summary" in result

        # Verify episodic memory was created
        memory = await db.episodicmemory.find_unique(
            where={"id": result["episodic_memory_id"]}
        )
        assert memory is not None
        assert memory.title == result["title"]
        assert memory.summary == result["summary"]
        assert memory.sessionId == session.id
        assert memory.userId == "test_user_123"
        assert len(memory.keyInsights) > 0
        assert memory.embedded is True  # Should be automatically embedded

        # Verify messages were marked as compressed
        compressed_count = result["messages_compressed"]
        compressed_messages = await db.chatmessage.find_many(
            where={
                "sessionId": session.id,
                "isCompressed": True
            }
        )
        assert len(compressed_messages) == compressed_count

        # Verify session compression count incremented
        updated_session = await db.chatsession.find_unique(
            where={"id": session.id}
        )
        assert updated_session.compressionCount == 1

    @pytest.mark.asyncio
    async def test_get_session_memories(self, compression_service, test_session):
        """Test retrieving session memories"""
        session = test_session["session"]

        # Compress conversation to create memory
        await compression_service.compress_conversation(
            session_id=session.id,
            user_id="test_user_123",
            clerk_user_id="clerk_test_123"
        )

        # Retrieve memories
        memories = await compression_service.get_session_memories(session.id)

        assert len(memories) == 1
        assert "title" in memories[0]
        assert "summary" in memories[0]
        assert "key_insights" in memories[0]

    @pytest.mark.asyncio
    async def test_get_user_memories(self, compression_service, test_session):
        """Test retrieving user memories across sessions"""
        session = test_session["session"]

        # Compress conversation
        await compression_service.compress_conversation(
            session_id=session.id,
            user_id="test_user_123",
            clerk_user_id="clerk_test_123"
        )

        # Retrieve user memories
        memories = await compression_service.get_user_memories(
            clerk_user_id="clerk_test_123",
            limit=5
        )

        assert len(memories) >= 1
        assert "title" in memories[0]
        assert "summary" in memories[0]


class TestMemorySearchService:
    """Test memory search service functionality"""

    @pytest.fixture
    def memory_search_service(self):
        return MemorySearchService()

    @pytest.fixture
    async def embedded_memory(self, test_session):
        """Create and embed a test memory"""
        session = test_session["session"]

        # Compress to create memory
        compression_service = CompressionService()
        result = await compression_service.compress_conversation(
            session_id=session.id,
            user_id="test_user_123",
            clerk_user_id="clerk_test_123"
        )

        memory_id = result["episodic_memory_id"]

        # Verify it was embedded automatically
        memory = await db.episodicmemory.find_unique(
            where={"id": memory_id}
        )
        assert memory.embedded is True

        return memory

    @pytest.mark.asyncio
    async def test_embed_memory(self, memory_search_service, test_session):
        """Test memory embedding"""
        from datetime import datetime
        from prisma import Json

        session = test_session["session"]

        # Create unembedded memory
        memory = await db.episodicmemory.create(
            data={
                "userId": "test_user_123",
                "sessionId": session.id,
                "title": "Test Memory",
                "summary": "This is a test memory about SELVE dimensions.",
                "keyInsights": Json(["User shows high LUMEN", "Interested in social aspects"]),
                "unresolvedTopics": Json([]),
                "emotionalState": "curious",
                "sourceMessageIds": [],
                "compressionModel": "gpt-4o-mini",
                "compressionCost": 0.0001,
                "spanStart": datetime.utcnow(),
                "spanEnd": datetime.utcnow(),
                "embedded": False
            }
        )

        # Embed the memory
        success = await memory_search_service.embed_memory(memory.id)
        assert success is True

        # Verify memory is marked as embedded
        updated_memory = await db.episodicmemory.find_unique(
            where={"id": memory.id}
        )
        assert updated_memory.embedded is True

    @pytest.mark.asyncio
    async def test_search_memories(self, memory_search_service, embedded_memory):
        """Test vector search through memories"""
        # Search for related content
        results = await memory_search_service.search_memories(
            query="Tell me about SELVE dimensions and personality",
            user_id="test_user_123",
            top_k=5,
            score_threshold=0.3
        )

        # Should find at least one relevant memory
        assert len(results) >= 1
        assert "memory_id" in results[0]
        assert "title" in results[0]
        assert "summary" in results[0]
        assert "relevance_score" in results[0]
        assert results[0]["relevance_score"] >= 0.3

    @pytest.mark.asyncio
    async def test_search_memories_with_user_filter(self, memory_search_service, embedded_memory):
        """Test user-filtered search"""
        # Verify embedded_memory exists
        assert embedded_memory is not None
        print(f"Embedded memory ID: {embedded_memory.id}")

        # Search with user filter
        results = await memory_search_service.search_memories(
            query="SELVE personality insights",
            user_id="test_user_123",
            top_k=5
        )

        print(f"Search returned {len(results)} results")
        if results:
            print(f"First result memory_id: {results[0].get('memory_id')}")

        # Should find the embedded memory
        assert len(results) >= 1

        # All results should belong to the test user
        for result in results:
            print(f"Checking memory_id: {result['memory_id']}")
            memory = await db.episodicmemory.find_unique(
                where={"id": result["memory_id"]}
            )
            if memory is None:
                print(f"⚠️ Memory {result['memory_id']} not found in database!")
                # List all memories to debug
                all_memories = await db.episodicmemory.find_many()
                print(f"All memories in DB: {[m.id for m in all_memories]}")
            assert memory is not None, f"Memory {result['memory_id']} not found"
            assert memory.userId == "test_user_123"

    @pytest.mark.asyncio
    async def test_format_search_results_for_context(self, memory_search_service, embedded_memory):
        """Test formatting search results for LLM context"""
        results = await memory_search_service.search_memories(
            query="SELVE dimensions",
            user_id="test_user_123",
            top_k=3
        )

        formatted = memory_search_service.format_search_results_for_context(results)

        assert "RELEVANT PAST CONVERSATIONS" in formatted
        assert "relevance:" in formatted.lower()

        # Empty results should return empty string
        empty_formatted = memory_search_service.format_search_results_for_context([])
        assert empty_formatted == ""


class TestCompressionAPI:
    """Test compression API endpoints"""

    @pytest.mark.asyncio
    async def test_compress_endpoint(self, client, test_session):
        """Test POST /api/compression/compress"""
        session = test_session["session"]

        response = await client.post(
            "/api/compression/compress",
            json={
                "session_id": session.id,
                "user_id": "test_user_123",
                "clerk_user_id": "clerk_test_123"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["compressed"] is True
        assert "episodic_memory_id" in data
        assert "tokens_saved" in data

    @pytest.mark.asyncio
    async def test_check_compression_needed(self, client, test_session):
        """Test GET /api/compression/check/{session_id}"""
        session = test_session["session"]

        response = await client.get(f"/api/compression/check/{session.id}")

        assert response.status_code == 200
        data = response.json()
        assert "needs_compression" in data
        assert data["needs_compression"] is True  # Session has 75K tokens

    @pytest.mark.asyncio
    async def test_get_session_memories_endpoint(self, client, test_session):
        """Test GET /api/compression/memories/{session_id}"""
        session = test_session["session"]

        # First compress to create memories
        await client.post(
            "/api/compression/compress",
            json={
                "session_id": session.id,
                "user_id": "test_user_123",
                "clerk_user_id": "clerk_test_123"
            }
        )

        # Then retrieve memories
        response = await client.get(f"/api/compression/memories/{session.id}")

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert data["memory_count"] >= 1

    @pytest.mark.asyncio
    async def test_search_memories_endpoint(self, client, test_session):
        """Test POST /api/compression/search"""
        session = test_session["session"]

        # Compress to create searchable memory
        await client.post(
            "/api/compression/compress",
            json={
                "session_id": session.id,
                "user_id": "test_user_123",
                "clerk_user_id": "clerk_test_123"
            }
        )

        # Search memories
        response = await client.post(
            "/api/compression/search",
            json={
                "query": "SELVE dimensions personality",
                "user_id": "test_user_123",
                "top_k": 5,
                "score_threshold": 0.3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "result_count" in data

    @pytest.mark.asyncio
    async def test_search_memories_get_endpoint(self, client, test_session):
        """Test GET /api/compression/search"""
        session = test_session["session"]

        # Compress to create memory
        await client.post(
            "/api/compression/compress",
            json={
                "session_id": session.id,
                "user_id": "test_user_123",
                "clerk_user_id": "clerk_test_123"
            }
        )

        # Search via GET
        response = await client.get(
            "/api/compression/search",
            params={
                "query": "SELVE personality",
                "user_id": "test_user_123",
                "top_k": 3
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "query" in data
