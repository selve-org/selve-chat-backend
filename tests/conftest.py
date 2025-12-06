"""
Pytest configuration and shared fixtures
"""
import pytest
from httpx import AsyncClient
from app.main import app
from app.db import db, connect_db, disconnect_db


@pytest.fixture(scope="function", autouse=True)
async def setup_database():
    """Setup database connection for each test"""
    await connect_db()
    yield
    await disconnect_db()


@pytest.fixture
async def client():
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def test_session(cleanup_session):
    """Create a test chat session with messages"""
    # Create test session
    session = await db.chatsession.create(
        data={
            "userId": "test_user_123",
            "clerkUserId": "clerk_test_123",
            "title": "Test Conversation",
            "status": "active",
            "totalTokens": 75000  # Above 70% threshold
        }
    )

    # Create test messages (simulate conversation above threshold)
    messages = []
    for i in range(20):
        msg = await db.chatmessage.create(
            data={
                "sessionId": session.id,
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Test message {i}: This is a conversation about SELVE dimensions and personality insights.",
                "tokenCount": 3750  # 75,000 / 20 = 3,750 per message
            }
        )
        messages.append(msg)

    yield {"session": session, "messages": messages}

    # Cleanup handled by cleanup_session fixture


async def _cleanup_test_data():
    """Helper to clean up test data from database and Qdrant"""
    from app.services.memory_search_service import MemorySearchService
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Delete test sessions and related data
    test_sessions = await db.chatsession.find_many(
        where={
            "OR": [
                {"userId": {"contains": "test_"}},
                {"clerkUserId": {"contains": "test_"}}
            ]
        }
    )

    # Clean up Qdrant - delete all test user points
    memory_service = MemorySearchService()
    try:
        memory_service._ensure_collection_exists()
        # Delete all points with user_id containing "test_"
        memory_service.qdrant.delete(
            collection_name=memory_service.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value="test_user_123")
                    )
                ]
            )
        )
        print("✅ Cleaned up Qdrant test points")
    except Exception as e:
        print(f"⚠️ Error cleaning up Qdrant: {e}")

    for session in test_sessions:
        # Delete messages
        await db.chatmessage.delete_many(
            where={"sessionId": session.id}
        )

        # Delete episodic memories
        await db.episodicmemory.delete_many(
            where={"sessionId": session.id}
        )

        # Delete session
        await db.chatsession.delete(
            where={"id": session.id}
        )


@pytest.fixture
async def cleanup_session():
    """Cleanup test sessions before and after each test"""
    # Clean up before test (remove leftovers from previous runs)
    await _cleanup_test_data()

    yield

    # Clean up after test
    await _cleanup_test_data()
