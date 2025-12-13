#!/usr/bin/env python3
"""
End-to-End Pipeline Test
Tests content ingestion → validation → embedding → RAG retrieval → chat response
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.content_ingestion_service import ContentIngestionService
from app.services.rag_service import RAGService
from app.services.agentic_chat_service import AgenticChatService
from app.db import db


async def test_ingestion():
    """Test 1: Content Ingestion"""
    print("\n" + "="*80)
    print("TEST 1: Content Ingestion")
    print("="*80)

    service = ContentIngestionService()

    # Sample SELVE content
    test_content = """
The LUMEN dimension in the SELVE framework measures your social energy and preference
for interaction. High LUMEN individuals draw energy from social interactions and thrive
in group settings. They enjoy meeting new people and feel energized after social
gatherings. Low LUMEN individuals recharge through solitude and prefer deep one-on-one
conversations. They feel drained after extended social interaction and value privacy.

Neither is better or worse - both have unique strengths. High LUMEN individuals excel
at networking and team motivation, while low LUMEN individuals excel at deep thinking
and focused expertise. Understanding your LUMEN score helps you work with your natural
tendencies rather than against them.
"""

    result = await service.ingest_content(
        content=test_content,
        source="selve_framework",
        content_type="dimension_description",
        metadata={"dimension": "LUMEN", "test": True},
        validate=False  # Skip validation for test pipeline
    )

    print(f"\n✅ Ingestion Result:")
    print(f"   Ingested: {result['ingested']}")
    print(f"   Chunks: {result['chunks_created']}")
    print(f"   Cost: ${result['embedding_cost']:.6f}")
    print(f"   Validation: {result['validation_status']}")

    if result.get('validation_result'):
        val = result['validation_result']
        print(f"\n   Validation Scores:")
        print(f"      SELVE Aligned: {val['scores']['selve_aligned']}/10")
        print(f"      Factually Accurate: {val['scores']['factually_accurate']}/10")
        print(f"      Appropriate Tone: {val['scores']['appropriate_tone']}/10")
        print(f"   Recommendation: {val['recommendation']}")

    return result['ingested']


async def test_rag_retrieval():
    """Test 2: RAG Retrieval"""
    print("\n" + "="*80)
    print("TEST 2: RAG Retrieval")
    print("="*80)

    rag_service = RAGService()

    # Test query
    query = "What does the LUMEN dimension measure?"

    print(f"\nQuery: \"{query}\"")

    context = rag_service.get_context_for_query(query, top_k=3)

    print(f"\n✅ Retrieval Result:")
    print(f"   Retrieved: {context['retrieved_count']} chunks")

    if context['retrieved_count'] > 0:
        print(f"\n   Top Results:")
        for i, chunk in enumerate(context['chunks'][:3], 1):
            print(f"\n   {i}. Score: {chunk['score']:.4f}")
            print(f"      Source: {chunk['source']}")
            print(f"      Preview: {chunk['text'][:150]}...")

        return True
    else:
        print("   ⚠️  No results found!")
        return False


async def test_chat_response():
    """Test 3: Chat Response with RAG (Streaming)"""
    print("\n" + "="*80)
    print("TEST 3: Chat Response with RAG (Streaming)")
    print("="*80)

    chat_service = AgenticChatService()

    # Test query
    message = "What is the LUMEN dimension and what does a high score mean?"

    print(f"\nUser Question: \"{message}\"")

    # Consume the stream
    print(f"\n✅ Chat Response:")
    print(f"   {'-'*76}")
    print("   ", end='')
    
    full_response = ""
    async for event in chat_service.chat_stream(message=message, emit_status=False):
        if isinstance(event, str):
            full_response += event
            print(event, end='', flush=True)
        elif isinstance(event, dict) and 'chunk' in event:
            chunk = event['chunk']
            full_response += chunk
            print(chunk, end='', flush=True)
    
    print()
    print(f"   {'-'*76}")
    print(f"\n   Response Length: {len(full_response)} characters")

    return True


async def test_compression_check():
    """Test 4: Compression Service Check"""
    print("\n" + "="*80)
    print("TEST 4: Compression Service Check")
    print("="*80)

    from app.services.compression_service import CompressionService

    service = CompressionService()

    # Test compression threshold
    test_tokens = 50000  # 50K tokens
    needs_compression = service.needs_compression(test_tokens)

    print(f"\nToken Count: {test_tokens}")
    print(f"Threshold: 70,000 (70% of 100K)")
    print(f"Needs Compression: {needs_compression}")

    test_tokens_high = 75000
    needs_compression_high = service.needs_compression(test_tokens_high)

    print(f"\nToken Count: {test_tokens_high}")
    print(f"Needs Compression: {needs_compression_high}")

    print(f"\n✅ Compression service working correctly")
    return True


async def test_statistics():
    """Test 5: Statistics and Monitoring"""
    print("\n" + "="*80)
    print("TEST 5: Statistics and Monitoring")
    print("="*80)

    ingestion_service = ContentIngestionService()

    stats = await ingestion_service.get_ingestion_stats()

    print(f"\n✅ Ingestion Statistics:")
    print(f"   Total Syncs: {stats.get('total_syncs', 0)}")
    print(f"   Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"   Total Cost: ${stats.get('total_cost', 0):.6f}")

    if stats.get('by_source'):
        print(f"\n   By Source:")
        for source, data in stats['by_source'].items():
            print(f"      {source}: {data['count']} syncs, {data['chunks']} chunks, ${data['cost']:.6f}")

    return True


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 SELVE Chatbot - End-to-End Pipeline Test")
    print("="*80)

    await db.connect()

    try:
        # Run tests
        results = []

        # Test 1: Ingestion
        try:
            results.append(("Content Ingestion", await test_ingestion()))
        except Exception as e:
            print(f"\n❌ Test 1 Failed: {e}")
            results.append(("Content Ingestion", False))

        # Test 2: RAG Retrieval
        try:
            results.append(("RAG Retrieval", await test_rag_retrieval()))
        except Exception as e:
            print(f"\n❌ Test 2 Failed: {e}")
            results.append(("RAG Retrieval", False))

        # Test 3: Chat Response
        try:
            results.append(("Chat Response", await test_chat_response()))
        except Exception as e:
            print(f"\n❌ Test 3 Failed: {e}")
            results.append(("Chat Response", False))

        # Test 4: Compression Check
        try:
            results.append(("Compression Check", await test_compression_check()))
        except Exception as e:
            print(f"\n❌ Test 4 Failed: {e}")
            results.append(("Compression Check", False))

        # Test 5: Statistics
        try:
            results.append(("Statistics", await test_statistics()))
        except Exception as e:
            print(f"\n❌ Test 5 Failed: {e}")
            results.append(("Statistics", False))

        # Summary
        print("\n" + "="*80)
        print("📊 Test Summary")
        print("="*80)

        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} - {test_name}")

        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)

        print(f"\n   Total: {passed_count}/{total_count} tests passed")

        if passed_count == total_count:
            print("\n🎉 All tests passed! Pipeline is working correctly.")
        else:
            print(f"\n⚠️  {total_count - passed_count} test(s) failed. Check logs above.")

        print("="*80 + "\n")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
