#!/usr/bin/env python3
"""
Content Ingestion Script
Ingest SELVE content into the RAG knowledge base
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.content_ingestion_service import ContentIngestionService
from app.db import db


async def ingest_file(service: ContentIngestionService, file_path: str, source: str, content_type: str):
    """Ingest a single file"""
    print(f"\n{'='*80}")
    print(f"📄 Ingesting: {file_path}")
    print(f"{'='*80}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metadata = {
        "filename": os.path.basename(file_path),
        "file_path": file_path
    }

    # Add dimension name for dimension descriptions
    if content_type == "dimension_description":
        dimension = os.path.basename(file_path).replace('.md', '').upper()
        metadata["dimension"] = dimension

    result = await service.ingest_content(
        content=content,
        source=source,
        content_type=content_type,
        metadata=metadata,
        validate=True
    )

    print(f"\n✅ Result:")
    print(f"   Ingested: {result.get('ingested')}")
    print(f"   Chunks: {result.get('chunks_created', 0)}")
    print(f"   Embedding Cost: ${result.get('embedding_cost', 0):.6f}")
    print(f"   Validation: {result.get('validation_status')}")

    if result.get('validation_result'):
        val = result['validation_result']
        if 'scores' in val:
            print(f"\n   Validation Scores:")
            print(f"      - SELVE Aligned: {val['scores'].get('selve_aligned', 'N/A')}/10")
            print(f"      - Factually Accurate: {val['scores'].get('factually_accurate', 'N/A')}/10")
            print(f"      - Appropriate Tone: {val['scores'].get('appropriate_tone', 'N/A')}/10")
        print(f"   Recommendation: {val.get('recommendation', 'N/A')}")

        if val.get('suggestions'):
            print(f"\n   Suggestions:")
            for suggestion in val['suggestions'][:3]:
                print(f"      • {suggestion}")

    return result


async def ingest_directory(service: ContentIngestionService, directory: str, source: str, content_type: str):
    """Ingest all markdown files from a directory"""
    print(f"\n{'='*80}")
    print(f"📁 Ingesting directory: {directory}")
    print(f"{'='*80}")

    results = []
    for file_path in Path(directory).glob("*.md"):
        result = await ingest_file(service, str(file_path), source, content_type)
        results.append(result)

    return results


async def show_stats(service: ContentIngestionService):
    """Show ingestion statistics"""
    print(f"\n{'='*80}")
    print("📊 Ingestion Statistics")
    print(f"{'='*80}")

    stats = await service.get_ingestion_stats()

    print(f"\nTotal Syncs: {stats.get('total_syncs', 0)}")
    print(f"Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"Total Cost: ${stats.get('total_cost', 0):.6f}")

    if stats.get('by_source'):
        print(f"\nBy Source:")
        for source, data in stats['by_source'].items():
            print(f"   {source}:")
            print(f"      - Count: {data['count']}")
            print(f"      - Chunks: {data['chunks']}")
            print(f"      - Cost: ${data['cost']:.6f}")

    if stats.get('recent_syncs'):
        print(f"\nRecent Syncs:")
        for sync in stats['recent_syncs'][:5]:
            print(f"   • {sync['source']} / {sync['content_type']}: {sync['chunks']} chunks (${sync['cost']:.6f})")


async def main():
    """Main ingestion function"""
    print("🚀 SELVE Content Ingestion Script")
    print("="*80)

    # Initialize services
    await db.connect()
    service = ContentIngestionService()

    try:
        # Ingest sample dimension description
        sample_dir = Path(__file__).parent.parent / "sample_content" / "dimensions"

        if sample_dir.exists():
            print(f"\n✅ Found sample content directory: {sample_dir}")

            # Ingest all dimension files
            await ingest_directory(
                service=service,
                directory=str(sample_dir),
                source="selve_framework",
                content_type="dimension_description"
            )
        else:
            print(f"\n⚠️  Sample content directory not found: {sample_dir}")
            print("   Create sample content first!")

        # Show statistics
        await show_stats(service)

        print(f"\n{'='*80}")
        print("✅ Ingestion complete!")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
