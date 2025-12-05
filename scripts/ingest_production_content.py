#!/usr/bin/env python3
"""
Production Content Ingestion Script
Ingest SELVE blog posts and framework content into RAG knowledge base
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
    print(f"📄 Ingesting: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract frontmatter metadata if exists
    metadata = {
        "filename": os.path.basename(file_path),
        "file_path": file_path
    }

    # Add dimension name for dimension blogs
    if content_type == "dimension_blog":
        dimension = os.path.basename(file_path).replace('.md', '').upper()
        metadata["dimension"] = dimension

        # Extract frontmatter if present
        if content.startswith('---'):
            try:
                frontmatter_end = content.find('---', 3)
                if frontmatter_end > 0:
                    frontmatter = content[3:frontmatter_end].strip()
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip().strip('"')
            except Exception as e:
                print(f"   ⚠️  Could not parse frontmatter: {e}")

    result = await service.ingest_content(
        content=content,
        source=source,
        content_type=content_type,
        metadata=metadata,
        validate=False  # Skip validation for production content (already written by SELVE team)
    )

    print(f"\n✅ Result:")
    print(f"   Ingested: {result.get('ingested')}")
    print(f"   Chunks: {result.get('chunks_created', 0)}")
    print(f"   Embedding Cost: ${result.get('embedding_cost', 0):.6f}")
    print(f"   Content Hash: {result.get('content_hash', 'N/A')[:12]}...")

    return result


async def ingest_directory(service: ContentIngestionService, directory: str, source: str, content_type: str):
    """Ingest all markdown files from a directory"""
    print(f"\n{'='*80}")
    print(f"📁 Ingesting directory: {directory}")
    print(f"{'='*80}")

    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"\n❌ Directory not found: {directory}")
        return []

    # Find all .md files
    md_files = sorted(directory_path.glob("*.md"))

    if not md_files:
        print(f"\n⚠️  No .md files found in {directory}")
        return []

    print(f"\n📝 Found {len(md_files)} files to ingest:")
    for f in md_files:
        print(f"   • {f.name}")

    results = []
    total_cost = 0.0
    total_chunks = 0

    for file_path in md_files:
        result = await ingest_file(service, str(file_path), source, content_type)
        results.append(result)

        if result.get('ingested'):
            total_cost += result.get('embedding_cost', 0)
            total_chunks += result.get('chunks_created', 0)

    print(f"\n{'='*80}")
    print("📊 Batch Summary")
    print(f"{'='*80}")
    print(f"Total Files: {len(md_files)}")
    print(f"Successfully Ingested: {sum(1 for r in results if r.get('ingested'))}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Total Cost: ${total_cost:.6f}")

    return results


async def main():
    """Main ingestion function"""
    import sys

    print("🚀 SELVE Production Content Ingestion")
    print("="*80)

    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python ingest_production_content.py <directory_path> [source] [content_type]")
        print("\nExample:")
        print("  python ingest_production_content.py /path/to/dimensions/ selve_framework dimension_blog")
        print("\nDefault values:")
        print("  source: selve_framework")
        print("  content_type: dimension_blog")
        sys.exit(1)

    directory = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "selve_framework"
    content_type = sys.argv[3] if len(sys.argv) > 3 else "dimension_blog"

    print(f"\n📂 Directory: {directory}")
    print(f"📦 Source: {source}")
    print(f"🏷️  Content Type: {content_type}")

    # Initialize services
    await db.connect()
    service = ContentIngestionService()

    try:
        # Ingest directory
        results = await ingest_directory(
            service=service,
            directory=directory,
            source=source,
            content_type=content_type
        )

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
