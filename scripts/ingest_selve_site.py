#!/usr/bin/env python3
"""
Ingest SELVE Website Content

Crawls selve.me and chat.selve.me and ingests into Qdrant.

IMPORTANT: Bypasses validation for our own domains - no "cult initiation"
needed for SELVE's own content. We know our own house!

Usage:
    python scripts/ingest_selve_site.py [--production]
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.content_ingestion_service import ContentIngestionService
from app.tools.content_crawler import ToolRegistry, WebpageRequest


# Pages to crawl from selve.me
SELVE_PAGES = [
    "https://selve.me",
    "https://selve.me/how-it-works",
    "https://selve.me/privacy",
    "https://selve.me/terms",
    "https://selve.me/blog",
    "https://selve.me/assessment",
    "https://selve.me/about",
    "https://chat.selve.me",  # Chat interface
]


async def ingest_selve_content():
    """Crawl and ingest SELVE website content."""

    print("=" * 70)
    print("SELVE Website Content Ingestion")
    print("=" * 70)
    print()
    print("This script crawls selve.me and chat.selve.me and ingests")
    print("content into the 'selve_web_content' Qdrant collection.")
    print()
    print("🏠 NO VALIDATION REQUIRED - This is our own house!")
    print("   (Bypassing 'cult initiation' for selve.me/chat.selve.me)")
    print()
    print("=" * 70)
    print()

    ingestion_service = ContentIngestionService()
    webpage_tool = await ToolRegistry.webpage()

    total_ingested = 0
    total_chunks = 0

    for url in SELVE_PAGES:
        print(f"\n📄 Crawling: {url}")

        try:
            # Fetch page content using WebPageTool
            request = WebpageRequest(url=url)
            result = await webpage_tool.fetch(request)

            if not result or not result.content:
                print(f"   ⚠️  No content retrieved")
                continue

            # Extract text content
            content = result.content
            title = result.title or url.split("/")[-1].replace("-", " ").title()

            print(f"   ✅ Fetched: {title} ({len(content)} chars)")

            # Ingest WITHOUT VALIDATION (our own domain!)
            # Note: ContentIngestionService.ingest_content doesn't validate by default
            ingestion_result = await ingestion_service.ingest_content(
                content=content,
                title=title,
                source="selve_website",
                collection_name="selve_web_content",
                metadata={
                    "url": url,
                    "category": _categorize_url(url),
                    "domain": "selve.me" if "selve.me" in url else "chat.selve.me",
                    "crawled_date": "2025-12-30",
                },
            )

            chunks_created = ingestion_result.chunks_created
            total_chunks += chunks_created
            total_ingested += 1

            print(f"   ✨ Ingested: {chunks_created} chunks created")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    print()
    print("=" * 70)
    print(f"✅ COMPLETE!")
    print(f"   Pages ingested: {total_ingested}/{len(SELVE_PAGES)}")
    print(f"   Total chunks: {total_chunks}")
    print("=" * 70)

    # Verify collection
    from qdrant_client import QdrantClient
    from app.services.base import Config

    qdrant = QdrantClient(url=Config.QDRANT_URL)
    collection_info = qdrant.get_collection("selve_web_content")

    print()
    print("📊 Collection Status:")
    print(f"   Total points: {collection_info.points_count}")
    print(f"   Status: {collection_info.status}")
    print()


def _categorize_url(url: str) -> str:
    """Categorize URL for better organization."""
    if "terms" in url:
        return "legal"
    elif "privacy" in url:
        return "legal"
    elif "blog" in url:
        return "blog"
    elif "how-it-works" in url:
        return "product"
    elif "assessment" in url:
        return "product"
    elif "about" in url:
        return "about"
    elif "chat.selve.me" in url:
        return "chat"
    else:
        return "landing"


if __name__ == "__main__":
    asyncio.run(ingest_selve_content())
