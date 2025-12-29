"""
Production Initial Setup Script

Runs all initial data ingestion in production:
1. Crawls selve.me website
2. Ingests YouTube transcripts (from markdown files)
3. Verifies all collections are populated

This script should be run ONCE when deploying to production.
After initial setup, the automated crawler keeps content updated.

Usage:
    python scripts/production_setup.py
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def setup_web_content():
    """Crawl and ingest SELVE web content."""
    logger.info("\n" + "=" * 60)
    logger.info("1️⃣  Setting up SELVE Web Content")
    logger.info("=" * 60)

    try:
        from app.services.web_crawler_service import WebCrawlerService

        async with WebCrawlerService() as crawler:
            await crawler.run_full_crawl()

        logger.info("✅ SELVE web content ingested")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to ingest web content: {e}")
        import traceback
        traceback.print_exc()
        return False


async def setup_youtube_transcripts():
    """Ingest YouTube transcripts from markdown files."""
    logger.info("\n" + "=" * 60)
    logger.info("2️⃣  Setting up YouTube Transcripts")
    logger.info("=" * 60)

    try:
        # Check if transcript files exist
        transcript_dir = Path("/home/chris/selve-org/selve/data/youtube-transcripts")

        if not transcript_dir.exists():
            logger.warning(
                f"⚠️  YouTube transcript directory not found: {transcript_dir}\n"
                f"   Skipping YouTube ingestion.\n"
                f"   If you have transcripts, copy them to this directory and re-run."
            )
            return True  # Not a failure - just skip

        # Count markdown files
        md_files = list(transcript_dir.glob("*.md"))

        if not md_files:
            logger.warning("⚠️  No YouTube transcript files found - skipping")
            return True

        logger.info(f"Found {len(md_files)} YouTube transcript files")

        # Run ingestion script
        sys.path.insert(0, str(transcript_dir))

        from ingest_to_qdrant import YouTubeTranscriptIngestion

        ingestion = YouTubeTranscriptIngestion()
        ingestion.ingest_all()

        logger.info("✅ YouTube transcripts ingested")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to ingest YouTube transcripts: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_setup():
    """Verify all collections are properly set up."""
    logger.info("\n" + "=" * 60)
    logger.info("3️⃣  Verifying Setup")
    logger.info("=" * 60)

    try:
        from qdrant_client import QdrantClient
        from app.services.base import Config

        qdrant = QdrantClient(url=Config.QDRANT_URL)

        collections = {
            "selve_web_content": {
                "min_points": 5,  # At least 5 pages
                "description": "SELVE website content",
            },
            "youtube_transcripts": {
                "min_points": 50,  # At least 50 chunks (optional)
                "description": "Psychology YouTube transcripts",
                "optional": True,
            },
        }

        all_good = True

        for collection_name, config in collections.items():
            try:
                collection = qdrant.get_collection(collection_name)
                points_count = collection.points_count
                min_points = config["min_points"]
                optional = config.get("optional", False)

                if points_count >= min_points:
                    logger.info(
                        f"✅ {collection_name}: {points_count} points "
                        f"({config['description']})"
                    )
                elif optional:
                    logger.warning(
                        f"⚠️  {collection_name}: {points_count} points "
                        f"(expected {min_points}+, but optional)"
                    )
                else:
                    logger.error(
                        f"❌ {collection_name}: {points_count} points "
                        f"(expected {min_points}+)"
                    )
                    all_good = False

            except Exception as e:
                if config.get("optional"):
                    logger.warning(
                        f"⚠️  {collection_name}: Not found (optional) - {e}"
                    )
                else:
                    logger.error(f"❌ {collection_name}: Not found - {e}")
                    all_good = False

        return all_good

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run complete production setup."""
    logger.info("=" * 60)
    logger.info("🚀 SELVE Production Setup")
    logger.info("=" * 60)
    logger.info(f"Qdrant URL: {os.getenv('QDRANT_URL', 'NOT SET')}")
    logger.info(f"OpenAI API: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
    logger.info("")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set - cannot continue")
        return False

    if not os.getenv("QDRANT_URL"):
        logger.error("❌ QDRANT_URL not set - cannot continue")
        return False

    start_time = datetime.utcnow()

    # Step 1: Web content
    web_success = await setup_web_content()

    # Step 2: YouTube transcripts (optional)
    youtube_success = await setup_youtube_transcripts()

    # Step 3: Verify
    verify_success = await verify_setup()

    # Summary
    duration = (datetime.utcnow() - start_time).total_seconds()

    logger.info("\n" + "=" * 60)

    if web_success and verify_success:
        logger.info("✅ Production Setup Complete!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Automated crawler will run daily at 3AM UTC")
        logger.info("2. Check /api/crawler/status for scheduler status")
        logger.info("3. Manually trigger: POST /api/crawler/trigger")
    else:
        logger.error("⚠️  Setup completed with issues (see above)")
        logger.info("")
        logger.info("Troubleshooting:")
        logger.info("1. Check Qdrant connection")
        logger.info("2. Verify OpenAI API key")
        logger.info("3. Review error logs above")

    logger.info(f"\n⏱️  Total duration: {duration:.1f}s")
    logger.info("=" * 60)

    return web_success and verify_success


if __name__ == "__main__":
    success = asyncio.run(main())

    if success:
        print("\n🎉 Production is ready!")
        sys.exit(0)
    else:
        print("\n❌ Setup had issues - please review logs")
        sys.exit(1)
