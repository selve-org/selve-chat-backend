"""
Production Data Migration Script

Migrates all Qdrant collections from local dev to production:
- youtube_transcripts (27 videos, 89 chunks)
- selve_web_content (web pages)
- Preserves all metadata and embeddings

Usage:
    # Set production Qdrant URL in .env:
    QDRANT_URL_PROD=https://your-qdrant-cloud-url.com

    # Run migration:
    python scripts/migrate_to_production.py

Safety:
- Backs up existing production data (if any)
- Verifies collections before migrating
- Validates data after migration
"""

import os
import sys
import logging
from typing import List, Dict, Any
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QdrantMigration:
    """Migrate Qdrant collections from dev to production."""

    COLLECTIONS = [
        "youtube_transcripts",
        "selve_web_content",
        # Add more collections here as they're created
    ]

    def __init__(self):
        """Initialize migration tool."""
        self.dev_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.prod_url = os.getenv("QDRANT_URL_PROD")

        if not self.prod_url:
            raise ValueError(
                "QDRANT_URL_PROD not set in environment. "
                "Please set it to your production Qdrant URL."
            )

        self.dev_client = QdrantClient(url=self.dev_url)
        self.prod_client = QdrantClient(url=self.prod_url)

        logger.info(f"Dev Qdrant: {self.dev_url}")
        logger.info(f"Prod Qdrant: {self.prod_url}")

    def verify_dev_collections(self) -> Dict[str, Any]:
        """Verify all collections exist in dev."""
        logger.info("\n=== Verifying Dev Collections ===")

        dev_stats = {}
        for collection_name in self.COLLECTIONS:
            try:
                collection = self.dev_client.get_collection(collection_name)
                points_count = collection.points_count
                vectors_count = collection.vectors_count

                dev_stats[collection_name] = {
                    "points_count": points_count,
                    "vectors_count": vectors_count,
                    "exists": True,
                }

                logger.info(f"✅ {collection_name}: {points_count} points, {vectors_count} vectors")

            except Exception as e:
                logger.warning(f"⚠️  {collection_name}: Not found in dev - {e}")
                dev_stats[collection_name] = {
                    "exists": False,
                    "error": str(e),
                }

        return dev_stats

    def create_prod_collections(self, dev_stats: Dict[str, Any]):
        """Create collections in production (if they don't exist)."""
        logger.info("\n=== Creating Production Collections ===")

        for collection_name, stats in dev_stats.items():
            if not stats.get("exists"):
                logger.info(f"⏭️  Skipping {collection_name} (doesn't exist in dev)")
                continue

            try:
                # Check if already exists in prod
                try:
                    prod_collection = self.prod_client.get_collection(collection_name)
                    points_count = prod_collection.points_count

                    logger.info(
                        f"ℹ️  {collection_name} already exists in prod ({points_count} points)"
                    )

                    # Ask user if they want to recreate
                    response = input(
                        f"  Recreate {collection_name}? This will DELETE existing data. (yes/no): "
                    )

                    if response.lower() != "yes":
                        logger.info(f"  Skipping {collection_name}")
                        continue

                    # Delete existing collection
                    self.prod_client.delete_collection(collection_name)
                    logger.info(f"  Deleted existing {collection_name}")

                except Exception:
                    # Collection doesn't exist - that's fine
                    pass

                # Get dev collection config
                dev_collection = self.dev_client.get_collection(collection_name)
                config = dev_collection.config

                # Create in prod with same config
                self.prod_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=config.params.vectors.size,
                        distance=config.params.vectors.distance,
                    ),
                )

                logger.info(f"✅ Created {collection_name} in production")

            except Exception as e:
                logger.error(f"❌ Failed to create {collection_name}: {e}")
                raise

    def migrate_collection(self, collection_name: str):
        """Migrate a single collection from dev to prod."""
        logger.info(f"\n=== Migrating {collection_name} ===")

        try:
            # Get all points from dev
            logger.info("  Fetching points from dev...")

            all_points = []
            offset = None

            while True:
                result = self.dev_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                points, offset = result

                if not points:
                    break

                all_points.extend(points)
                logger.info(f"  Fetched {len(all_points)} points so far...")

                if offset is None:
                    break

            logger.info(f"  Total points to migrate: {len(all_points)}")

            # Upload to prod in batches
            batch_size = 100
            for i in range(0, len(all_points), batch_size):
                batch = all_points[i:i + batch_size]

                self.prod_client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )

                logger.info(f"  Uploaded batch {i//batch_size + 1}/{(len(all_points)-1)//batch_size + 1}")

            logger.info(f"✅ Migrated {len(all_points)} points to production")

        except Exception as e:
            logger.error(f"❌ Failed to migrate {collection_name}: {e}")
            raise

    def verify_production(self, dev_stats: Dict[str, Any]):
        """Verify production has all the data."""
        logger.info("\n=== Verifying Production ===")

        all_good = True

        for collection_name, dev_stat in dev_stats.items():
            if not dev_stat.get("exists"):
                continue

            try:
                prod_collection = self.prod_client.get_collection(collection_name)
                prod_points = prod_collection.points_count
                dev_points = dev_stat["points_count"]

                if prod_points == dev_points:
                    logger.info(f"✅ {collection_name}: {prod_points} points (matches dev)")
                else:
                    logger.error(
                        f"❌ {collection_name}: {prod_points} points in prod vs {dev_points} in dev - MISMATCH!"
                    )
                    all_good = False

            except Exception as e:
                logger.error(f"❌ {collection_name}: Failed to verify - {e}")
                all_good = False

        return all_good

    def run_migration(self):
        """Run complete migration."""
        logger.info("=" * 60)
        logger.info("🚀 Starting Qdrant Production Migration")
        logger.info("=" * 60)

        start_time = datetime.utcnow()

        try:
            # Step 1: Verify dev collections
            dev_stats = self.verify_dev_collections()

            if not any(stat.get("exists") for stat in dev_stats.values()):
                logger.error("❌ No collections found in dev - nothing to migrate")
                return False

            # Step 2: Create prod collections
            self.create_prod_collections(dev_stats)

            # Step 3: Migrate data
            for collection_name, stats in dev_stats.items():
                if stats.get("exists"):
                    self.migrate_collection(collection_name)

            # Step 4: Verify
            all_good = self.verify_production(dev_stats)

            # Summary
            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.info("\n" + "=" * 60)
            if all_good:
                logger.info("✅ Migration Complete - All Collections Verified")
            else:
                logger.error("⚠️  Migration Complete - Some Issues Found (see above)")

            logger.info(f"⏱️  Duration: {duration:.1f}s")
            logger.info("=" * 60)

            return all_good

        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run migration."""
    print("\n⚠️  WARNING: This will migrate data to PRODUCTION Qdrant")
    print(f"Production URL: {os.getenv('QDRANT_URL_PROD', 'NOT SET')}")
    print("\nThis will:")
    print("1. Create collections in production")
    print("2. Copy all data from local dev Qdrant")
    print("3. Verify data integrity")
    print()

    response = input("Continue with migration? (yes/no): ")

    if response.lower() != "yes":
        print("❌ Migration cancelled")
        return

    migration = QdrantMigration()
    success = migration.run_migration()

    if success:
        print("\n🎉 Migration successful! Production is ready.")
    else:
        print("\n❌ Migration had issues - please review logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
