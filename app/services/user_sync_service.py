"""
User Sync Service - Syncs user data from main SELVE API to chat backend database
"""
import httpx
import logging
import os
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from app.db import db

logger = logging.getLogger(__name__)

MAIN_SELVE_API_URL = os.getenv("MAIN_SELVE_API_URL", "http://localhost:8000")
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")


class UserSyncService:
    """Syncs user data from main SELVE database to chat backend database"""

    async def sync_user_from_main_selve(self, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user from main SELVE API and create/update in chat backend database.

        Args:
            clerk_user_id: Clerk user ID

        Returns:
            User data if successful, None otherwise
        """
        if not INTERNAL_API_SECRET:
            logger.error("INTERNAL_API_SECRET not configured")
            return None

        try:
            # Fetch user from main SELVE API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MAIN_SELVE_API_URL}/api/internal/users/{clerk_user_id}",
                    headers={"X-Internal-Secret": INTERNAL_API_SECRET},
                    timeout=10.0
                )

                if response.status_code == 404:
                    logger.warning(f"User not found in main SELVE: {clerk_user_id}")
                    return None

                if response.status_code != 200:
                    logger.error(f"Failed to fetch user from main SELVE: {response.status_code}")
                    return None

                selve_user_data = response.json()

            # Create or update user in chat backend database
            user = await self._upsert_user(clerk_user_id, selve_user_data)

            logger.info(f"✅ Synced user from main SELVE: {clerk_user_id}")
            return user

        except httpx.TimeoutException:
            logger.error(f"Timeout fetching user from main SELVE: {clerk_user_id}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error fetching user from main SELVE: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error syncing user: {e}", exc_info=True)
            return None

    async def _upsert_user(self, clerk_user_id: str, selve_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update user in chat backend database based on main SELVE data.

        Args:
            clerk_user_id: Clerk user ID
            selve_data: User data from main SELVE API

        Returns:
            User record from database
        """
        # Check if user exists
        existing_user = await db.user.find_unique(
            where={"clerkId": clerk_user_id}
        )

        now = datetime.now(timezone.utc)

        # Prepare user data
        user_data = {
            "email": selve_data.get("email"),
            "name": selve_data.get("name"),
            "subscriptionPlan": selve_data.get("subscriptionPlan", "free"),
            "hasCompletedAssessment": selve_data.get("hasCompletedAssessment", False),
            "updatedAt": now,
        }

        if existing_user:
            # Update existing user
            user = await db.user.update(
                where={"clerkId": clerk_user_id},
                data=user_data
            )
            logger.info(f"📝 Updated existing user: {clerk_user_id}")
        else:
            # Create new user with period initialization
            user_data["clerkId"] = clerk_user_id
            user_data["currentPeriodStart"] = now
            user_data["currentPeriodCost"] = 0.0
            user_data["createdAt"] = now

            user = await db.user.create(data=user_data)
            logger.info(f"✨ Created new user: {clerk_user_id}")

        return user

    async def get_or_sync_user(self, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user from local database, syncing from main SELVE if not found.

        Args:
            clerk_user_id: Clerk user ID

        Returns:
            User data if found or synced successfully, None otherwise
        """
        # Try to get from local database first
        user = await db.user.find_unique(
            where={"clerkId": clerk_user_id}
        )

        if user:
            return user

        # User not found locally, sync from main SELVE
        logger.info(f"🔄 User not found locally, syncing from main SELVE: {clerk_user_id}")
        return await self.sync_user_from_main_selve(clerk_user_id)


# Singleton instance
user_sync_service = UserSyncService()
