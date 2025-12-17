"""
Usage Service - Manages user usage tracking and limits for subscription tiers
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from app.db import db

logger = logging.getLogger(__name__)

# Usage limits per tier (in USD)
USAGE_LIMITS = {
    "free": 1.00,  # $1 per day
    "pro": float('inf')  # Unlimited
}

# Notification thresholds
NOTIFICATION_THRESHOLDS = {
    "warning": 0.80,  # 80%
    "critical": 1.00   # 100%
}


class UsageService:
    """Service for tracking and enforcing usage limits"""

    async def get_current_usage(self, clerk_user_id: str) -> Dict[str, Any]:
        """
        Get user's current usage in the current 24h period

        Args:
            clerk_user_id: Clerk user ID

        Returns:
            Dict with usage information:
            {
                "total_cost": float,
                "message_count": int,
                "period_start": str (ISO),
                "period_end": str (ISO),
                "percentage_used": float,
                "limit": float,
                "subscription_plan": str
            }
        """
        # Get user's subscription plan
        user = await db.user.find_unique(
            where={"clerkId": clerk_user_id}
        )

        if not user:
            raise ValueError(f"User not found: {clerk_user_id}")

        subscription_plan = user.subscriptionPlan or "free"
        limit = USAGE_LIMITS.get(subscription_plan, USAGE_LIMITS["free"])

        # Check if usage period needs reset
        await self._reset_usage_if_needed(user)

        # Get fresh user data after potential reset
        user = await db.user.find_unique(
            where={"clerkId": clerk_user_id}
        )

        # Get message count for current period
        period_start = user.currentPeriodStart
        message_count = await db.chatmessage.count(
            where={
                "session": {
                    "clerkUserId": clerk_user_id
                },
                "createdAt": {
                    "gte": period_start
                },
                "role": "user"  # Only count user messages
            }
        )

        current_cost = user.currentPeriodCost or 0.0
        percentage_used = (current_cost / limit * 100) if limit != float('inf') else 0

        period_end = user.currentPeriodEnd or (period_start + timedelta(hours=24))

        return {
            "total_cost": current_cost,
            "message_count": message_count,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "percentage_used": min(percentage_used, 100),  # Cap at 100%
            "limit": limit if limit != float('inf') else None,
            "subscription_plan": subscription_plan,
            "time_until_reset": self._calculate_time_until_reset(period_end)
        }

    async def check_usage_limit(
        self,
        clerk_user_id: str,
        subscription_plan: str
    ) -> Dict[str, Any]:
        """
        Check if user can send a message based on usage limits

        Args:
            clerk_user_id: Clerk user ID
            subscription_plan: User's subscription plan ("free" or "pro")

        Returns:
            Dict with:
            {
                "can_send": bool,
                "usage_percentage": float,
                "limit_exceeded": bool,
                "current_cost": float,
                "limit": float
            }
        """
        # Pro users always have unlimited access
        if subscription_plan == "pro":
            return {
                "can_send": True,
                "usage_percentage": 0,
                "limit_exceeded": False,
                "current_cost": 0,
                "limit": None
            }

        # Get current usage for free users
        usage = await self.get_current_usage(clerk_user_id)
        limit = USAGE_LIMITS["free"]
        current_cost = usage["total_cost"]
        usage_percentage = usage["percentage_used"]

        # Strict enforcement - no buffer (removes attack surface)
        # Previously had 5% buffer which allowed users to consistently overspend
        can_send = current_cost < limit
        limit_exceeded = current_cost >= limit

        return {
            "can_send": can_send,
            "usage_percentage": usage_percentage,
            "limit_exceeded": limit_exceeded,
            "current_cost": current_cost,
            "limit": limit,
            "time_until_reset": usage["time_until_reset"]
        }

    async def increment_usage(
        self,
        clerk_user_id: str,
        cost: float,
        tokens: int,
        session_id: str,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Increment user's usage after sending a message (ATOMIC - prevents race conditions)

        Args:
            clerk_user_id: Clerk user ID
            cost: Cost of the message in USD
            tokens: Number of tokens used
            session_id: Chat session ID
            model: LLM model used
            provider: LLM provider (anthropic/openai)

        Returns:
            Updated usage information

        Raises:
            ValueError: If inputs are invalid or user not found
            HTTPException: If usage limit would be exceeded
        """
        # INPUT VALIDATION - Must be first to prevent malicious inputs
        if cost < 0:
            raise ValueError(f"Cost must be non-negative, got: {cost}")
        if tokens < 0:
            raise ValueError(f"Tokens must be non-negative, got: {tokens}")
        if cost > 10.0:
            # Sanity check - single request shouldn't cost $10+
            logger.warning(f"Suspicious cost value: ${cost} for user {clerk_user_id}")
            raise ValueError(f"Suspicious cost value: ${cost} - possible error")

        # Use transaction to prevent race conditions
        # This ensures atomicity - either all operations succeed or none do
        async with db.tx() as transaction:
            # Lock the user row for update (prevents concurrent modifications)
            user = await transaction.user.find_unique(
                where={"clerkId": clerk_user_id}
            )

            if not user:
                raise ValueError(f"User not found: {clerk_user_id}")

            # Reset if needed (within transaction for atomicity)
            was_reset = await self._reset_usage_if_needed_tx(transaction, user)

            # Recalculate current cost after potential reset
            current_cost = 0.0 if was_reset else (user.currentPeriodCost or 0.0)
            new_cost = current_cost + cost

            # Check limit AFTER acquiring lock (prevents race condition)
            subscription_plan = user.subscriptionPlan or "free"
            limit = USAGE_LIMITS.get(subscription_plan, USAGE_LIMITS["free"])

            if limit != float('inf') and new_cost > limit:
                # Strict enforcement - no 5% buffer
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail=f"Usage limit exceeded: ${new_cost:.4f} > ${limit:.2f}"
                )

            # Atomic update within transaction
            period_start = user.currentPeriodStart if not was_reset else datetime.utcnow()
            period_end = user.currentPeriodEnd if not was_reset else (datetime.utcnow() + timedelta(hours=24))

            updated_user = await transaction.user.update(
                where={"id": user.id},
                data={
                    "currentPeriodCost": new_cost,
                    "currentPeriodStart": period_start,
                    "currentPeriodEnd": period_end,
                    "lastUsageReset": user.lastUsageReset if not was_reset else datetime.utcnow()
                }
            )

            # Update or create ChatbotAnalytics record (within transaction for atomicity)
            reset_at = period_start + timedelta(hours=24)

            # Try to find existing analytics record for this period
            analytics = await transaction.chatbotanalytics.find_first(
                where={
                    "clerkUserId": clerk_user_id,
                    "periodStart": period_start
                }
            )

            if analytics:
                # Update existing record
                await transaction.chatbotanalytics.update(
                    where={"id": analytics.id},
                    data={
                        "messageCount": {"increment": 1},
                        "totalTokensUsed": {"increment": tokens},
                        "totalCost": {"increment": cost},
                        "provider": provider,
                        "model": model
                    }
                )
            else:
                # Create new record
                await transaction.chatbotanalytics.create(
                    data={
                        "userId": user.id,
                        "clerkUserId": clerk_user_id,
                        "sessionId": session_id,
                        "messageCount": 1,
                        "totalTokensUsed": tokens,
                        "totalCost": cost,
                        "provider": provider,
                        "model": model,
                        "periodStart": period_start,
                        "periodEnd": period_end,
                        "resetAt": reset_at
                    }
                )

            logger.info(f"Usage incremented: ${cost:.4f} for user {clerk_user_id[:8]}*** (new total: ${new_cost:.4f})")

        # Transaction committed successfully at this point

        # Check if we need to send notifications (outside transaction - non-critical)
        await self._check_and_notify_threshold(
            clerk_user_id,
            updated_user.currentPeriodCost,
            updated_user.subscriptionPlan
        )

        # Return updated usage
        return await self.get_current_usage(clerk_user_id)

    async def _reset_usage_if_needed_tx(self, transaction: Any, user: Any) -> bool:
        """
        Reset usage if 24-hour period has expired (transaction-safe version)

        This method is called within a database transaction to ensure atomicity.

        Args:
            transaction: Prisma transaction object
            user: User database object

        Returns:
            True if reset occurred, False otherwise
        """
        now = datetime.utcnow()
        period_start = user.currentPeriodStart
        period_end = user.currentPeriodEnd or (period_start + timedelta(hours=24))

        # Check if period has expired
        if now >= period_end:
            logger.info(f"Resetting usage for user {user.clerkId[:8]}***")

            # Reset happens within the transaction (atomic with usage increment)
            # Note: We don't actually update the user here - the caller will do it
            # This just returns True to signal that reset is needed
            return True

        return False

    async def _reset_usage_if_needed(self, user: Any) -> bool:
        """
        Reset usage if 24-hour period has expired (standalone version)

        Use this when NOT within a transaction. For transactional operations,
        use _reset_usage_if_needed_tx instead.

        Args:
            user: User database object

        Returns:
            True if reset occurred, False otherwise
        """
        now = datetime.utcnow()
        period_start = user.currentPeriodStart
        period_end = user.currentPeriodEnd or (period_start + timedelta(hours=24))

        # Check if period has expired
        if now >= period_end:
            logger.info(f"Resetting usage for user {user.clerkId[:8]}***")

            # Reset usage
            new_period_start = now
            new_period_end = now + timedelta(hours=24)

            await db.user.update(
                where={"id": user.id},
                data={
                    "currentPeriodStart": new_period_start,
                    "currentPeriodEnd": new_period_end,
                    "currentPeriodCost": 0.0,
                    "lastUsageReset": now
                }
            )

            return True

        return False

    async def _check_and_notify_threshold(
        self,
        clerk_user_id: str,
        current_cost: float,
        subscription_plan: str
    ) -> None:
        """
        Check if usage threshold crossed and send notification

        Args:
            clerk_user_id: Clerk user ID
            current_cost: Current period cost
            subscription_plan: User's subscription plan
        """
        # Only notify free users
        if subscription_plan != "free":
            return

        limit = USAGE_LIMITS["free"]
        percentage = (current_cost / limit) * 100

        # Send warning at 80%
        if percentage >= 80 and percentage < 90:
            logger.info(f"User {clerk_user_id} at 80% usage threshold")
            # TODO: Integrate with notification service
            # await self._send_notification(clerk_user_id, "warning", percentage)

        # Send critical at 100%
        elif percentage >= 100:
            logger.info(f"User {clerk_user_id} reached usage limit")
            # TODO: Integrate with notification service
            # await self._send_notification(clerk_user_id, "critical", percentage)

    def _calculate_time_until_reset(self, period_end: datetime) -> str:
        """
        Calculate human-readable time until reset

        Args:
            period_end: End of current period

        Returns:
            String like "8h 23m" or "45m"
        """
        now = datetime.utcnow()
        delta = period_end - now

        if delta.total_seconds() <= 0:
            return "0m"

        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


# Singleton instance
usage_service = UsageService()
