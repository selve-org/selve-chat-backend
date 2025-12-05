"""
User Profile Service - Fetches SELVE personality scores from main app database
"""
from typing import Dict, Any, Optional
from app.db import db


class UserProfileService:
    """Service for fetching user personality profiles and SELVE scores"""

    async def get_user_scores(self, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user's current SELVE personality scores from AssessmentResult table

        Args:
            clerk_user_id: Clerk authentication ID

        Returns:
            Dict containing scores and profile info, or None if no results found
        """
        try:
            # Query the most recent assessment result for this user
            result = await db.assessmentresult.find_first(
                where={
                    "clerkUserId": clerk_user_id,
                    "isCurrent": True
                },
                order={
                    "createdAt": "desc"
                }
            )

            if not result:
                return None

            # Format scores for easy consumption
            scores = {
                "LUMEN": result.scoreLumen,
                "AETHER": result.scoreAether,
                "ORPHEUS": result.scoreOrpheus,
                "ORIN": result.scoreOrin,
                "LYRA": result.scoreLyra,
                "VARA": result.scoreVara,
                "CHRONOS": result.scoreChronos,
                "KAEL": result.scoreKael,
            }

            return {
                "clerk_user_id": result.clerkUserId,
                "scores": scores,
                "archetype": result.archetype,
                "profile_pattern": result.profilePattern,
                "consistency_score": result.consistencyScore,
                "attention_score": result.attentionScore,
                "created_at": result.createdAt.isoformat() if result.createdAt else None
            }

        except Exception as e:
            print(f"Error fetching user scores: {str(e)}")
            return None

    async def get_dimension_description(self, dimension: str) -> str:
        """
        Get a brief description of a SELVE dimension

        Args:
            dimension: Dimension name (LUMEN, AETHER, etc.)

        Returns:
            Brief description of the dimension
        """
        descriptions = {
            "LUMEN": "Mindful Curiosity - Social energy and how you recharge",
            "AETHER": "Rational Reflection - How you process information",
            "ORPHEUS": "Compassionate Connection - How you approach decisions",
            "ORIN": "Structured Harmony - Your approach to planning and structure",
            "LYRA": "Creative Expression - Your openness to new experiences",
            "VARA": "Purposeful Commitment - Your emotional stability",
            "CHRONOS": "Adaptive Spontaneity - Your agreeableness and cooperation",
            "KAEL": "Bold Resilience - Your conscientiousness and discipline"
        }

        return descriptions.get(dimension, "Unknown dimension")
