"""
User Profile Service - Fetch SELVE assessment data for personalization
"""
from typing import Optional, Dict, Any
from app.db import db


class UserProfileService:
    """
    Service for fetching and formatting user SELVE assessment data

    Retrieves:
    - 8 dimension scores (LUMEN, AETHER, ORPHEUS, VARA, CHRONOS, KAEL, ORIN, LYRA)
    - Narrative summaries
    - Archetype
    - Demographics
    - Friend insights (blind spots)
    """

    # SELVE Dimension names mapping
    DIMENSIONS = {
        "scoreLumen": "LUMEN (Social Energy/Extraversion)",
        "scoreAether": "AETHER (Emotional Stability)",
        "scoreOrpheus": "ORPHEUS (Empathy/Warmth)",
        "scoreVara": "VARA (Honesty/Integrity)",
        "scoreChronos": "CHRONOS (Patience/Flexibility)",
        "scoreKael": "KAEL (Assertiveness/Dominance)",
        "scoreOrin": "ORIN (Conscientiousness/Organization)",
        "scoreLyra": "LYRA (Openness/Curiosity)"
    }

    async def get_user_profile(
        self,
        clerk_user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch complete user profile with SELVE scores

        Args:
            clerk_user_id: Clerk authentication ID

        Returns:
            {
                "has_assessment": bool,
                "dimensions": {...},
                "archetype": str,
                "narrative": {...},
                "demographics": {...},
                "friend_insights": {...}
            }
        """
        try:
            # Get latest assessment result
            result = await db.assessmentresult.find_first(
                where={
                    "clerkUserId": clerk_user_id,
                    "isCurrent": True
                },
                include={
                    "session": {
                        "include": {
                            "friendInsightGenerations": {
                                "where": {"isCurrent": True},
                                "take": 1
                            }
                        }
                    }
                }
            )

            if not result:
                return {
                    "has_assessment": False,
                    "message": "No SELVE assessment found. Please complete your assessment first."
                }

            # Extract dimension scores
            dimensions = {
                "LUMEN": round(result.scoreLumen, 1),
                "AETHER": round(result.scoreAether, 1),
                "ORPHEUS": round(result.scoreOrpheus, 1),
                "VARA": round(result.scoreVara, 1),
                "CHRONOS": round(result.scoreChronos, 1),
                "KAEL": round(result.scoreKael, 1),
                "ORIN": round(result.scoreOrin, 1),
                "LYRA": round(result.scoreLyra, 1)
            }

            # Get friend insights if available
            friend_insights = None
            if result.session and result.session.friendInsightGenerations:
                friend_gen = result.session.friendInsightGenerations[0]
                friend_insights = {
                    "narrative": friend_gen.narrative,
                    "blind_spots": friend_gen.blindSpots,
                    "friend_count": friend_gen.friendCount
                }

            # Get demographics
            demographics = None
            if result.session and result.session.demographics:
                demographics = result.session.demographics

            return {
                "has_assessment": True,
                "dimensions": dimensions,
                "archetype": result.archetype,
                "profile_pattern": result.profilePattern,
                "narrative": result.narrative,
                "demographics": demographics,
                "friend_insights": friend_insights,
                "consistency_score": result.consistencyScore,
                "completed_at": result.createdAt.isoformat()
            }

        except Exception as e:
            print(f"❌ Error fetching user profile: {e}")
            return None

    def format_profile_for_context(
        self,
        profile: Dict[str, Any]
    ) -> str:
        """
        Format user profile as context string for LLM

        Args:
            profile: User profile dict from get_user_profile()

        Returns:
            Formatted context string
        """
        if not profile or not profile.get("has_assessment"):
            return ""

        dimensions = profile["dimensions"]

        # Build context string
        context_parts = [
            "=== USER'S SELVE PROFILE ===",
            "",
            "Dimension Scores (0-100):"
        ]

        # Add dimension scores with interpretation
        for dim, score in dimensions.items():
            interpretation = self._interpret_score(score)
            context_parts.append(f"- {dim}: {score}/100 ({interpretation})")

        # Add archetype if available
        if profile.get("archetype"):
            context_parts.append(f"\nPrimary Archetype: {profile['archetype']}")

        # Add friend insights if available
        if profile.get("friend_insights") and profile["friend_insights"].get("narrative"):
            context_parts.append("\nFriend Insights:")
            context_parts.append(f"- Based on {profile['friend_insights']['friend_count']} friend(s)")

            # Add blind spots
            blind_spots = profile["friend_insights"].get("blind_spots", [])
            if blind_spots:
                context_parts.append("- Notable Blind Spots:")
                for spot in blind_spots[:3]:  # Top 3 blind spots
                    dim = spot.get("dimension")
                    diff = spot.get("diff", 0)
                    context_parts.append(f"  • {dim}: Friends see you {abs(diff)} points {'higher' if diff > 0 else 'lower'}")

        context_parts.append("\n===")

        return "\n".join(context_parts)

    def _interpret_score(self, score: float) -> str:
        """Interpret dimension score"""
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Moderate"
        else:
            return "Low"

    async def get_user_dimensions_summary(
        self,
        clerk_user_id: str
    ) -> Optional[str]:
        """
        Get a brief summary of user's dimensions for quick context

        Returns a single-line summary like:
        "High LUMEN (82), Moderate AETHER (55), Low CHRONOS (32)..."
        """
        profile = await self.get_user_profile(clerk_user_id)

        if not profile or not profile.get("has_assessment"):
            return None

        dimensions = profile["dimensions"]

        # Find highest and lowest dimensions
        sorted_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)

        top_3 = sorted_dims[:3]
        bottom_2 = sorted_dims[-2:]

        summary_parts = ["Top dimensions:"]
        for dim, score in top_3:
            summary_parts.append(f"{dim} ({int(score)})")

        summary_parts.append("| Lower dimensions:")
        for dim, score in bottom_2:
            summary_parts.append(f"{dim} ({int(score)})")

        return " ".join(summary_parts)
