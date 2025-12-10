"""
User Profile Service - Fetches SELVE personality scores from main app database
"""
import os
from typing import Dict, Any, Optional
from app.db import db


DEFAULT_ACTIVE_PLAN = os.getenv("DEFAULT_SUBSCRIPTION_PLAN", "Pro plan")
DEFAULT_FREE_PLAN = os.getenv("DEFAULT_FREE_PLAN", "Starter plan")
ASSESSMENT_URL = (
    os.getenv("ASSESSMENT_URL")
    or os.getenv("APP_URL")
    or os.getenv("NEXT_PUBLIC_APP_URL")
    or os.getenv("MAIN_APP_URL")
    or os.getenv("MAIN_APP_URL_PROD")
    or os.getenv("MAIN_APP_URL_DEV")
    or "http://localhost:3000"
)


class UserProfileService:
    """Service for fetching user personality profiles and SELVE scores"""

    # Dimension descriptions for context building
    DIMENSION_DESCRIPTIONS = {
        "LUMEN": "Mindful Curiosity - Social energy and how you recharge",
        "AETHER": "Rational Reflection - How you process information",
        "ORPHEUS": "Compassionate Connection - How you approach decisions",
        "ORIN": "Structured Harmony - Your approach to planning and structure",
        "LYRA": "Creative Expression - Your openness to new experiences",
        "VARA": "Purposeful Commitment - Your emotional stability",
        "CHRONOS": "Adaptive Spontaneity - Your agreeableness and cooperation",
        "KAEL": "Bold Resilience - Your conscientiousness and discipline"
    }

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

    async def get_user_account(self, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch core user account details for UI surfaces (sidebar, header, etc.).

        Args:
            clerk_user_id: Clerk authentication ID

        Returns:
            Dict containing user identity and inferred plan info, or None if not found
        """
        try:
            user = await db.user.find_first(where={"clerkId": clerk_user_id})

            if not user:
                return None

            # Check if the user has a current assessment to infer engagement level
            latest_result = await db.assessmentresult.find_first(
                where={
                    "clerkUserId": clerk_user_id,
                    "isCurrent": True
                },
                order={"createdAt": "desc"}
            )

            has_assessment = bool(latest_result)

            # We don't yet store billing plans; infer a plan label from engagement
            subscription_plan = DEFAULT_ACTIVE_PLAN if has_assessment else DEFAULT_FREE_PLAN

            user_name = user.name or (user.email.split("@")[0] if user.email else None)

            return {
                "user_id": user.id,
                "clerk_user_id": clerk_user_id,
                "user_name": user_name,
                "email": user.email,
                "has_assessment": has_assessment,
                "subscription_plan": subscription_plan,
            }

        except Exception as e:
            print(f"Error fetching user account: {str(e)}")
            return None

    async def get_user_profile(self, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch complete user profile including personality data and user info.
        This is the main method used by chat_service for personalization.

        Args:
            clerk_user_id: Clerk authentication ID

        Returns:
            Complete profile dict with has_assessment flag, scores, and user info
        """
        try:
            # First get the user's basic info
            user = await db.user.find_first(
                where={"clerkId": clerk_user_id}
            )

            user_name = None
            if user:
                user_name = user.name or (user.email.split("@")[0] if user.email else None)

            # Get assessment results
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
                return {
                    "has_assessment": False,
                    "user_name": user_name,
                    "clerk_user_id": clerk_user_id
                }

            # Format scores
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

            # Get narrative summary if available
            narrative_summary = None
            if result.narrative and isinstance(result.narrative, dict):
                narrative_summary = result.narrative.get("summary") or result.narrative.get("overview")

            return {
                "has_assessment": True,
                "clerk_user_id": clerk_user_id,
                "user_name": user_name,
                "scores": scores,
                "archetype": result.archetype,
                "profile_pattern": result.profilePattern,
                "narrative_summary": narrative_summary,
                "consistency_score": result.consistencyScore,
                "attention_score": result.attentionScore,
                "completed_at": result.createdAt.isoformat() if result.createdAt else None
            }

        except Exception as e:
            print(f"Error fetching user profile: {str(e)}")
            return None

    def format_profile_for_context(self, profile: Dict[str, Any]) -> str:
        """
        Format user profile into context string for LLM system prompt.
        
        Args:
            profile: Profile dict from get_user_profile
            
        Returns:
            Formatted string for LLM context
        """
        if not profile or not profile.get("has_assessment"):
            user_name = profile.get("user_name", "there") if profile else "there"
            assessment_base = ASSESSMENT_URL.rstrip("/")
            assessment_link = (
                assessment_base
                if assessment_base.lower().endswith("/assessment")
                else f"{assessment_base}/assessment"
            )

            return (
                """USER CONTEXT:
User {user_name} has not completed their SELVE personality assessment yet.
When relevant, encourage them to take the assessment to get personalized insights.
Assessment available at: {assessment_link}"""
            ).format(user_name=user_name, assessment_link=assessment_link)

        # Build personalized context
        user_name = profile.get("user_name", "this user")
        scores = profile.get("scores", {})
        archetype = profile.get("archetype", "Unknown")
        profile_pattern = profile.get("profile_pattern", "")
        
        # Sort dimensions by score to identify strengths
        sorted_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_dims[:3]
        bottom_2 = sorted_dims[-2:]

        context_parts = [
            "═══════════════════════════════════════════════════════════",
            f"USER PERSONALITY PROFILE: {user_name}",
            "═══════════════════════════════════════════════════════════",
            ""
        ]

        if archetype:
            context_parts.append(f"🎭 Archetype: {archetype}")
        if profile_pattern:
            context_parts.append(f"📊 Profile Pattern: {profile_pattern}")
        
        context_parts.append("")
        context_parts.append("DIMENSION SCORES (0-100):")
        
        for dim, score in sorted_dims:
            desc = self.DIMENSION_DESCRIPTIONS.get(dim, "")
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            context_parts.append(f"  {dim}: [{bar}] {int(score)} - {desc}")

        context_parts.extend([
            "",
            "KEY STRENGTHS (Top 3):",
        ])
        for dim, score in top_3:
            desc = self.DIMENSION_DESCRIPTIONS.get(dim, "")
            context_parts.append(f"  • {dim} ({int(score)}): {desc}")

        context_parts.extend([
            "",
            "GROWTH AREAS (Lower scores):",
        ])
        for dim, score in bottom_2:
            desc = self.DIMENSION_DESCRIPTIONS.get(dim, "")
            context_parts.append(f"  • {dim} ({int(score)}): {desc}")

        if profile.get("narrative_summary"):
            context_parts.extend([
                "",
                "PROFILE SUMMARY:",
                profile["narrative_summary"]
            ])

        context_parts.extend([
            "",
            "═══════════════════════════════════════════════════════════",
            "PERSONALIZATION GUIDELINES:",
            "- Reference their specific scores when relevant to the conversation",
            "- Tailor insights to their archetype and profile pattern",
            "- Be encouraging about growth areas - these are opportunities, not flaws",
            "- Connect their questions to their personality profile when helpful",
            "═══════════════════════════════════════════════════════════"
        ])

        return "\n".join(context_parts)

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
