"""
Friend Insights Tool

Allows the chatbot to fetch user's friend insights data on-demand.

Use cases:
- User asks "What are my blind spots?"
- User wants to know how friends see them differently
- User asks about friend perception vs self perception
- Chatbot needs to reference friend comparison data
"""

import logging
from typing import Optional, Dict, Any, List
import json

from app.db import db


class FriendInsightsTool:
    """
    Tool for fetching user's friend insights and blind spots.

    Provides the chatbot with on-demand access to:
    - Blind spots (where friends see you differently)
    - Aggregated friend scores vs self scores
    - Friend insights narrative
    - Number of friends who responded
    - Individual friend response metadata
    """

    def __init__(self):
        """Initialize friend insights tool."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.BLIND_SPOT_THRESHOLD = 15.0  # Points difference to qualify as blind spot

    async def get_user_friend_insights(
        self,
        user_id: str,
        include_narrative: bool = True,
        include_individual_responses: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch user's friend insights data.

        Args:
            user_id: User's Clerk ID
            include_narrative: Include friend insights narrative summary
            include_individual_responses: Include individual friend response details

        Returns:
            Dictionary with friend insights data or error message
        """
        try:
            self.logger.info(f"🔍 Querying friend insights for user_id: {user_id[:20]}...")

            # Get user's assessment session
            assessment = await db.assessmentresult.find_first(
                where={
                    "clerkUserId": user_id,
                    "isCurrent": True,
                },
                order={"createdAt": "desc"},
            )

            if not assessment:
                self.logger.warning(f"❌ No assessment found for user_id: {user_id[:20]}...")
                return {
                    "status": "not_found",
                    "message": "User has not completed the SELVE assessment yet.",
                    "has_assessment": False,
                }

            session_id = assessment.sessionId

            # Get user record
            user = await db.user.find_unique(where={"clerkId": user_id})

            if not user:
                return {
                    "status": "error",
                    "message": "User record not found",
                    "has_friend_insights": False,
                }

            # Get friend responses via invite links
            invites = await db.invitelink.find_many(
                where={"inviterId": user.id},
                include={"friendResponse": True},
            )

            friend_responses = []
            friend_response_ids = []

            for invite in invites:
                if invite.friendResponse:
                    response_data = {
                        "id": invite.friendResponse.id,
                        "qualityScore": invite.friendResponse.qualityScore,
                        "completedAt": invite.friendResponse.completedAt.isoformat() if invite.friendResponse.completedAt else None,
                    }

                    if include_individual_responses:
                        response_data["responses"] = invite.friendResponse.responses
                        response_data["totalTime"] = invite.friendResponse.totalTime

                    friend_responses.append(response_data)
                    friend_response_ids.append(invite.friendResponse.id)

            if not friend_responses:
                return {
                    "status": "success",
                    "has_friend_insights": False,
                    "friend_count": 0,
                    "message": "No friends have completed the assessment yet. Invite friends to get insights!",
                    "blind_spots": [],
                }

            # Calculate aggregated friend scores
            aggregated_scores = self._calculate_aggregated_scores(friend_responses)

            # Get user's self scores
            self_scores = {
                "LUMEN": assessment.scoreLumen,
                "AETHER": assessment.scoreAether,
                "ORPHEUS": assessment.scoreOrpheus,
                "VARA": assessment.scoreVara,
                "CHRONOS": assessment.scoreChronos,
                "KAEL": assessment.scoreKael,
                "ORIN": assessment.scoreOrin,
                "LYRA": assessment.scoreLyra,
            }

            # Calculate blind spots
            blind_spots = self._identify_blind_spots(self_scores, aggregated_scores)

            result = {
                "status": "success",
                "has_friend_insights": True,
                "friend_count": len(friend_responses),
                "blind_spots": blind_spots,
                "self_scores": self_scores,
                "friend_scores": aggregated_scores,
            }

            # Add comparison summary
            result["comparison_summary"] = self._generate_comparison_summary(
                self_scores, aggregated_scores, blind_spots
            )

            # Get friend insights narrative from FriendInsightGeneration table
            if include_narrative and session_id:
                insight_generation = await db.friendinsightgeneration.find_first(
                    where={
                        "sessionId": session_id,
                        "isCurrent": True
                    },
                    order={"createdAt": "desc"}
                )

                if insight_generation:
                    result["narrative"] = insight_generation.narrative
                    result["narrative_generated_at"] = insight_generation.createdAt.isoformat() if insight_generation.createdAt else None
                    result["narrative_friend_count"] = insight_generation.friendCount

                    if insight_generation.generationError:
                        result["narrative_error"] = insight_generation.generationError

            # Add individual friend responses if requested
            if include_individual_responses:
                result["friend_responses"] = friend_responses

            self.logger.info(f"✅ Fetched friend insights for user {user_id[:8]}... ({len(friend_responses)} friends, {len(blind_spots)} blind spots)")
            return result

        except Exception as e:
            self.logger.error(f"Error fetching friend insights: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to fetch friend insights: {str(e)}",
                "has_friend_insights": False,
            }

    def _calculate_aggregated_scores(self, friend_responses: List[Dict]) -> Dict[str, float]:
        """
        Calculate average scores across all friend responses.

        Args:
            friend_responses: List of friend response dicts

        Returns:
            Dictionary of dimension -> average score
        """
        dimensions = ["LUMEN", "AETHER", "ORPHEUS", "VARA", "CHRONOS", "KAEL", "ORIN", "LYRA"]
        aggregated = {}

        for dimension in dimensions:
            scores = []
            for response in friend_responses:
                if "responses" in response and response["responses"]:
                    # Friend responses are stored as JSON
                    responses_data = response["responses"]
                    if isinstance(responses_data, str):
                        responses_data = json.loads(responses_data)

                    # Extract score for this dimension
                    if dimension in responses_data:
                        scores.append(float(responses_data[dimension]))

            if scores:
                aggregated[dimension] = round(sum(scores) / len(scores), 1)
            else:
                aggregated[dimension] = None

        return aggregated

    def _identify_blind_spots(
        self,
        self_scores: Dict[str, float],
        friend_scores: Dict[str, float]
    ) -> List[Dict]:
        """
        Identify blind spots where self-perception differs significantly from friends.

        Args:
            self_scores: Self-assessment scores
            friend_scores: Aggregated friend scores

        Returns:
            List of blind spot dicts with dimension, selfScore, friendScore, diff
        """
        blind_spots = []

        for dimension in self_scores:
            if dimension not in friend_scores or friend_scores[dimension] is None:
                continue

            self_score = self_scores[dimension]
            friend_score = friend_scores[dimension]
            diff = friend_score - self_score

            if abs(diff) >= self.BLIND_SPOT_THRESHOLD:
                blind_spots.append({
                    "dimension": dimension,
                    "selfScore": round(self_score, 1),
                    "friendScore": round(friend_score, 1),
                    "diff": round(diff, 1),
                    "type": "underestimate" if diff > 0 else "overestimate",
                    "description": self._get_blind_spot_description(dimension, diff)
                })

        # Sort by absolute difference (biggest gaps first)
        blind_spots.sort(key=lambda x: abs(x["diff"]), reverse=True)

        return blind_spots

    def _get_blind_spot_description(self, dimension: str, diff: float) -> str:
        """
        Get a human-readable description of a blind spot.

        Args:
            dimension: Dimension name
            diff: Difference (friend_score - self_score)

        Returns:
            Description string
        """
        dimension_names = {
            "LUMEN": "Social Energy",
            "AETHER": "Conceptual Depth",
            "ORPHEUS": "Emotional Sensitivity",
            "VARA": "Purposeful Commitment",
            "CHRONOS": "Adaptive Spontaneity",
            "KAEL": "Bold Resilience",
            "ORIN": "Structured Harmony",
            "LYRA": "Creative Expression",
        }

        name = dimension_names.get(dimension, dimension)
        points = abs(diff)

        if diff > 0:
            return f"Your friends see you {points:.0f} points higher in {name} than you see yourself. You may be underestimating this trait."
        else:
            return f"Your friends see you {points:.0f} points lower in {name} than you see yourself. You may be overestimating this trait."

    def _generate_comparison_summary(
        self,
        self_scores: Dict[str, float],
        friend_scores: Dict[str, float],
        blind_spots: List[Dict]
    ) -> str:
        """
        Generate a brief text summary of self vs friend comparison.

        Args:
            self_scores: Self-assessment scores
            friend_scores: Aggregated friend scores
            blind_spots: List of blind spots

        Returns:
            Summary string
        """
        if not blind_spots:
            return "Your self-perception aligns closely with how your friends see you. No major blind spots detected."

        # Find biggest gaps
        biggest_underestimate = None
        biggest_overestimate = None

        for bs in blind_spots:
            if bs["type"] == "underestimate" and not biggest_underestimate:
                biggest_underestimate = bs
            elif bs["type"] == "overestimate" and not biggest_overestimate:
                biggest_overestimate = bs

        summary_parts = []

        if biggest_underestimate:
            summary_parts.append(
                f"You underestimate your {biggest_underestimate['dimension']} "
                f"({biggest_underestimate['diff']:.0f} point gap)"
            )

        if biggest_overestimate:
            summary_parts.append(
                f"You overestimate your {biggest_overestimate['dimension']} "
                f"({abs(biggest_overestimate['diff']):.0f} point gap)"
            )

        return " | ".join(summary_parts) if summary_parts else "Mixed perception differences across dimensions."

    async def get_blind_spots_only(self, user_id: str) -> Dict[str, Any]:
        """
        Get only blind spots data for quick queries.

        Args:
            user_id: User's Clerk ID

        Returns:
            Dictionary with blind spots data
        """
        insights = await self.get_user_friend_insights(
            user_id=user_id,
            include_narrative=False,
            include_individual_responses=False,
        )

        if insights["status"] != "success":
            return insights

        return {
            "status": "success",
            "blind_spots": insights.get("blind_spots", []),
            "friend_count": insights.get("friend_count", 0),
            "has_blind_spots": len(insights.get("blind_spots", [])) > 0,
        }

    async def get_dimension_comparison(
        self,
        user_id: str,
        dimension: str
    ) -> Dict[str, Any]:
        """
        Compare self vs friend scores for a specific dimension.

        Args:
            user_id: User's Clerk ID
            dimension: Dimension name (LUMEN, AETHER, etc.)

        Returns:
            Dictionary with comparison data for that dimension
        """
        dimension = dimension.upper()

        valid_dimensions = ["LUMEN", "AETHER", "ORPHEUS", "VARA", "CHRONOS", "KAEL", "ORIN", "LYRA"]
        if dimension not in valid_dimensions:
            return {
                "status": "error",
                "message": f"Invalid dimension: {dimension}. Must be one of: {', '.join(valid_dimensions)}",
            }

        insights = await self.get_user_friend_insights(
            user_id=user_id,
            include_narrative=False,
            include_individual_responses=False,
        )

        if insights["status"] != "success":
            return insights

        self_score = insights["self_scores"].get(dimension)
        friend_score = insights["friend_scores"].get(dimension)

        if friend_score is None:
            return {
                "status": "no_data",
                "message": f"No friend data available for {dimension}",
            }

        diff = friend_score - self_score
        is_blind_spot = abs(diff) >= self.BLIND_SPOT_THRESHOLD

        return {
            "status": "success",
            "dimension": dimension,
            "self_score": self_score,
            "friend_score": friend_score,
            "difference": round(diff, 1),
            "is_blind_spot": is_blind_spot,
            "perception_type": "underestimate" if diff > 0 else "overestimate" if diff < 0 else "aligned",
            "friend_count": insights["friend_count"],
        }
