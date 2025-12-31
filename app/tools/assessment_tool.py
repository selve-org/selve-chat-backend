"""
Assessment Tool

Allows the chatbot to fetch user assessment scores and narrative on-demand.

Use cases:
- User asks "What are my scores?"
- User wants to know their personality profile
- Chatbot needs to reference specific assessment details
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from app.db import db


class AssessmentTool:
    """
    Tool for fetching user assessment scores and narrative.

    Provides the chatbot with on-demand access to:
    - 8 dimension scores (LUMEN, AETHER, ORPHEUS, VARA, CHRONOS, KAEL, ORIN, LYRA)
    - Complete personality narrative
    - Archetype and profile pattern
    - Assessment metadata
    """

    def __init__(self):
        """Initialize assessment tool."""
        self.logger = logging.getLogger(self.__class__.__name__)

    async def get_user_assessment(
        self,
        user_id: str,
        include_narrative: bool = True,
        include_scores: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch user's current assessment results.

        Args:
            user_id: User's Clerk ID
            include_narrative: Include full narrative text
            include_scores: Include dimension scores

        Returns:
            Dictionary with assessment data or error message
        """
        try:
            # Get user's most recent assessment result
            assessment = await db.assessmentresult.find_first(
                where={
                    "clerkUserId": user_id,
                    "isCurrent": True,
                },
                order_by={"createdAt": "desc"},
            )

            if not assessment:
                return {
                    "status": "not_found",
                    "message": "User has not completed the SELVE assessment yet.",
                    "has_assessment": False,
                }

            result = {
                "status": "success",
                "has_assessment": True,
                "archetype": assessment.archetype,
                "profile_pattern": assessment.profilePattern,
                "completed_at": assessment.createdAt.isoformat() if assessment.createdAt else None,
            }

            # Add quality metrics (non-personal metadata about assessment reliability)
            if assessment.consistencyScore is not None or assessment.attentionScore is not None:
                quality_notes = []
                if assessment.consistencyScore is not None:
                    if assessment.consistencyScore >= 80:
                        quality_notes.append(f"High consistency ({assessment.consistencyScore:.0f}%)")
                    elif assessment.consistencyScore >= 60:
                        quality_notes.append(f"Good consistency ({assessment.consistencyScore:.0f}%)")
                    else:
                        quality_notes.append(f"Lower consistency ({assessment.consistencyScore:.0f}%) - results may be less reliable")

                if assessment.attentionScore is not None:
                    if assessment.attentionScore >= 90:
                        quality_notes.append(f"Excellent attention ({assessment.attentionScore:.0f}%)")
                    elif assessment.attentionScore >= 70:
                        quality_notes.append(f"Good attention ({assessment.attentionScore:.0f}%)")
                    else:
                        quality_notes.append(f"Attention flags ({assessment.attentionScore:.0f}%) - user may have rushed")

                if quality_notes:
                    result["quality_info"] = " | ".join(quality_notes)

            # Include scores if requested
            if include_scores:
                result["scores"] = {
                    "LUMEN": assessment.scoreLumen,
                    "AETHER": assessment.scoreAether,
                    "ORPHEUS": assessment.scoreOrpheus,
                    "VARA": assessment.scoreVara,
                    "CHRONOS": assessment.scoreChronos,
                    "KAEL": assessment.scoreKael,
                    "ORIN": assessment.scoreOrin,
                    "LYRA": assessment.scoreLyra,
                }

                # Calculate average score
                scores_list = list(result["scores"].values())
                result["average_score"] = sum(scores_list) / len(scores_list) if scores_list else 0

            # Include narrative if requested
            if include_narrative and assessment.narrative:
                if isinstance(assessment.narrative, dict):
                    result["narrative"] = assessment.narrative
                else:
                    # Try to parse if it's a JSON string
                    import json
                    try:
                        result["narrative"] = json.loads(assessment.narrative)
                    except:
                        result["narrative"] = {"raw": str(assessment.narrative)}

            self.logger.info(f"✅ Fetched assessment for user {user_id[:8]}...")
            return result

        except Exception as e:
            self.logger.error(f"Error fetching assessment: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to fetch assessment: {str(e)}",
                "has_assessment": False,
            }

    async def get_archived_assessments(
        self,
        user_id: str,
        include_narrative: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch user's archived (previous) assessment results.
        Useful for tracking personality changes over time.

        Args:
            user_id: User's Clerk ID
            include_narrative: Include full narrative text for each archived result

        Returns:
            Dictionary with archived assessments or empty list
        """
        try:
            # Get all archived assessments (isCurrent = False)
            archived = await db.assessmentresult.find_many(
                where={
                    "clerkUserId": user_id,
                    "isCurrent": False,
                },
                order_by={"createdAt": "desc"},  # Most recent first
            )

            if not archived:
                return {
                    "status": "success",
                    "has_archived": False,
                    "count": 0,
                    "assessments": [],
                    "message": "No archived assessments found. User has only taken the assessment once."
                }

            results = []
            for assessment in archived:
                result = {
                    "archetype": assessment.archetype,
                    "profile_pattern": assessment.profilePattern,
                    "completed_at": assessment.createdAt.isoformat() if assessment.createdAt else None,
                    "scores": {
                        "LUMEN": assessment.scoreLumen,
                        "AETHER": assessment.scoreAether,
                        "ORPHEUS": assessment.scoreOrpheus,
                        "VARA": assessment.scoreVara,
                        "CHRONOS": assessment.scoreChronos,
                        "KAEL": assessment.scoreKael,
                        "ORIN": assessment.scoreOrin,
                        "LYRA": assessment.scoreLyra,
                    }
                }

                # Calculate average score
                scores_list = list(result["scores"].values())
                result["average_score"] = sum(scores_list) / len(scores_list) if scores_list else 0

                # Include narrative if requested
                if include_narrative and assessment.narrative:
                    if isinstance(assessment.narrative, dict):
                        result["narrative"] = assessment.narrative
                    else:
                        import json
                        try:
                            result["narrative"] = json.loads(assessment.narrative)
                        except:
                            result["narrative"] = {"raw": str(assessment.narrative)}

                results.append(result)

            self.logger.info(f"✅ Fetched {len(results)} archived assessments for user {user_id[:8]}...")
            return {
                "status": "success",
                "has_archived": True,
                "count": len(results),
                "assessments": results,
            }

        except Exception as e:
            self.logger.error(f"Error fetching archived assessments: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to fetch archived assessments: {str(e)}",
                "has_archived": False,
                "count": 0,
                "assessments": [],
            }

    async def compare_assessments(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Compare user's current assessment with their most recent archived assessment.
        Shows how personality has changed over time.

        Args:
            user_id: User's Clerk ID

        Returns:
            Dictionary with comparison data and insights
        """
        try:
            # Get current assessment
            current_result = await self.get_user_assessment(
                user_id=user_id,
                include_narrative=True,
                include_scores=True,
            )

            if current_result["status"] != "success":
                return {
                    "status": "error",
                    "message": "No current assessment found to compare.",
                }

            # Get most recent archived assessment
            archived_result = await self.get_archived_assessments(
                user_id=user_id,
                include_narrative=True,
            )

            if not archived_result["has_archived"]:
                return {
                    "status": "no_comparison",
                    "message": "No previous assessment found. User has only taken the assessment once.",
                }

            # Get the most recent archived assessment (first in list)
            previous = archived_result["assessments"][0]
            current = current_result

            # Calculate score changes
            score_changes = {}
            for dimension in ["LUMEN", "AETHER", "ORPHEUS", "VARA", "CHRONOS", "KAEL", "ORIN", "LYRA"]:
                current_score = current["scores"][dimension]
                previous_score = previous["scores"][dimension]
                change = current_score - previous_score

                score_changes[dimension] = {
                    "current": current_score,
                    "previous": previous_score,
                    "change": round(change, 2),
                    "percent_change": round((change / previous_score * 100) if previous_score > 0 else 0, 1),
                }

            # Identify biggest changes
            sorted_changes = sorted(
                score_changes.items(),
                key=lambda x: abs(x[1]["change"]),
                reverse=True
            )

            biggest_increase = None
            biggest_decrease = None
            for dimension, data in sorted_changes:
                if data["change"] > 0 and not biggest_increase:
                    biggest_increase = {"dimension": dimension, **data}
                elif data["change"] < 0 and not biggest_decrease:
                    biggest_decrease = {"dimension": dimension, **data}
                if biggest_increase and biggest_decrease:
                    break

            self.logger.info(f"✅ Compared assessments for user {user_id[:8]}...")
            return {
                "status": "success",
                "current_archetype": current["archetype"],
                "previous_archetype": previous["archetype"],
                "archetype_changed": current["archetype"] != previous["archetype"],
                "current_completed_at": current["completed_at"],
                "previous_completed_at": previous["completed_at"],
                "score_changes": score_changes,
                "biggest_increase": biggest_increase,
                "biggest_decrease": biggest_decrease,
                "total_archived": archived_result["count"],
            }

        except Exception as e:
            self.logger.error(f"Error comparing assessments: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to compare assessments: {str(e)}",
            }

    async def get_dimension_score(
        self,
        user_id: str,
        dimension: str,
    ) -> Dict[str, Any]:
        """
        Get user's score for a specific dimension.

        Args:
            user_id: User's Clerk ID
            dimension: Dimension name (LUMEN, AETHER, ORPHEUS, VARA, CHRONOS, KAEL, ORIN, LYRA)

        Returns:
            Dictionary with dimension score and description
        """
        dimension = dimension.upper()

        # Dimension descriptions
        dimension_info = {
            "LUMEN": "Social Energy - The radiant energy you bring to social situations",
            "AETHER": "Conceptual Depth - Your attraction to abstract thinking and philosophy",
            "ORPHEUS": "Emotional Sensitivity - How deeply you experience and express emotions",
            "VARA": "Purposeful Commitment - Your dedication to long-term goals and values",
            "CHRONOS": "Adaptive Spontaneity - Your flexibility and openness to change",
            "KAEL": "Bold Resilience - Your courage in facing challenges and taking risks",
            "ORIN": "Structured Harmony - Your preference for order and systematic approaches",
            "LYRA": "Creative Expression - Your drive to create and express uniqueness",
        }

        if dimension not in dimension_info:
            return {
                "status": "error",
                "message": f"Invalid dimension: {dimension}. Must be one of: {', '.join(dimension_info.keys())}",
            }

        try:
            # Fetch full assessment
            assessment = await self.get_user_assessment(user_id, include_narrative=False, include_scores=True)

            if assessment["status"] != "success":
                return assessment

            score = assessment["scores"].get(dimension)

            return {
                "status": "success",
                "dimension": dimension,
                "description": dimension_info[dimension],
                "score": score,
                "score_interpretation": self._interpret_score(score),
            }

        except Exception as e:
            self.logger.error(f"Error fetching dimension score: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to fetch dimension score: {str(e)}",
            }

    def _interpret_score(self, score: float) -> str:
        """
        Interpret a dimension score.

        Args:
            score: Score value (0-100)

        Returns:
            Interpretation string
        """
        if score >= 70:
            return "High - This dimension is a prominent part of your personality"
        elif score >= 30:
            return "Moderate - This dimension is balanced in your personality"
        else:
            return "Low - This dimension is less prominent in your personality"

    async def get_narrative_section(
        self,
        user_id: str,
        section: str,
    ) -> Dict[str, Any]:
        """
        Get a specific section from the user's narrative.

        Args:
            user_id: User's Clerk ID
            section: Section name (e.g., "overview", "core_traits", "strengths", "growth_areas")

        Returns:
            Dictionary with narrative section
        """
        try:
            assessment = await self.get_user_assessment(user_id, include_narrative=True, include_scores=False)

            if assessment["status"] != "success":
                return assessment

            narrative = assessment.get("narrative", {})

            if section in narrative:
                return {
                    "status": "success",
                    "section": section,
                    "content": narrative[section],
                }
            else:
                available_sections = list(narrative.keys())
                return {
                    "status": "not_found",
                    "message": f"Section '{section}' not found in narrative.",
                    "available_sections": available_sections,
                }

        except Exception as e:
            self.logger.error(f"Error fetching narrative section: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to fetch narrative section: {str(e)}",
            }
