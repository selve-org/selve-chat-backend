"""
Content Validation Service - Validate external content against SELVE framework
"""
from typing import Dict, Any, Optional
from datetime import datetime
from app.db import db
from .llm_service import LLMService
import json


class ContentValidationService:
    """
    Service for validating content against SELVE framework principles

    Validates:
    - SELVE alignment (uses framework vocabulary correctly)
    - Factual accuracy (no false claims about dimensions/assessments)
    - Appropriate tone (warm, non-judgmental, growth-oriented)
    - Citation requirements (external claims must be cited)
    """

    def __init__(self):
        """Initialize with LLM for validation"""
        self.llm_service = LLMService()
        self.validation_prompt = self._load_validation_prompt()

    def _load_validation_prompt(self) -> str:
        """Load content validation prompt"""
        return """You are a content validator for the SELVE personality framework.

Your task is to validate content against SELVE principles:

**SELVE Framework Core Principles:**
1. **8 Dimensions**: LUMEN (Social Energy), AETHER (Emotional Stability), ORPHEUS (Empathy), VARA (Honesty), CHRONOS (Patience), KAEL (Assertiveness), ORIN (Conscientiousness), LYRA (Openness)
2. **Non-judgmental**: No scores are "good" or "bad" - all personalities have value
3. **Growth-oriented**: Focus on self-understanding and development, not labeling
4. **Evidence-based**: Claims about personality must be grounded in research or framework
5. **Warm tone**: Supportive, empathetic, never harsh or critical

**Validation Criteria:**
1. **SELVE Aligned** (score 0-10):
   - Uses SELVE vocabulary correctly
   - Doesn't contradict framework principles
   - Aligns with dimension definitions

2. **Factually Accurate** (score 0-10):
   - No false claims about SELVE assessments
   - External claims are reasonable and verifiable
   - No pseudoscience or unsupported generalizations

3. **Appropriate Tone** (score 0-10):
   - Warm, supportive, non-judgmental
   - Growth-oriented, not labeling
   - No harsh criticism or negative framing

4. **Citation Needed** (yes/no):
   - Does content make external claims requiring citations?

5. **Recommendation** (approve/reject/needs_revision):
   - approve: Content meets all criteria (8+ on all scores)
   - needs_revision: Minor issues (6-7 on any score)
   - reject: Major issues (<6 on any score)

Format your response as JSON:
{
    "selve_aligned": {"score": 0-10, "reasoning": "..."},
    "factually_accurate": {"score": 0-10, "reasoning": "..."},
    "appropriate_tone": {"score": 0-10, "reasoning": "..."},
    "citation_needed": true/false,
    "recommendation": "approve|needs_revision|reject",
    "suggestions": ["suggestion 1", "suggestion 2", ...],
    "confidence": 0.0-1.0
}

Be thorough and objective."""

    async def validate_content(
        self,
        content: str,
        source: str,
        content_hash: str
    ) -> Dict[str, Any]:
        """
        Validate content against SELVE framework

        Args:
            content: Content text to validate
            source: Content source
            content_hash: Hash for tracking

        Returns:
            {
                "validation_id": str,
                "status": "approved|needs_revision|rejected",
                "scores": {...},
                "recommendation": str,
                "suggestions": [...],
                "confidence": float
            }
        """
        try:
            # Truncate content if too long (first 2000 chars for validation)
            validation_content = content[:2000] if len(content) > 2000 else content

            messages = [
                {"role": "system", "content": self.validation_prompt},
                {"role": "user", "content": f"Validate this content from '{source}':\n\n{validation_content}"}
            ]

            # Use LLM for validation
            result = self.llm_service.generate_response(
                messages=messages,
                temperature=0.3,  # Lower temp for consistent validation
                max_tokens=800
            )

            # Parse JSON response
            validation_data = json.loads(result["content"])

            # Determine status from recommendation
            recommendation = validation_data.get("recommendation", "needs_revision")
            status_map = {
                "approve": "approved",
                "needs_revision": "needs_revision",
                "reject": "rejected"
            }
            status = status_map.get(recommendation, "needs_revision")

            # Store validation in database
            validation_record = await db.contentvalidation.create(
                data={
                    "contentHash": content_hash,
                    "source": source,
                    "status": status,
                    "validationCriteria": {
                        "selve_aligned": validation_data.get("selve_aligned"),
                        "factually_accurate": validation_data.get("factually_accurate"),
                        "appropriate_tone": validation_data.get("appropriate_tone"),
                        "citation_needed": validation_data.get("citation_needed", False)
                    },
                    "validatorType": "llm",
                    "validatorId": self.llm_service.model,
                    "validationCost": result["cost"],
                    "recommendations": validation_data.get("suggestions", [])
                }
            )

            return {
                "validation_id": validation_record.id,
                "status": status,
                "scores": {
                    "selve_aligned": validation_data["selve_aligned"]["score"],
                    "factually_accurate": validation_data["factually_accurate"]["score"],
                    "appropriate_tone": validation_data["appropriate_tone"]["score"]
                },
                "recommendation": recommendation,
                "suggestions": validation_data.get("suggestions", []),
                "confidence": validation_data.get("confidence", 0.8),
                "reasoning": {
                    "selve_aligned": validation_data["selve_aligned"]["reasoning"],
                    "factually_accurate": validation_data["factually_accurate"]["reasoning"],
                    "appropriate_tone": validation_data["appropriate_tone"]["reasoning"]
                }
            }

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse validation JSON: {e}")
            return {
                "validation_id": None,
                "status": "error",
                "error": "Failed to parse validation response"
            }
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return {
                "validation_id": None,
                "status": "error",
                "error": str(e)
            }

    async def approve_validation(
        self,
        validation_id: str,
        admin_notes: Optional[str] = None
    ) -> bool:
        """
        Manually approve a validation (admin override)

        Args:
            validation_id: Validation record ID
            admin_notes: Optional admin notes

        Returns:
            True if successful
        """
        try:
            await db.contentvalidation.update(
                where={"id": validation_id},
                data={
                    "status": "approved",
                    "adminApproved": True,
                    "adminNotes": admin_notes,
                    "validatedAt": datetime.utcnow()
                }
            )
            return True

        except Exception as e:
            print(f"❌ Failed to approve validation: {e}")
            return False

    async def reject_validation(
        self,
        validation_id: str,
        admin_notes: Optional[str] = None
    ) -> bool:
        """
        Manually reject a validation (admin override)

        Args:
            validation_id: Validation record ID
            admin_notes: Optional admin notes

        Returns:
            True if successful
        """
        try:
            await db.contentvalidation.update(
                where={"id": validation_id},
                data={
                    "status": "rejected",
                    "adminApproved": False,
                    "adminNotes": admin_notes,
                    "validatedAt": datetime.utcnow()
                }
            )
            return True

        except Exception as e:
            print(f"❌ Failed to reject validation: {e}")
            return False

    async def get_pending_validations(
        self,
        limit: int = 10
    ) -> list[Dict[str, Any]]:
        """
        Get pending validations for admin review

        Args:
            limit: Maximum number to return

        Returns:
            List of pending validation records
        """
        try:
            validations = await db.contentvalidation.find_many(
                where={"status": "needs_revision"},
                order_by={"createdAt": "desc"},
                take=limit
            )

            return [
                {
                    "id": v.id,
                    "content_hash": v.contentHash,
                    "source": v.source,
                    "status": v.status,
                    "criteria": v.validationCriteria,
                    "recommendations": v.recommendations,
                    "created_at": v.createdAt.isoformat()
                }
                for v in validations
            ]

        except Exception as e:
            print(f"❌ Failed to get pending validations: {e}")
            return []

    async def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics

        Returns:
            {
                "total": int,
                "approved": int,
                "rejected": int,
                "pending": int,
                "by_source": {...}
            }
        """
        try:
            validations = await db.contentvalidation.find_many()

            total = len(validations)
            approved = sum(1 for v in validations if v.status == "approved")
            rejected = sum(1 for v in validations if v.status == "rejected")
            pending = sum(1 for v in validations if v.status == "needs_revision")

            # Group by source
            by_source = {}
            for v in validations:
                source = v.source
                if source not in by_source:
                    by_source[source] = {
                        "total": 0,
                        "approved": 0,
                        "rejected": 0,
                        "pending": 0
                    }
                by_source[source]["total"] += 1
                by_source[source][v.status] += 1

            return {
                "total": total,
                "approved": approved,
                "rejected": rejected,
                "pending": pending,
                "by_source": by_source
            }

        except Exception as e:
            print(f"❌ Failed to get validation stats: {e}")
            return {}
