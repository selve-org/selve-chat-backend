"""
Content Validation Service - Validate external content against SELVE framework.

Validates content for:
- SELVE alignment (uses framework vocabulary correctly)
- Factual accuracy (no false claims about dimensions/assessments)
- Appropriate tone (warm, non-judgmental, growth-oriented)
- Citation requirements (external claims must be cited)

Security & Robustness:
- Content truncation for LLM calls
- Safe JSON parsing with validation
- Rate limiting ready
- Admin override controls
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import (
    BaseService,
    Result,
    Validator,
    safe_json_parse,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


class ValidationStatus(Enum):
    """Content validation statuses."""

    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class ValidationScore:
    """Score for a single validation criterion."""

    score: int  # 0-10
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "reasoning": self.reasoning}


@dataclass
class ValidationResult:
    """Complete validation result."""

    status: ValidationStatus
    selve_aligned: ValidationScore
    factually_accurate: ValidationScore
    appropriate_tone: ValidationScore
    citation_needed: bool
    suggestions: List[str]
    confidence: float
    validation_id: Optional[str] = None
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "validation_id": self.validation_id,
            "scores": {
                "selve_aligned": self.selve_aligned.score,
                "factually_accurate": self.factually_accurate.score,
                "appropriate_tone": self.appropriate_tone.score,
            },
            "citation_needed": self.citation_needed,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "reasoning": {
                "selve_aligned": self.selve_aligned.reasoning,
                "factually_accurate": self.factually_accurate.reasoning,
                "appropriate_tone": self.appropriate_tone.reasoning,
            },
        }


# =============================================================================
# CONTENT VALIDATION SERVICE
# =============================================================================


class ContentValidationService(BaseService):
    """
    Service for validating content against SELVE framework principles.

    Validation Criteria:
    1. SELVE Aligned: Uses framework vocabulary correctly
    2. Factually Accurate: No false claims about SELVE
    3. Appropriate Tone: Warm, supportive, non-judgmental
    4. Citation Needed: External claims require sources

    Recommendations:
    - approve: All scores 8+
    - needs_revision: Any score 6-7
    - reject: Any score <6
    """

    # Limits
    MAX_CONTENT_FOR_VALIDATION = 2000
    APPROVAL_THRESHOLD = 8
    REVISION_THRESHOLD = 6

    # LLM settings
    VALIDATION_TEMPERATURE = 0.3
    VALIDATION_MAX_TOKENS = 800

    def __init__(self, llm_service=None, db=None):
        """
        Initialize content validation service.

        Args:
            llm_service: LLM service for validation
            db: Database client
        """
        super().__init__()
        self._llm_service = llm_service
        self._db = db
        self._validation_prompt = self._build_validation_prompt()

    @property
    def llm_service(self):
        """Lazy-loaded LLM service."""
        if self._llm_service is None:
            from .llm_service import LLMService

            self._llm_service = LLMService()
        return self._llm_service

    @property
    def db(self):
        """Lazy-loaded database client."""
        if self._db is None:
            from app.db import db

            self._db = db
        return self._db

    def _build_validation_prompt(self) -> str:
        """Build the content validation prompt."""
        return """You are a content validator for the SELVE personality framework.

**SELVE Framework Core Principles:**
1. **8 Dimensions**: LUMEN (Social Energy), AETHER (Emotional Stability), ORPHEUS (Empathy), VARA (Honesty), CHRONOS (Patience), KAEL (Assertiveness), ORIN (Conscientiousness), LYRA (Openness)
2. **Non-judgmental**: No scores are "good" or "bad" - all personalities have value
3. **Growth-oriented**: Focus on self-understanding and development, not labeling
4. **Evidence-based**: Claims about personality must be grounded in research or framework
5. **Warm tone**: Supportive, empathetic, never harsh or critical

**Validation Criteria (score 0-10):**

1. **SELVE Aligned**:
   - Uses SELVE vocabulary correctly
   - Doesn't contradict framework principles
   - Aligns with dimension definitions

2. **Factually Accurate**:
   - No false claims about SELVE assessments
   - External claims are reasonable and verifiable
   - No pseudoscience or unsupported generalizations

3. **Appropriate Tone**:
   - Warm, supportive, non-judgmental
   - Growth-oriented, not labeling
   - No harsh criticism or negative framing

4. **Citation Needed**: Does content make external claims requiring citations?

**Recommendations:**
- approve: All scores 8+ (meets standards)
- needs_revision: Any score 6-7 (minor issues)
- reject: Any score <6 (major issues)

Format your response as JSON ONLY:
{
    "selve_aligned": {"score": 0-10, "reasoning": "..."},
    "factually_accurate": {"score": 0-10, "reasoning": "..."},
    "appropriate_tone": {"score": 0-10, "reasoning": "..."},
    "citation_needed": true/false,
    "recommendation": "approve|needs_revision|reject",
    "suggestions": ["suggestion 1", "suggestion 2"],
    "confidence": 0.0-1.0
}"""

    def _validate_llm_response(
        self,
        data: Dict[str, Any],
    ) -> Optional[ValidationResult]:
        """
        Validate and sanitize LLM response.

        Args:
            data: Parsed JSON response

        Returns:
            ValidationResult or None if invalid
        """
        if not isinstance(data, dict):
            return None

        def extract_score(value: Any, default_score: int = 5) -> ValidationScore:
            if isinstance(value, dict):
                score = value.get("score", default_score)
                reasoning = value.get("reasoning", "")
            else:
                score = default_score
                reasoning = ""

            # Clamp score to valid range
            try:
                score = max(0, min(10, int(score)))
            except (TypeError, ValueError):
                score = default_score

            return ValidationScore(
                score=score,
                reasoning=str(reasoning)[:500],  # Limit reasoning length
            )

        selve_aligned = extract_score(data.get("selve_aligned"))
        factually_accurate = extract_score(data.get("factually_accurate"))
        appropriate_tone = extract_score(data.get("appropriate_tone"))

        # Determine status from scores
        min_score = min(
            selve_aligned.score,
            factually_accurate.score,
            appropriate_tone.score,
        )

        if min_score >= self.APPROVAL_THRESHOLD:
            status = ValidationStatus.APPROVED
        elif min_score >= self.REVISION_THRESHOLD:
            status = ValidationStatus.NEEDS_REVISION
        else:
            status = ValidationStatus.REJECTED

        # Extract other fields
        citation_needed = bool(data.get("citation_needed", False))

        suggestions = data.get("suggestions", [])
        if isinstance(suggestions, list):
            suggestions = [str(s)[:200] for s in suggestions[:5]]
        else:
            suggestions = []

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
        except (TypeError, ValueError):
            confidence = 0.7

        return ValidationResult(
            status=status,
            selve_aligned=selve_aligned,
            factually_accurate=factually_accurate,
            appropriate_tone=appropriate_tone,
            citation_needed=citation_needed,
            suggestions=suggestions,
            confidence=confidence,
        )

    async def validate_content(
        self,
        content: str,
        source: str,
        content_hash: str,
    ) -> Dict[str, Any]:
        """
        Validate content against SELVE framework.

        Args:
            content: Content text to validate
            source: Content source identifier
            content_hash: Hash for tracking

        Returns:
            Validation result dictionary
        """
        # Validate inputs
        try:
            content = Validator.validate_string(
                content,
                "content",
                min_length=10,
                max_length=100_000,
            )
            source = Validator.validate_string(
                source,
                "source",
                min_length=1,
                max_length=100,
            )
            content_hash = Validator.validate_string(
                content_hash,
                "content_hash",
                min_length=64,
                max_length=64,
            )
        except Exception as e:
            return {
                "validation_id": None,
                "status": "error",
                "error": str(e),
            }

        # Truncate content for validation
        validation_content = content[: self.MAX_CONTENT_FOR_VALIDATION]

        try:
            messages = [
                {"role": "system", "content": self._validation_prompt},
                {
                    "role": "user",
                    "content": f"Validate this content from '{source}':\n\n{validation_content}",
                },
            ]

            # Call LLM
            result = self.llm_service.generate_response(
                messages=messages,
                temperature=self.VALIDATION_TEMPERATURE,
                max_tokens=self.VALIDATION_MAX_TOKENS,
            )

            # Parse response
            response_content = result.get("content", "")
            data = safe_json_parse(response_content)

            if data is None:
                self.logger.warning(
                    f"Failed to parse validation response: {response_content[:200]}"
                )
                return {
                    "validation_id": None,
                    "status": "error",
                    "error": "Failed to parse validation response",
                }

            # Validate and create result
            validation_result = self._validate_llm_response(data)
            if validation_result is None:
                return {
                    "validation_id": None,
                    "status": "error",
                    "error": "Invalid validation response structure",
                }

            validation_result.cost = result.get("cost", 0)

            # Store in database
            try:
                validation_record = await self.db.contentvalidation.create(
                    data={
                        "contentHash": content_hash,
                        "source": source,
                        "status": validation_result.status.value,
                        "validationCriteria": {
                            "selve_aligned": validation_result.selve_aligned.to_dict(),
                            "factually_accurate": validation_result.factually_accurate.to_dict(),
                            "appropriate_tone": validation_result.appropriate_tone.to_dict(),
                            "citation_needed": validation_result.citation_needed,
                        },
                        "validatorType": "llm",
                        "validatorId": self.llm_service.model,
                        "validationCost": validation_result.cost,
                        "recommendations": validation_result.suggestions,
                    }
                )
                validation_result.validation_id = validation_record.id

            except Exception as e:
                self.logger.warning(f"Failed to store validation: {e}")
                # Continue without storing - validation still valid

            return validation_result.to_dict()

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                "validation_id": None,
                "status": "error",
                "error": str(e),
            }

    async def approve_validation(
        self,
        validation_id: str,
        admin_user_id: str,
        admin_notes: Optional[str] = None,
    ) -> Result[bool]:
        """
        Manually approve a validation (admin override).

        Args:
            validation_id: Validation record ID
            admin_user_id: Admin user performing the action
            admin_notes: Optional admin notes

        Returns:
            Result indicating success
        """
        try:
            validation_id = Validator.validate_string(
                validation_id, "validation_id", min_length=5
            )
            admin_user_id = Validator.validate_user_id(admin_user_id, "admin_user_id")
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            await self.db.contentvalidation.update(
                where={"id": validation_id},
                data={
                    "status": ValidationStatus.APPROVED.value,
                    "adminApproved": True,
                    "adminNotes": admin_notes[:1000] if admin_notes else None,
                    "validatedAt": datetime.utcnow(),
                },
            )

            self.logger.info(
                f"Validation {validation_id} approved by admin {admin_user_id[:8]}..."
            )
            return Result.success(True)

        except Exception as e:
            self.logger.error(f"Failed to approve validation: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    async def reject_validation(
        self,
        validation_id: str,
        admin_user_id: str,
        admin_notes: Optional[str] = None,
    ) -> Result[bool]:
        """
        Manually reject a validation (admin override).

        Args:
            validation_id: Validation record ID
            admin_user_id: Admin user performing the action
            admin_notes: Optional admin notes

        Returns:
            Result indicating success
        """
        try:
            validation_id = Validator.validate_string(
                validation_id, "validation_id", min_length=5
            )
            admin_user_id = Validator.validate_user_id(admin_user_id, "admin_user_id")
        except Exception as e:
            return Result.validation_error(str(e))

        try:
            await self.db.contentvalidation.update(
                where={"id": validation_id},
                data={
                    "status": ValidationStatus.REJECTED.value,
                    "adminApproved": False,
                    "adminNotes": admin_notes[:1000] if admin_notes else None,
                    "validatedAt": datetime.utcnow(),
                },
            )

            self.logger.info(
                f"Validation {validation_id} rejected by admin {admin_user_id[:8]}..."
            )
            return Result.success(True)

        except Exception as e:
            self.logger.error(f"Failed to reject validation: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    async def get_pending_validations(
        self,
        limit: int = 10,
    ) -> Result[List[Dict[str, Any]]]:
        """
        Get pending validations for admin review.

        Args:
            limit: Maximum number to return

        Returns:
            Result containing list of pending validations
        """
        limit = min(limit, 100)  # Cap at 100

        try:
            validations = await self.db.contentvalidation.find_many(
                where={"status": ValidationStatus.NEEDS_REVISION.value},
                order={"createdAt": "desc"},
                take=limit,
            )

            return Result.success([
                {
                    "id": v.id,
                    "content_hash": v.contentHash,
                    "source": v.source,
                    "status": v.status,
                    "criteria": v.validationCriteria,
                    "recommendations": v.recommendations,
                    "created_at": v.createdAt.isoformat(),
                }
                for v in validations
            ])

        except Exception as e:
            self.logger.error(f"Failed to get pending validations: {e}")
            return Result.failure(str(e), error_code="DATABASE_ERROR")

    async def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Statistics dictionary
        """
        try:
            validations = await self.db.contentvalidation.find_many()

            total = len(validations)
            by_status = {}

            for v in validations:
                status = v.status
                by_status[status] = by_status.get(status, 0) + 1

            # Group by source
            by_source = {}
            for v in validations:
                source = v.source
                if source not in by_source:
                    by_source[source] = {"total": 0}
                by_source[source]["total"] += 1
                by_source[source][v.status] = by_source[source].get(v.status, 0) + 1

            return {
                "total": total,
                "approved": by_status.get("approved", 0),
                "rejected": by_status.get("rejected", 0),
                "pending": by_status.get("needs_revision", 0),
                "by_source": by_source,
            }

        except Exception as e:
            self.logger.error(f"Failed to get validation stats: {e}")
            return {"error": str(e)}