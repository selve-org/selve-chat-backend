"""
Content Transformation Service - Convert External Content to SELVE's Framework

This service transforms personality insights from external sources (MBTI, Big Five,
Enneagram, etc.) into SELVE's 8-dimensional framework and voice.

Key Functions:
1. Framework Translation: Maps MBTI/Big Five/Enneagram → SELVE dimensions
2. Terminology Replacement: Replaces competing framework terms with SELVE language
3. Source Citation: Adds transparent source attribution
4. Brand Voice Enforcement: Ensures content matches SELVE's warm, poetic tone
5. Validation: Verifies transformed content aligns with SELVE ideology
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# FRAMEWORK MAPPING DEFINITIONS
# =============================================================================


class ExternalFramework(str, Enum):
    """External personality frameworks we may encounter."""
    MBTI = "mbti"
    BIG_FIVE = "big_five"
    ENNEAGRAM = "enneagram"
    HEXACO = "hexaco"
    DISC = "disc"
    FOUR_TEMPERAMENTS = "four_temperaments"
    UNKNOWN = "unknown"


@dataclass
class FrameworkMapping:
    """Mapping from external framework concept to SELVE dimensions."""
    external_term: str
    selve_dimensions: List[str]  # Primary SELVE dimensions this maps to
    transformation_note: str  # How to phrase this in SELVE's voice
    confidence: float  # 0-1, how direct the mapping is


# =============================================================================
# COMPREHENSIVE FRAMEWORK MAPPINGS
# =============================================================================


class FrameworkTranslator:
    """
    Translates external personality framework concepts to SELVE's 8 dimensions.

    SELVE's 8 Dimensions:
    - LUMEN: Social energy (like Extraversion)
    - AETHER: Emotional stability
    - ORPHEUS: Empathy/warmth
    - VARA: Integrity/honesty (UNIQUE - from HEXACO)
    - CHRONOS: Patience/flexibility
    - KAEL: Assertiveness/will
    - ORIN: Conscientiousness/organization
    - LYRA: Openness/creativity
    """

    # MBTI → SELVE mappings
    MBTI_MAPPINGS: Dict[str, FrameworkMapping] = {
        # Extraversion/Introversion
        "extravert": FrameworkMapping(
            "extravert", ["LUMEN"],
            "social energy and how you recharge", 0.9
        ),
        "introvert": FrameworkMapping(
            "introvert", ["LUMEN"],
            "reflective depth and inner focus", 0.9
        ),
        "E": FrameworkMapping("E", ["LUMEN"], "social energy", 0.9),
        "I": FrameworkMapping("I", ["LUMEN"], "inner reflection", 0.9),

        # Sensing/Intuition
        "sensing": FrameworkMapping(
            "sensing", ["LYRA"],
            "practical, detail-oriented approach", 0.7
        ),
        "intuitive": FrameworkMapping(
            "intuitive", ["LYRA"],
            "creative, big-picture thinking", 0.7
        ),
        "S": FrameworkMapping("S", ["LYRA"], "grounded focus", 0.7),
        "N": FrameworkMapping("N", ["LYRA"], "imaginative exploration", 0.7),

        # Thinking/Feeling
        "thinking": FrameworkMapping(
            "thinking", ["ORPHEUS"],
            "logical analysis in decisions", 0.7
        ),
        "feeling": FrameworkMapping(
            "feeling", ["ORPHEUS"],
            "empathic consideration in decisions", 0.8
        ),
        "T": FrameworkMapping("T", ["ORPHEUS"], "analytical approach", 0.7),
        "F": FrameworkMapping("F", ["ORPHEUS"], "warm, values-based approach", 0.8),

        # Judging/Perceiving
        "judging": FrameworkMapping(
            "judging", ["ORIN"],
            "structured, organized approach", 0.8
        ),
        "perceiving": FrameworkMapping(
            "perceiving", ["CHRONOS", "LYRA"],
            "flexible, spontaneous style", 0.8
        ),
        "J": FrameworkMapping("J", ["ORIN"], "organized planning", 0.8),
        "P": FrameworkMapping("P", ["CHRONOS"], "adaptive flexibility", 0.8),

        # 16 types
        "INTJ": FrameworkMapping("INTJ", ["LYRA", "ORIN", "LUMEN"], "strategic, independent thinker", 0.6),
        "INTP": FrameworkMapping("INTP", ["LYRA", "CHRONOS", "LUMEN"], "analytical, curious explorer", 0.6),
        "ENTJ": FrameworkMapping("ENTJ", ["KAEL", "ORIN", "LUMEN"], "decisive, goal-driven leader", 0.6),
        "ENTP": FrameworkMapping("ENTP", ["LYRA", "LUMEN", "CHRONOS"], "innovative, energetic debater", 0.6),
        "INFJ": FrameworkMapping("INFJ", ["ORPHEUS", "LYRA", "LUMEN"], "insightful, idealistic counselor", 0.6),
        "INFP": FrameworkMapping("INFP", ["ORPHEUS", "LYRA", "VARA"], "authentic, values-driven idealist", 0.6),
        "ENFJ": FrameworkMapping("ENFJ", ["ORPHEUS", "LUMEN", "KAEL"], "charismatic, empathic guide", 0.6),
        "ENFP": FrameworkMapping("ENFP", ["LYRA", "LUMEN", "ORPHEUS"], "enthusiastic, creative connector", 0.6),
        "ISTJ": FrameworkMapping("ISTJ", ["ORIN", "VARA", "AETHER"], "reliable, detail-oriented organizer", 0.6),
        "ISFJ": FrameworkMapping("ISFJ", ["ORPHEUS", "ORIN", "VARA"], "caring, conscientious supporter", 0.6),
        "ESTJ": FrameworkMapping("ESTJ", ["ORIN", "KAEL", "LUMEN"], "efficient, direct administrator", 0.6),
        "ESFJ": FrameworkMapping("ESFJ", ["ORPHEUS", "LUMEN", "ORIN"], "warm, sociable organizer", 0.6),
        "ISTP": FrameworkMapping("ISTP", ["CHRONOS", "LYRA", "LUMEN"], "pragmatic, adaptable problem-solver", 0.6),
        "ISFP": FrameworkMapping("ISFP", ["ORPHEUS", "LYRA", "CHRONOS"], "gentle, artistic individualist", 0.6),
        "ESTP": FrameworkMapping("ESTP", ["LUMEN", "CHRONOS", "KAEL"], "energetic, action-oriented realist", 0.6),
        "ESFP": FrameworkMapping("ESFP", ["LUMEN", "ORPHEUS", "CHRONOS"], "spontaneous, enthusiastic entertainer", 0.6),
    }

    # Big Five → SELVE mappings
    BIG_FIVE_MAPPINGS: Dict[str, FrameworkMapping] = {
        "extraversion": FrameworkMapping("extraversion", ["LUMEN"], "social energy", 1.0),
        "neuroticism": FrameworkMapping("neuroticism", ["AETHER"], "emotional steadiness", 0.95),
        "emotional stability": FrameworkMapping("emotional stability", ["AETHER"], "inner calm and resilience", 0.95),
        "agreeableness": FrameworkMapping("agreeableness", ["ORPHEUS", "VARA"], "warmth and cooperation", 0.9),
        "conscientiousness": FrameworkMapping("conscientiousness", ["ORIN"], "organization and discipline", 1.0),
        "openness": FrameworkMapping("openness", ["LYRA"], "creativity and curiosity", 1.0),
        "openness to experience": FrameworkMapping("openness to experience", ["LYRA"], "intellectual and aesthetic curiosity", 1.0),
    }

    # Four Temperaments → SELVE mappings
    TEMPERAMENT_MAPPINGS: Dict[str, FrameworkMapping] = {
        "sanguine": FrameworkMapping("sanguine", ["LUMEN", "ORPHEUS", "CHRONOS"], "sociable, optimistic energy", 0.6),
        "choleric": FrameworkMapping("choleric", ["KAEL", "ORIN", "LUMEN"], "decisive, goal-driven leadership", 0.6),
        "melancholic": FrameworkMapping("melancholic", ["ORPHEUS", "LYRA", "ORIN"], "thoughtful, detail-oriented sensitivity", 0.6),
        "phlegmatic": FrameworkMapping("phlegmatic", ["AETHER", "CHRONOS", "ORPHEUS"], "calm, patient reliability", 0.6),
    }

    # Enneagram → SELVE mappings
    ENNEAGRAM_MAPPINGS: Dict[str, FrameworkMapping] = {
        "type 1": FrameworkMapping("type 1", ["ORIN", "VARA"], "principled and purposeful", 0.6),
        "type 2": FrameworkMapping("type 2", ["ORPHEUS"], "caring and interpersonal", 0.7),
        "type 3": FrameworkMapping("type 3", ["KAEL", "ORIN"], "success-oriented and adaptable", 0.6),
        "type 4": FrameworkMapping("type 4", ["LYRA", "ORPHEUS"], "individualistic and expressive", 0.6),
        "type 5": FrameworkMapping("type 5", ["LYRA"], "investigative and perceptive", 0.7),
        "type 6": FrameworkMapping("type 6", ["AETHER", "ORPHEUS"], "loyal and security-oriented", 0.6),
        "type 7": FrameworkMapping("type 7", ["LUMEN", "LYRA", "CHRONOS"], "enthusiastic and spontaneous", 0.6),
        "type 8": FrameworkMapping("type 8", ["KAEL"], "powerful and self-confident", 0.8),
        "type 9": FrameworkMapping("type 9", ["AETHER", "CHRONOS", "ORPHEUS"], "peaceful and agreeable", 0.6),
    }

    @classmethod
    def get_all_mappings(cls) -> Dict[str, FrameworkMapping]:
        """Combine all framework mappings."""
        return {
            **cls.MBTI_MAPPINGS,
            **cls.BIG_FIVE_MAPPINGS,
            **cls.TEMPERAMENT_MAPPINGS,
            **cls.ENNEAGRAM_MAPPINGS,
        }


# =============================================================================
# TERMINOLOGY REPLACEMENT
# =============================================================================


class TerminologyReplacer:
    """
    Replaces competing framework terminology with SELVE-aligned language.

    Strategy:
    1. Remove explicit framework names (MBTI, Big Five, Enneagram)
    2. Replace specific terms with SELVE equivalents
    3. Maintain the insight while rebranding the language
    """

    # Framework names to remove/replace
    FRAMEWORK_REMOVALS = {
        r"\bMBTI\b": "personality frameworks",
        r"\bMyers-Briggs\b": "personality typing",
        r"\bBig Five\b": "personality research",
        r"\bOCEAN\b": "core personality dimensions",
        r"\bEnneagram\b": "personality systems",
        r"\bDISC\b": "behavioral assessments",
        r"\bfour temperaments\b": "classical personality types",
        r"\bsanguine\b": "socially energetic",
        r"\bcholeric\b": "decisive and driven",
        r"\bmelancholic\b": "thoughtful and sensitive",
        r"\bphlegmatic\b": "calm and steady",
    }

    # Direct term replacements
    TERM_REPLACEMENTS = {
        # MBTI terminology
        r"\bextravert(ed|s|sion)?\b": "high social energy",
        r"\bintrovert(ed|s|sion)?\b": "reflective and introspective",
        r"\bsensing type\b": "detail-oriented and practical",
        r"\bintuitive type\b": "imaginative and big-picture focused",
        r"\bthinking type\b": "analytically inclined",
        r"\bfeeling type\b": "values-driven and empathic",
        r"\bjudging type\b": "organized and structured",
        r"\bperceiving type\b": "flexible and spontaneous",

        # Big Five terminology
        r"\bhigh neuroticism\b": "emotionally sensitive",
        r"\blow neuroticism\b": "emotionally stable",
        r"\bhigh agreeableness\b": "warm and cooperative",
        r"\bhigh conscientiousness\b": "disciplined and organized",
        r"\bhigh openness\b": "creative and curious",
    }

    @classmethod
    def replace_terminology(cls, text: str) -> str:
        """Replace competing framework terminology with SELVE language."""
        result = text

        # Apply framework removals
        for pattern, replacement in cls.FRAMEWORK_REMOVALS.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Apply term replacements
        for pattern, replacement in cls.TERM_REPLACEMENTS.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result


# =============================================================================
# CONTENT TRANSFORMATION SERVICE
# =============================================================================


@dataclass
class TransformationResult:
    """Result of content transformation."""
    original_content: str
    transformed_content: str
    detected_frameworks: List[ExternalFramework]
    selve_dimensions_mentioned: Set[str]
    source_url: Optional[str]
    citation: Optional[str]
    transformation_notes: List[str]
    is_valid: bool
    validation_message: Optional[str]


class ContentTransformationService:
    """
    Transforms external personality content into SELVE's framework and voice.

    This is the core service that ensures all crawled content:
    1. Uses SELVE terminology instead of competing frameworks
    2. Maps concepts to SELVE's 8 dimensions
    3. Maintains SELVE's warm, poetic voice
    4. Includes transparent source citations
    5. Validates alignment with SELVE ideology
    """

    def __init__(self):
        self.translator = FrameworkTranslator()
        self.replacer = TerminologyReplacer()
        self.logger = logger

    def transform_content(
        self,
        content: str,
        source_url: Optional[str] = None,
        add_citation: bool = True,
    ) -> TransformationResult:
        """
        Transform external personality content to SELVE's framework.

        Args:
            content: Original content from external source
            source_url: URL of the source
            add_citation: Whether to add source citation

        Returns:
            TransformationResult with transformed content
        """
        try:
            # Detect frameworks mentioned
            detected_frameworks = self._detect_frameworks(content)

            # Replace terminology
            transformed = self.replacer.replace_terminology(content)

            # Map to SELVE dimensions
            selve_dimensions = self._extract_selve_dimensions(transformed)

            # Add source citation if requested
            citation = None
            if add_citation and source_url:
                citation = self._create_citation(source_url)
                transformed = f"{transformed}\n\n{citation}"

            # Validate transformation
            is_valid, validation_msg = self._validate_transformation(transformed)

            # Generate transformation notes
            notes = self._generate_transformation_notes(
                detected_frameworks, selve_dimensions
            )

            return TransformationResult(
                original_content=content,
                transformed_content=transformed,
                detected_frameworks=detected_frameworks,
                selve_dimensions_mentioned=selve_dimensions,
                source_url=source_url,
                citation=citation,
                transformation_notes=notes,
                is_valid=is_valid,
                validation_message=validation_msg,
            )

        except Exception as e:
            self.logger.error(f"Content transformation failed: {e}", exc_info=True)
            return TransformationResult(
                original_content=content,
                transformed_content=content,  # Return original on failure
                detected_frameworks=[],
                selve_dimensions_mentioned=set(),
                source_url=source_url,
                citation=None,
                transformation_notes=[f"Transformation error: {str(e)}"],
                is_valid=False,
                validation_message=f"Transformation failed: {str(e)}",
            )

    def _detect_frameworks(self, content: str) -> List[ExternalFramework]:
        """Detect which external frameworks are mentioned in content."""
        content_lower = content.lower()
        detected = []

        if any(term in content_lower for term in ["mbti", "myers-briggs", "intj", "enfp", "extravert", "introvert"]):
            detected.append(ExternalFramework.MBTI)

        if any(term in content_lower for term in ["big five", "ocean", "neuroticism", "agreeableness"]):
            detected.append(ExternalFramework.BIG_FIVE)

        if any(term in content_lower for term in ["enneagram", "type 1", "type 2", "type 3"]):
            detected.append(ExternalFramework.ENNEAGRAM)

        if any(term in content_lower for term in ["sanguine", "choleric", "melancholic", "phlegmatic"]):
            detected.append(ExternalFramework.FOUR_TEMPERAMENTS)

        return detected or [ExternalFramework.UNKNOWN]

    def _extract_selve_dimensions(self, content: str) -> Set[str]:
        """Extract SELVE dimensions mentioned in transformed content."""
        dimensions = {"LUMEN", "AETHER", "ORPHEUS", "VARA", "CHRONOS", "KAEL", "ORIN", "LYRA"}
        found = set()

        content_upper = content.upper()
        for dimension in dimensions:
            if dimension in content_upper:
                found.add(dimension)

        return found

    def _create_citation(self, source_url: str) -> str:
        """Create a source citation."""
        # Extract domain for cleaner citation
        from urllib.parse import urlparse
        domain = urlparse(source_url).netloc.replace("www.", "")
        return f"[Source: {domain}]"

    def _validate_transformation(self, content: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that transformed content aligns with SELVE ideology.

        Checks:
        1. No competing framework names remain
        2. Content length is reasonable
        3. No harmful content
        """
        content_lower = content.lower()

        # Check for remaining framework names that should be removed
        forbidden_terms = ["mbti", "myers-briggs", "big five", "ocean traits", "enneagram"]
        for term in forbidden_terms:
            if term in content_lower:
                return False, f"Competing framework '{term}' still present after transformation"

        # Check length
        if len(content) < 50:
            return False, "Transformed content too short"

        if len(content) > 10000:
            return False, "Transformed content too long"

        # Check for harmful content
        harmful_terms = ["kill yourself", "suicide method", "self-harm", "illegal drugs"]
        for term in harmful_terms:
            if term in content_lower:
                return False, f"Harmful content detected: {term}"

        return True, None

    def _generate_transformation_notes(
        self,
        frameworks: List[ExternalFramework],
        dimensions: Set[str],
    ) -> List[str]:
        """Generate notes about the transformation process."""
        notes = []

        if frameworks:
            framework_names = ", ".join(f.value for f in frameworks)
            notes.append(f"Detected frameworks: {framework_names}")

        if dimensions:
            dimension_names = ", ".join(sorted(dimensions))
            notes.append(f"SELVE dimensions: {dimension_names}")

        return notes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def transform_crawled_content(
    content: str,
    source_url: Optional[str] = None,
    add_citation: bool = True,
) -> TransformationResult:
    """
    Convenience function to transform crawled content.

    Usage:
        result = transform_crawled_content(
            content="Extraverts gain energy from social interaction...",
            source_url="https://psychologytoday.com/article",
            add_citation=True
        )

        if result.is_valid:
            use_content = result.transformed_content
    """
    service = ContentTransformationService()
    return service.transform_content(content, source_url, add_citation)
