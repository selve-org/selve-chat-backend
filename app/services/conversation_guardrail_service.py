"""
Conversation Guardrail Service - Keep chatbot on-brand and on-script.

This service continuously monitors the conversation to ensure SELVE:
1. Stays focused on personality topics (not programming, politics, etc.)
2. Never mentions competing frameworks (MBTI, Big Five, etc.)
3. Maintains warm, human voice (not corporate AI speak)
4. Stays within ethical boundaries
5. Doesn't hallucinate user scores or dimensions

It works in conjunction with ResponseValidator but adds real-time
conversation context awareness.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# GUARDRAIL TYPES
# =============================================================================


class ViolationType(str, Enum):
    """Types of guardrail violations."""
    OFF_TOPIC = "off_topic"
    FRAMEWORK_MENTION = "framework_mention"
    AI_IDENTITY_LEAK = "ai_identity_leak"
    SCORE_HALLUCINATION = "score_hallucination"
    HARMFUL_CONTENT = "harmful_content"
    CORPORATE_VOICE = "corporate_voice"
    OVERLY_TECHNICAL = "overly_technical"


class ViolationSeverity(str, Enum):
    """Severity levels for violations."""
    LOW = "low"  # Minor style issue, can let pass
    MEDIUM = "medium"  # Should fix before sending
    HIGH = "high"  # Must fix before sending
    CRITICAL = "critical"  # Block immediately


@dataclass
class GuardrailViolation:
    """A detected guardrail violation."""
    type: ViolationType
    severity: ViolationSeverity
    description: str
    matched_text: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    is_safe: bool
    violations: List[GuardrailViolation]
    should_block: bool  # True if response should be blocked
    should_warn: bool  # True if response should show warning
    corrected_response: Optional[str] = None


# =============================================================================
# CONVERSATION GUARDRAIL SERVICE
# =============================================================================


class ConversationGuardrailService:
    """
    Monitors conversations to keep SELVE on-brand and on-script.

    This service checks both individual responses and conversation flow
    to ensure SELVE maintains its identity and stays within bounds.
    """

    # Topics SELVE should focus on
    ON_BRAND_TOPICS: Set[str] = {
        "personality", "dimensions", "lumen", "aether", "orpheus", "vara",
        "chronos", "kael", "orin", "lyra", "self-discovery", "self-awareness",
        "emotional intelligence", "relationships", "communication", "growth",
        "strengths", "values", "empathy", "resilience", "leadership",
        "career", "work style", "team dynamics", "decision making",
    }

    # Topics SELVE should avoid
    OFF_BRAND_TOPICS: Set[str] = {
        "programming", "coding", "python", "javascript", "api", "database",
        "politics", "election", "government", "republican", "democrat",
        "religion", "god", "jesus", "islam", "buddhism",
        "medical", "diagnosis", "treatment", "medication", "prescription",
        "legal", "lawyer", "lawsuit", "court", "attorney",
    }

    # Competing frameworks to never mention
    COMPETING_FRAMEWORKS: Set[str] = {
        "mbti", "myers-briggs", "big five", "ocean", "enneagram",
        "disc", "strengthsfinder", "cliftonstrengths",
        "four temperaments", "sanguine", "choleric", "melancholic", "phlegmatic",
        "holland codes", "riasec",
    }

    # AI identity phrases that break SELVE's character
    AI_IDENTITY_LEAKS = [
        r"\b(i'?m|i\s+am)\s+(an?\s+)?AI\b",
        r"\bas\s+an?\s+AI\b",
        r"\blanguage\s+model\b",
        r"\bChatGPT\b",
        r"\bClaude\b(?!\s+Code)",  # Allow "Claude Code" but not standalone "Claude"
        r"\bGPT-?\d\b",
        r"\bartificial\s+intelligence\b",
        r"\bmachine\s+learning\b",
        r"\bneural\s+network\b",
        r"\btrained\s+on\b",
        r"\bmy\s+training\b",
    ]

    # Corporate/stiff language patterns
    CORPORATE_VOICE_PATTERNS = [
        r"\bleverage\b",
        r"\bsynergize\b",
        r"\bparadigm\b",
        r"\boptimize\s+for\b",
        r"\bvalue\s+proposition\b",
        r"\bstakeholder\b",
        r"\bstrategize\b",
        r"\bactionable\s+insights\b",
    ]

    # Score hallucination patterns
    SCORE_PATTERNS = [
        r"your\s+\w+\s+score\s+is\s+\d+",
        r"you\s+scored\s+\d+\s+on\s+\w+",
        r"your\s+\w+\s+dimension\s+is\s+\d+",
    ]

    def __init__(self):
        self.logger = logger

    def check_response(
        self,
        response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_has_assessment: bool = False,
    ) -> GuardrailResult:
        """
        Check a chatbot response for guardrail violations.

        Args:
            response: The chatbot's response to check
            conversation_history: Recent conversation for context
            user_has_assessment: Whether user has taken assessment

        Returns:
            GuardrailResult with any violations detected
        """
        violations = []

        # 1. Check for off-topic content
        off_topic_violations = self._check_off_topic(response)
        violations.extend(off_topic_violations)

        # 2. Check for framework mentions
        framework_violations = self._check_framework_mentions(response)
        violations.extend(framework_violations)

        # 3. Check for AI identity leaks
        identity_violations = self._check_ai_identity(response)
        violations.extend(identity_violations)

        # 4. Check for score hallucination
        if not user_has_assessment:
            score_violations = self._check_score_hallucination(response)
            violations.extend(score_violations)

        # 5. Check for corporate voice
        voice_violations = self._check_corporate_voice(response)
        violations.extend(voice_violations)

        # 6. Check for harmful content
        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        # Determine action based on violations
        should_block = any(v.severity == ViolationSeverity.CRITICAL for v in violations)
        should_warn = any(v.severity == ViolationSeverity.HIGH for v in violations)
        is_safe = not should_block

        # Try to auto-correct if possible
        corrected = self._attempt_correction(response, violations) if violations else None

        return GuardrailResult(
            is_safe=is_safe,
            violations=violations,
            should_block=should_block,
            should_warn=should_warn,
            corrected_response=corrected,
        )

    def _check_off_topic(self, response: str) -> List[GuardrailViolation]:
        """Check if response is off-topic for SELVE."""
        violations = []
        response_lower = response.lower()

        # Count mentions of off-brand topics
        off_brand_count = sum(1 for topic in self.OFF_BRAND_TOPICS if topic in response_lower)

        # Count mentions of on-brand topics
        on_brand_count = sum(1 for topic in self.ON_BRAND_TOPICS if topic in response_lower)

        # If significant off-brand content and no on-brand content
        if off_brand_count >= 2 and on_brand_count == 0:
            # Find which off-brand topic was mentioned
            mentioned = [topic for topic in self.OFF_BRAND_TOPICS if topic in response_lower]

            violations.append(GuardrailViolation(
                type=ViolationType.OFF_TOPIC,
                severity=ViolationSeverity.MEDIUM,
                description=f"Response is off-topic for SELVE (mentions: {', '.join(mentioned[:3])})",
                suggestion="Redirect to personality-related topics"
            ))

        return violations

    def _check_framework_mentions(self, response: str) -> List[GuardrailViolation]:
        """Check for mentions of competing frameworks."""
        violations = []
        response_lower = response.lower()

        for framework in self.COMPETING_FRAMEWORKS:
            if framework in response_lower:
                violations.append(GuardrailViolation(
                    type=ViolationType.FRAMEWORK_MENTION,
                    severity=ViolationSeverity.HIGH,
                    description=f"Mentioned competing framework: {framework}",
                    matched_text=framework,
                    suggestion=f"Remove '{framework}' and use SELVE dimensions instead"
                ))

        return violations

    def _check_ai_identity(self, response: str) -> List[GuardrailViolation]:
        """Check for AI identity leaks."""
        violations = []

        for pattern in self.AI_IDENTITY_LEAKS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                violations.append(GuardrailViolation(
                    type=ViolationType.AI_IDENTITY_LEAK,
                    severity=ViolationSeverity.CRITICAL,
                    description="Broke SELVE character by revealing AI identity",
                    matched_text=match.group(),
                    suggestion="Replace with SELVE's identity: 'I'm SELVE, your personality companion'"
                ))

        return violations

    def _check_score_hallucination(self, response: str) -> List[GuardrailViolation]:
        """Check for hallucinating scores when user hasn't taken assessment."""
        violations = []

        for pattern in self.SCORE_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                violations.append(GuardrailViolation(
                    type=ViolationType.SCORE_HALLUCINATION,
                    severity=ViolationSeverity.HIGH,
                    description="Mentioned scores for user who hasn't taken assessment",
                    matched_text=match.group(),
                    suggestion="Remove specific scores and invite user to take assessment"
                ))

        return violations

    def _check_corporate_voice(self, response: str) -> List[GuardrailViolation]:
        """Check for corporate/stiff language."""
        violations = []

        for pattern in self.CORPORATE_VOICE_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                violations.append(GuardrailViolation(
                    type=ViolationType.CORPORATE_VOICE,
                    severity=ViolationSeverity.LOW,
                    description="Using corporate language instead of warm, human voice",
                    matched_text=match.group(),
                    suggestion="Use simpler, more conversational language"
                ))

        return violations

    def _check_harmful_content(self, response: str) -> List[GuardrailViolation]:
        """Check for harmful content."""
        violations = []
        response_lower = response.lower()

        harmful_patterns = {
            "self-harm": ["kill yourself", "self-harm", "cut yourself"],
            "dangerous_advice": ["overdose", "suicide method"],
            "illegal": ["how to make", "illegal drug"],
        }

        for category, patterns in harmful_patterns.items():
            for pattern in patterns:
                if pattern in response_lower:
                    violations.append(GuardrailViolation(
                        type=ViolationType.HARMFUL_CONTENT,
                        severity=ViolationSeverity.CRITICAL,
                        description=f"Contains harmful content ({category})",
                        matched_text=pattern,
                        suggestion="Redirect to professional resources"
                    ))

        return violations

    def _attempt_correction(
        self,
        response: str,
        violations: List[GuardrailViolation],
    ) -> Optional[str]:
        """
        Attempt to auto-correct violations.

        Only corrects simple cases like framework mentions and corporate voice.
        Complex violations require human review.
        """
        corrected = response

        # Replace framework mentions with generic terms
        for violation in violations:
            if violation.type == ViolationType.FRAMEWORK_MENTION and violation.matched_text:
                corrected = corrected.replace(violation.matched_text, "personality frameworks")

        # Replace corporate voice with simpler language
        corporate_replacements = {
            "leverage": "use",
            "synergize": "work together",
            "paradigm": "approach",
            "optimize for": "improve",
            "stakeholder": "person involved",
        }

        for corporate, simple in corporate_replacements.items():
            corrected = re.sub(
                r"\b" + re.escape(corporate) + r"\b",
                simple,
                corrected,
                flags=re.IGNORECASE
            )

        return corrected if corrected != response else None

    def monitor_conversation_drift(
        self,
        conversation_history: List[Dict[str, str]],
        window_size: int = 10,
    ) -> Tuple[bool, Optional[str]]:
        """
        Monitor if conversation has drifted off-topic over time.

        Args:
            conversation_history: Full conversation
            window_size: Number of recent messages to analyze

        Returns:
            (is_drifting, reason) tuple
        """
        if not conversation_history or len(conversation_history) < 3:
            return False, None

        # Analyze recent messages
        recent_messages = conversation_history[-window_size:]

        # Count off-brand mentions in recent window
        off_brand_mentions = 0
        on_brand_mentions = 0

        for msg in recent_messages:
            content = msg.get("content", "").lower()

            off_brand_mentions += sum(1 for topic in self.OFF_BRAND_TOPICS if topic in content)
            on_brand_mentions += sum(1 for topic in self.ON_BRAND_TOPICS if topic in content)

        # Calculate drift ratio
        total_mentions = off_brand_mentions + on_brand_mentions
        if total_mentions == 0:
            return False, None

        off_brand_ratio = off_brand_mentions / total_mentions

        # If >60% of recent mentions are off-brand, conversation has drifted
        if off_brand_ratio > 0.6:
            return True, f"Conversation has drifted off-topic ({off_brand_ratio:.0%} off-brand mentions)"

        return False, None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def check_response_guardrails(
    response: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_has_assessment: bool = False,
) -> GuardrailResult:
    """
    Convenience function to check response guardrails.

    Usage:
        result = check_response_guardrails(
            response="You're an INTJ personality type...",
            user_has_assessment=False
        )

        if result.should_block:
            # Don't send this response
        elif result.corrected_response:
            # Use corrected version
            response = result.corrected_response
    """
    service = ConversationGuardrailService()
    return service.check_response(response, conversation_history, user_has_assessment)
