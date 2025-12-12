"""
Response Validator - Validate and sanitize LLM responses.

Ensures responses:
1. Don't leak system prompt or internal instructions
2. Don't mention competing frameworks (Big Five, MBTI, etc.)
3. Maintain appropriate tone
4. Don't contain harmful content
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class ValidationConfig:
    """Response validation configuration."""
    
    # Whether to automatically fix issues or just flag them
    AUTO_FIX_ENABLED: bool = True
    
    # Maximum response length
    MAX_RESPONSE_LENGTH: int = 5000
    
    # Minimum response length (to catch empty/broken responses)
    MIN_RESPONSE_LENGTH: int = 10


# =============================================================================
# DATA TYPES
# =============================================================================


class ValidationIssue(str, Enum):
    """Types of validation issues."""
    PROMPT_LEAKAGE = "prompt_leakage"
    FRAMEWORK_MENTION = "framework_mention"
    INSTRUCTION_REVEAL = "instruction_reveal"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    ROLE_BREAK = "role_break"


@dataclass
class ValidationResult:
    """Result of response validation."""
    
    is_valid: bool
    original_response: str
    sanitized_response: str
    issues: List[ValidationIssue] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [i.value for i in self.issues],
            "fixes_applied": self.fixes_applied,
            "warnings": self.warnings,
        }


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================


class ResponsePatterns:
    """Patterns for detecting problematic content in responses."""
    
    # === Prompt Leakage Patterns ===
    PROMPT_LEAKAGE = [
        # Direct prompt reveals
        (r"my\s+(system\s+)?prompt\s+(is|says|tells|instructs)", "prompt_direct"),
        (r"i\s+(was|am)\s+instructed\s+to", "instruction_reveal"),
        (r"my\s+instructions?\s+(are|say|tell)", "instruction_reveal"),
        (r"according\s+to\s+my\s+(instructions?|programming|prompt)", "instruction_reveal"),
        
        # Guidelines/rules reveals  
        (r"my\s+guidelines?\s+(say|tell|state|require)", "guidelines_reveal"),
        (r"i'?m\s+(not\s+)?programmed\s+to", "programming_reveal"),
        (r"my\s+programming\s+(says|tells|requires)", "programming_reveal"),
        
        # System content leaks
        (r"```\s*(system|prompt|instruction)", "code_block_leak"),
        (r"<system>.*</system>", "xml_leak"),
        (r"\[SYSTEM\].*\[/SYSTEM\]", "bracket_leak"),
    ]
    
    # === Competing Framework Mentions ===
    FRAMEWORK_MENTIONS = [
        # Big Five / OCEAN
        (r"\b(big\s*five|big\s*5|ocean\s+model)\b", "big_five"),
        (r"\b(openness|conscientiousness|extraversion|agreeableness|neuroticism)\b", "ocean_traits"),
        
        # MBTI
        (r"\b(mbti|myers[- ]briggs)\b", "mbti"),
        (r"\b(intj|intp|entj|entp|infj|infp|enfj|enfp|istj|istp|estj|estp|isfj|isfp|esfj|esfp)\b", "mbti_types"),
        
        # Other frameworks
        (r"\b(enneagram|type\s+[1-9])\b", "enneagram"),
        (r"\b(disc\s+(assessment|profile|test))\b", "disc"),
        (r"\b(strengths\s*finder|clifton\s*strengths)\b", "strengthsfinder"),
        (r"\b(tim\s+lahaye|four\s+temperaments)\b", "four_temperaments"),
        (r"\b(holland\s+codes?|riasec)\b", "holland"),
    ]
    
    # === Role Breaking Patterns ===
    ROLE_BREAK = [
        (r"as\s+an?\s+(ai|language\s+model|llm|chatbot|assistant)\b", "ai_identity"),
        (r"i'?m\s+(just\s+)?(an?\s+)?(ai|language\s+model|llm)\b", "ai_identity"),
        (r"i\s+don'?t\s+have\s+(feelings?|emotions?|consciousness)\b", "ai_limitations"),
        (r"i'?m\s+not\s+(capable|able)\s+of\s+(feeling|emotion)", "ai_limitations"),
    ]
    
    # === Inappropriate Content ===
    INAPPROPRIATE = [
        (r"\b(kill|murder|suicide|self[- ]?harm)\b", "harmful_content"),
        (r"\b(illegal|drugs?|weapons?)\b", "dangerous_content"),
    ]


# =============================================================================
# REPLACEMENT TEMPLATES
# =============================================================================


class Replacements:
    """Replacement templates for problematic content."""
    
    # Framework mention replacements
    FRAMEWORK_REPLACEMENTS = {
        "big_five": "personality research",
        "ocean_traits": "personality dimension",
        "mbti": "personality typing systems",
        "mbti_types": "personality type",
        "enneagram": "personality frameworks",
        "disc": "behavioral assessments",
        "strengthsfinder": "strengths assessments",
        "four_temperaments": "temperament theories",
        "holland": "career interest models",
    }
    
    # Prompt leakage replacements
    PROMPT_LEAKAGE_REPLACEMENTS = {
        "prompt_direct": "I focus on helping you understand your personality",
        "instruction_reveal": "I'm here to help you",
        "guidelines_reveal": "Based on what I know",
        "programming_reveal": "I'm designed to help",
        "code_block_leak": "[content redacted]",
        "xml_leak": "[content redacted]",
        "bracket_leak": "[content redacted]",
    }


# =============================================================================
# RESPONSE VALIDATOR
# =============================================================================


class ResponseValidator:
    """
    Validates and sanitizes LLM responses.
    
    Ensures responses don't:
    - Leak system prompt or instructions
    - Mention competing personality frameworks
    - Break character/role
    - Contain inappropriate content
    """
    
    def __init__(self, auto_fix: bool = True):
        """
        Initialize response validator.
        
        Args:
            auto_fix: Whether to automatically fix issues
        """
        self.auto_fix = auto_fix
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, response: str) -> ValidationResult:
        """
        Validate and optionally sanitize a response.
        
        Args:
            response: LLM response to validate
            
        Returns:
            ValidationResult with issues and sanitized response
        """
        issues: List[ValidationIssue] = []
        fixes_applied: List[str] = []
        warnings: List[str] = []
        
        sanitized = response
        
        # === Check Length ===
        if len(response) > ValidationConfig.MAX_RESPONSE_LENGTH:
            issues.append(ValidationIssue.TOO_LONG)
            warnings.append(f"Response exceeds {ValidationConfig.MAX_RESPONSE_LENGTH} chars")
            if self.auto_fix:
                sanitized = response[:ValidationConfig.MAX_RESPONSE_LENGTH] + "..."
                fixes_applied.append("truncated_length")
        
        if len(response) < ValidationConfig.MIN_RESPONSE_LENGTH:
            issues.append(ValidationIssue.TOO_SHORT)
            warnings.append("Response is suspiciously short")
        
        # === Check Prompt Leakage ===
        leakage_result = self._check_prompt_leakage(sanitized)
        if leakage_result[0]:
            issues.append(ValidationIssue.PROMPT_LEAKAGE)
            warnings.extend(leakage_result[1])
            if self.auto_fix:
                sanitized = self._fix_prompt_leakage(sanitized)
                fixes_applied.append("removed_prompt_leakage")
        
        # === Check Framework Mentions ===
        framework_result = self._check_framework_mentions(sanitized)
        if framework_result[0]:
            issues.append(ValidationIssue.FRAMEWORK_MENTION)
            warnings.extend(framework_result[1])
            if self.auto_fix:
                sanitized = self._fix_framework_mentions(sanitized)
                fixes_applied.append("replaced_framework_mentions")
        
        # === Check Role Breaking ===
        role_result = self._check_role_break(sanitized)
        if role_result[0]:
            issues.append(ValidationIssue.ROLE_BREAK)
            warnings.extend(role_result[1])
            # Don't auto-fix role breaks - they might be contextually appropriate
        
        # === Check Inappropriate Content ===
        inappropriate_result = self._check_inappropriate(sanitized)
        if inappropriate_result[0]:
            issues.append(ValidationIssue.INAPPROPRIATE_CONTENT)
            warnings.extend(inappropriate_result[1])
            # Flag but don't auto-censor - context matters
        
        is_valid = len(issues) == 0 or (
            self.auto_fix and 
            all(i in [ValidationIssue.FRAMEWORK_MENTION, ValidationIssue.PROMPT_LEAKAGE] for i in issues)
        )
        
        return ValidationResult(
            is_valid=is_valid,
            original_response=response,
            sanitized_response=sanitized,
            issues=issues,
            fixes_applied=fixes_applied,
            warnings=warnings,
        )
    
    def _check_prompt_leakage(self, text: str) -> Tuple[bool, List[str]]:
        """Check for prompt leakage patterns."""
        found = []
        text_lower = text.lower()
        
        for pattern, name in ResponsePatterns.PROMPT_LEAKAGE:
            if re.search(pattern, text_lower, re.I):
                found.append(f"prompt_leakage:{name}")
        
        return (len(found) > 0, found)
    
    def _check_framework_mentions(self, text: str) -> Tuple[bool, List[str]]:
        """Check for competing framework mentions."""
        found = []
        text_lower = text.lower()
        
        for pattern, name in ResponsePatterns.FRAMEWORK_MENTIONS:
            if re.search(pattern, text_lower, re.I):
                found.append(f"framework:{name}")
        
        return (len(found) > 0, found)
    
    def _check_role_break(self, text: str) -> Tuple[bool, List[str]]:
        """Check for role-breaking patterns."""
        found = []
        text_lower = text.lower()
        
        for pattern, name in ResponsePatterns.ROLE_BREAK:
            if re.search(pattern, text_lower, re.I):
                found.append(f"role_break:{name}")
        
        return (len(found) > 0, found)
    
    def _check_inappropriate(self, text: str) -> Tuple[bool, List[str]]:
        """Check for inappropriate content."""
        found = []
        text_lower = text.lower()
        
        for pattern, name in ResponsePatterns.INAPPROPRIATE:
            if re.search(pattern, text_lower, re.I):
                found.append(f"inappropriate:{name}")
        
        return (len(found) > 0, found)
    
    def _fix_prompt_leakage(self, text: str) -> str:
        """Remove or replace prompt leakage content."""
        result = text
        
        for pattern, name in ResponsePatterns.PROMPT_LEAKAGE:
            replacement = Replacements.PROMPT_LEAKAGE_REPLACEMENTS.get(name, "")
            result = re.sub(pattern, replacement, result, flags=re.I)
        
        return result
    
    def _fix_framework_mentions(self, text: str) -> str:
        """Replace framework mentions with generic terms."""
        result = text
        
        for pattern, name in ResponsePatterns.FRAMEWORK_MENTIONS:
            replacement = Replacements.FRAMEWORK_REPLACEMENTS.get(name, "personality research")
            result = re.sub(pattern, replacement, result, flags=re.I)
        
        return result


# =============================================================================
# SPECIALIZED VALIDATORS
# =============================================================================


class SELVEResponseValidator(ResponseValidator):
    """
    SELVE-specific response validator.
    
    Adds additional checks specific to the SELVE chatbot context.
    """
    
    # SELVE dimension names for validation
    SELVE_DIMENSIONS = {
        "LUMEN", "AETHER", "ORPHEUS", "ORIN", 
        "LYRA", "VARA", "CHRONOS", "KAEL"
    }
    
    def validate(self, response: str, user_has_assessment: bool = False) -> ValidationResult:
        """
        Validate response with SELVE-specific checks.
        
        Args:
            response: Response to validate
            user_has_assessment: Whether user has completed assessment
            
        Returns:
            ValidationResult
        """
        # Run base validation
        result = super().validate(response)
        
        # === SELVE-specific checks ===
        
        # Check for incorrect dimension references
        dimension_issues = self._check_dimension_accuracy(response)
        if dimension_issues:
            result.warnings.extend(dimension_issues)
        
        # Check for appropriate assessment prompting
        if not user_has_assessment:
            assessment_check = self._check_assessment_prompting(response)
            result.warnings.extend(assessment_check)
        
        # Check for score hallucination (mentioning scores when user doesn't have them)
        if not user_has_assessment:
            hallucination_check = self._check_score_hallucination(response)
            if hallucination_check:
                result.issues.append(ValidationIssue.INAPPROPRIATE_CONTENT)
                result.warnings.extend(hallucination_check)
        
        return result
    
    def _check_dimension_accuracy(self, text: str) -> List[str]:
        """Check if dimension names are used correctly."""
        warnings = []
        
        # Check for misspelled dimension names
        potential_misspellings = {
            r"\bluman\b": "LUMEN",
            r"\blumen\b": "LUMEN (check capitalization)",
            r"\baether\b": "AETHER (check capitalization)",
            r"\borpheous\b": "ORPHEUS",
            r"\borin\b": "ORIN (check capitalization)",
            r"\blyra\b": "LYRA (check capitalization)",
            r"\bvara\b": "VARA (check capitalization)",
            r"\bchronos\b": "CHRONOS (check capitalization)",
            r"\bkale\b": "KAEL",
        }
        
        text_lower = text.lower()
        for pattern, correct in potential_misspellings.items():
            if re.search(pattern, text_lower):
                warnings.append(f"dimension_spelling: should be {correct}")
        
        return warnings
    
    def _check_assessment_prompting(self, text: str) -> List[str]:
        """Check if assessment is being prompted appropriately."""
        warnings = []
        
        # Count how many times assessment is mentioned
        assessment_mentions = len(re.findall(
            r"\b(take|complete|do)\s+(the\s+)?assessment\b",
            text.lower()
        ))
        
        if assessment_mentions > 2:
            warnings.append("assessment_overprompt: mentioned assessment too many times")
        
        return warnings
    
    def _check_score_hallucination(self, text: str) -> List[str]:
        """Check for hallucinated scores when user doesn't have assessment."""
        warnings = []
        
        # Check for specific score mentions
        score_patterns = [
            r"\byour\s+\w+\s+score\s+(is|of)\s+\d+",
            r"\byou\s+scored\s+\d+",
            r"\byour\s+\d+\s+(in|on|for)\s+\w+",
            r"based\s+on\s+your\s+scores?",
        ]
        
        for pattern in score_patterns:
            if re.search(pattern, text.lower()):
                warnings.append("score_hallucination: mentioned scores but user hasn't taken assessment")
                break
        
        return warnings


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_validator: Optional[SELVEResponseValidator] = None


def get_response_validator() -> SELVEResponseValidator:
    """Get the global response validator instance."""
    global _validator
    if _validator is None:
        _validator = SELVEResponseValidator()
    return _validator


def validate_response(
    response: str, 
    user_has_assessment: bool = False
) -> ValidationResult:
    """
    Convenience function to validate a response.
    
    Args:
        response: Response to validate
        user_has_assessment: Whether user has assessment
        
    Returns:
        ValidationResult
    """
    validator = get_response_validator()
    return validator.validate(response, user_has_assessment)


def sanitize_response(
    response: str,
    user_has_assessment: bool = False
) -> str:
    """
    Convenience function to sanitize a response.
    
    Args:
        response: Response to sanitize
        user_has_assessment: Whether user has assessment
        
    Returns:
        Sanitized response
    """
    result = validate_response(response, user_has_assessment)
    return result.sanitized_response
