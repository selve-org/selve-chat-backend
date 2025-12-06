"""Prompts module for SELVE Chatbot"""
from .system_prompt import (
    SYSTEM_PROMPT,
    OFF_TOPIC_PATTERNS,
    SENSITIVE_PATTERNS,
    OFF_TOPIC_RESPONSE,
    SENSITIVE_RESPONSE,
    classify_message,
    is_off_topic,
    is_sensitive,
    get_canned_response,
)

__all__ = [
    "SYSTEM_PROMPT",
    "OFF_TOPIC_PATTERNS",
    "SENSITIVE_PATTERNS",
    "OFF_TOPIC_RESPONSE",
    "SENSITIVE_RESPONSE",
    "classify_message",
    "is_off_topic",
    "is_sensitive",
    "get_canned_response",
]
