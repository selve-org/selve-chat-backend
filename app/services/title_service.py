"""
Title generation helper for SELVE Chat
Handles provider-aware model selection and safe fallbacks.
"""
import asyncio
from typing import Optional

from .llm_service import LLMService

# Timeouts
TITLE_GENERATION_TIMEOUT_SECONDS = 10.0


class InputValidationError(ValueError):
    """Raised when input validation fails"""
    pass


class TitleService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _validate_message(self, message: str) -> str:
        if not message:
            raise InputValidationError("Message cannot be empty")
        if not isinstance(message, str):
            raise InputValidationError("Message must be a string")
        message = message.strip()
        if not message:
            raise InputValidationError("Message cannot be empty")
        return message

    async def generate_title(self, first_message: str, assistant_response: Optional[str] = None) -> str:
        try:
            first_message = self._validate_message(first_message)
        except InputValidationError:
            return "New Conversation"

        assistant_text = assistant_response or ""

        # Choose a title model that exists for the configured provider
        if getattr(self.llm_service, "provider", "openai") == "anthropic":
            title_model = self.llm_service.model  # assume configured Anthropic model
        else:
            title_model = "gpt-4o-mini"

        prompt = (
            "Generate a short, descriptive title (max 5 words) for a conversation "
            "that starts with this exchange.\n"
            "Summarize both the user's first question and the assistant's first reply.\n\n"
            f"User's first message:\n\"{first_message[:500]}\"\n\n"
            f"Assistant's first reply:\n\"{assistant_text[:500]}\"\n\n"
            "Return ONLY the title text, nothing else. Make it specific and meaningful."
        )

        try:
            response = await asyncio.wait_for(
                self.llm_service.generate_response_async(
                    messages=[{"role": "user", "content": prompt}],
                    model=title_model,
                    temperature=0.7,
                    max_tokens=20,
                ),
                timeout=TITLE_GENERATION_TIMEOUT_SECONDS,
            )

            title = response.get("content", "").strip().strip('"').strip("'")
            if not title:
                raise ValueError("Empty title generated")
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        except asyncio.TimeoutError:
            return self._fallback_title(first_message, assistant_text)
        except Exception:
            return self._fallback_title(first_message, assistant_text)

    def _fallback_title(self, first_message: str, assistant_text: str) -> str:
        fallback = first_message
        if assistant_text:
            fallback = f"{first_message} - {assistant_text}"
        return fallback[:50] + "..." if len(fallback) > 50 else fallback
