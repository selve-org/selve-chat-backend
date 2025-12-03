"""
Unified LLM Service - Supports OpenAI and Anthropic
Environment-based provider switching with tiered model routing
"""
import os
from typing import List, Dict, Any
from openai import OpenAI
from anthropic import Anthropic


class LLMService:
    """
    Unified service for LLM interactions with dual provider support

    Supports:
    - OpenAI (GPT-4o-mini, GPT-5-nano)
    - Anthropic (Claude Haiku 4.5, Sonnet 4.5, Opus 4.5)

    Environment variables:
    - LLM_PROVIDER: "openai" | "anthropic" (default: "anthropic")
    - ANTHROPIC_MODEL: Model to use (default: "claude-3-5-haiku-20241022")
    - OPENAI_MODEL: Model to use (default: "gpt-4o-mini")
    """

    MODEL_PRICING = {
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-opus-4-20250514": (15.00, 75.00),
        "gpt-4o-mini": (0.150, 0.600),
        "gpt-5-nano": (0.200, 0.800),
    }

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        
        if self.provider == "anthropic":
            self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        elif self.provider == "openai":
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {self.provider}")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        if self.provider == "anthropic":
            return self._generate_anthropic(messages, temperature, max_tokens)
        else:
            return self._generate_openai(messages, temperature, max_tokens)

    def _generate_anthropic(self, messages, temperature, max_tokens):
        system_msg = None
        conv = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                conv.append(msg)
        
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg or "",
            messages=conv
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_price, output_price = self.MODEL_PRICING.get(self.model, (0, 0))
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        
        return {
            "content": response.content[0].text,
            "model": self.model,
            "provider": "anthropic",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost": cost
        }

    def _generate_openai(self, messages, temperature, max_tokens):
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        input_price, output_price = self.MODEL_PRICING.get(self.model, (0, 0))
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        
        return {
            "content": response.choices[0].message.content,
            "model": self.model,
            "provider": "openai",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "cost": cost
        }
