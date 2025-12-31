"""
Unified LLM Service - Supports OpenAI, Anthropic, and Google Gemini
Environment-based provider switching with tiered model routing
GPT-5 support with reasoning_effort and text_verbosity parameters
Gemini 3 support with function calling
Retry logic with exponential backoff
Proper Langfuse v3 tracing with clean inputs/outputs
"""
import os
import time
import asyncio
import logging
import json
from typing import List, Dict, Any, AsyncGenerator, Optional, Callable, TypeVar, Union
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from anthropic import Anthropic, APIError as AnthropicAPIError
from langfuse import observe, get_client

# Import Google Generative AI SDK
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)


# Type variable for generic retry function
T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(self):
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.initial_delay = float(os.getenv("INITIAL_RETRY_DELAY", "2.0"))
        self.max_delay = float(os.getenv("MAX_RETRY_DELAY", "30.0"))
        self.backoff_multiplier = 2.0
        self.enable_fallback = os.getenv("ENABLE_FALLBACK", "true").lower() == "true"
        self.timeout = float(os.getenv("API_TIMEOUT", "60"))


class LLMService:
    """
    Unified service for LLM interactions with multi-provider support

    Supports:
    - OpenAI (GPT-4o-mini, GPT-5-nano, GPT-5-mini, GPT-5, GPT-5.1, GPT-5.1-nano)
    - Anthropic (Claude Haiku 4.5, Sonnet 4.5, Opus 4.5)
    - Google Gemini (Gemini 3 Pro, Gemini 3 Flash)

    Environment variables:
    - LLM_PROVIDER: "openai" | "anthropic" | "gemini" (default: "openai")
    - OPENAI_MODEL: Model to use (default: "gpt-4o-mini")
    - ANTHROPIC_MODEL: Model to use (default: "claude-3-5-haiku-20241022")
    - GEMINI_MODEL: Model to use (default: "gemini-3-flash")
    - OPENAI_REASONING_EFFORT: "minimal" | "low" | "medium" | "high" (default: "high")
    - OPENAI_TEXT_VERBOSITY: "low" | "medium" | "high" (default: "medium")
    - ENABLE_DYNAMIC_SWITCHING: Enable dynamic model selection (default: "false")
    """

    MODEL_PRICING = {
        # Anthropic models (full API names)
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-opus-4-20250514": (15.00, 75.00),
        # Anthropic models (simplified aliases)
        "claude-haiku-4-5": (0.80, 4.00),
        "claude-sonnet-4-5": (3.00, 15.00),
        "claude-opus-4-5": (15.00, 75.00),
        # OpenAI GPT-4 models
        "gpt-4o-mini": (0.150, 0.600),
        # OpenAI GPT-5 models
        "gpt-5-nano": (0.200, 0.800),
        "gpt-5-mini": (0.400, 1.600),
        "gpt-5": (2.00, 8.00),
        "gpt-5.1-nano": (0.250, 1.000),
        "gpt-5.1": (2.50, 10.00),
        # Google Gemini models (pricing per 1M tokens)
        "gemini-3-pro": (2.00, 12.00),
        "gemini-3-flash": (0.50, 3.00),
        "gemini-2.0-flash-exp": (0.00, 0.00),  # Free tier during preview
    }

    OPENAI_PREFIXES = ("gpt-4", "gpt-5")
    ANTHROPIC_PREFIXES = ("claude",)
    GEMINI_PREFIXES = ("gemini",)

    # GPT-5 models that support reasoning_effort parameter
    GPT5_REASONING_MODELS = {
        "gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5.1-nano", "gpt-5.1"
    }

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.enable_dynamic_switching = os.getenv("ENABLE_DYNAMIC_SWITCHING", "false").lower() == "true"

        # Retry configuration
        self.retry_config = RetryConfig()

        # GPT-5 specific parameters
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "high")
        self.text_verbosity = os.getenv("OPENAI_TEXT_VERBOSITY", "medium")

        # Model tier configuration for dynamic switching
        self.tier_models = {
            1: os.getenv("TIER_1_MODEL", "gpt-5.1-nano"),   # Simple queries
            2: os.getenv("TIER_2_MODEL", "gpt-5-mini"),     # Standard (default)
            3: os.getenv("TIER_3_MODEL", "gpt-5.1"),        # Complex queries
        }

        # Initialize clients for all providers when keys exist so we can swap by model automatically
        openai_key = os.getenv("OPENAI_API_KEY")
        self.openai = OpenAI(api_key=openai_key) if openai_key else None

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic = Anthropic(api_key=anthropic_key) if anthropic_key else None

        # Initialize Gemini client
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=gemini_key)
            self.gemini = genai
        else:
            self.gemini = None
            if not GEMINI_AVAILABLE and self.provider == "gemini":
                logger.warning("Gemini provider selected but google-generativeai package not installed")

        # Default model comes from provider-specific env, but provider can be overridden per-call by model detection
        if self.provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        elif self.provider == "gemini":
            self.model = os.getenv("GEMINI_MODEL", "gemini-3-flash")
        else:
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _with_retry(self, func: Callable[[], T], operation: str = "LLM call") -> T:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: The function to execute
            operation: Description for logging
            
        Returns:
            The result of the function
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        delay = self.retry_config.initial_delay
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return func()
            except (RateLimitError, APIConnectionError) as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    print(f"⚠️ {operation} failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}")
                    print(f"   Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * self.retry_config.backoff_multiplier, self.retry_config.max_delay)
            except APIError as e:
                # Don't retry on 4xx errors (except rate limit)
                if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                    raise
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    print(f"⚠️ {operation} failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}")
                    print(f"   Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * self.retry_config.backoff_multiplier, self.retry_config.max_delay)
            except AnthropicAPIError as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    print(f"⚠️ {operation} failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}")
                    print(f"   Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * self.retry_config.backoff_multiplier, self.retry_config.max_delay)
        
        # All retries exhausted - try fallback if enabled
        if self.retry_config.enable_fallback and last_exception:
            print(f"❌ All retries exhausted for {operation}. Last error: {last_exception}")
        
        raise last_exception

    async def _with_retry_async(self, coro_func: Callable, operation: str = "LLM call"):
        """
        Execute an async function with exponential backoff retry logic.
        
        Args:
            coro_func: A callable that returns a coroutine
            operation: Description for logging
            
        Returns:
            The result of the coroutine
        """
        last_exception = None
        delay = self.retry_config.initial_delay
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return await coro_func()
            except (RateLimitError, APIConnectionError) as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    print(f"⚠️ {operation} failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}")
                    print(f"   Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * self.retry_config.backoff_multiplier, self.retry_config.max_delay)
            except APIError as e:
                if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                    raise
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    print(f"⚠️ {operation} failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}")
                    print(f"   Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * self.retry_config.backoff_multiplier, self.retry_config.max_delay)
            except AnthropicAPIError as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    print(f"⚠️ {operation} failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}")
                    print(f"   Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * self.retry_config.backoff_multiplier, self.retry_config.max_delay)
        
        if self.retry_config.enable_fallback and last_exception:
            print(f"❌ All retries exhausted for {operation}. Last error: {last_exception}")
        
        raise last_exception

    def _is_gpt5_model(self, model: str) -> bool:
        """Check if model supports GPT-5 reasoning parameters"""
        return model in self.GPT5_REASONING_MODELS

    def _resolve_provider_for_model(self, model: str) -> str:
        """Infer provider from model name; fallback to configured provider."""
        model_lower = (model or "").lower()
        if model_lower.startswith(self.OPENAI_PREFIXES):
            return "openai"
        if model_lower.startswith(self.ANTHROPIC_PREFIXES):
            return "anthropic"
        if model_lower.startswith(self.GEMINI_PREFIXES):
            return "gemini"
        return self.provider
    
    def _get_gpt5_extra_body(self) -> Dict[str, Any]:
        """Build extra_body params for GPT-5 models"""
        return {
            "reasoning_effort": self.reasoning_effort,
            "text": {
                "verbosity": self.text_verbosity
            }
        }

    def _extract_user_message(self, messages: List[Dict[str, str]]) -> str:
        """Extract the last user message for clean Langfuse input display."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def select_model_for_query(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Select appropriate model based on query complexity.
        Only used when ENABLE_DYNAMIC_SWITCHING is true.
        """
        if not self.enable_dynamic_switching:
            return self.model
        
        # Simple heuristics for model selection
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Simple patterns -> Tier 1 (nano model)
        simple_patterns = ["what is", "tell me about", "explain", "define", "who am i"]
        if word_count < 15 and any(p in query_lower for p in simple_patterns):
            return self.tier_models[1]
        
        # Complex patterns -> Tier 3 (full model)
        complex_patterns = [
            "compare", "analyze", "deep dive", "comprehensive",
            "how do i improve", "career advice", "relationship",
            "multiple", "steps", "plan", "strategy"
        ]
        if word_count > 30 or any(p in query_lower for p in complex_patterns):
            return self.tier_models[3]
        
        # Default -> Tier 2 (mini model)
        return self.tier_models[2]

    async def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call LLM with function calling tools across all providers.

        Args:
            messages: List of message dicts with role and content
            tools: List of tool definitions in OpenAI format
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override

        Returns:
            Dict containing:
                - message: The assistant message (may contain tool_calls)
                - tool_calls: List of tool calls if any (None if no calls)
                - content: Text content if any (None if only tool calls)
                - usage: Token usage statistics
                - cost: Generation cost
        """
        model = model or self.model
        provider = self._resolve_provider_for_model(model)

        if provider == "anthropic":
            return await self._call_anthropic_with_tools(messages, tools, temperature, max_tokens, model)
        elif provider == "gemini":
            return await self._call_gemini_with_tools(messages, tools, temperature, max_tokens, model)
        else:
            return await self._call_openai_with_tools(messages, tools, temperature, max_tokens, model)

    async def _call_openai_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        model: str
    ) -> Dict[str, Any]:
        """Call OpenAI with function calling"""
        if not self.openai:
            raise ValueError("OpenAI client not configured")

        request_params = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self._is_gpt5_model(model):
            request_params["extra_body"] = self._get_gpt5_extra_body()

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.openai.chat.completions.create(**request_params)
        )

        assistant_message = response.choices[0].message

        # Parse tool calls if present
        tool_calls = None
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            tool_calls = []
            for tc in assistant_message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments  # JSON string
                    }
                })

        # Calculate costs
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))
        input_cost = (input_tokens * input_price) / 1_000_000
        output_cost = (output_tokens * output_price) / 1_000_000
        total_cost = input_cost + output_cost

        return {
            "content": assistant_message.content,
            "tool_calls": tool_calls,
            "model": model,
            "provider": "openai",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost
        }

    async def _call_anthropic_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        model: str
    ) -> Dict[str, Any]:
        """Call Anthropic with tool use"""
        if not self.anthropic:
            raise ValueError("Anthropic client not configured")

        # Import converter
        from app.tools.function_definitions import convert_to_anthropic_format

        # Convert tools to Anthropic format
        anthropic_tools = convert_to_anthropic_format(tools)

        # Separate system message and convert tool messages to Anthropic format
        system_msg = None
        conv = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "system":
                system_msg = msg["content"]
                i += 1
            elif msg["role"] == "tool":
                # Tool results must be in user message with tool_result blocks
                # This happens after an assistant message with tool_use
                tool_result_content = []

                # Collect all consecutive tool messages
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_msg = messages[i]
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_msg.get("tool_call_id"),
                        "content": tool_msg["content"]
                    })
                    i += 1

                # Add as user message with tool_result blocks
                conv.append({
                    "role": "user",
                    "content": tool_result_content
                })
            elif msg["role"] == "assistant":
                # Convert assistant message with tool_calls to Anthropic format
                if msg.get("tool_calls"):
                    content_blocks = []

                    # Add text content if present
                    if msg.get("content"):
                        content_blocks.append({
                            "type": "text",
                            "text": msg["content"]
                        })

                    # Add tool_use blocks
                    for tc in msg["tool_calls"]:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"])
                        })

                    conv.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                else:
                    # Regular assistant message
                    conv.append(msg)
                i += 1
            else:
                # User or other messages
                conv.append(msg)
                i += 1

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg or "",
                messages=conv,
                tools=anthropic_tools
            )
        )

        # Parse tool calls from response
        tool_calls = None
        text_content = None

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })

        # Calculate costs
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))
        input_cost = (input_tokens * input_price) / 1_000_000
        output_cost = (output_tokens * output_price) / 1_000_000
        total_cost = input_cost + output_cost

        return {
            "content": text_content,
            "tool_calls": tool_calls,
            "model": model,
            "provider": "anthropic",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost
        }

    async def _call_gemini_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        model: str
    ) -> Dict[str, Any]:
        """Call Gemini with function calling"""
        if not self.gemini:
            raise ValueError("Gemini client not configured")

        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})

        # Convert tools to Gemini format
        from app.tools.function_definitions import convert_to_gemini_format
        gemini_tools = convert_to_gemini_format(tools)

        # Create generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        gemini_model = self.gemini.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction,
        )

        # Run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(gemini_messages, tools=gemini_tools)
        )

        # Parse function calls from response
        tool_calls = None
        text_content = None

        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                text_content = part.text
            elif hasattr(part, 'function_call'):
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args))
                    }
                })

        # Calculate costs
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))
        input_cost = (input_tokens * input_price) / 1_000_000
        output_cost = (output_tokens * output_price) / 1_000_000
        total_cost = input_cost + output_cost

        return {
            "content": text_content,
            "tool_calls": tool_calls,
            "model": model,
            "provider": "gemini",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost
        }

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response using the configured LLM provider"""
        model = model or self.model
        provider = self._resolve_provider_for_model(model)

        if provider == "anthropic":
            result = self._generate_anthropic(messages, temperature, max_tokens, model)
        elif provider == "gemini":
            result = self._generate_gemini(messages, temperature, max_tokens, model)
        else:
            result = self._generate_openai(messages, temperature, max_tokens, model)

        return result

    def _generate_anthropic(self, messages, temperature, max_tokens, model: str = None):
        """Generate response using Anthropic Claude with retry logic"""
        model = model or self.model
        if not self.anthropic:
            raise ValueError("Anthropic client not configured; set ANTHROPIC_API_KEY or use an OpenAI model")
        system_msg = None
        conv = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                conv.append(msg)
        
        def make_request():
            return self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg or "",
                messages=conv,
                timeout=self.retry_config.timeout  # ROB-3: Add timeout
            )
        
        response = self._with_retry(make_request, f"Anthropic ({model})")

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))

        # Calculate costs separately for better visibility
        input_cost = (input_tokens * input_price) / 1_000_000
        output_cost = (output_tokens * output_price) / 1_000_000
        total_cost = input_cost + output_cost

        return {
            "content": response.content[0].text,
            "model": model,
            "provider": "anthropic",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost": total_cost,  # Total cost for backward compatibility
            "input_cost": input_cost,  # Separate input cost
            "output_cost": output_cost  # Separate output cost
        }

    def _generate_openai(self, messages, temperature, max_tokens, model: str = None):
        """Generate response using OpenAI - supports GPT-5 reasoning parameters with retry logic"""
        model = model or self.model
        if not self.openai:
            raise ValueError("OpenAI client not configured; set OPENAI_API_KEY or use an Anthropic model")
        
        # Build request params
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add GPT-5 specific parameters if applicable
        if self._is_gpt5_model(model):
            request_params["extra_body"] = self._get_gpt5_extra_body()
        
        def make_request():
            # ROB-3: Add timeout to prevent hanging requests
            request_params["timeout"] = self.retry_config.timeout
            return self.openai.chat.completions.create(**request_params)
        
        response = self._with_retry(make_request, f"OpenAI ({model})")

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))

        # Calculate costs separately for better visibility
        input_cost = (input_tokens * input_price) / 1_000_000
        output_cost = (output_tokens * output_price) / 1_000_000
        total_cost = input_cost + output_cost

        return {
            "content": response.choices[0].message.content,
            "model": model,
            "provider": "openai",
            "reasoning_effort": self.reasoning_effort if self._is_gpt5_model(model) else None,
            "text_verbosity": self.text_verbosity if self._is_gpt5_model(model) else None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "cost": total_cost,  # Total cost for backward compatibility
            "input_cost": input_cost,  # Separate input cost
            "output_cost": output_cost  # Separate output cost
        }

    def _generate_gemini(self, messages, temperature, max_tokens, model: str = None):
        """Generate response using Google Gemini with retry logic"""
        model = model or self.model
        if not self.gemini:
            raise ValueError("Gemini client not configured; set GEMINI_API_KEY or use an OpenAI/Anthropic model")

        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})

        # Create generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Safety settings (permissive for psychology content)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        def make_request():
            gemini_model = self.gemini.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_instruction,
            )
            response = gemini_model.generate_content(gemini_messages)
            return response

        response = self._with_retry(make_request, f"Gemini ({model})")

        # Extract token usage
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))

        # Calculate costs separately
        input_cost = (input_tokens * input_price) / 1_000_000
        output_cost = (output_tokens * output_price) / 1_000_000
        total_cost = input_cost + output_cost

        return {
            "content": response.text,
            "model": model,
            "provider": "gemini",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost
        }

    async def generate_response_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async wrapper around generate_response for convenience."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            ),
        )

    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        model: Optional[str] = None
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Generate streaming response from LLM

        Yields individual text chunks as they arrive, followed by metadata dict
        """
        model = model or self.model
        provider = self._resolve_provider_for_model(model)

        if provider == "anthropic":
            async for chunk in self._generate_anthropic_stream(messages, temperature, max_tokens, model):
                yield chunk
        elif provider == "gemini":
            async for chunk in self._generate_gemini_stream(messages, temperature, max_tokens, model):
                yield chunk
        else:
            async for chunk in self._generate_openai_stream(messages, temperature, max_tokens, model):
                yield chunk

    async def _generate_anthropic_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Stream responses from Anthropic.
        Yields text chunks, then a final dict with usage metadata.
        """
        model = model or self.model
        system_msg = None
        conv = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                conv.append(msg)

        # Anthropic streaming
        with self.anthropic.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg or "",
            messages=conv
        ) as stream:
            for text in stream.text_stream:
                yield text
            
            # After stream completes, yield usage metadata as a dict
            final_message = stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens
            input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))

            # Calculate costs separately for better visibility
            input_cost = (input_tokens * input_price) / 1_000_000
            output_cost = (output_tokens * output_price) / 1_000_000
            total_cost = input_cost + output_cost

            metadata = {
                "__metadata__": True,
                "model": model,
                "provider": "anthropic",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "cost": total_cost,  # Total cost for backward compatibility
                "input_cost": input_cost,  # Separate input cost
                "output_cost": output_cost  # Separate output cost
            }
            
            logger.info(f"🔍 Anthropic stream metadata: {metadata}")
            # Yield metadata marker at the end
            yield metadata

    async def _generate_openai_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Stream responses from OpenAI - supports GPT-5 reasoning parameters.
        Yields text chunks, then a final dict with usage metadata.
        """
        model = model or self.model
        
        # Build request params
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}  # Request usage data in stream
        }
        
        # Add GPT-5 specific parameters if applicable
        if self._is_gpt5_model(model):
            request_params["extra_body"] = self._get_gpt5_extra_body()
        
        # ROB-3: Add timeout to prevent hanging streams
        request_params["timeout"] = self.retry_config.timeout
        stream = self.openai.chat.completions.create(**request_params)
        
        # Track usage from stream
        usage_data = None

        for chunk in stream:
            # Check for content
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            
            # Capture usage data (comes in final chunk when include_usage=True)
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                usage_data = chunk.usage
        
        # After stream completes, yield usage metadata
        if usage_data:
            input_tokens = usage_data.prompt_tokens
            output_tokens = usage_data.completion_tokens
            input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))

            # Calculate costs separately for better visibility
            input_cost = (input_tokens * input_price) / 1_000_000
            output_cost = (output_tokens * output_price) / 1_000_000
            total_cost = input_cost + output_cost

            yield {
                "__metadata__": True,
                "model": model,
                "provider": "openai",
                "reasoning_effort": self.reasoning_effort if self._is_gpt5_model(model) else None,
                "text_verbosity": self.text_verbosity if self._is_gpt5_model(model) else None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": usage_data.total_tokens
                },
                "cost": total_cost,  # Total cost for backward compatibility
                "input_cost": input_cost,  # Separate input cost
                "output_cost": output_cost  # Separate output cost
            }

    async def _generate_gemini_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Stream responses from Google Gemini.
        Yields text chunks, then a final dict with usage metadata.
        """
        model = model or self.model

        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})

        # Create generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        gemini_model = self.gemini.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction,
        )

        # Stream the response
        response = gemini_model.generate_content(gemini_messages, stream=True)

        # Track usage
        input_tokens = 0
        output_tokens = 0

        for chunk in response:
            if chunk.text:
                yield chunk.text

        # Get final usage metadata after stream completes
        # Note: Gemini streaming doesn't provide usage metadata per chunk
        # We need to make a separate call or estimate based on the response
        # For now, we'll try to get it from the response object if available
        try:
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
        except:
            # If usage metadata not available, set to 0
            logger.warning(f"Could not retrieve usage metadata for Gemini stream")

        input_price, output_price = self.MODEL_PRICING.get(model, (0, 0))
        input_cost = (input_tokens * input_price) / 1_000_000
        output_cost = (output_tokens * output_price) / 1_000_000
        total_cost = input_cost + output_cost

        yield {
            "__metadata__": True,
            "model": model,
            "provider": "gemini",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost
        }