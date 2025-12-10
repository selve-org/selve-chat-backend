"""
Chat Service for SELVE Chatbot
Integrates RAG with Dual LLM Support (OpenAI + Anthropic)
Includes personality-focused guardrails and off-topic detection
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Set
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum

from langfuse import observe, get_client, propagate_attributes

from .llm_service import LLMService
from .rag_service import RAGService
from .user_profile_service import UserProfileService
from .compression_service import CompressionService
from .conversation_state_service import ConversationStateService
from .semantic_memory_service import SemanticMemoryService
from .title_service import TitleService
from .context_service import ContextService
from ..prompts import SYSTEM_PROMPT, get_canned_response, classify_message


logger = logging.getLogger(__name__)


# Configuration constants
MAX_MESSAGE_LENGTH = 10_000
MAX_HISTORY_MESSAGES = 50
MAX_HISTORY_TOTAL_CHARS = 100_000
LLM_TIMEOUT_SECONDS = 30.0


class StatusType(str, Enum):
    """Status event types for thinking UI"""
    RETRIEVING_CONTEXT = "retrieving_context"
    PERSONALIZING = "personalizing"
    GENERATING = "generating"
    CITING_SOURCES = "citing_sources"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StatusEvent:
    """Status events emitted during response generation for thinking UI"""
    status: StatusType
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "type": "status",
            "status": self.status.value,
            "message": self.message,
            "details": self.details
        }
    
    @classmethod
    def retrieving_context(cls, phase: int = 1, total_phases: int = 4) -> "StatusEvent":
        return cls(
            status=StatusType.RETRIEVING_CONTEXT,
            message="Searching knowledge base...",
            details={"phase": phase, "total_phases": total_phases}
        )
    
    @classmethod
    def personalizing(cls, phase: int = 2, total_phases: int = 4) -> "StatusEvent":
        return cls(
            status=StatusType.PERSONALIZING,
            message="Loading your personality profile...",
            details={"phase": phase, "total_phases": total_phases}
        )
    
    @classmethod
    def generating(cls, model: str, phase: int = 3, total_phases: int = 4) -> "StatusEvent":
        return cls(
            status=StatusType.GENERATING,
            message="Crafting your response...",
            details={"phase": phase, "total_phases": total_phases, "model": model}
        )
    
    @classmethod
    def citing_sources(cls, sources: List[Dict], phase: int = 4, total_phases: int = 4) -> "StatusEvent":
        return cls(
            status=StatusType.CITING_SOURCES,
            message="Adding sources...",
            details={"phase": phase, "total_phases": total_phases, "sources": sources}
        )
    
    @classmethod
    def complete(cls, sources: List[Dict], guardrail: Optional[str] = None) -> "StatusEvent":
        details = {"sources": sources}
        if guardrail:
            details["guardrail"] = guardrail
        return cls(
            status=StatusType.COMPLETE,
            message="Response complete",
            details=details
        )
    
    @classmethod
    def error(cls, error_message: str) -> "StatusEvent":
        return cls(
            status=StatusType.ERROR,
            message=error_message,
            details={"error": error_message}
        )


@dataclass
class ChatResponse:
    """Structured response from chat generation"""
    response: str
    context_used: bool
    retrieved_chunks: List[Dict[str, Any]]
    model: str
    provider: str
    usage: Dict[str, int]
    cost: float
    compression_needed: bool


class InputValidationError(ValueError):
    """Raised when input validation fails"""
    pass


class ChatService:
    """
    Service for handling chat interactions with RAG and dual LLM support.
    
    Features:
    - RAG-enhanced responses with knowledge base retrieval
    - Personality-based personalization using SELVE scores
    - Episodic and semantic memory integration
    - Conversation state tracking
    - Streaming response support
    - Background task management with graceful shutdown
    """

    def __init__(self):
        """Initialize all services and background task tracking"""
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self.user_profile_service = UserProfileService()
        self.compression_service = CompressionService()
        self.conversation_state_service = ConversationStateService()
        self.semantic_memory_service = SemanticMemoryService()
        self.assessment_url = self._resolve_assessment_url()
        self.system_prompt = self._load_system_prompt()

        self.title_service = TitleService(self.llm_service)
        self.context_service = ContextService(
            rag_service=self.rag_service,
            compression_service=self.compression_service,
            semantic_memory_service=self.semantic_memory_service,
            system_prompt=self.system_prompt,
            assessment_url=self.assessment_url,
        )
        
        # Background task management
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

    def _load_system_prompt(self) -> str:
        """Load the SELVE chatbot system prompt from prompts module"""
        return SYSTEM_PROMPT

    def _resolve_assessment_url(self) -> str:
        """Resolve assessment URL from environment with sensible defaults."""
        # Priority: explicit assessment URL
        env_url = os.getenv("ASSESSMENT_URL")
        if env_url:
            return env_url.rstrip("/") + "/assessment"

        # Fallbacks from app URLs (frontend/backends share envs)
        app_url = (
            os.getenv("APP_URL")
            or os.getenv("NEXT_PUBLIC_APP_URL")
            or os.getenv("MAIN_APP_URL")
            or os.getenv("MAIN_APP_URL_PROD")
            or os.getenv("MAIN_APP_URL_DEV")
        )
        if app_url:
            return f"{app_url.rstrip('/')}/assessment"

        return "http://localhost:3000/assessment"

    # =========================================================================
    # Background Task Management
    # =========================================================================

    def _handle_task_result(self, task: asyncio.Task) -> None:
        """
        Handle completed background task, log any errors.
        
        This callback is attached to all background tasks to ensure:
        1. Task is removed from tracking set
        2. Any exceptions are logged (not silently swallowed)
        """
        self._background_tasks.discard(task)
        
        if task.cancelled():
            logger.debug(f"Background task '{task.get_name()}' was cancelled")
            return
            
        try:
            # This will raise if the task had an exception
            task.result()
        except Exception as e:
            logger.error(
                f"Background task '{task.get_name()}' failed: {e}",
                exc_info=True,
                extra={"task_name": task.get_name()}
            )

    def _schedule_background_task(self, coro, name: str) -> asyncio.Task:
        """
        Schedule a background task with proper tracking and error handling.
        
        Args:
            coro: Coroutine to run
            name: Descriptive name for logging/debugging
            
        Returns:
            The created task
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._handle_task_result)
        logger.debug(f"Scheduled background task: {name}")
        return task

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Gracefully shutdown the service, cancelling all background tasks.
        
        Args:
            timeout: Maximum seconds to wait for tasks to complete
        """
        self._shutdown_event.set()
        
        if not self._background_tasks:
            logger.info("No background tasks to cancel during shutdown")
            return
        
        logger.info(f"Shutting down ChatService, cancelling {len(self._background_tasks)} background tasks")
        
        # Cancel all tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for all tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for background tasks to cancel after {timeout}s")
        
        self._background_tasks.clear()
        logger.info("ChatService shutdown complete")

    @asynccontextmanager
    async def lifespan(self):
        """
        Async context manager for service lifecycle.
        
        Usage:
            async with chat_service.lifespan():
                # Use the service
                pass
            # Service is gracefully shutdown
        """
        try:
            yield self
        finally:
            await self.shutdown()

    # =========================================================================
    # Input Validation
    # =========================================================================

    def _validate_message(self, message: str) -> str:
        """
        Validate and sanitize user message.
        
        Args:
            message: Raw user message
            
        Returns:
            Sanitized message
            
        Raises:
            InputValidationError: If message is invalid
        """
        if not message:
            raise InputValidationError("Message cannot be empty")
        
        if not isinstance(message, str):
            raise InputValidationError(f"Message must be a string, got {type(message).__name__}")
        
        # Strip whitespace
        message = message.strip()
        
        if not message:
            raise InputValidationError("Message cannot be empty after trimming whitespace")
        
        if len(message) > MAX_MESSAGE_LENGTH:
            raise InputValidationError(
                f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters "
                f"(got {len(message)})"
            )
        
        return message

    def _validate_conversation_history(
        self,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Validate and sanitize conversation history.
        
        Args:
            conversation_history: List of message dicts with 'role' and 'content'
            
        Returns:
            Validated (possibly truncated) history
            
        Raises:
            InputValidationError: If history format is invalid
        """
        if conversation_history is None:
            return []
        
        if not isinstance(conversation_history, list):
            raise InputValidationError(
                f"Conversation history must be a list, got {type(conversation_history).__name__}"
            )
        
        valid_roles = {"user", "assistant", "system"}
        validated_history = []
        total_chars = 0
        
        for i, msg in enumerate(conversation_history):
            if not isinstance(msg, dict):
                raise InputValidationError(
                    f"History message at index {i} must be a dict, got {type(msg).__name__}"
                )
            
            if "role" not in msg:
                raise InputValidationError(f"History message at index {i} missing 'role' field")
            
            if "content" not in msg:
                raise InputValidationError(f"History message at index {i} missing 'content' field")
            
            role = msg["role"]
            content = msg["content"]
            
            if role not in valid_roles:
                raise InputValidationError(
                    f"Invalid role '{role}' at index {i}. Must be one of: {valid_roles}"
                )
            
            if not isinstance(content, str):
                raise InputValidationError(
                    f"Content at index {i} must be a string, got {type(content).__name__}"
                )
            
            total_chars += len(content)
            validated_history.append({"role": role, "content": content})
        
        # Truncate if too many messages
        if len(validated_history) > MAX_HISTORY_MESSAGES:
            logger.warning(
                f"Truncating conversation history from {len(validated_history)} "
                f"to {MAX_HISTORY_MESSAGES} messages"
            )
            validated_history = validated_history[-MAX_HISTORY_MESSAGES:]
        
        # Truncate if total content too large
        if total_chars > MAX_HISTORY_TOTAL_CHARS:
            logger.warning(
                f"Conversation history exceeds {MAX_HISTORY_TOTAL_CHARS} chars, "
                f"truncating oldest messages"
            )
            # Remove oldest messages until under limit
            while total_chars > MAX_HISTORY_TOTAL_CHARS and len(validated_history) > 1:
                removed = validated_history.pop(0)
                total_chars -= len(removed["content"])
        
        return validated_history

    def _validate_selve_scores(
        self,
        scores: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """
        Validate SELVE personality scores.
        
        Args:
            scores: Dictionary of dimension names to scores
            
        Returns:
            Validated scores or None
            
        Raises:
            InputValidationError: If scores format is invalid
        """
        if scores is None:
            return None
        
        if not isinstance(scores, dict):
            raise InputValidationError(
                f"SELVE scores must be a dict, got {type(scores).__name__}"
            )
        
        valid_dimensions = {
            "LUMEN", "AETHER", "ORPHEUS", "ORIN",
            "LYRA", "VARA", "CHRONOS", "KAEL"
        }
        
        validated_scores = {}
        
        for dim, score in scores.items():
            if dim not in valid_dimensions:
                logger.warning(f"Unknown SELVE dimension '{dim}', skipping")
                continue
            
            try:
                score_float = float(score)
            except (TypeError, ValueError):
                raise InputValidationError(
                    f"Score for '{dim}' must be numeric, got {type(score).__name__}"
                )
            
            # Clamp to valid range
            score_float = max(0.0, min(100.0, score_float))
            validated_scores[dim] = score_float
        
        return validated_scores if validated_scores else None

    async def _get_scores_from_profile(self, clerk_user_id: str) -> Optional[Dict[str, float]]:
        """Fetch SELVE scores from the user profile service when not provided."""
        try:
            profile = await self.user_profile_service.get_user_scores(clerk_user_id)
            if not profile:
                return None

            profile_scores = profile.get("scores")
            if not profile_scores:
                return None

            return self._validate_selve_scores(profile_scores)
        except Exception as exc:
            logger.warning(
                "Failed to fetch SELVE scores for user %s: %s",
                clerk_user_id,
                exc,
                exc_info=True,
            )
            return None

    # =========================================================================
    # Conversation State Management
    # =========================================================================

    async def _analyze_and_update_state(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """
        Analyze conversation state and update session.
        
        This runs as a background task and handles its own errors.
        
        Args:
            session_id: Chat session ID
            user_message: User's message
            assistant_response: Assistant's response
            conversation_history: Previous messages
        """
        try:
            state = await self.conversation_state_service.analyze_conversation_state(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_history=conversation_history
            )
            
            await self.conversation_state_service.update_session_state(
                session_id=session_id,
                state=state
            )
            
            logger.debug(f"Updated conversation state for session {session_id}")
            
        except Exception as e:
            # Error is logged but not re-raised (background task)
            logger.error(
                f"Failed to update conversation state for session {session_id}: {e}",
                exc_info=True
            )

    # =========================================================================
    # Title Generation
    # =========================================================================

    async def generate_conversation_title(
        self,
        first_message: str,
        assistant_response: Optional[str] = None
    ) -> str:
        """
        Generate a concise title for a conversation based on the opening exchange.
        
        Args:
            first_message: User's first message
            assistant_response: Optional assistant's first reply
            
        Returns:
            Generated title (max 50 chars)
        """
        return await self.title_service.generate_title(
            first_message=first_message,
            assistant_response=assistant_response,
        )

    # =========================================================================
    # Response Generation (Non-Streaming)
    # =========================================================================

    async def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        clerk_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        selve_scores: Optional[Dict[str, float]] = None,
        assessment_url: Optional[str] = None,
    ) -> ChatResponse:
        """
        Generate a chat response with optional RAG context.
        
        Args:
            message: User's current message
            conversation_history: Previous messages [{"role": "user/assistant", "content": "..."}]
            use_rag: Whether to retrieve context from RAG
            clerk_user_id: User ID for personalization
            session_id: Session ID for state tracking
            selve_scores: SELVE personality scores
            assessment_url: URL for users to take the assessment when scores are missing
            
        Returns:
            ChatResponse with response and metadata
            
        Raises:
            InputValidationError: If inputs are invalid
        """
        # Validate inputs
        message = self._validate_message(message)
        conversation_history = self._validate_conversation_history(conversation_history)
        selve_scores = self._validate_selve_scores(selve_scores)
        
        # Get Langfuse client for tracing
        langfuse = get_client()

        # Propagate trace attributes for all nested observations
        propagate_attributes(
            user_id=clerk_user_id or "anonymous",
            session_id=session_id,
            metadata={
                "use_rag": str(use_rag),
                "has_scores": str(bool(selve_scores)),
                "message_length": str(len(message))
            },
            tags=["chat", "selve-chatbot"]
        )

        if not assessment_url:
            assessment_url = self.assessment_url

        if not selve_scores and clerk_user_id:
            selve_scores = await self._get_scores_from_profile(clerk_user_id)
        
        # Check guardrails first
        canned_response = get_canned_response(message)
        if canned_response:
            classification, _ = classify_message(message)
            logger.info(f"Message classified as '{classification}', returning canned response")
            
            return ChatResponse(
                response=canned_response,
                context_used=False,
                retrieved_chunks=[],
                model="guardrail",
                provider="internal",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                cost=0.0,
                compression_needed=False
            )
        
        # Build context (RAG, memories, personality)
        context_result = await self.context_service.build_context(
            message=message,
            clerk_user_id=clerk_user_id,
            selve_scores=selve_scores,
            use_rag=use_rag,
            assessment_url=assessment_url,
        )

        # Build messages for LLM
        messages = self.context_service.build_messages(
            message=message,
            system_content=context_result.system_content,
            conversation_history=conversation_history,
            context_info=context_result.context_info,
        )
        
        # Use context manager for proper Langfuse generation tracing
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="chat-response",
            model=self.llm_service.model,
            input=message,  # Clean string input - just the user message
        ) as generation:
            try:
                # Generate response with timeout
                llm_response = await asyncio.wait_for(
                    self.llm_service.generate_response_async(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500
                    ),
                    timeout=LLM_TIMEOUT_SECONDS
                )
                
                # Update generation with clean output and usage
                generation.update(
                    output=llm_response["content"],  # Clean string output
                    usage_details={
                        "input": llm_response["usage"].get("input_tokens", 0),
                        "output": llm_response["usage"].get("output_tokens", 0),
                    },
                    model=llm_response["model"],
                    metadata={
                        "provider": llm_response["provider"],
                        "cost": llm_response["cost"],
                        "context_used": context_result.context_info is not None,
                    }
                )
                
            except asyncio.TimeoutError:
                logger.error(f"LLM response generation timed out after {LLM_TIMEOUT_SECONDS}s")
                generation.update(
                    level="ERROR",
                    status_message=f"Timeout after {LLM_TIMEOUT_SECONDS}s"
                )
                raise
        
        # Check if compression needed
        compression_needed = False
        if clerk_user_id:
            total_tokens = llm_response.get("usage", {}).get("total_tokens", 0)
            compression_needed = self.compression_service.needs_compression(
                total_tokens,
                llm_response.get("model", "")
            )
        
        # Schedule background state update
        if session_id:
            self._schedule_background_task(
                self._analyze_and_update_state(
                    session_id=session_id,
                    user_message=message,
                    assistant_response=llm_response["content"],
                    conversation_history=conversation_history
                ),
                name=f"state_update_{session_id}"
            )
        
        # Determine if context was used
        context_used = (
            context_result.context_info is not None and
            context_result.context_info.get("retrieved_count", 0) > 0
        )
        
        return ChatResponse(
            response=llm_response["content"],
            context_used=context_used,
            retrieved_chunks=context_result.context_info.get("chunks", []) if context_result.context_info else [],
            model=llm_response["model"],
            provider=llm_response["provider"],
            usage=llm_response["usage"],
            cost=llm_response["cost"],
            compression_needed=compression_needed
        )

    # =========================================================================
    # Response Generation (Streaming)
    # =========================================================================

    async def generate_response_stream(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        clerk_user_id: Optional[str] = None,
        selve_scores: Optional[Dict[str, float]] = None,
        assessment_url: Optional[str] = None,
        emit_status: bool = True,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Generate a streaming chat response with optional RAG context and status events.
        
        Yields response chunks and status events as they're generated.
        
        Args:
            message: User's current message
            conversation_history: Previous messages
            use_rag: Whether to retrieve context from RAG
            clerk_user_id: Clerk user ID for profile personalization
            selve_scores: User's SELVE personality scores
            assessment_url: URL for users to take the assessment when scores are missing
            emit_status: Whether to emit status events for thinking UI
            session_id: Session ID for tracing
            
        Yields:
            Either text chunks (str) or status event dicts
            
        Raises:
            InputValidationError: If inputs are invalid
        """
        # Validate inputs upfront
        try:
            message = self._validate_message(message)
            conversation_history = self._validate_conversation_history(conversation_history)
            selve_scores = self._validate_selve_scores(selve_scores)
            
            if not assessment_url:
                assessment_url = self.assessment_url

            if not selve_scores and clerk_user_id:
                selve_scores = await self._get_scores_from_profile(clerk_user_id)
        except InputValidationError as e:
            if emit_status:
                yield StatusEvent.error(str(e)).to_dict()
            raise
        
        # Check guardrails first
        canned_response = get_canned_response(message)
        if canned_response:
            classification, _ = classify_message(message)
            logger.info(f"Message classified as '{classification}', returning canned response")
            
            if emit_status:
                yield StatusEvent.complete(sources=[], guardrail=classification).to_dict()
            
            # Stream canned response character by character
            for char in canned_response:
                yield char
            return
        
        sources_used: List[Dict[str, str]] = []
        
        # Get Langfuse client and propagate attributes
        langfuse = get_client()
        
        propagate_attributes(
            user_id=clerk_user_id or "anonymous",
            session_id=session_id,
            metadata={
                "use_rag": str(use_rag),
                "has_scores": str(bool(selve_scores)),
                "streaming": "true",
                "message_length": str(len(message))
            },
            tags=["chat", "streaming", "selve-chatbot"]
        )
        
        # Use context manager for the entire streaming operation
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="chat-response-stream",
            model=self.llm_service.model,
            input=message,  # Clean string input - just the user message
        ) as generation:
            try:
                # Phase 1: Retrieve Context
                if emit_status and use_rag:
                    yield StatusEvent.retrieving_context().to_dict()
                
                # Phase 2: Personalization
                if emit_status:
                    yield StatusEvent.personalizing().to_dict()
                
                # Build all context (parallel fetching)
                context_result = await self.context_service.build_context(
                    message=message,
                    clerk_user_id=clerk_user_id,
                    selve_scores=selve_scores,
                    use_rag=use_rag,
                    assessment_url=assessment_url,
                )
                
                sources_used = context_result.sources_used
                
                # Phase 3: Generation
                if emit_status:
                    yield StatusEvent.generating(model=self.llm_service.model).to_dict()
                
                # Build messages
                messages = self.context_service.build_messages(
                    message=message,
                    system_content=context_result.system_content,
                    conversation_history=conversation_history,
                    context_info=context_result.context_info,
                )
                
                # Stream response and collect for Langfuse
                full_response_chunks = []
                async for chunk in self.llm_service.generate_response_stream(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                ):
                    # Check for shutdown
                    if self._shutdown_event.is_set():
                        logger.info("Shutdown requested, stopping stream")
                        break
                    if isinstance(chunk, str):
                        full_response_chunks.append(chunk)
                    yield chunk
                
                # Update generation with collected output
                full_response = "".join(full_response_chunks)
                generation.update(
                    output=full_response,  # Clean string output
                    metadata={
                        "provider": self.llm_service.provider,
                        "sources_count": len(sources_used),
                        "context_used": context_result.context_info is not None,
                    }
                )
                
                # Phase 4: Citation
                if emit_status and sources_used:
                    yield StatusEvent.citing_sources(sources=sources_used).to_dict()
                
                # Complete
                if emit_status:
                    yield StatusEvent.complete(sources=sources_used).to_dict()
                    
            except asyncio.CancelledError:
                logger.info("Stream generation was cancelled")
                generation.update(
                    level="WARNING",
                    status_message="Request was cancelled"
                )
                if emit_status:
                    yield StatusEvent.error("Request was cancelled").to_dict()
                raise
                
            except Exception as e:
                logger.error(f"Error during stream generation: {e}", exc_info=True)
                generation.update(
                    level="ERROR",
                    status_message=str(e)
                )
                if emit_status:
                    yield StatusEvent.error("An error occurred while generating the response").to_dict()
                raise