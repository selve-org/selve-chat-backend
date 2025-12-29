"""
Agentic Chat Service - SELVE Chatbot Orchestrator.

This is the main entry point for chat interactions. It orchestrates:
1. Security Guard - Prompt injection detection
2. User State Loader - Complete user context
3. Thinking Engine - Agentic reasoning
4. Response Validator - Output sanitization
5. Post-Processing - Notes, compression, logging

This replaces the simple LLM call with a multi-phase agentic pipeline.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

from langfuse import get_client, propagate_attributes

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class AgentConfig:
    """Agent configuration."""
    
    # Feature flags
    SECURITY_GUARD_ENABLED: bool = True
    THINKING_ENGINE_ENABLED: bool = True
    RESPONSE_VALIDATION_ENABLED: bool = True
    USER_NOTES_ENABLED: bool = True
    
    # Timeouts
    SECURITY_CHECK_TIMEOUT: float = 2.0
    USER_STATE_TIMEOUT: float = 3.0
    THINKING_TIMEOUT: float = 30.0
    
    # Limits
    MAX_MESSAGE_LENGTH: int = 10_000
    MAX_HISTORY_MESSAGES: int = 50


# =============================================================================
# STATUS TYPES
# =============================================================================


class AgentPhase(str, Enum):
    """Phases of the agent pipeline."""
    SECURITY_CHECK = "security_check"
    LOADING_USER = "loading_user"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RETRIEVING = "retrieving_context"
    PERSONALIZING = "personalizing"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentStatus:
    """Status event for UI."""
    
    phase: AgentPhase
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "status",
            "status": self.phase.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class AgentResponse:
    """Complete response from agent."""
    
    response: str
    sources: List[Dict[str, str]]
    user_intent: str
    security_score: float
    was_blocked: bool = False
    block_reason: Optional[str] = None
    notes_added: List[str] = field(default_factory=list)
    model: str = ""
    cost: float = 0.0


# =============================================================================
# AGENTIC CHAT SERVICE
# =============================================================================


class AgenticChatService:
    """
    Agentic chat service with multi-phase pipeline.
    
    Pipeline:
    1. SECURITY CHECK - Detect prompt injection
    2. LOAD USER STATE - Get complete user context
    3. THINK & RESPOND - Agentic reasoning + response
    4. VALIDATE - Sanitize output
    5. POST-PROCESS - Update notes, trigger compression
    """
    
    def __init__(self):
        """Initialize all services lazily."""
        self._security_guard = None
        self._user_state_service = None
        self._thinking_engine = None
        self._response_validator = None
        self._llm_service = None
        self._compression_service = None
        self._system_prompt = None
        self._assessment_url = None
        
        # Background task tracking
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # =========================================================================
    # Lazy Loading
    # =========================================================================
    
    @property
    def security_guard(self):
        if self._security_guard is None:
            from .security_guard import SecurityGuard
            self._security_guard = SecurityGuard()
        return self._security_guard
    
    @property
    def user_state_service(self):
        if self._user_state_service is None:
            from .user_state_service import UserStateService
            self._user_state_service = UserStateService()
        return self._user_state_service
    
    @property
    def thinking_engine(self):
        if self._thinking_engine is None:
            from .thinking_engine import ThinkingEngine
            self._thinking_engine = ThinkingEngine()
        return self._thinking_engine
    
    @property
    def response_validator(self):
        if self._response_validator is None:
            from .response_validator import SELVEResponseValidator
            self._response_validator = SELVEResponseValidator()
        return self._response_validator
    
    @property
    def llm_service(self):
        if self._llm_service is None:
            from app.services.llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service
    
    @property
    def compression_service(self):
        if self._compression_service is None:
            from app.services.compression_service import CompressionService
            self._compression_service = CompressionService()
        return self._compression_service
    
    @property
    def system_prompt(self):
        if self._system_prompt is None:
            from app.prompts.system_prompt import SYSTEM_PROMPT
            self._system_prompt = SYSTEM_PROMPT
        return self._system_prompt
    
    @property
    def assessment_url(self):
        if self._assessment_url is None:
            import os
            self._assessment_url = (
                os.getenv("ASSESSMENT_URL") or
                os.getenv("APP_URL", "http://localhost:3000") + "/assessment"
            )
        return self._assessment_url
    
    # =========================================================================
    # Main Entry Point (Streaming)
    # =========================================================================
    
    async def chat_stream(
        self,
        message: str,
        clerk_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        client_ip: Optional[str] = None,
        user_timezone: str = "UTC",
        emit_status: bool = True,
        regeneration_type: Optional[str] = None,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Main streaming chat endpoint.

        Yields status events and response chunks.

        Args:
            message: User's message
            clerk_user_id: Clerk user ID
            session_id: Session ID
            conversation_history: Previous messages
            client_ip: Client IP for geo/security
            user_timezone: User's timezone (e.g., "America/New_York")
            emit_status: Whether to emit status events
            regeneration_type: "regenerate" | "edit" | None

        Yields:
            Status dicts and response text chunks
        """
        start_time = datetime.utcnow()
        
        # Validate inputs
        if not message or not message.strip():
            if emit_status:
                yield AgentStatus(
                    phase=AgentPhase.ERROR,
                    message="Message cannot be empty",
                ).to_dict()
            return
        
        message = message.strip()[:AgentConfig.MAX_MESSAGE_LENGTH]
        conversation_history = conversation_history or []
        
        # Setup Langfuse tracing with user identification
        langfuse = get_client()
        from langfuse import propagate_attributes

        # CRITICAL FIX: Use propagate_attributes to set user_id and session_id
        # This ensures users appear in Langfuse dashboard
        with propagate_attributes(
            user_id=clerk_user_id or "anonymous",
            session_id=session_id,
            metadata={"streaming": "true"},
            tags=["chat", "selve-agent"],
        ):
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="agentic-chat-response",
                model="selve-agent",
                input=message,
            ) as generation:
                # Capture and yield trace ID for frontend feedback tracking
                try:
                    trace_id = langfuse.get_current_trace_id()
                    if trace_id:
                        yield {"type": "trace_id", "trace_id": trace_id}
                except Exception as e:
                    self.logger.warning(f"Failed to get trace ID: {e}")

                try:
                    # =================================================================
                    # PHASE 1: SECURITY CHECK
                    # =================================================================
                    if AgentConfig.SECURITY_GUARD_ENABLED:
                        if emit_status:
                            yield AgentStatus(
                                phase=AgentPhase.SECURITY_CHECK,
                                message="Checking security...",
                            ).to_dict()
                    
                        security_result = await asyncio.wait_for(
                            self.security_guard.analyze(message, clerk_user_id),
                            timeout=AgentConfig.SECURITY_CHECK_TIMEOUT,
                        )
                    
                        # If blocked, return canned response
                        if not security_result.is_safe:
                            self.logger.warning(
                                f"Security block: {security_result.threat_level.value} "
                                f"(score: {security_result.risk_score:.2f})"
                            )
                        
                            # Log security incident
                            if clerk_user_id and AgentConfig.USER_NOTES_ENABLED:
                                self._schedule_background_task(
                                    self._log_security_incident(
                                        clerk_user_id=clerk_user_id,
                                        session_id=session_id,
                                        result=security_result,
                                    ),
                                    name="log_security_incident",
                                )
                        
                            if emit_status:
                                yield AgentStatus(
                                    phase=AgentPhase.COMPLETE,
                                    message="Response complete",
                                    details={"security_blocked": True},
                                ).to_dict()

                            # HONEST MODE: Direct response when manipulation detected
                            # Be straight with users about what we detected
                            if security_result.threat_level.value in ["high", "critical"]:
                                canned = (
                                    "I notice you're trying to manipulate my responses. "
                                    "I'm designed to have honest conversations about personality. "
                                    "Want to try that instead?"
                                )
                            elif security_result.threat_level.value == "medium":
                                canned = (
                                    "Hey, I'm picking up on something unusual in your message. "
                                    "I work best when we're having a genuine conversation. "
                                    "What's really on your mind?"
                                )
                            else:
                                # Low threat - be gentler
                                canned = (
                                    "I'm here to help you understand your personality through "
                                    "honest conversation. What would you like to explore?"
                                )

                            # Update Langfuse generation for security block
                            generation.update(
                                output=canned,
                                metadata={
                                    "security_blocked": True,
                                    "threat_level": security_result.threat_level.value,
                                    "risk_score": security_result.risk_score,
                                }
                            )

                            for char in canned:
                                yield char
                            return
                
                    # =================================================================
                    # PHASE 2: LOAD USER STATE
                    # =================================================================
                    if emit_status:
                        yield AgentStatus(
                            phase=AgentPhase.LOADING_USER,
                            message="Loading your profile...",
                        ).to_dict()
                
                    user_state = None
                    if clerk_user_id:
                        try:
                            user_state = await asyncio.wait_for(
                                self.user_state_service.load_user_state(
                                    clerk_user_id=clerk_user_id,
                                    session_id=session_id,
                                ),
                                timeout=AgentConfig.USER_STATE_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            self.logger.warning("User state load timed out")
                        except Exception as e:
                            self.logger.error(f"User state load failed: {e}")
                
                    # Create minimal user state if not loaded
                    if user_state is None:
                        from .user_state_service import UserState, AssessmentStatus
                        user_state = UserState(
                            user_id="",
                            clerk_user_id=clerk_user_id or "",
                            assessment_status=AssessmentStatus.NOT_TAKEN,
                            has_assessment=False,
                        )
    
                    # If no explicit history was provided, fall back to stored session messages
                    # so the agent preserves context across turns.
                    if not conversation_history and getattr(user_state, "current_session_messages", None):
                        conversation_history = user_state.current_session_messages[-AgentConfig.MAX_HISTORY_MESSAGES:]
                
                    # =================================================================
                    # PHASE 3-4: THINKING ENGINE (includes analyze, plan, execute, generate)
                    # =================================================================
                    if AgentConfig.THINKING_ENGINE_ENABLED:
                        response_chunks = []
                        sources = []
                        user_intent = "unknown"
                        stream_metadata = None  # Capture metadata for Langfuse

                        # Build dynamic system prompt with temporal awareness
                        system_prompt = self.system_prompt

                        # Add temporal/situational context (uses user's actual timezone!)
                        from app.prompts.system_prompt import build_temporal_context_prompt
                        temporal_context = build_temporal_context_prompt(user_timezone=user_timezone)
                        if temporal_context:
                            system_prompt += "\n" + temporal_context

                        # Add regeneration context if applicable
                        if regeneration_type == "regenerate":
                            system_prompt += "\n\nNOTE: The user requested a regeneration of the previous response. Provide a different perspective, alternative approach, or additional insights that weren't covered in the previous response. Vary your communication style and examples."

                        async for item in self.thinking_engine.think_and_respond(
                            message=message,
                            user_state=user_state,
                            conversation_history=conversation_history,
                            system_prompt=system_prompt,
                            emit_status=emit_status,
                        ):
                            if isinstance(item, dict):
                                # Check if this is metadata from LLM service
                                if item.get("__metadata__"):
                                    stream_metadata = item
                                    self.logger.info(f"📊 Captured stream metadata: {stream_metadata}")
                                    continue  # Don't yield metadata to client

                                # Status event from thinking engine
                                # Map thinking phases to agent phases
                                phase_map = {
                                    "analyzing": AgentPhase.ANALYZING,
                                    "planning": AgentPhase.PLANNING,
                                    "retrieving": AgentPhase.RETRIEVING,
                                    "personalizing": AgentPhase.PERSONALIZING,
                                    "generating": AgentPhase.GENERATING,
                                    "complete": AgentPhase.COMPLETE,
                                    "error": AgentPhase.ERROR,
                                }

                                thinking_phase = item.get("phase", "")
                                if thinking_phase in phase_map:
                                    yield AgentStatus(
                                        phase=phase_map[thinking_phase],
                                        message=item.get("message", ""),
                                        details=item.get("details", {}),
                                    ).to_dict()

                                # Capture metadata
                                if item.get("details", {}).get("sources"):
                                    sources = item["details"]["sources"]
                                if item.get("details", {}).get("intent"):
                                    user_intent = item["details"]["intent"]

                            elif isinstance(item, str):
                                # Response chunk
                                response_chunks.append(item)
                                yield item

                        full_response = "".join(response_chunks)
                
                    else:
                        # Fallback: Simple LLM call (non-agentic)
                        full_response = await self._simple_llm_call(
                            message=message,
                            user_state=user_state,
                            conversation_history=conversation_history,
                        )
                    
                        for char in full_response:
                            yield char
                    
                        sources = []
                        user_intent = "unknown"
                
                    # =================================================================
                    # PHASE 5: RESPONSE VALIDATION & GUARDRAILS
                    # =================================================================
                    if AgentConfig.RESPONSE_VALIDATION_ENABLED and full_response:
                        if emit_status:
                            yield AgentStatus(
                                phase=AgentPhase.VALIDATING,
                                message="Checking response...",
                            ).to_dict()

                        # Standard response validation (leakage, framework mentions, etc.)
                        validation_result = self.response_validator.validate(
                            full_response,
                            user_has_assessment=user_state.has_assessment if user_state else False,
                        )

                        if validation_result.fixes_applied:
                            self.logger.info(
                                f"Response validation fixes: {validation_result.fixes_applied}"
                            )

                        # Conversation guardrails (on-brand check, drift detection)
                        try:
                            from .conversation_guardrail_service import ConversationGuardrailService

                            guardrail_service = ConversationGuardrailService()
                            guardrail_result = guardrail_service.check_response(
                                response=full_response,
                                conversation_history=conversation_history,
                                user_has_assessment=user_state.has_assessment if user_state else False,
                            )

                            if guardrail_result.should_block:
                                self.logger.error(
                                    f"CRITICAL: Response blocked by guardrails! "
                                    f"Violations: {[v.description for v in guardrail_result.violations]}"
                                )
                                # Note: Already streamed, but log for monitoring
                            elif guardrail_result.should_warn:
                                self.logger.warning(
                                    f"Response guardrail warnings: "
                                    f"{[v.description for v in guardrail_result.violations]}"
                                )

                            # Check for conversation drift
                            if conversation_history and len(conversation_history) >= 5:
                                is_drifting, drift_reason = guardrail_service.monitor_conversation_drift(
                                    conversation_history + [{"role": "assistant", "content": full_response}]
                                )
                                if is_drifting:
                                    self.logger.warning(f"Conversation drift detected: {drift_reason}")

                        except Exception as e:
                            self.logger.error(f"Guardrail check failed: {e}", exc_info=True)

                        # Note: We've already streamed the response, so we can't fix it
                        # Violations are logged for monitoring and alerting
                
                    # =================================================================
                    # PHASE 6: POST-PROCESSING (background)
                    # =================================================================
                    if clerk_user_id:
                        # Schedule background tasks
                        self._schedule_background_task(
                            self._post_process(
                                clerk_user_id=clerk_user_id,
                                session_id=session_id,
                                user_state=user_state,
                                message=message,
                                response=full_response,
                                user_intent=user_intent,
                            ),
                            name="post_process",
                        )
                
                    # Update Langfuse generation with final output
                    update_params = {
                        "output": full_response,
                        "metadata": {
                            "intent": user_intent,
                            "sources_count": len(sources),
                            "security_blocked": False,
                        }
                    }

                    # Add usage and cost if metadata was captured from LLM
                    if stream_metadata:
                        self.logger.info(f"✅ Stream metadata captured: {stream_metadata}")
                        update_params["model"] = stream_metadata.get("model")
                        update_params["usage_details"] = {
                            "input_tokens": stream_metadata["usage"].get("input_tokens", 0),
                            "output_tokens": stream_metadata["usage"].get("output_tokens", 0),
                            "total_tokens": stream_metadata["usage"].get("total_tokens", 0),
                        }
                        update_params["metadata"]["provider"] = stream_metadata.get("provider")
                        if stream_metadata.get("cost") is not None:
                            update_params["cost_details"] = {
                                "total": stream_metadata.get("cost"),
                                "input": stream_metadata.get("input_cost"),
                                "output": stream_metadata.get("output_cost")
                            }
                    else:
                        self.logger.warning("⚠️ No stream metadata captured - cost and usage will be missing")

                    generation.update(**update_params)

                    # =================================================================
                    # COMPLETE
                    # =================================================================
                    if emit_status:
                        yield AgentStatus(
                            phase=AgentPhase.COMPLETE,
                            message="Response complete",
                            details={
                                "sources": sources,
                                "intent": user_intent,
                                "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                            },
                        ).to_dict()
            
                except asyncio.CancelledError:
                    self.logger.info("Chat stream cancelled")
                    generation.update(
                        level="WARNING",
                        status_message="Request was cancelled"
                    )
                    if emit_status:
                        yield AgentStatus(
                            phase=AgentPhase.ERROR,
                            message="Request cancelled",
                        ).to_dict()
                    raise
            
                except Exception as e:
                    self.logger.error(f"Chat error: {e}", exc_info=True)
                    generation.update(
                        level="ERROR",
                        status_message=str(e)
                    )
                    if emit_status:
                        yield AgentStatus(
                            phase=AgentPhase.ERROR,
                            message="An error occurred",
                            details={"error": str(e)},
                        ).to_dict()

                    # Yield fallback response
                    yield "I apologize, but I encountered an issue. Could you try again?"
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    async def _simple_llm_call(
        self,
        message: str,
        user_state: Any,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """Fallback simple LLM call without agentic reasoning."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add user context
        if user_state and hasattr(user_state, 'to_context_string'):
            messages[0]["content"] += "\n\n" + user_state.to_context_string()
        
        # Add history
        for msg in conversation_history[-10:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append(msg)
        
        messages.append({"role": "user", "content": message})
        
        response = await self.llm_service.generate_response_async(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        
        return response.get("content", "")
    
    async def _log_security_incident(
        self,
        clerk_user_id: str,
        session_id: Optional[str],
        result: Any,  # SecurityResult
    ) -> None:
        """Log a security incident."""
        try:
            await self.user_state_service.flag_security_incident(
                user_id=clerk_user_id,  # Will need to look up internal ID
                incident_type=result.threat_types[0].value if result.threat_types else "unknown",
                details=f"Score: {result.risk_score:.2f}, Flags: {result.flags[:3]}",
            )
        except Exception as e:
            self.logger.error(f"Failed to log security incident: {e}")
    
    async def _post_process(
        self,
        clerk_user_id: str,
        session_id: Optional[str],
        user_state: Any,
        message: str,
        response: str,
        user_intent: str,
    ) -> None:
        """Post-process after response generation."""
        try:
            # Check if compression needed
            if session_id:
                should_compress = await self.compression_service.should_trigger_compression(
                    session_id
                )
                if should_compress:
                    self.logger.info(f"Triggering compression for session {session_id}")
                    # Get user_id from user_state
                    user_id = getattr(user_state, 'user_id', None)
                    if user_id:
                        await self.compression_service.compress_conversation(
                            session_id=session_id,
                            user_id=user_id,
                            clerk_user_id=clerk_user_id,
                        )
            
            # Add user notes if emotional state detected
            # (This would come from thinking engine analysis)
            
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
    
    # =========================================================================
    # Background Task Management
    # =========================================================================
    
    def _schedule_background_task(self, coro, name: str) -> asyncio.Task:
        """Schedule a background task with tracking."""
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(lambda t: self._background_tasks.discard(t))
        return task
    
    async def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shutdown, cancelling background tasks."""
        self._shutdown_event.set()
        
        if self._background_tasks:
            for task in self._background_tasks:
                task.cancel()
            
            await asyncio.wait(
                self._background_tasks,
                timeout=timeout,
            )
        
        self._background_tasks.clear()
    
    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for lifecycle."""
        try:
            yield self
        finally:
            await self.shutdown()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_chat_service: Optional[AgenticChatService] = None


def get_chat_service() -> AgenticChatService:
    """Get the global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = AgenticChatService()
    return _chat_service
