"""
Langfuse Service for LLM Observability
Provides tracing, cost tracking, and prompt management.
Updated for Langfuse Python SDK v3 with clean input/output handling.
"""
import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


class LangfuseService:
    """
    Service for Langfuse LLM observability and tracing.
    
    Updated for Langfuse v3 SDK which uses OpenTelemetry under the hood.
    
    Provides:
    - Trace creation and management via context managers
    - Generation tracking for LLM calls with clean inputs/outputs
    - Span tracking for non-LLM operations
    - Cost tracking
    - Attribute propagation across nested observations
    
    Key v3 changes:
    - Uses `get_client()` instead of `Langfuse()` constructor
    - Context managers (`start_as_current_observation`) instead of manual trace/span creation
    - `propagate_attributes()` for user_id, session_id, etc.
    - Clean string inputs/outputs instead of nested dicts
    """
    
    def __init__(self):
        self.enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
        self._client = None
        
        if self.enabled:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Langfuse client using v3 API."""
        try:
            from langfuse import get_client
            
            # v3 uses environment variables automatically:
            # LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            
            if not secret_key or not public_key:
                logger.warning("Langfuse keys not configured, tracing disabled")
                self.enabled = False
                return
            
            # get_client() returns a singleton configured via environment variables
            self._client = get_client()
            logger.info("Langfuse tracing initialized (v3 SDK)")
            
        except ImportError:
            logger.warning("Langfuse package not installed, tracing disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Langfuse initialization failed: {e}")
            self.enabled = False
    
    @property
    def client(self):
        """Get the Langfuse client instance."""
        return self._client
    
    @contextmanager
    def trace_generation(
        self,
        name: str,
        model: str,
        input_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Context manager for tracing an LLM generation with clean input/output.
        
        This is the PRIMARY method for tracing LLM calls. It creates a generation
        observation and yields a context object that can be updated with output.
        
        Args:
            name: Generation name (e.g., "chat-response")
            model: Model name (e.g., "gpt-5-mini", "claude-3-5-haiku")
            input_text: Clean string input - the user's message, NOT a dict
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            tags: Tags for filtering
            
        Usage:
            with langfuse.trace_generation(
                name="chat-response",
                model="gpt-5-mini",
                input_text="What careers fit my personality?",
                user_id="user_123",
                session_id="session_456"
            ) as gen:
                # Make LLM call
                response = await llm.generate(...)
                
                # Update with output
                gen.update(
                    output=response["content"],
                    usage_details={
                        "input": response["usage"]["input_tokens"],
                        "output": response["usage"]["output_tokens"],
                    }
                )
        """
        if not self.enabled or not self._client:
            # Yield a no-op context when disabled
            yield _NoOpContext()
            return
        
        try:
            from langfuse import propagate_attributes
            
            # Propagate trace-level attributes to all nested observations
            with propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            ):
                # Create the generation observation
                with self._client.start_as_current_observation(
                    as_type="generation",
                    name=name,
                    model=model,
                    input=input_text,  # Clean string, not a dict!
                ) as generation:
                    yield generation
                    
        except Exception as e:
            logger.error(f"Failed to create generation trace: {e}", exc_info=True)
            yield _NoOpContext()
    
    @contextmanager
    def trace_span(
        self,
        name: str,
        input_data: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Context manager for tracing a non-LLM operation (span).
        
        Use this for operations like RAG retrieval, data processing, etc.
        
        Args:
            name: Span name (e.g., "rag-retrieval", "context-building")
            input_data: Optional input description (keep it clean/short)
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            tags: Tags for filtering
            
        Usage:
            with langfuse.trace_span(
                name="rag-retrieval",
                input_data="personality careers",
                user_id="user_123"
            ) as span:
                # Do retrieval
                chunks = await rag.retrieve(query)
                
                # Update with output
                span.update(
                    output=f"Retrieved {len(chunks)} chunks",
                    metadata={"chunk_count": len(chunks)}
                )
        """
        if not self.enabled or not self._client:
            yield _NoOpContext()
            return
        
        try:
            from langfuse import propagate_attributes
            
            with propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            ):
                with self._client.start_as_current_observation(
                    as_type="span",
                    name=name,
                    input=input_data,
                ) as span:
                    yield span
                    
        except Exception as e:
            logger.error(f"Failed to create span trace: {e}", exc_info=True)
            yield _NoOpContext()
    
    @contextmanager 
    def trace_retrieval(
        self,
        name: str = "rag-retrieval",
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager specifically for RAG retrieval operations.
        
        Args:
            name: Retrieval name
            query: The search query
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Usage:
            with langfuse.trace_retrieval(query="career advice") as retrieval:
                chunks = await vector_store.search(query)
                retrieval.update(
                    output=f"Found {len(chunks)} relevant chunks",
                    metadata={"sources": [c["source"] for c in chunks]}
                )
        """
        if not self.enabled or not self._client:
            yield _NoOpContext()
            return
        
        try:
            from langfuse import propagate_attributes
            
            with propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            ):
                with self._client.start_as_current_observation(
                    as_type="retriever",
                    name=name,
                    input=query,
                ) as retrieval:
                    yield retrieval
                    
        except Exception as e:
            logger.error(f"Failed to create retrieval trace: {e}", exc_info=True)
            yield _NoOpContext()
    
    def update_current_trace(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Update the currently active trace with additional attributes.
        
        Call this from within a traced context to add information.
        """
        if not self.enabled or not self._client:
            return
        
        try:
            update_kwargs = {}
            if user_id:
                update_kwargs["user_id"] = user_id
            if session_id:
                update_kwargs["session_id"] = session_id
            if metadata:
                update_kwargs["metadata"] = metadata
            if tags:
                update_kwargs["tags"] = tags
            
            if update_kwargs:
                self._client.update_current_trace(**update_kwargs)
        except Exception as e:
            logger.warning(f"Failed to update current trace: {e}")
    
    def update_current_span(
        self,
        output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None
    ):
        """
        Update the currently active span/generation.
        
        Call this from within a traced context to update the current observation.
        """
        if not self.enabled or not self._client:
            return
        
        try:
            update_kwargs = {}
            if output is not None:
                update_kwargs["output"] = output
            if metadata:
                update_kwargs["metadata"] = metadata
            if level:
                update_kwargs["level"] = level
            if status_message:
                update_kwargs["status_message"] = status_message
            
            if update_kwargs:
                self._client.update_current_span(**update_kwargs)
        except Exception as e:
            logger.warning(f"Failed to update current span: {e}")
    
    def score_current_trace(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """
        Add a score to the current trace for evaluation.
        
        Args:
            name: Score name (e.g., "relevance", "helpfulness")
            value: Score value (typically 0-1 or 1-5)
            comment: Optional comment explaining the score
        """
        if not self.enabled or not self._client:
            return
        
        try:
            trace_id = self._client.get_current_trace_id()
            if trace_id:
                self._client.score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment
                )
        except Exception as e:
            logger.warning(f"Failed to score trace: {e}")
    
    def add_feedback_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Add a feedback score to a specific trace.

        Args:
            trace_id: The trace ID to score
            name: Score name (e.g., "user-feedback", "helpfulness")
            value: Score value (1 for helpful, 0 for not helpful)
            comment: Optional comment explaining the score
            user_id: Optional user ID who provided the feedback
        """
        if not self.enabled or not self._client:
            return

        try:
            # Langfuse v3 SDK: Use create_score method
            self._client.create_score(
                name=name,
                value=value,
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=comment
            )
            logger.info(f"Added feedback score to trace {trace_id}: {name}={value}")
        except Exception as e:
            logger.error(f"Failed to add feedback score: {e}", exc_info=True)
            raise
    
    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled and self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse events: {e}")
    
    def shutdown(self):
        """Shutdown the Langfuse client gracefully."""
        if self.enabled and self._client:
            try:
                self._client.flush()
                # v3 client doesn't have explicit shutdown, flush is sufficient
            except Exception as e:
                logger.warning(f"Failed to shutdown Langfuse: {e}")


class _NoOpContext:
    """No-op context for when Langfuse is disabled."""
    
    def update(self, **kwargs):
        """No-op update."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# Singleton instance
_langfuse_service: Optional[LangfuseService] = None


def get_langfuse_service() -> LangfuseService:
    """Get the singleton Langfuse service instance."""
    global _langfuse_service
    if _langfuse_service is None:
        _langfuse_service = LangfuseService()
    return _langfuse_service


def trace_llm_call(
    name: str = "llm-call",
    capture_messages: bool = False
):
    """
    Decorator for tracing LLM calls with clean input/output.
    
    This decorator extracts the user message from the messages list
    and uses it as the clean input for Langfuse display.
    
    Args:
        name: Name for the generation trace
        capture_messages: If True, store full messages in metadata (for debugging)
    
    Usage:
        @trace_llm_call("chat-completion")
        async def generate_response(messages, model, **kwargs):
            ...
            return {"content": "...", "model": "...", "usage": {...}}
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = get_langfuse_service()
            
            if not langfuse.enabled:
                return await func(*args, **kwargs)
            
            # Extract messages and find the last user message for clean input
            messages = kwargs.get("messages", [])
            if not messages and args:
                # Try to find messages in positional args
                for arg in args:
                    if isinstance(arg, list) and arg and isinstance(arg[0], dict):
                        messages = arg
                        break
            
            # Get clean user message for input display
            user_message = ""
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            # Get model from kwargs or use default
            model = kwargs.get("model", "unknown")
            
            # Build metadata
            metadata = {}
            if capture_messages:
                metadata["messages_count"] = len(messages)
            
            start_time = time.time()
            
            with langfuse.trace_generation(
                name=name,
                model=model,
                input_text=user_message,  # Clean string input
                metadata=metadata
            ) as generation:
                try:
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Update with clean output
                    if isinstance(result, dict):
                        generation.update(
                            output=result.get("content", ""),  # Clean string output
                            model=result.get("model", model),
                            usage_details={
                                "input": result.get("usage", {}).get("input_tokens", 0),
                                "output": result.get("usage", {}).get("output_tokens", 0),
                            },
                            metadata={
                                "provider": result.get("provider", "unknown"),
                                "cost": result.get("cost", 0),
                                "duration_seconds": round(end_time - start_time, 3),
                            }
                        )
                    
                    return result
                    
                except Exception as e:
                    generation.update(
                        level="ERROR",
                        status_message=str(e)
                    )
                    raise
        
        return wrapper
    return decorator


def trace_retrieval_call(name: str = "retrieval"):
    """
    Decorator for tracing RAG retrieval calls.
    
    Usage:
        @trace_retrieval_call("knowledge-base-search")
        async def retrieve_context(query: str, top_k: int = 5):
            ...
            return [{"content": "...", "source": "..."}, ...]
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = get_langfuse_service()
            
            if not langfuse.enabled:
                return await func(*args, **kwargs)
            
            # Extract query from args/kwargs
            query = kwargs.get("query", "")
            if not query and args:
                query = str(args[0]) if args else ""
            
            with langfuse.trace_retrieval(
                name=name,
                query=query
            ) as retrieval:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Update with result summary
                    if isinstance(result, list):
                        retrieval.update(
                            output=f"Retrieved {len(result)} chunks",
                            metadata={"chunk_count": len(result)}
                        )
                    
                    return result
                    
                except Exception as e:
                    retrieval.update(
                        level="ERROR",
                        status_message=str(e)
                    )
                    raise
        
        return wrapper
    return decorator