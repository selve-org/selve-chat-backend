"""
Langfuse Service for LLM Observability
Provides tracing, cost tracking, and prompt management.
"""
import os
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from functools import wraps
import time


class LangfuseService:
    """
    Service for Langfuse LLM observability and tracing.
    
    Provides:
    - Trace creation and management
    - Span tracking for LLM calls
    - Cost tracking
    - Prompt versioning (optional)
    """
    
    def __init__(self):
        self.enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
        self.langfuse = None
        
        if self.enabled:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Langfuse client."""
        try:
            from langfuse import Langfuse
            
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if not secret_key or not public_key:
                print("⚠️ Langfuse keys not configured, tracing disabled")
                self.enabled = False
                return
            
            self.langfuse = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host
            )
            print("✅ Langfuse tracing initialized")
            
        except ImportError:
            print("⚠️ Langfuse package not installed, tracing disabled")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ Langfuse initialization failed: {e}")
            self.enabled = False
    
    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Create a new trace for tracking a complete request.
        
        Args:
            name: Trace name (e.g., "chat_request")
            user_id: User identifier (clerk_user_id)
            session_id: Session identifier
            metadata: Additional metadata
            tags: Tags for filtering
            
        Returns:
            Trace object or None if disabled
        """
        if not self.enabled or not self.langfuse:
            return None
        
        try:
            return self.langfuse.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            )
        except Exception as e:
            print(f"⚠️ Failed to create trace: {e}")
            return None
    
    def create_generation(
        self,
        trace,
        name: str,
        model: str,
        input_messages: List[Dict[str, str]],
        output: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """
        Create a generation (LLM call) within a trace.
        
        Args:
            trace: Parent trace object
            name: Generation name
            model: Model used
            input_messages: Input messages
            output: Output text
            usage: Token usage dict
            metadata: Additional metadata
            start_time: Start timestamp
            end_time: End timestamp
        """
        if not self.enabled or not trace:
            return None
        
        try:
            return trace.generation(
                name=name,
                model=model,
                input=input_messages,
                output=output,
                usage=usage,
                metadata=metadata or {},
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            print(f"⚠️ Failed to create generation: {e}")
            return None
    
    def create_span(
        self,
        trace,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """
        Create a span within a trace for non-LLM operations.
        
        Args:
            trace: Parent trace object
            name: Span name (e.g., "rag_retrieval")
            input: Input data
            output: Output data
            metadata: Additional metadata
        """
        if not self.enabled or not trace:
            return None
        
        try:
            return trace.span(
                name=name,
                input=input,
                output=output,
                metadata=metadata or {},
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            print(f"⚠️ Failed to create span: {e}")
            return None
    
    @contextmanager
    def trace_context(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing a complete operation.
        
        Usage:
            with langfuse.trace_context("chat_request", user_id="123") as trace:
                # Do work
                trace.span(name="retrieval", ...)
        """
        trace = self.create_trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        try:
            yield trace
        finally:
            if trace and self.enabled:
                try:
                    self.langfuse.flush()
                except Exception:
                    pass
    
    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                print(f"⚠️ Failed to flush Langfuse events: {e}")
    
    def shutdown(self):
        """Shutdown the Langfuse client gracefully."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.shutdown()
            except Exception as e:
                print(f"⚠️ Failed to shutdown Langfuse: {e}")


# Singleton instance
_langfuse_service: Optional[LangfuseService] = None


def get_langfuse_service() -> LangfuseService:
    """Get the singleton Langfuse service instance."""
    global _langfuse_service
    if _langfuse_service is None:
        _langfuse_service = LangfuseService()
    return _langfuse_service


def trace_llm_call(name: str = "llm_call"):
    """
    Decorator for tracing LLM calls.
    
    Usage:
        @trace_llm_call("chat_completion")
        async def generate_response(messages, model):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            langfuse = get_langfuse_service()
            
            if not langfuse.enabled:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            trace = langfuse.create_trace(
                name=name,
                metadata={"function": func.__name__}
            )
            
            try:
                result = await func(*args, **kwargs)
                end_time = time.time()
                
                # Try to extract usage info from result
                if isinstance(result, dict):
                    langfuse.create_generation(
                        trace=trace,
                        name=name,
                        model=result.get("model", "unknown"),
                        input_messages=kwargs.get("messages", []),
                        output=result.get("content"),
                        usage=result.get("usage"),
                        start_time=start_time,
                        end_time=end_time
                    )
                
                return result
            except Exception as e:
                if trace:
                    trace.update(metadata={"error": str(e)})
                raise
            finally:
                langfuse.flush()
        
        return wrapper
    return decorator
