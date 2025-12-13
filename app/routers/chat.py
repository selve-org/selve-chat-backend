"""
Chat API Router
"""
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.agentic_chat_service import AgenticChatService, get_chat_service as get_agentic_chat_service
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.services.compression_service import CompressionService
from app.services.geoip_service import GeoIPService
from app.services.semantic_memory_service import SemanticMemoryService
from app.services.security_service import SecurityService
from slowapi import Limiter
from slowapi.util import get_remote_address
import os
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# Get limiter from app state (will be injected at runtime)
limiter = Limiter(key_func=get_remote_address)


def get_compression_service():
    """Get CompressionService instance"""
    return CompressionService()


def get_geoip_service():
    """Get GeoIPService instance"""
    return GeoIPService()


# Dependency injection for services
def get_agent_chat_service():
    """Get AgenticChatService instance"""
    return get_agentic_chat_service()


def get_rag_service():
    """Get RAGService instance"""
    return RAGService()


def get_session_service():
    """Get SessionService instance"""
    return SessionService()


def get_semantic_memory_service():
    """Get SemanticMemoryService instance"""
    return SemanticMemoryService()


def get_security_service():
    """Get SecurityService instance"""
    return SecurityService()


async def extract_and_store_memory(
    clerk_user_id: str,
    session_id: str,
    enable_memory: bool = True
):
    """
    Background task to extract and store semantic memory from conversation.
    """
    if not enable_memory or not clerk_user_id:
        return
    
    try:
        from app.db import db
        
        session = await db.chatsession.find_unique(where={"id": session_id})
        if not session or not session.userId:
            logger.warning(f"Session {session_id} not found or missing userId")
            return
        
        existing_episodes = await db.episodicmemory.count(
            where={"sessionId": session_id}
        )
        
        if existing_episodes < 5:
            try:
                recent_msgs = await db.chatmessage.find_many(
                    where={"sessionId": session_id},
                    order={"createdAt": "desc"},
                    take=4
                )
                
                if len(recent_msgs) >= 2:
                    turn_content = "\n".join(
                        f"{msg.role.upper()}: {msg.content}"
                        for msg in reversed(recent_msgs)
                    )
                    
                    from prisma import Json
                    await db.episodicmemory.create(
                        data={
                            "userId": session.userId,
                            "sessionId": session_id,
                            "title": f"Conversation Turn {existing_episodes + 1}",
                            "summary": turn_content[:500],
                            "keyInsights": Json([]),
                            "unresolvedTopics": Json([]),
                            "emotionalState": "engaged",
                            "sourceMessageIds": [msg.id for msg in recent_msgs],
                            "compressionModel": "chat-turn",
                            "compressionCost": 0.0,
                            "spanStart": recent_msgs[-1].createdAt,
                            "spanEnd": recent_msgs[0].createdAt,
                            "embedded": False
                        }
                    )
                    logger.info(f"✓ Created episodic memory for session {session_id}")
            except Exception as e:
                logger.debug(f"Could not create episodic memory: {e}")
        
        memory_service = SemanticMemoryService()
        should_extract_result = await memory_service.should_extract(clerk_user_id)
        
        if should_extract_result:
            extract_result = await memory_service.extract_and_save(
                clerk_user_id=clerk_user_id,
                user_id=session.userId
            )
            if extract_result.is_success:
                logger.info(f"✓ Semantic memory extracted: {extract_result.data}")
            else:
                logger.debug(f"Memory extraction failed: {extract_result.error}")
        else:
            ep_count = await db.episodicmemory.count()
            logger.debug(f"Not yet time for semantic memory extraction ({ep_count} total episodes)")
    except Exception as e:
        logger.error(f"Error extracting memory for session {session_id}: {e}", exc_info=True)


@router.post("/chat/stream")
@limiter.limit("50/hour")  # Per-IP limit: 50 chat requests per hour
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    chat_service: AgenticChatService = Depends(get_agent_chat_service),
    session_service: SessionService = Depends(get_session_service),
    compression_service: CompressionService = Depends(get_compression_service),
    geoip_service: GeoIPService = Depends(get_geoip_service),
    security_service: SecurityService = Depends(get_security_service),
):
    """
    Streaming chat endpoint with RAG support and user profile personalization.
    Returns Server-Sent Events (SSE) stream of response chunks.

    Rate limit: 50 requests/hour per IP to prevent API credit exhaustion.
    """
    try:
        # Extract client IP
        client_ip = geoip_service.extract_client_ip(dict(http_request.headers))
        if not client_ip and http_request.client:
            client_ip = http_request.client.host
        
        # Security check: detect prompt injection attempts
        security_check = await security_service.check_message(
            message=request.message,
            user_id=getattr(request, 'user_id', None),
            clerk_user_id=request.clerk_user_id,
            session_id=request.session_id,
            ip_address=client_ip
        )

        # Handle banned users - return immediately with ban message
        if not security_check["is_safe"] and security_check["is_banned"]:
            ban_msg = security_check["message"]
            ban_expires = security_check.get("ban_expires_at")
            
            async def banned_response():
                # Send ban notification
                yield f"data: {json.dumps({'type': 'ban', 'message': ban_msg, 'expires_at': ban_expires.isoformat() if ban_expires else None})}\n\n"
                # Send content so frontend can finalize
                yield f"data: {json.dumps({'chunk': ban_msg})}\n\n"
                # Signal completion
                yield f"data: {json.dumps({'done': True})}\n\n"
                yield f"data: [DONE]\n\n"
            
            return StreamingResponse(
                banned_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
        
        # Handle security warnings (not banned, but suspicious)
        # IMPORTANT: For warnings, we still process the message but notify the user
        security_warning = None
        if not security_check["is_safe"] and not security_check["is_banned"]:
            security_warning = {
                "type": "warning",
                "message": security_check["message"],
                "incident_count": security_check.get("incident_count", 0)
            }
            logger.warning(f"Security warning for user {request.clerk_user_id}: {security_check['message']}")
        
        # Convert conversation history
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Save user message to session
        if request.session_id:
            try:
                await session_service.add_message(
                    session_id=request.session_id,
                    role="user",
                    content=request.message
                )
            except Exception as e:
                logger.warning(f"Error saving user message to session: {e}")

        # Response container for background task
        class ResponseContainer:
            def __init__(self):
                self.content = ""
                self.compression_needed = False
                self.total_tokens = None

        response_container = ResponseContainer()

        async def generate():
            # Send security warning first if there is one
            if security_warning:
                yield f"data: {json.dumps(security_warning)}\n\n"
            
            # Stream the actual response
            try:
                async for event in chat_service.chat_stream(
                    message=request.message,
                    conversation_history=conversation_history,
                    clerk_user_id=request.clerk_user_id,
                    session_id=request.session_id,
                    client_ip=client_ip,
                    emit_status=True,
                ):
                    if isinstance(event, dict):
                        yield f"data: {json.dumps(event)}\n\n"
                    else:
                        response_container.content += event
                        yield f"data: {json.dumps({'chunk': event})}\n\n"
            except Exception as e:
                logger.error(f"Error in chat stream: {e}")
                error_msg = "I apologize, but I encountered an issue processing your request. Please try again."
                response_container.content = error_msg
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                yield f"data: {json.dumps({'chunk': error_msg})}\n\n"

            # Check compression status
            if request.session_id:
                try:
                    response_container.compression_needed = await compression_service.should_trigger_compression(request.session_id)
                    from app.db import db
                    session = await db.chatsession.find_unique(where={"id": request.session_id})
                    if session:
                        response_container.total_tokens = session.totalTokens
                except Exception as e:
                    logger.warning(f"Error checking compression: {e}")

            # Send completion signal
            yield f"data: {json.dumps({'done': True, 'compression_needed': response_container.compression_needed, 'total_tokens': response_container.total_tokens})}\n\n"
            yield f"data: [DONE]\n\n"

        # Background task to save assistant message
        async def save_assistant_message():
            await asyncio.sleep(0.5)
            if request.session_id and response_container.content:
                try:
                    await session_service.add_message(
                        session_id=request.session_id,
                        role="assistant",
                        content=response_container.content
                    )
                except Exception as e:
                    logger.error(f"Error saving assistant message: {e}")

        # Register background tasks
        if request.session_id:
            background_tasks.add_task(save_assistant_message)
            enable_memory = os.getenv("ENABLE_SEMANTIC_MEMORY", "true").lower() == "true"
            background_tasks.add_task(
                extract_and_store_memory,
                clerk_user_id=request.clerk_user_id,
                session_id=request.session_id,
                enable_memory=enable_memory
            )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        import traceback
        logger.error(f"Chat stream error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating streaming response: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """Health check endpoint"""
    try:
        collection_info = rag_service.qdrant.get_collection(rag_service.collection_name)
        qdrant_connected = True
        collection_points = collection_info.points_count
    except Exception as e:
        qdrant_connected = False
        collection_points = 0

    openai_configured = bool(os.getenv("OPENAI_API_KEY"))

    return HealthResponse(
        status="healthy" if qdrant_connected and openai_configured else "degraded",
        qdrant_connected=qdrant_connected,
        collection_points=collection_points,
        services={
            "qdrant": qdrant_connected,
            "openai": openai_configured
        }
    )


@router.get("/context")
async def get_context(
    query: str,
    top_k: int = 3,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Test endpoint to retrieve RAG context without generating a response"""
    try:
        context_info = rag_service.get_context_for_query(query, top_k)
        return context_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving context: {str(e)}"
        )