"""
Chat API Router
"""
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.chat_service import ChatService
from app.services.agentic_chat_service import AgenticChatService, get_chat_service as get_agentic_chat_service
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.services.compression_service import CompressionService
from app.services.geoip_service import GeoIPService
from app.services.semantic_memory_service import SemanticMemoryService
import os
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def get_compression_service():
    """Get CompressionService instance"""
    return CompressionService()


def get_geoip_service():
    """Get GeoIPService instance"""
    return GeoIPService()


# Dependency injection for services
def get_chat_service():
    """Get legacy ChatService instance"""
    return ChatService()


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


async def extract_and_store_memory(
    clerk_user_id: str,
    session_id: str,
    enable_memory: bool = True
):
    """
    Background task to extract and store semantic memory from conversation.
    
    Also creates episodic memory entries from recent turns to enable
    semantic memory extraction.
    
    Args:
        clerk_user_id: User's Clerk ID
        session_id: Chat session ID
        enable_memory: Whether to enable memory extraction
    """
    if not enable_memory or not clerk_user_id:
        return
    
    try:
        from app.db import db
        
        # Get session to retrieve user ID
        session = await db.chatsession.find_unique(where={"id": session_id})
        if not session or not session.userId:
            logger.warning(f"Session {session_id} not found or missing userId")
            return
        
        # Check if we should create an episodic memory for this turn
        # Count existing episodic memories for this session
        existing_episodes = await db.episodicmemory.count(
            where={"sessionId": session_id}
        )
        
        # Create episodic memory for every 2-3 turns to enable semantic extraction
        if existing_episodes < 5:  # Keep building up episodic memories
            try:
                # Get recent messages from this session
                recent_msgs = await db.chatmessage.find_many(
                    where={"sessionId": session_id},
                    order={"createdAt": "desc"},
                    take=4  # Last 2 turns (user + assistant)
                )
                
                if len(recent_msgs) >= 2:
                    # Create episodic memory from these turns
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
                            "summary": turn_content[:500],  # Truncate if needed
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
        
        # Try to extract semantic memory
        memory_service = SemanticMemoryService()
        should_extract_result = await memory_service.should_extract(clerk_user_id)
        
        # should_extract returns a bool, not Result
        if should_extract_result:
            # Extract and save semantic memory
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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service),
    geoip_service: GeoIPService = Depends(get_geoip_service),
):
    """
    Chat endpoint with RAG support and user profile personalization

    Generates responses using dual LLM with optional context from the SELVE knowledge base
    and user's personality assessment data.
    """
    try:
        # Extract client IP from request headers
        client_ip = geoip_service.extract_client_ip(dict(http_request.headers))
        if not client_ip and http_request.client:
            client_ip = http_request.client.host
        
        # Convert Pydantic messages to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Generate response with user context and geolocation
        result = await chat_service.generate_response(
            message=request.message,
            conversation_history=conversation_history,
            use_rag=request.use_rag,
            clerk_user_id=request.clerk_user_id,
            selve_scores=request.selve_scores,
            assessment_url=request.assessment_url,
            session_id=request.session_id,
            client_ip=client_ip,
        )

        # Save messages to session if session_id provided
        if request.session_id:
            await session_service.add_message(
                session_id=request.session_id,
                role="user",
                content=request.message
            )
            await session_service.add_message(
                session_id=request.session_id,
                role="assistant",
                content=result["response"]
            )

        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    chat_service: AgenticChatService = Depends(get_agent_chat_service),
    session_service: SessionService = Depends(get_session_service),
    compression_service: CompressionService = Depends(get_compression_service),
    geoip_service: GeoIPService = Depends(get_geoip_service),
):
    """
    Streaming chat endpoint with RAG support and user profile personalization

    Returns Server-Sent Events (SSE) stream of response chunks
    """
    try:
        # Extract client IP from request headers
        client_ip = geoip_service.extract_client_ip(dict(http_request.headers))
        if not client_ip and http_request.client:
            client_ip = http_request.client.host
        
        # Convert Pydantic messages to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Save user message to session immediately if session_id provided
        if request.session_id:
            try:
                await session_service.add_message(
                    session_id=request.session_id,
                    role="user",
                    content=request.message
                )
            except Exception as e:
                print(f"⚠️ Error saving user message to session: {e}")

        # Containers to store response and compression state
        class ResponseContainer:
            def __init__(self):
                self.content = ""
                self.compression_needed = False
                self.total_tokens = None
                self.session_data = None

        response_container = ResponseContainer()

        # Generate streaming response with status events
        async def generate():
            async for event in chat_service.chat_stream(
                message=request.message,
                conversation_history=conversation_history,
                clerk_user_id=request.clerk_user_id,
                session_id=request.session_id,
                client_ip=client_ip,
                emit_status=True,
            ):
                # Check if this is a status event (dict) or text chunk (str)
                if isinstance(event, dict):
                    # Status event for thinking UI
                    yield f"data: {json.dumps(event)}\n\n"
                else:
                    # Text chunk from LLM
                    response_container.content += event
                    yield f"data: {json.dumps({'chunk': event})}\n\n"

            # Check if compression is needed (informational only; compression handled in agent post-process)
            if request.session_id:
                try:
                    response_container.compression_needed = await compression_service.should_trigger_compression(request.session_id)
                    # Get session to retrieve total tokens and user ID
                    from app.db import db
                    session = await db.chatsession.find_unique(where={"id": request.session_id})
                    if session:
                        response_container.total_tokens = session.totalTokens
                        response_container.session_data = {
                            "userId": session.userId,
                            "clerkUserId": session.clerkUserId
                        }
                except Exception as e:
                    print(f"⚠️ Error checking compression: {e}")

            # Send completion signal with metadata
            yield f"data: {json.dumps({'done': True, 'compression_needed': response_container.compression_needed, 'total_tokens': response_container.total_tokens})}\n\n"

        # Background task to save assistant message after stream completes
        async def save_assistant_message():
            # Wait a moment to ensure streaming is complete
            await asyncio.sleep(0.5)
            if request.session_id and response_container.content:
                await session_service.add_message(
                    session_id=request.session_id,
                    role="assistant",
                    content=response_container.content
                )

        # Register background tasks
        if request.session_id:
            background_tasks.add_task(save_assistant_message)
            # Also trigger memory extraction
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
        print(f"❌ Chat stream error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating streaming response: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """
    Health check endpoint

    Verifies that all services (Qdrant, OpenAI) are operational.
    """
    try:
        # Check Qdrant connection
        collection_info = rag_service.qdrant.get_collection(rag_service.collection_name)
        qdrant_connected = True
        collection_points = collection_info.points_count
    except Exception as e:
        qdrant_connected = False
        collection_points = 0

    # Check OpenAI API key
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
    """
    Test endpoint to retrieve RAG context without generating a response

    Useful for debugging and testing the retrieval system.
    """
    try:
        context_info = rag_service.get_context_for_query(query, top_k)
        return context_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving context: {str(e)}"
        )
