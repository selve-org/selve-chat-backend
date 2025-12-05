"""
Chat API Router
"""
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.services.compression_service import CompressionService
import os
import json
import asyncio

router = APIRouter(prefix="/api", tags=["chat"])


def get_compression_service():
    """Get CompressionService instance"""
    return CompressionService()


# Dependency injection for services
def get_chat_service():
    """Get ChatService instance"""
    return ChatService()


def get_rag_service():
    """Get RAGService instance"""
    return RAGService()


def get_session_service():
    """Get SessionService instance"""
    return SessionService()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Chat endpoint with RAG support and user profile personalization

    Generates responses using dual LLM with optional context from the SELVE knowledge base
    and user's personality assessment data.
    """
    try:
        # Convert Pydantic messages to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Generate response with user context
        result = await chat_service.generate_response(
            message=request.message,
            conversation_history=conversation_history,
            use_rag=request.use_rag,
            clerk_user_id=request.clerk_user_id
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
    background_tasks: BackgroundTasks,
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service),
    compression_service: CompressionService = Depends(get_compression_service)
):
    """
    Streaming chat endpoint with RAG support and user profile personalization

    Returns Server-Sent Events (SSE) stream of response chunks
    """
    try:
        # Convert Pydantic messages to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Save user message to session immediately if session_id provided
        if request.session_id:
            await session_service.add_message(
                session_id=request.session_id,
                role="user",
                content=request.message
            )

        # Container to store accumulated response
        class ResponseContainer:
            def __init__(self):
                self.content = ""

        response_container = ResponseContainer()

        # Generate streaming response
        async def generate():
            async for chunk in chat_service.generate_response_stream(
                message=request.message,
                conversation_history=conversation_history,
                use_rag=request.use_rag,
                clerk_user_id=request.clerk_user_id,
                selve_scores=request.selve_scores
            ):
                response_container.content += chunk
                # Send as Server-Sent Event format
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Check if compression is needed
            compression_needed = False
            total_tokens = None
            if request.session_id:
                try:
                    compression_needed = await compression_service.should_trigger_compression(request.session_id)
                    # Get session to retrieve total tokens
                    from app.db import db
                    session = await db.chatsession.find_unique(where={"id": request.session_id})
                    if session:
                        total_tokens = session.totalTokens
                except Exception as e:
                    print(f"⚠️ Error checking compression: {e}")

            # Send completion signal with metadata
            yield f"data: {json.dumps({'done': True, 'compression_needed': compression_needed, 'total_tokens': total_tokens})}\n\n"

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

        # Register background task
        if request.session_id:
            background_tasks.add_task(save_assistant_message)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
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
