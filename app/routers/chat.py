"""
Chat API Router
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.models.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
import os

router = APIRouter(prefix="/api", tags=["chat"])


# Dependency injection for services
def get_chat_service():
    """Get ChatService instance"""
    return ChatService()


def get_rag_service():
    """Get RAGService instance"""
    return RAGService()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Chat endpoint with RAG support

    Generates responses using OpenAI with optional context from the SELVE knowledge base.
    """
    try:
        # Convert Pydantic messages to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Generate response
        result = chat_service.generate_response(
            message=request.message,
            conversation_history=conversation_history,
            use_rag=request.use_rag
        )

        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
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
