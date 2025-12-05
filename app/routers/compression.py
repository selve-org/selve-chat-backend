"""
Compression API Router
Endpoints for managing conversation compression
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.compression_service import CompressionService


router = APIRouter(prefix="/api/compression", tags=["compression"])


class CompressionRequest(BaseModel):
    """Request to compress a conversation"""
    session_id: str
    user_id: str
    clerk_user_id: str


class CompressionResponse(BaseModel):
    """Response from compression operation"""
    compressed: bool
    episodic_memory_id: str | None = None
    messages_compressed: int | None = None
    messages_kept: int | None = None
    tokens_saved: int | None = None
    compression_ratio: float | None = None
    summary: str | None = None
    title: str | None = None
    error: str | None = None


class CompressionCheckResponse(BaseModel):
    """Response for compression check"""
    needs_compression: bool
    total_tokens: int | None = None
    threshold: int | None = None


compression_service = CompressionService()


@router.post("/compress", response_model=CompressionResponse)
async def compress_conversation(request: CompressionRequest):
    """
    Compress a conversation session

    Compresses oldest 70% of messages into episodic memory,
    keeping most recent 30% uncompressed for context.
    """
    try:
        result = await compression_service.compress_conversation(
            session_id=request.session_id,
            user_id=request.user_id,
            clerk_user_id=request.clerk_user_id
        )

        return CompressionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compression failed: {str(e)}"
        )


@router.get("/check/{session_id}", response_model=CompressionCheckResponse)
async def check_compression_needed(session_id: str):
    """
    Check if a session needs compression

    Returns whether conversation has reached 70% of context window.
    """
    try:
        needs_compression = await compression_service.should_trigger_compression(session_id)

        return CompressionCheckResponse(
            needs_compression=needs_compression
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compression check failed: {str(e)}"
        )


@router.get("/memories/{session_id}")
async def get_session_memories(session_id: str):
    """
    Get all episodic memories for a session

    Returns compressed conversation summaries.
    """
    try:
        memories = await compression_service.get_session_memories(session_id)

        return {
            "session_id": session_id,
            "memory_count": len(memories),
            "memories": memories
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}"
        )
