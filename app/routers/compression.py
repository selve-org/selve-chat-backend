"""
Compression API Router
Endpoints for managing conversation compression and memory search
"""
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.services.compression_service import CompressionService
from app.services.memory_search_service import MemorySearchService


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
memory_search_service = MemorySearchService()


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


class MemorySearchRequest(BaseModel):
    """Request to search memories"""
    query: str = Field(..., description="Search query", min_length=1)
    user_id: Optional[str] = Field(None, description="Internal user ID to filter by")
    clerk_user_id: Optional[str] = Field(None, description="Clerk user ID to filter by")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    score_threshold: float = Field(0.5, description="Minimum similarity score", ge=0.0, le=1.0)


class MemorySearchResponse(BaseModel):
    """Response from memory search"""
    query: str
    result_count: int
    memories: List[Dict[str, Any]]


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(request: MemorySearchRequest):
    """
    Search episodic memories using vector similarity

    Uses semantic search to find relevant past conversations based on query.
    """
    try:
        memories = await memory_search_service.search_memories(
            query=request.query,
            user_id=request.user_id,
            clerk_user_id=request.clerk_user_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )

        return MemorySearchResponse(
            query=request.query,
            result_count=len(memories),
            memories=memories
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory search failed: {str(e)}"
        )


@router.get("/search")
async def search_memories_get(
    query: str = Query(..., description="Search query", min_length=1),
    user_id: Optional[str] = Query(None, description="Internal user ID to filter by"),
    clerk_user_id: Optional[str] = Query(None, description="Clerk user ID to filter by"),
    top_k: int = Query(5, description="Number of results", ge=1, le=20),
    score_threshold: float = Query(0.5, description="Minimum similarity score", ge=0.0, le=1.0)
):
    """
    Search episodic memories using vector similarity (GET endpoint)

    Uses semantic search to find relevant past conversations based on query.
    """
    try:
        memories = await memory_search_service.search_memories(
            query=query,
            user_id=user_id,
            clerk_user_id=clerk_user_id,
            top_k=top_k,
            score_threshold=score_threshold
        )

        return {
            "query": query,
            "result_count": len(memories),
            "memories": memories
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory search failed: {str(e)}"
        )


@router.get("/related/{memory_id}")
async def get_related_memories(
    memory_id: str,
    top_k: int = Query(3, description="Number of similar memories", ge=1, le=10)
):
    """
    Find memories similar to a given memory

    Returns memories with similar content or themes.
    """
    try:
        related = await memory_search_service.get_related_memories(
            memory_id=memory_id,
            top_k=top_k
        )

        return {
            "memory_id": memory_id,
            "result_count": len(related),
            "related_memories": related
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find related memories: {str(e)}"
        )
