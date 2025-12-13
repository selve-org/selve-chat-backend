"""
Sessions API Router
Manages chat sessions - create, retrieve, list, update
"""
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List, Optional
from app.services.session_service import SessionService
from app.services.title_service import TitleService

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


# Pydantic models
class CreateSessionRequest(BaseModel):
    userId: str
    clerkUserId: str
    title: Optional[str] = None


class GenerateTitleRequest(BaseModel):
    message: str
    assistant_response: Optional[str] = None


class SessionSummaryResponse(BaseModel):
    """Session summary without user IDs (for list endpoint)"""
    id: str
    title: str
    status: str
    totalTokens: Optional[int] = 0
    createdAt: str
    lastMessageAt: str


class SessionDetailResponse(BaseModel):
    """Full session details with user IDs"""
    id: str
    userId: str
    clerkUserId: str
    title: str
    status: str
    totalTokens: Optional[int] = 0
    createdAt: str
    lastMessageAt: str


class SessionWithMessagesResponse(SessionDetailResponse):
    messages: List[dict]


class UpdateSessionTitleRequest(BaseModel):
    title: str


# Dependency injection
def get_session_service():
    """Get SessionService instance"""
    return SessionService()


def get_title_service():
    """Get TitleService instance"""
    return TitleService()


@router.post("/", response_model=SessionDetailResponse)
async def create_session(
    request: CreateSessionRequest,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Create a new chat session

    Args:
        request: Session creation data (userId, clerkUserId, optional title)

    Returns:
        Created session data
    """
    try:
        session = await session_service.create_session(
            user_id=request.userId,
            clerk_user_id=request.clerkUserId,
            title=request.title
        )
        return SessionDetailResponse(**session)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating session: {str(e)}"
        )


@router.get("/{session_id}", response_model=SessionWithMessagesResponse)
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Get session by ID with all messages

    Args:
        session_id: Session ID to retrieve

    Returns:
        Session data with messages
    """
    try:
        session = await session_service.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        return SessionWithMessagesResponse(**session)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving session: {str(e)}"
        )


@router.get("/user/{clerk_user_id}", response_model=List[SessionSummaryResponse])
async def list_user_sessions(
    clerk_user_id: str,
    limit: int = 50,
    status_filter: str = "active",
    session_service: SessionService = Depends(get_session_service)
):
    """
    List all sessions for a user

    Args:
        clerk_user_id: Clerk user ID
        limit: Maximum number of sessions to return
        status_filter: Session status filter (default: "active")

    Returns:
        List of session summaries (without messages)
    """
    try:
        sessions = await session_service.list_sessions(
            clerk_user_id=clerk_user_id,
            limit=limit,
            status=status_filter
        )

        return [SessionSummaryResponse(**session) for session in sessions]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing sessions: {str(e)}"
        )


@router.patch("/{session_id}/title", response_model=SessionDetailResponse)
async def update_session_title(
    session_id: str,
    request: UpdateSessionTitleRequest,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Update session title

    Args:
        session_id: Session ID to update
        request: New title

    Returns:
        Updated session data
    """
    try:
        session = await session_service.update_session_title(
            session_id=session_id,
            title=request.title
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        return SessionDetailResponse(**session)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating session title: {str(e)}"
        )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Archive (soft delete) a session

    Args:
        session_id: Session ID to archive

    Returns:
        204 No Content on success
    """
    try:
        success = await session_service.archive_session(session_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error archiving session: {str(e)}"
        )


@router.post("/{session_id}/generate-title", response_model=SessionDetailResponse)
async def generate_and_update_title(
    session_id: str,
    request: GenerateTitleRequest,
    title_service: TitleService = Depends(get_title_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Auto-generate a title for the conversation based on the first message

    Args:
        session_id: Session ID to update
        request: Contains the first user message

    Returns:
        Session details (title update runs in background)
    """
    try:
        # Ensure the session exists before starting background work
        session = await session_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Set a lightweight placeholder so the UI can show a loading state (similar to Google's blinking stub)
        if session.get("title") in (None, "", "New Conversation"):
            placeholder_title = "Generating title..."
            await session_service.update_session_title(session_id=session_id, title=placeholder_title)
            session["title"] = placeholder_title

        async def _generate_and_save_title():
            try:
                title = await title_service.generate_title(
                    request.message,
                    request.assistant_response,
                )
                await session_service.update_session_title(
                    session_id=session_id,
                    title=title
                )
            except Exception as e:
                logger.warning(f"Background title generation failed for {session_id}: {e}")

        # Fire and forget - run in background
        import asyncio
        asyncio.create_task(_generate_and_save_title())

        # Return current session state immediately (with placeholder if applied)
        return SessionDetailResponse(**session)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating title: {str(e)}"
        )
