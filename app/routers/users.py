"""
User API Router - Fetch user profiles and SELVE scores
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional
from app.services.user_profile_service import UserProfileService

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/{clerk_user_id}/scores")
async def get_user_scores(clerk_user_id: str):
    """
    Get user's SELVE personality scores

    Fetches the current assessment result for a user from the database.

    Args:
        clerk_user_id: Clerk user ID

    Returns:
        User scores and profile information
    """
    service = UserProfileService()

    try:
        scores = await service.get_user_scores(clerk_user_id)

        if not scores:
            return {
                "has_scores": False,
                "message": "No assessment results found for this user"
            }

        return {
            "has_scores": True,
            **scores
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user scores: {str(e)}"
        )
