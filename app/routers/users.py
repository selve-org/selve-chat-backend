"""
User API Router - Fetch user profiles and SELVE scores
"""
from fastapi import APIRouter, HTTPException, status
from app.services.user_profile_service import UserProfileService
from app.services.usage_service import usage_service

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


@router.get("/{clerk_user_id}")
async def get_user_details(clerk_user_id: str):
    """
    Get user account details for UI surfaces

    Args:
        clerk_user_id: Clerk user ID

    Returns:
        Basic user info and inferred subscription plan
    """
    service = UserProfileService()

    try:
        user = await service.get_user_account(clerk_user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return user

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user details: {str(e)}"
        )


@router.get("/{clerk_user_id}/usage")
async def get_user_usage(clerk_user_id: str):
    """
    Get user's chatbot usage and limits

    Fetches current usage in the 24-hour period, subscription plan,
    and whether the user can send messages.

    Args:
        clerk_user_id: Clerk user ID

    Returns:
        {
            "subscription_plan": "free" | "pro",
            "current_period": {
                "start": str (ISO),
                "end": str (ISO),
                "total_cost": float,
                "message_count": int,
                "percentage_used": float,
                "limit": float | null
            },
            "can_send_message": bool,
            "time_until_reset": str
        }
    """
    try:
        # Get current usage
        usage = await usage_service.get_current_usage(clerk_user_id)

        subscription_plan = usage["subscription_plan"]

        # Check if user can send message
        limit_check = await usage_service.check_usage_limit(
            clerk_user_id,
            subscription_plan
        )

        return {
            "subscription_plan": subscription_plan,
            "current_period": {
                "start": usage["period_start"],
                "end": usage["period_end"],
                "total_cost": usage["total_cost"],
                "message_count": usage["message_count"],
                "percentage_used": usage["percentage_used"],
                "limit": usage["limit"]
            },
            "can_send_message": limit_check["can_send"],
            "limit_exceeded": limit_check["limit_exceeded"],
            "time_until_reset": usage["time_until_reset"]
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching usage: {str(e)}"
        )
