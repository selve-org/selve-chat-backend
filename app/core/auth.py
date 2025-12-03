"""
Clerk Authentication and Toll Gate Implementation
Verifies users exist in main SELVE database before allowing chatbot access
"""

from fastapi import HTTPException, Header, Depends
from jose import jwt, JWTError
import httpx
import os
from typing import Dict

# Environment variables
CLERK_PUBLIC_KEY = os.getenv("CLERK_PUBLIC_KEY")
MAIN_SELVE_API_URL = os.getenv("MAIN_SELVE_API_URL", "http://localhost:8000")
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")


async def verify_clerk_token(authorization: str = Header(None)) -> Dict:
    """
    Step 1: Verify Clerk JWT token is valid

    Args:
        authorization: Bearer token from request header

    Returns:
        JWT payload containing user information

    Raises:
        HTTPException: If token is missing or invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header"
        )

    if not CLERK_PUBLIC_KEY:
        raise HTTPException(
            status_code=500,
            detail="Clerk public key not configured"
        )

    try:
        # Remove "Bearer " prefix
        token = authorization.replace("Bearer ", "")

        # Verify JWT signature and decode payload
        payload = jwt.decode(token, CLERK_PUBLIC_KEY, algorithms=["RS256"])

        return payload

    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}"
        )


async def verify_user_in_main_db(
    clerk_payload: Dict = Depends(verify_clerk_token)
) -> Dict:
    """
    Step 2: Toll Gate - Verify user exists in main SELVE database

    This prevents random Clerk users from accessing the chatbot.
    Only users who have registered with the main SELVE app can use the chatbot.

    Args:
        clerk_payload: Decoded JWT payload from Clerk

    Returns:
        User data including clerk_user_id, user_id, email, has_assessment

    Raises:
        HTTPException: If user not found, API call fails, or secret missing
    """
    clerk_user_id = clerk_payload.get("sub")

    if not clerk_user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid token: missing subject"
        )

    if not INTERNAL_API_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Internal API secret not configured"
        )

    # Call main SELVE API to check if user exists
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{MAIN_SELVE_API_URL}/api/users/verify/{clerk_user_id}",
                headers={"X-Internal-Secret": INTERNAL_API_SECRET},
                timeout=5.0
            )

            # User not found in main SELVE database
            if response.status_code == 404:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied. Please complete your SELVE assessment first at selve.me"
                )

            # Other API error
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to verify user"
                )

            user_data = response.json()

            return {
                "clerk_user_id": clerk_user_id,
                "user_id": user_data["id"],
                "email": user_data.get("email"),
                "has_assessment": user_data.get("has_completed_assessment", False)
            }

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=500,
                detail="User verification timeout"
            )

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"User verification failed: {str(e)}"
            )


async def get_current_user(
    user: Dict = Depends(verify_user_in_main_db)
) -> Dict:
    """
    Complete authentication flow for protected routes

    This dependency combines:
    1. Clerk JWT token verification
    2. Toll gate check (user exists in main SELVE database)

    Use this as a dependency in protected endpoints:

    ```python
    @router.post("/api/chat/message")
    async def send_message(
        message: str,
        user: dict = Depends(get_current_user)
    ):
        user_id = user["user_id"]
        clerk_user_id = user["clerk_user_id"]
        # ... rest of endpoint logic
    ```

    Args:
        user: User data from toll gate verification

    Returns:
        User data dictionary
    """
    return user
