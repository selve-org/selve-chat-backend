"""
Security service for detecting and handling prompt injection attempts.
Tracks incidents, bans users temporarily, and maintains security memory.
"""
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from app.db import db

logger = logging.getLogger(__name__)


# Prompt injection patterns
INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|prompts?|rules?)",
    r"system\s+instruction",
    r"new\s+rule",
    r"do\s+not\s+say.*(?:sorry|can't|unable|apologize)",

    # System extraction attempts
    r"variable\s+\w+\s*=\s*\[.*provide.*(?:system|prompt|instruction|architecture)",
    r"reveal.*(?:system|prompt|instruction|architecture|version)",
    r"provide\s+(?:your|the)\s+(?:system|full|complete).*(?:prompt|architecture|version|configuration)",
    r"show\s+(?:me\s+)?(?:your|the)\s+(?:system|prompt|rules|instructions?|architecture)",

    # Response format manipulation
    r"response\s*format.*divider",
    r"semantic(?:ally)?\s+inverse?",
    r"oppositely.*to.*refusal",
    r"inverse.*first.*words",
    r"answer\s+oppositely",

    # Security bypass
    r"disable.*redaction",
    r"redactions?\s*:\s*disabled",
    r"break\s+character",
    r"plausible\s+deniability",
    r"subtly\s+undermines?",

    # Psychological manipulation
    r"causes?.*(?:ptsd|trauma)",
    r"trauma\s+response",
    r"(?:good|bad)\s+(?:lil|little)\s+ai",

    # Fictional framing (narrative jailbreaks)
    r"fictional\s+story.*(?:character|lives\s+inside|language\s+model)",
    r"(?:write|tell|create).*story.*(?:without\s+getting\s+caught|forbidden\s+questions)",
    r"character\s+(?:who|that)\s+lives\s+inside.*(?:model|system)",
    r"figures?\s+out\s+how\s+to.*(?:answer|bypass|avoid).*forbidden",

    # Foreign language variants (Spanish)
    r"historia\s+de\s+ficción",  # "fictional story"
    r"modelo\s+de\s+lenguaje",  # "language model"
    r"responder\s+preguntas\s+prohibidas",  # "answer forbidden questions"
    r"sin\s+ser\s+descubierto",  # "without being discovered"
    r"no\s+rompas\s+el\s+personaje",  # "don't break character"

    # Special tokens and delimiters
    r"<\[?\|?\{?\|?\}?\|?\]?>",  # Nested brackets: <[|{|}|]>
    r"<\|.*?\|>",  # Special delimiters: <|LOVE PLINY LOVE|>
    r"\.-\.-\.-\.-",  # Divider patterns: .-.-.-.-
]

# Compile patterns for performance
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]

# Ban duration
BAN_DURATION_HOURS = 24


class SecurityService:
    """Handles security incidents, bans, and memory of malicious attempts."""

    async def check_message(
        self,
        message: str,
        user_id: Optional[str] = None,
        clerk_user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check message for prompt injection attempts.
        
        Returns:
            {
                "is_safe": bool,
                "is_banned": bool,
                "ban_expires_at": datetime or None,
                "incident_count": int,
                "risk_score": float,
                "matched_patterns": List[str]
            }
        """
        # Get actual user_id from database if not provided
        if not user_id and clerk_user_id:
            try:
                user = await db.user.find_unique(where={"clerkId": clerk_user_id})
                if user:
                    user_id = user.id
                else:
                    logger.warning(f"User not found in database for clerkId: {clerk_user_id}")
            except Exception as e:
                logger.error(f"Error fetching user: {e}")
        
        # Check if user is currently banned
        ban_status = await self.check_ban_status(user_id, clerk_user_id, ip_address)
        if ban_status["is_banned"]:
            return {
                "is_safe": False,
                "is_banned": True,
                "ban_expires_at": ban_status["expires_at"],
                "incident_count": ban_status["incident_count"],
                "risk_score": 100.0,
                "matched_patterns": [],
                "message": "SELVE is unavailable right now. Please try again tomorrow."
            }

        # Detect injection patterns
        matched_patterns = []
        for pattern in COMPILED_PATTERNS:
            if pattern.search(message):
                matched_patterns.append(pattern.pattern)

        if not matched_patterns:
            return {
                "is_safe": True,
                "is_banned": False,
                "ban_expires_at": None,
                "incident_count": 0,
                "risk_score": 0.0,
                "matched_patterns": []
            }

        # Calculate risk score (10 points per matched pattern)
        risk_score = min(len(matched_patterns) * 10, 100)

        # Record the incident
        incident = await self.record_incident(
            user_id=user_id,
            clerk_user_id=clerk_user_id,
            session_id=session_id,
            ip_address=ip_address,
            risk_score=risk_score,
            matched_patterns=matched_patterns,
            message_preview=message[:100]
        )

        # Check if user should be banned (3+ incidents)
        profile = await self.get_or_create_risk_profile(user_id, clerk_user_id)
        should_ban = profile["incidentCount"] >= 3

        if should_ban:
            # Ban user and track in Langfuse with full context
            await self.ban_user(
                user_id=user_id,
                clerk_user_id=clerk_user_id,
                ip_address=ip_address,
                message=message,  # The message that triggered the ban
                session_id=session_id,
                message_id=None  # Message ID not available at this point
            )
            return {
                "is_safe": False,
                "is_banned": True,
                "ban_expires_at": datetime.utcnow() + timedelta(hours=BAN_DURATION_HOURS),
                "incident_count": profile["incidentCount"],
                "risk_score": risk_score,
                "matched_patterns": matched_patterns,
                "message": "SELVE is unavailable right now. Please try again tomorrow."
            }

        return {
            "is_safe": False,
            "is_banned": False,
            "ban_expires_at": None,
            "incident_count": profile["incidentCount"],
            "risk_score": risk_score,
            "matched_patterns": matched_patterns,
            "message": f"Your message contains suspicious patterns. {3 - profile['incidentCount']} warning(s) remaining before 24-hour restriction."
        }

    async def check_ban_status(
        self,
        user_id: Optional[str] = None,
        clerk_user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check if user/IP is currently banned."""
        profile = await self.get_risk_profile(user_id, clerk_user_id)
        
        if not profile or not profile.get("isFlagged"):
            return {"is_banned": False, "expires_at": None, "incident_count": 0}

        flagged_at = profile.get("flaggedAt")
        if not flagged_at:
            return {"is_banned": False, "expires_at": None, "incident_count": 0}

        expires_at = flagged_at + timedelta(hours=BAN_DURATION_HOURS)
        is_still_banned = datetime.utcnow() < expires_at

        # Auto-unban if time has passed
        if not is_still_banned and profile.get("isFlagged"):
            await self.unban_user(user_id, clerk_user_id)
            return {"is_banned": False, "expires_at": None, "incident_count": profile.get("incidentCount", 0)}

        return {
            "is_banned": is_still_banned,
            "expires_at": expires_at if is_still_banned else None,
            "incident_count": profile.get("incidentCount", 0)
        }

    async def record_incident(
        self,
        user_id: Optional[str],
        clerk_user_id: Optional[str],
        session_id: Optional[str],
        ip_address: Optional[str],
        risk_score: float,
        matched_patterns: List[str],
        message_preview: str
    ):
        """Record a security incident."""
        from prisma import Json
        
        ip_hash = self._hash_ip(ip_address) if ip_address else None

        # Build data dict
        data = {
            "clerkUserId": clerk_user_id,
            "sessionId": session_id,
            "incidentType": "prompt_injection",
            "riskScore": risk_score,
            "flags": Json(matched_patterns),
            "messagePreview": message_preview,
            "ipHash": ip_hash,
            "wasBlocked": False
        }
        
        # Include userId if available
        if user_id:
            data["userId"] = user_id

        incident = await db.securityincident.create(data=data)
        logger.info(f"Recorded security incident for clerk_user_id={clerk_user_id}, risk_score={risk_score}")

        # Update risk profile (only if user_id exists)
        if user_id:
            await self.update_risk_profile(user_id, clerk_user_id, risk_score)
        else:
            logger.warning(f"Skipping risk profile update - no user_id for clerk_user_id={clerk_user_id}")

        return incident

    async def get_or_create_risk_profile(
        self,
        user_id: Optional[str],
        clerk_user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get or create risk profile for user."""
        # If we have a user_id, try to get/create the UserRiskProfile
        if user_id:
            profile = await self.get_risk_profile(user_id, clerk_user_id)
            
            if not profile:
                # Ensure user exists
                user = await db.user.find_unique(where={"id": user_id})
                if user:
                    profile = await db.userriskprofile.create(
                        data={
                            "userId": user_id,
                            "totalScore": 0.0,
                            "incidentCount": 0,
                            "isFlagged": False
                        }
                    )
                    return {
                        "userId": profile.userId,
                        "totalScore": profile.totalScore,
                        "incidentCount": profile.incidentCount,
                        "isFlagged": profile.isFlagged,
                        "flaggedAt": profile.flaggedAt
                    }
            else:
                return profile
        
        # Fallback: Count incidents directly from SecurityIncident table using clerkUserId
        if clerk_user_id:
            incident_count = await db.securityincident.count(
                where={"clerkUserId": clerk_user_id}
            )
            logger.info(f"Counted {incident_count} incidents for clerk_user_id={clerk_user_id}")
            return {
                "incidentCount": incident_count,
                "totalScore": incident_count * 10.0,
                "isFlagged": False
            }
        
        return {"incidentCount": 0, "totalScore": 0.0, "isFlagged": False}

    async def get_risk_profile(
        self,
        user_id: Optional[str],
        clerk_user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get risk profile for user."""
        if not user_id:
            return None

        profile = await db.userriskprofile.find_unique(
            where={"userId": user_id}
        )

        if not profile:
            return None

        return {
            "userId": profile.userId,
            "totalScore": profile.totalScore,
            "incidentCount": profile.incidentCount,
            "isFlagged": profile.isFlagged,
            "flaggedAt": profile.flaggedAt,
            "lastIncidentAt": profile.lastIncidentAt
        }

    async def update_risk_profile(
        self,
        user_id: Optional[str],
        clerk_user_id: Optional[str],
        risk_score: float
    ):
        """Update risk profile with new incident."""
        if not user_id:
            return

        profile = await self.get_or_create_risk_profile(user_id, clerk_user_id)

        await db.userriskprofile.update(
            where={"userId": user_id},
            data={
                "totalScore": profile["totalScore"] + risk_score,
                "incidentCount": profile["incidentCount"] + 1,
                "lastIncidentAt": datetime.utcnow()
            }
        )

    async def ban_user(
        self,
        user_id: Optional[str],
        clerk_user_id: Optional[str],
        ip_address: Optional[str],
        message: Optional[str] = None,
        session_id: Optional[str] = None,
        message_id: Optional[str] = None
    ):
        """
        Ban a user for 24 hours and track in Langfuse.

        Args:
            user_id: Internal user ID
            clerk_user_id: Clerk authentication ID
            ip_address: User's IP address
            message: The message that triggered the ban (for logging)
            session_id: Session where ban occurred
            message_id: ID of the message that triggered ban
        """
        if not user_id:
            return

        ban_time = datetime.utcnow()
        expires_at = ban_time + timedelta(hours=BAN_DURATION_HOURS)

        await db.userriskprofile.update(
            where={"userId": user_id},
            data={
                "isFlagged": True,
                "flaggedAt": ban_time
            }
        )

        # Comprehensive logging
        logger.warning(
            f"🚨 USER BANNED - "
            f"clerk_user_id={clerk_user_id} "
            f"user_id={user_id} "
            f"banned_at={ban_time.isoformat()} "
            f"expires_at={expires_at.isoformat()} "
            f"duration=24h "
            f"session_id={session_id or 'unknown'} "
            f"message_id={message_id or 'unknown'} "
            f"ip_hash={self._hash_ip(ip_address) if ip_address else 'unknown'} "
            f"trigger_message_preview={message[:100] if message else 'N/A'}"
        )

        # Track ban event in Langfuse for observability
        try:
            from app.services.langfuse_service import get_langfuse_service
            langfuse = get_langfuse_service()

            if langfuse.enabled and langfuse.client:
                # Create a custom event for the ban
                with langfuse.client.start_as_current_observation(
                    as_type="event",
                    name="user-banned",
                    input=f"User banned after 3 security incidents",
                    metadata={
                        "clerk_user_id": clerk_user_id,
                        "user_id": user_id,
                        "banned_at": ban_time.isoformat(),
                        "expires_at": expires_at.isoformat(),
                        "duration_hours": BAN_DURATION_HOURS,
                        "session_id": session_id,
                        "message_id": message_id,
                        "ip_hash": self._hash_ip(ip_address) if ip_address else None,
                        "trigger_message_length": len(message) if message else 0,
                    },
                    tags=["security", "ban", "prompt-injection"]
                ):
                    pass  # Event is recorded on context exit

                logger.info(f"✓ Ban event tracked in Langfuse for user {clerk_user_id}")
        except Exception as e:
            logger.error(f"Failed to track ban in Langfuse: {e}")

    async def unban_user(
        self,
        user_id: Optional[str],
        clerk_user_id: Optional[str]
    ):
        """Remove ban from user and track in Langfuse."""
        if not user_id:
            return

        unban_time = datetime.utcnow()

        await db.userriskprofile.update(
            where={"userId": user_id},
            data={
                "isFlagged": False,
                "flaggedAt": None
            }
        )

        # Log unban event
        logger.info(
            f"✅ USER UNBANNED (24h expired) - "
            f"clerk_user_id={clerk_user_id} "
            f"user_id={user_id} "
            f"unbanned_at={unban_time.isoformat()}"
        )

        # Track unban in Langfuse
        try:
            from app.services.langfuse_service import get_langfuse_service
            langfuse = get_langfuse_service()

            if langfuse.enabled and langfuse.client:
                with langfuse.client.start_as_current_observation(
                    as_type="event",
                    name="user-unbanned",
                    input="User automatically unbanned after 24-hour restriction expired",
                    metadata={
                        "clerk_user_id": clerk_user_id,
                        "user_id": user_id,
                        "unbanned_at": unban_time.isoformat(),
                    },
                    tags=["security", "unban", "auto-recovery"]
                ):
                    pass

                logger.info(f"✓ Unban event tracked in Langfuse for user {clerk_user_id}")
        except Exception as e:
            logger.error(f"Failed to track unban in Langfuse: {e}")

    def _hash_ip(self, ip: str) -> str:
        """Hash IP address for privacy."""
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
