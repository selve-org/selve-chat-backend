"""
Session Service - Manages chat sessions and message persistence
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.db import db


class SessionService:
    """Service for managing chat sessions and messages"""

    async def create_session(
        self,
        user_id: str,
        clerk_user_id: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new chat session

        Args:
            user_id: Internal user ID
            clerk_user_id: Clerk authentication ID
            title: Optional session title

        Returns:
            Session data dict
        """
        session = await db.chatsession.create(
            data={
                "userId": user_id,
                "clerkUserId": clerk_user_id,
                "title": title or "New Conversation",
                "status": "active"
            }
        )

        return {
            "id": session.id,
            "userId": session.userId,
            "clerkUserId": session.clerkUserId,
            "title": session.title,
            "status": session.status,
            "createdAt": session.createdAt.isoformat(),
            "lastMessageAt": session.lastMessageAt.isoformat()
        }

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID with active messages only"""
        session = await db.chatsession.find_unique(
            where={"id": session_id},
            include={
                "messages": {
                    "where": {"isActive": True},  # Filter to active messages only
                    "orderBy": {"createdAt": "asc"}
                }
            }
        )

        if not session:
            return None

        return {
            "id": session.id,
            "userId": session.userId,
            "clerkUserId": session.clerkUserId,
            "title": session.title,
            "status": session.status,
            "totalTokens": session.totalTokens,
            "createdAt": session.createdAt.isoformat(),
            "lastMessageAt": session.lastMessageAt.isoformat(),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "tokenCount": msg.tokenCount,
                    "model": msg.model,
                    "provider": msg.provider,
                    "cost": msg.cost,
                    "langfuseTraceId": msg.langfuseTraceId,
                    "isActive": msg.isActive,
                    "regenerationIndex": msg.regenerationIndex,
                    "groupId": msg.groupId,
                    "createdAt": msg.createdAt.isoformat()
                }
                for msg in session.messages
            ]
        }

    async def list_sessions(
        self,
        clerk_user_id: str,
        limit: int = 20,
        status: str = "active"
    ) -> List[Dict[str, Any]]:
        """List sessions for a user"""
        sessions = await db.chatsession.find_many(
            where={
                "clerkUserId": clerk_user_id,
                "status": status
            },
            order={"lastMessageAt": "desc"},
            take=limit
        )

        return [
            {
                "id": s.id,
                "title": s.title,
                "status": s.status,
                "totalTokens": s.totalTokens,
                "createdAt": s.createdAt.isoformat(),
                "lastMessageAt": s.lastMessageAt.isoformat()
            }
            for s in sessions
        ]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        cost: Optional[float] = None,
        langfuse_trace_id: Optional[str] = None,
        is_active: bool = True,
        regeneration_index: int = 1,
        parent_message_id: Optional[str] = None,
        group_id: Optional[str] = None,
        regeneration_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a message to a session with versioning support

        Args:
            session_id: Session ID
            role: "user" | "assistant" | "system"
            content: Message content
            token_count: Number of tokens
            model: LLM model used
            provider: LLM provider
            cost: Cost in USD
            langfuse_trace_id: Langfuse trace ID for feedback tracking
            is_active: Whether this is the active version
            regeneration_index: Version number (1, 2, 3...)
            parent_message_id: ID of parent message if this is a regeneration
            group_id: Group ID for versioning
            regeneration_type: "regenerate" | "edit" | None

        Returns:
            Message data dict
        """
        message = await db.chatmessage.create(
            data={
                "sessionId": session_id,
                "role": role,
                "content": content,
                "tokenCount": token_count,
                "model": model,
                "provider": provider,
                "cost": cost,
                "langfuseTraceId": langfuse_trace_id,
                "isActive": is_active,
                "regenerationIndex": regeneration_index,
                "parentMessageId": parent_message_id,
                "groupId": group_id,
                "regenerationType": regeneration_type
            }
        )

        # Update session totals and lastMessageAt
        await db.chatsession.update(
            where={"id": session_id},
            data={
                "totalTokens": {"increment": token_count},
                "lastMessageAt": datetime.utcnow()
            }
        )

        return {
            "id": message.id,
            "sessionId": message.sessionId,
            "role": message.role,
            "content": message.content,
            "tokenCount": message.tokenCount,
            "model": message.model,
            "provider": message.provider,
            "cost": message.cost,
            "langfuseTraceId": message.langfuseTraceId,
            "isActive": message.isActive,
            "regenerationIndex": message.regenerationIndex,
            "parentMessageId": message.parentMessageId,
            "groupId": message.groupId,
            "regenerationType": message.regenerationType,
            "createdAt": message.createdAt.isoformat()
        }

    async def mark_messages_inactive(
        self,
        session_id: str,
        group_id: str,
        exclude_message_id: Optional[str] = None
    ) -> int:
        """
        Mark all messages in a group as inactive except the specified one.

        Used when creating a new version to deactivate old versions.

        Args:
            session_id: Session ID
            group_id: Group ID to filter by
            exclude_message_id: Message ID to keep active (new version)

        Returns:
            Number of messages marked inactive
        """
        where_clause = {
            "sessionId": session_id,
            "groupId": group_id,
        }

        if exclude_message_id:
            where_clause["id"] = {"not": exclude_message_id}

        result = await db.chatmessage.update_many(
            where=where_clause,
            data={"isActive": False}
        )

        return result

    async def update_session_title(
        self,
        session_id: str,
        title: str
    ) -> bool:
        """Update session title"""
        try:
            await db.chatsession.update(
                where={"id": session_id},
                data={"title": title}
            )
            return True
        except Exception:
            return False

    async def archive_session(self, session_id: str) -> bool:
        """Archive a session"""
        try:
            await db.chatsession.update(
                where={"id": session_id},
                data={"status": "archived"}
            )
            return True
        except Exception:
            return False
