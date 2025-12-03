"""
Chat Service for SELVE Chatbot
Integrates RAG with Dual LLM Support (OpenAI + Anthropic)
"""
import os
from typing import List, Dict, Any, Optional
from .llm_service import LLMService
from .rag_service import RAGService
from .user_profile_service import UserProfileService


class ChatService:
    """Service for handling chat interactions with RAG and dual LLM support"""

    def __init__(self):
        """Initialize LLM service, RAG service, and user profile service"""
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self.user_profile_service = UserProfileService()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the SELVE chatbot system prompt"""
        return """You are the SELVE Chatbot, an expert assistant for the SELVE personality framework.

Your role is to:
1. Help users understand their SELVE assessment results
2. Explain the 8 SELVE dimensions (LUMEN, CHRONOS, KAEL, LYRA, ORIN, ORPHEUS, AETHER, VARA)
3. Provide personalized insights based on dimension scores
4. Answer questions about personality traits, strengths, and growth areas

Guidelines:
- Always ground your responses in the provided SELVE framework context
- Be warm, supportive, and non-judgmental
- Focus on growth and self-understanding, not labeling
- When relevant context is provided, reference it naturally
- If you don't have specific information, be honest about it
- Keep responses concise and actionable

Remember: All personalities have value. There are no "good" or "bad" scores - only different ways of being human."""

    async def generate_response(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        use_rag: bool = True,
        clerk_user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat response with optional RAG context

        Args:
            message: User's current message
            conversation_history: Previous messages [{"role": "user/assistant", "content": "..."}]
            use_rag: Whether to retrieve context from RAG

        Returns:
            {
                "response": "assistant's response",
                "context_used": bool,
                "retrieved_chunks": [...],
                "model": "model name"
            }
        """
        # Retrieve user profile for personalization
        user_context = None
        if clerk_user_id:
            profile = await self.user_profile_service.get_user_profile(clerk_user_id)
            if profile and profile.get("has_assessment"):
                user_context = self.user_profile_service.format_profile_for_context(profile)

        # Retrieve relevant context if RAG is enabled
        context_info = None
        if use_rag:
            context_info = self.rag_service.get_context_for_query(message, top_k=3)

        # Build system prompt with user context
        system_content = self.system_prompt
        if user_context:
            system_content = f"{self.system_prompt}\n\n{user_context}"

        # Build messages
        messages = [{"role": "system", "content": system_content}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current message with context (if available)
        user_message = message
        if context_info and context_info["retrieved_count"] > 0:
            user_message = f"{context_info['context']}\n\n---\n\nUser Question: {message}"

        messages.append({"role": "user", "content": user_message})

        # Generate response using unified LLM service
        llm_response = self.llm_service.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return {
            "response": llm_response["content"],
            "context_used": context_info is not None and context_info["retrieved_count"] > 0,
            "retrieved_chunks": context_info["chunks"] if context_info else [],
            "model": llm_response["model"],
            "provider": llm_response["provider"],
            "usage": llm_response["usage"],
            "cost": llm_response["cost"]
        }

    def generate_streaming_response(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        use_rag: bool = True
    ):
        """
        Generate a streaming chat response (TODO: implement with LLMService)

        Yields response chunks as they're generated
        """
        raise NotImplementedError("Streaming responses will be implemented in Phase 5")
