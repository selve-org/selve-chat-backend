"""
Chat Service for SELVE Chatbot
Integrates RAG with OpenAI Chat Completion
"""
import os
from typing import List, Dict, Any
from openai import OpenAI
from .rag_service import RAGService


class ChatService:
    """Service for handling chat interactions with RAG"""

    def __init__(self):
        """Initialize OpenAI client and RAG service"""
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag_service = RAGService()
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
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

    def generate_response(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        use_rag: bool = True
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
        # Retrieve relevant context if RAG is enabled
        context_info = None
        if use_rag:
            context_info = self.rag_service.get_context_for_query(message, top_k=3)

        # Build messages for OpenAI
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current message with context (if available)
        user_message = message
        if context_info and context_info["retrieved_count"] > 0:
            user_message = f"{context_info['context']}\n\n---\n\nUser Question: {message}"

        messages.append({"role": "user", "content": user_message})

        # Generate response from OpenAI
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content

        return {
            "response": assistant_message,
            "context_used": context_info is not None and context_info["retrieved_count"] > 0,
            "retrieved_chunks": context_info["chunks"] if context_info else [],
            "model": self.model
        }

    def generate_streaming_response(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        use_rag: bool = True
    ):
        """
        Generate a streaming chat response (for future use)

        Yields response chunks as they're generated
        """
        # Retrieve context
        context_info = None
        if use_rag:
            context_info = self.rag_service.get_context_for_query(message, top_k=3)

        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)

        user_message = message
        if context_info and context_info["retrieved_count"] > 0:
            user_message = f"{context_info['context']}\n\n---\n\nUser Question: {message}"

        messages.append({"role": "user", "content": user_message})

        # Stream response
        stream = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
