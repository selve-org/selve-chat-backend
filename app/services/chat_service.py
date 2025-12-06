"""
Chat Service for SELVE Chatbot
Integrates RAG with Dual LLM Support (OpenAI + Anthropic)
Includes personality-focused guardrails and off-topic detection
"""
import os
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from .llm_service import LLMService
from .rag_service import RAGService
from .user_profile_service import UserProfileService
from .compression_service import CompressionService
from .conversation_state_service import ConversationStateService
from .semantic_memory_service import SemanticMemoryService
from ..prompts import SYSTEM_PROMPT, get_canned_response, classify_message


# Status event types for thinking UI
class StatusEvent:
    """Status events emitted during response generation for thinking UI"""
    RETRIEVING_CONTEXT = "retrieving_context"
    PERSONALIZING = "personalizing"
    GENERATING = "generating"
    CITING_SOURCES = "citing_sources"
    COMPLETE = "complete"
    ERROR = "error"
    
    @staticmethod
    def create(status: str, message: str = "", details: Optional[Dict] = None) -> Dict:
        """Create a status event payload"""
        return {
            "type": "status",
            "status": status,
            "message": message,
            "details": details or {}
        }


class ChatService:
    """Service for handling chat interactions with RAG and dual LLM support"""

    def __init__(self):
        """Initialize LLM service, RAG service, user profile service, compression service, and state service"""
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self.user_profile_service = UserProfileService()
        self.compression_service = CompressionService()
        self.conversation_state_service = ConversationStateService()
        self.semantic_memory_service = SemanticMemoryService()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the SELVE chatbot system prompt from prompts module"""
        return SYSTEM_PROMPT

    def _format_scores_for_context(self, scores: Dict[str, float]) -> str:
        """Format SELVE scores into context for personalization"""
        dimension_descriptions = {
            "LUMEN": "Mindful Curiosity (social energy and recharging)",
            "AETHER": "Rational Reflection (information processing)",
            "ORPHEUS": "Compassionate Connection (decision-making approach)",
            "ORIN": "Structured Harmony (planning and structure)",
            "LYRA": "Creative Expression (openness to experiences)",
            "VARA": "Purposeful Commitment (emotional stability)",
            "CHRONOS": "Adaptive Spontaneity (agreeableness and cooperation)",
            "KAEL": "Bold Resilience (conscientiousness and discipline)"
        }

        # Identify strongest and weakest dimensions
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]

        context_parts = [
            "USER'S SELVE PROFILE:",
            ""
        ]

        # Add top 3 dimensions
        context_parts.append("Strongest dimensions:")
        for dim, score in top_3:
            desc = dimension_descriptions.get(dim, dim)
            context_parts.append(f"  • {dim}: {int(score)}/100 - {desc}")

        context_parts.extend([
            "",
            "When responding:",
            "- Reference their specific scores when relevant to the question",
            "- Provide personalized insights based on their profile",
            "- Help them understand how their scores influence their behavior"
        ])

        return "\n".join(context_parts)

    async def generate_conversation_title(self, first_message: str) -> str:
        """Generate a concise title for a conversation based on the first message"""
        try:
            prompt = f"""Generate a short, descriptive title (max 5 words) for a conversation that starts with this message:

"{first_message}"

Return ONLY the title, nothing else. Make it specific and meaningful."""

            response = await self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=20
            )

            title = response.strip().strip('"').strip("'")
            # Truncate if too long
            if len(title) > 50:
                title = title[:47] + "..."

            return title
        except Exception as e:
            # Fallback to truncated first message
            return first_message[:50] + "..." if len(first_message) > 50 else first_message

    async def _analyze_and_update_state(
        self,
        session_id: Optional[str],
        user_message: str,
        assistant_response: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Analyze conversation state and update session (async background task)

        Args:
            session_id: Chat session ID (optional)
            user_message: User's message
            assistant_response: Assistant's response
            conversation_history: Previous messages
        """
        if not session_id:
            return

        try:
            # Analyze conversation state
            state = await self.conversation_state_service.analyze_conversation_state(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_history=conversation_history
            )

            # Update session with state
            await self.conversation_state_service.update_session_state(
                session_id=session_id,
                state=state
            )
        except Exception as e:
            print(f"⚠️ Failed to update conversation state: {e}")

    async def generate_response(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        use_rag: bool = True,
        clerk_user_id: Optional[str] = None,
        session_id: Optional[str] = None
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

        # Retrieve episodic memories for conversation continuity
        memory_context = None
        if clerk_user_id:
            try:
                memories = await self.compression_service.get_user_memories(clerk_user_id, limit=3)
                if memories:
                    memory_context = self.compression_service.format_memories_for_context(memories)
            except Exception as e:
                print(f"⚠️ Error retrieving episodic memories: {e}")

        # Retrieve semantic memory (long-term patterns)
        semantic_context = None
        if clerk_user_id:
            try:
                semantic_mem = await self.semantic_memory_service.get_user_semantic_memory(clerk_user_id=clerk_user_id)
                if semantic_mem:
                    semantic_context = self.semantic_memory_service.format_semantic_memory_for_context(semantic_mem)
            except Exception as e:
                print(f"⚠️ Error retrieving semantic memory: {e}")

        # Retrieve relevant context if RAG is enabled
        context_info = None
        if use_rag:
            context_info = self.rag_service.get_context_for_query(message, top_k=3)

        # Build system prompt with user context, memories, and patterns
        system_content = self.system_prompt
        if user_context:
            system_content = f"{self.system_prompt}\n\n{user_context}"
        if semantic_context:
            system_content = f"{system_content}\n\n{semantic_context}"
        if memory_context:
            system_content = f"{system_content}\n\n{memory_context}"

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

        # Check if compression needed (for next message)
        compression_needed = False
        if clerk_user_id:
            total_tokens = llm_response["usage"]["total_tokens"]
            compression_needed = self.compression_service.needs_compression(
                total_tokens,
                llm_response["model"]
            )

        # Analyze and update conversation state (async background)
        if session_id:
            # Don't await this - let it run in background
            import asyncio
            asyncio.create_task(
                self._analyze_and_update_state(
                    session_id=session_id,
                    user_message=message,
                    assistant_response=llm_response["content"],
                    conversation_history=conversation_history
                )
            )

        return {
            "response": llm_response["content"],
            "context_used": context_info is not None and context_info["retrieved_count"] > 0,
            "retrieved_chunks": context_info["chunks"] if context_info else [],
            "model": llm_response["model"],
            "provider": llm_response["provider"],
            "usage": llm_response["usage"],
            "cost": llm_response["cost"],
            "compression_needed": compression_needed
        }

    async def generate_response_stream(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        use_rag: bool = True,
        clerk_user_id: Optional[str] = None,
        selve_scores: Optional[Dict[str, float]] = None,
        emit_status: bool = True
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """
        Generate a streaming chat response with optional RAG context and status events

        Yields response chunks and status events as they're generated by the LLM

        Args:
            message: User's current message
            conversation_history: Previous messages
            use_rag: Whether to retrieve context from RAG
            clerk_user_id: Clerk user ID for profile personalization
            selve_scores: User's SELVE personality scores for personalization
            emit_status: Whether to emit status events for thinking UI

        Yields:
            Union[str, Dict]: Either text chunks or status event dicts
        """
        # --- Guardrails Check ---
        # Check if message is off-topic or sensitive before processing
        canned_response = get_canned_response(message)
        if canned_response:
            classification, _ = classify_message(message)
            print(f"⚠️ Message classified as '{classification}', returning canned response")
            
            # Emit complete status with no sources
            if emit_status:
                yield StatusEvent.create(
                    StatusEvent.COMPLETE,
                    "Response complete",
                    {"sources": [], "guardrail": classification}
                )
            
            # Stream the canned response
            for char in canned_response:
                yield char
            return
        
        sources_used = []
        
        # --- Phase 1: Retrieve Context (RAG) ---
        if emit_status and use_rag:
            yield StatusEvent.create(
                StatusEvent.RETRIEVING_CONTEXT,
                "Searching knowledge base...",
                {"phase": 1, "total_phases": 4}
            )
        
        # Retrieve relevant context if RAG is enabled
        context_info = None
        if use_rag:
            context_info = self.rag_service.get_context_for_query(message, top_k=3)
            if context_info and context_info.get("chunks"):
                sources_used = [
                    {"title": c.get("title", "SELVE Knowledge"), "source": c.get("source", "knowledge_base")}
                    for c in context_info["chunks"]
                ]
        
        # --- Phase 2: Personalization ---
        if emit_status:
            yield StatusEvent.create(
                StatusEvent.PERSONALIZING,
                "Loading your personality profile...",
                {"phase": 2, "total_phases": 4}
            )
        
        # Build user context from SELVE scores
        user_context = None
        if selve_scores:
            user_context = self._format_scores_for_context(selve_scores)

        # Retrieve episodic memories for conversation continuity
        memory_context = None
        if clerk_user_id:
            try:
                memories = await self.compression_service.get_user_memories(clerk_user_id, limit=3)
                if memories:
                    memory_context = self.compression_service.format_memories_for_context(memories)
            except Exception as e:
                print(f"⚠️ Error retrieving episodic memories: {e}")

        # Retrieve semantic memory (long-term patterns)
        semantic_context = None
        if clerk_user_id:
            try:
                semantic_mem = await self.semantic_memory_service.get_user_semantic_memory(clerk_user_id=clerk_user_id)
                if semantic_mem:
                    semantic_context = self.semantic_memory_service.format_semantic_memory_for_context(semantic_mem)
            except Exception as e:
                print(f"⚠️ Error retrieving semantic memory: {e}")

        # --- Phase 3: Generation ---
        if emit_status:
            yield StatusEvent.create(
                StatusEvent.GENERATING,
                "Crafting your response...",
                {"phase": 3, "total_phases": 4, "model": self.llm_service.model}
            )

        # Build system prompt with user context, memories, and patterns
        system_content = self.system_prompt
        if user_context:
            system_content = f"{self.system_prompt}\n\n{user_context}"
        if semantic_context:
            system_content = f"{system_content}\n\n{semantic_context}"
        if memory_context:
            system_content = f"{system_content}\n\n{memory_context}"

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

        # Stream response using LLM service
        async for chunk in self.llm_service.generate_response_stream(
            messages=messages,
            temperature=0.7,
            max_tokens=500
        ):
            yield chunk
        
        # --- Phase 4: Citation ---
        if emit_status and sources_used:
            yield StatusEvent.create(
                StatusEvent.CITING_SOURCES,
                "Adding sources...",
                {"phase": 4, "total_phases": 4, "sources": sources_used}
            )
        
        # --- Complete ---
        if emit_status:
            yield StatusEvent.create(
                StatusEvent.COMPLETE,
                "Response complete",
                {"sources": sources_used}
            )
