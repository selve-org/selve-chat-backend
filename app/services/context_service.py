"""
Context Service for SELVE Chat.
Handles RAG retrieval, episodic/semantic memory fetch, and message assembly.

This is the orchestration layer that brings together:
- RAG knowledge retrieval
- Episodic memory search
- Semantic memory patterns
- SELVE profile context
- System prompt building

Security & Robustness:
- Graceful degradation on service failures
- Timeout handling for all external calls
- Clean separation of concerns
- No sensitive data in context strings
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Config, Result

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class ContextResult:
    """Result of context building operation."""

    system_content: str
    context_info: Optional[Dict[str, Any]] = None
    sources_used: List[Dict[str, str]] = field(default_factory=list)
    user_context: Optional[str] = None
    memory_context: Optional[str] = None
    semantic_context: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def has_rag_context(self) -> bool:
        return bool(self.context_info and self.context_info.get("retrieved_count", 0) > 0)

    @property
    def has_memory_context(self) -> bool:
        return bool(self.memory_context)

    @property
    def has_semantic_context(self) -> bool:
        return bool(self.semantic_context)


# =============================================================================
# SELVE DIMENSION DESCRIPTIONS
# =============================================================================


DIMENSION_DESCRIPTIONS = {
    "LUMEN": "Mindful Curiosity (social energy and recharging)",
    "AETHER": "Rational Reflection (information processing)",
    "ORPHEUS": "Compassionate Connection (decision-making approach)",
    "ORIN": "Structured Harmony (planning and structure)",
    "LYRA": "Creative Expression (openness to experiences)",
    "VARA": "Purposeful Commitment (emotional stability)",
    "CHRONOS": "Adaptive Spontaneity (agreeableness and cooperation)",
    "KAEL": "Bold Resilience (conscientiousness and discipline)",
}


# =============================================================================
# CONTEXT SERVICE
# =============================================================================


class ContextService:
    """
    Orchestration service for building chat context.

    Responsibilities:
    - Fetch RAG context for knowledge grounding
    - Retrieve episodic memories for conversation continuity
    - Get semantic patterns for personalization
    - Format SELVE scores for context
    - Build complete system prompt with all context
    - Assemble final message list for LLM

    Design Principles:
    - Graceful degradation: Context components fail independently
    - Concurrent fetching: All context sources fetched in parallel
    - Clean interfaces: Services interact through defined contracts
    - No coupling: Services injected via constructor
    """

    def __init__(
        self,
        rag_service,
        compression_service,
        semantic_memory_service,
        system_prompt: str,
        assessment_url: str,
    ):
        """
        Initialize context service.

        Args:
            rag_service: RAG retrieval service
            compression_service: Episodic memory/compression service
            semantic_memory_service: Semantic memory service
            system_prompt: Base system prompt
            assessment_url: URL for SELVE assessment
        """
        self.rag_service = rag_service
        self.compression_service = compression_service
        self.semantic_memory_service = semantic_memory_service
        self.system_prompt = system_prompt
        self.assessment_url = assessment_url.rstrip("/")
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _fetch_rag_context(
        self,
        message: str,
        top_k: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch RAG context with timeout.

        Args:
            message: User message for similarity search
            top_k: Number of chunks to retrieve

        Returns:
            Context info dict or None on failure
        """
        try:
            # RAG service is synchronous, run in executor
            loop = asyncio.get_event_loop()
            context_info = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.rag_service.get_context_for_query(message, top_k=top_k),
                ),
                timeout=Config.RAG_TIMEOUT_SECONDS,
            )
            return context_info

        except asyncio.TimeoutError:
            self.logger.warning("RAG retrieval timed out")
            return None
        except Exception as e:
            self.logger.error(f"RAG retrieval failed: {e}")
            return None

    async def _fetch_episodic_memories(
        self,
        clerk_user_id: str,
        limit: int = 3,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch episodic memories with timeout.

        Args:
            clerk_user_id: User identifier
            limit: Maximum memories to retrieve

        Returns:
            List of memory dicts or None on failure
        """
        try:
            memories = await asyncio.wait_for(
                self.compression_service.get_user_memories(clerk_user_id, limit=limit),
                timeout=Config.MEMORY_TIMEOUT_SECONDS,
            )
            return memories

        except asyncio.TimeoutError:
            self.logger.warning("Episodic memory fetch timed out")
            return None
        except Exception as e:
            self.logger.error(f"Episodic memory fetch failed: {e}")
            return None

    async def _fetch_semantic_memory(
        self,
        clerk_user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch semantic memory with timeout.

        Args:
            clerk_user_id: User identifier

        Returns:
            Semantic memory dict or None on failure
        """
        try:
            result = await asyncio.wait_for(
                self.semantic_memory_service.get_user_semantic_memory(
                    clerk_user_id=clerk_user_id
                ),
                timeout=Config.MEMORY_TIMEOUT_SECONDS,
            )

            # Handle Result wrapper if present
            if hasattr(result, "is_success"):
                return result.data if result.is_success else None
            return result

        except asyncio.TimeoutError:
            self.logger.warning("Semantic memory fetch timed out")
            return None
        except Exception as e:
            self.logger.error(f"Semantic memory fetch failed: {e}")
            return None

    def _format_scores_for_context(
        self,
        scores: Dict[str, float],
    ) -> str:
        """
        Format SELVE scores as context string.

        Args:
            scores: Dict of dimension name to score (0-100)

        Returns:
            Formatted context string
        """
        if not scores:
            return ""

        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        parts = [
            "USER'S SELVE SCORES (Complete Profile):",
            "",
        ]

        for dim, score in sorted_scores:
            desc = DIMENSION_DESCRIPTIONS.get(dim, dim)
            parts.append(f"  • {dim}: {int(score)}/100 - {desc}")

        parts.extend([
            "",
            "CRITICAL INSTRUCTIONS:",
            "- When user asks about 'strongest', 'weakest', 'high', 'low', or 'dimensions', ALWAYS reference these exact scores",
            "- If user asks about personality, strengths, or how they are, proactively mention these scores",
            "- ALWAYS show the complete list when discussing their profile",
            "- Use these scores to provide personalized, specific insights",
            "- Connect their actual scores to real-world implications",
        ])

        return "\n".join(parts)

    async def build_context(
        self,
        message: str,
        clerk_user_id: Optional[str],
        selve_scores: Optional[Dict[str, float]],
        use_rag: bool = True,
        assessment_url: Optional[str] = None,
        rag_top_k: int = 3,
    ) -> ContextResult:
        """
        Build complete context for a chat request.

        Fetches all context sources concurrently and assembles
        the final system prompt with graceful degradation.

        Args:
            message: User's message
            clerk_user_id: User identifier (for personalization)
            selve_scores: User's SELVE dimension scores
            use_rag: Whether to fetch RAG context
            assessment_url: Override assessment URL
            rag_top_k: Number of RAG chunks to retrieve

        Returns:
            ContextResult with all context components
        """
        errors: List[str] = []
        tasks: Dict[str, asyncio.Task] = {}

        # Schedule concurrent fetches
        if use_rag:
            tasks["rag"] = asyncio.create_task(
                self._fetch_rag_context(message, top_k=rag_top_k),
                name="fetch_rag",
            )

        if clerk_user_id:
            tasks["memories"] = asyncio.create_task(
                self._fetch_episodic_memories(clerk_user_id),
                name="fetch_memories",
            )
            tasks["semantic"] = asyncio.create_task(
                self._fetch_semantic_memory(clerk_user_id),
                name="fetch_semantic",
            )

        # Gather results
        results: Dict[str, Any] = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
                self.logger.error(f"Task {key} failed: {e}")
                results[key] = None
                errors.append(f"{key}: {str(e)}")

        # Format user context from SELVE scores
        user_context = None
        if selve_scores:
            user_context = self._format_scores_for_context(selve_scores)
            self.logger.info(f"✅ User context formatted with {len(selve_scores)} dimension scores")
        else:
            self.logger.info("⚠️ No selve_scores provided to build_context")

        # Format episodic memory context
        memory_context = None
        if results.get("memories"):
            try:
                memory_context = self.compression_service.format_memories_for_context(
                    results["memories"]
                )
            except Exception as e:
                self.logger.warning(f"Failed to format memories: {e}")

        # Format semantic memory context
        semantic_context = None
        semantic_mem = results.get("semantic")
        if semantic_mem:
            try:
                # Handle both SemanticPattern objects and dicts
                if hasattr(self.semantic_memory_service, "format_for_context"):
                    semantic_context = self.semantic_memory_service.format_for_context(
                        semantic_mem
                    )
                elif hasattr(self.semantic_memory_service, "format_semantic_memory_for_context"):
                    semantic_context = self.semantic_memory_service.format_semantic_memory_for_context(
                        semantic_mem
                    )
            except Exception as e:
                self.logger.warning(f"Failed to format semantic memory: {e}")

        # Extract RAG context info and sources
        context_info = results.get("rag")
        sources_used: List[Dict[str, str]] = []
        if context_info and context_info.get("chunks"):
            sources_used = [
                {
                    "title": chunk.get("title", "SELVE Knowledge"),
                    "source": chunk.get("source", "knowledge_base"),
                }
                for chunk in context_info["chunks"]
            ]

        # Build system prompt
        system_content = self._build_system_prompt(
            base_prompt=self.system_prompt,
            selve_scores=selve_scores,
            assessment_url=assessment_url or self.assessment_url,
            user_context=user_context,
            semantic_context=semantic_context,
            memory_context=memory_context,
        )

        return ContextResult(
            system_content=system_content,
            context_info=context_info,
            sources_used=sources_used,
            user_context=user_context,
            memory_context=memory_context,
            semantic_context=semantic_context,
            errors=errors,
        )

    def _build_system_prompt(
        self,
        base_prompt: str,
        selve_scores: Optional[Dict[str, float]],
        assessment_url: str,
        user_context: Optional[str],
        semantic_context: Optional[str],
        memory_context: Optional[str],
    ) -> str:
        """
        Build the complete system prompt.

        Args:
            base_prompt: Base system prompt
            selve_scores: User's SELVE scores (to determine if CTA needed)
            assessment_url: Assessment URL for CTA
            user_context: Formatted SELVE profile context
            semantic_context: Formatted semantic memory context
            memory_context: Formatted episodic memory context

        Returns:
            Complete system prompt
        """
        parts = [base_prompt]

        # Add assessment CTA if no scores
        if not selve_scores and assessment_url:
            cta = (
                "\n\nASSESSMENT CTA:\n"
                "- When you do not have the user's SELVE scores, invite them to take their assessment.\n"
                f"- Include a short call-to-action with this link: [Take the SELVE assessment]({assessment_url})\n"
                "- Keep it to one concise line before continuing with help."
            )
            parts.append(cta)

        # Add context sections
        if user_context:
            parts.append(f"\n\n{user_context}")

        if semantic_context:
            parts.append(f"\n\n{semantic_context}")

        if memory_context:
            parts.append(f"\n\n{memory_context}")

        return "".join(parts)

    def build_messages(
        self,
        message: str,
        system_content: str,
        conversation_history: List[Dict[str, str]],
        context_info: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Build the final message list for the LLM.

        Args:
            message: User's current message
            system_content: Complete system prompt
            conversation_history: Previous messages
            context_info: RAG context info

        Returns:
            List of message dicts ready for LLM
        """
        messages = [{"role": "system", "content": system_content}]

        # Add conversation history
        if conversation_history:
            # Validate and sanitize history
            for msg in conversation_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = msg["role"]
                    if role in ("user", "assistant"):
                        messages.append({
                            "role": role,
                            "content": str(msg["content"]),
                        })

        # Build user message with RAG context
        user_message = message
        if context_info and context_info.get("retrieved_count", 0) > 0:
            rag_context = context_info.get("context", "")
            if rag_context:
                user_message = (
                    f"<knowledge_context>\n{rag_context}\n</knowledge_context>\n\n"
                    f"User Question: {message}"
                )

        messages.append({"role": "user", "content": user_message})

        return messages

    async def build_and_assemble(
        self,
        message: str,
        clerk_user_id: Optional[str],
        selve_scores: Optional[Dict[str, float]],
        conversation_history: List[Dict[str, str]],
        use_rag: bool = True,
    ) -> tuple[List[Dict[str, str]], ContextResult]:
        """
        Convenience method to build context and assemble messages in one call.

        Args:
            message: User's message
            clerk_user_id: User identifier
            selve_scores: SELVE dimension scores
            conversation_history: Previous messages
            use_rag: Whether to use RAG

        Returns:
            Tuple of (messages list, context result)
        """
        context_result = await self.build_context(
            message=message,
            clerk_user_id=clerk_user_id,
            selve_scores=selve_scores,
            use_rag=use_rag,
        )

        messages = self.build_messages(
            message=message,
            system_content=context_result.system_content,
            conversation_history=conversation_history,
            context_info=context_result.context_info,
        )

        return messages, context_result