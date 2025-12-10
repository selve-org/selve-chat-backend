"""
Context building service for SELVE Chat.
Handles RAG retrieval, episodic/semantic memory fetch, and message assembly.
"""
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from .rag_service import RAGService
from .compression_service import CompressionService
from .semantic_memory_service import SemanticMemoryService

RAG_TIMEOUT_SECONDS = 5.0
MEMORY_TIMEOUT_SECONDS = 3.0


@dataclass
class ContextResult:
    system_content: str
    context_info: Optional[Dict[str, Any]]
    sources_used: List[Dict[str, str]]
    user_context: Optional[str] = None
    memory_context: Optional[str] = None
    semantic_context: Optional[str] = None


class ContextService:
    def __init__(
        self,
        rag_service: RAGService,
        compression_service: CompressionService,
        semantic_memory_service: SemanticMemoryService,
        system_prompt: str,
        assessment_url: str,
    ) -> None:
        self.rag_service = rag_service
        self.compression_service = compression_service
        self.semantic_memory_service = semantic_memory_service
        self.system_prompt = system_prompt
        self.assessment_url = assessment_url.rstrip("/")

    async def _fetch_rag_context(self, message: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
            context_info = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.rag_service.get_context_for_query(message, top_k=top_k),
                ),
                timeout=RAG_TIMEOUT_SECONDS,
            )
            return context_info
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    async def _fetch_episodic_memories(self, clerk_user_id: str, limit: int = 3) -> Optional[List[Dict[str, Any]]]:
        try:
            memories = await asyncio.wait_for(
                self.compression_service.get_user_memories(clerk_user_id, limit=limit),
                timeout=MEMORY_TIMEOUT_SECONDS,
            )
            return memories
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    async def _fetch_semantic_memory(self, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        try:
            semantic_mem = await asyncio.wait_for(
                self.semantic_memory_service.get_user_semantic_memory(clerk_user_id=clerk_user_id),
                timeout=MEMORY_TIMEOUT_SECONDS,
            )
            return semantic_mem
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    async def build_context(
        self,
        message: str,
        clerk_user_id: Optional[str],
        selve_scores: Optional[Dict[str, float]],
        use_rag: bool,
        assessment_url: Optional[str] = None,
    ) -> ContextResult:
        tasks: Dict[str, asyncio.Task] = {}

        if use_rag:
            tasks["rag"] = asyncio.create_task(self._fetch_rag_context(message), name="fetch_rag")

        if clerk_user_id:
            tasks["memories"] = asyncio.create_task(
                self._fetch_episodic_memories(clerk_user_id),
                name="fetch_memories",
            )
            tasks["semantic"] = asyncio.create_task(
                self._fetch_semantic_memory(clerk_user_id),
                name="fetch_semantic",
            )

        results: Dict[str, Any] = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception:
                results[key] = None

        user_context = None
        if selve_scores:
            user_context = self._format_scores_for_context(selve_scores)

        memory_context = None
        if results.get("memories"):
            memory_context = self.compression_service.format_memories_for_context(results["memories"])

        semantic_context = None
        if results.get("semantic"):
            semantic_context = self.semantic_memory_service.format_semantic_memory_for_context(
                results["semantic"]
            )

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

        system_content = self.system_prompt
        assessment_link = assessment_url or self.assessment_url
        if not selve_scores and assessment_link:
            system_content = (
                f"{system_content}\n\nASSESSMENT CTA:\n"
                "- When you do not have the user's SELVE scores, invite them to take their assessment.\n"
                f"- Include a short call-to-action with this link: [Take the SELVE assessment]({assessment_link})\n"
                "- Keep it to one concise line before continuing with help."
            )
        if user_context:
            system_content = f"{system_content}\n\n{user_context}"
        if semantic_context:
            system_content = f"{system_content}\n\n{semantic_context}"
        if memory_context:
            system_content = f"{system_content}\n\n{memory_context}"

        return ContextResult(
            system_content=system_content,
            context_info=context_info,
            sources_used=sources_used,
            user_context=user_context,
            memory_context=memory_context,
            semantic_context=semantic_context,
        )

    def build_messages(
        self,
        message: str,
        system_content: str,
        conversation_history: List[Dict[str, str]],
        context_info: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_content}]

        if conversation_history:
            messages.extend(conversation_history)

        user_message = message
        if context_info and context_info.get("retrieved_count", 0) > 0:
            rag_context = context_info.get("context", "")
            user_message = f"<knowledge_context>\n{rag_context}\n</knowledge_context>\n\nUser Question: {message}"

        messages.append({"role": "user", "content": user_message})
        return messages

    def _format_scores_for_context(self, scores: Dict[str, float]) -> str:
        dimension_descriptions = {
            "LUMEN": "Mindful Curiosity (social energy and recharging)",
            "AETHER": "Rational Reflection (information processing)",
            "ORPHEUS": "Compassionate Connection (decision-making approach)",
            "ORIN": "Structured Harmony (planning and structure)",
            "LYRA": "Creative Expression (openness to experiences)",
            "VARA": "Purposeful Commitment (emotional stability)",
            "CHRONOS": "Adaptive Spontaneity (agreeableness and cooperation)",
            "KAEL": "Bold Resilience (conscientiousness and discipline)",
        }

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]

        context_parts = ["USER'S SELVE PROFILE:", "", "Strongest dimensions:"]

        for dim, score in top_3:
            desc = dimension_descriptions.get(dim, dim)
            context_parts.append(f"  • {dim}: {int(score)}/100 - {desc}")

        context_parts.extend(
            [
                "",
                "When responding:",
                "- Reference their specific scores when relevant to the question",
                "- Provide personalized insights based on their profile",
                "- Help them understand how their scores influence their behavior",
            ]
        )

        return "\n".join(context_parts)
