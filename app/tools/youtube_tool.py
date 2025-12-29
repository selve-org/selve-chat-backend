"""
YouTube Transcript Search Tool

Searches psychology YouTube transcripts for information.
Uses semantic search via Qdrant to find relevant content.
"""

import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

from app.services.base import Config

logger = logging.getLogger(__name__)


class YouTubeTranscriptTool:
    """
    Tool for searching YouTube psychology transcripts.

    Use this when the user asks about psychological concepts
    that might be explained in educational videos.
    """

    COLLECTION_NAME = "youtube_transcripts"
    DEFAULT_TOP_K = 3
    SCORE_THRESHOLD = 0.4  # Balanced threshold for relevance

    def __init__(self):
        """Initialize YouTube transcript search tool."""
        self.qdrant = QdrantClient(url=Config.QDRANT_URL)
        self.openai = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.logger = logging.getLogger(self.__class__.__name__)

    def search_transcripts(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        channel_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search YouTube transcripts for relevant information.

        Args:
            query: What to search for
            top_k: Number of results to return (max 5)
            channel_filter: Optional channel name to filter by (e.g., "TED-Ed")

        Returns:
            Dict with:
                - context: Formatted context string
                - sources: List of source dicts with title, url, channel
                - videos: List of unique video IDs
        """
        try:
            # Validate inputs
            top_k = min(max(top_k, 1), 5)  # Clamp to 1-5

            # Generate query embedding
            embedding = self._generate_embedding(query)

            # Build filter (optional channel filter)
            search_filter = None
            if channel_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="channel",
                            match=MatchValue(value=channel_filter),
                        )
                    ]
                )

            # Search Qdrant
            results = self.qdrant.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k * 2,  # Get more, then filter
                query_filter=search_filter,
            )

            # Filter by score threshold and format results
            relevant_results = [r for r in results if r.score >= self.SCORE_THRESHOLD]

            if not relevant_results:
                self.logger.info(f"No YouTube transcripts found above threshold {self.SCORE_THRESHOLD}")
                return {
                    "context": None,
                    "sources": [],
                    "videos": [],
                }

            # Limit to top_k
            relevant_results = relevant_results[:top_k]

            # Format context
            context_parts = []
            sources = []
            seen_videos = set()

            for i, result in enumerate(relevant_results, 1):
                payload = result.payload
                video_id = payload.get("video_id", "")

                # Build context
                score_pct = int(result.score * 100)
                context_parts.append(
                    f"{i}. {payload.get('title', 'Unknown')} (relevance: {score_pct}%)\n"
                    f"   Channel: {payload.get('channel', 'Unknown')}\n"
                    f"   {payload.get('text', '')[:300]}...\n"
                )

                # Add to sources (deduplicate by video ID)
                if video_id not in seen_videos:
                    sources.append({
                        "title": payload.get("title", "Unknown"),
                        "url": payload.get("url", ""),
                        "channel": payload.get("channel", "Unknown"),
                        "video_id": video_id,
                        "relevance": int(result.score * 100),
                    })
                    seen_videos.add(video_id)

            context = "\n".join(context_parts)

            self.logger.info(
                f"Found {len(relevant_results)} relevant YouTube transcript chunks "
                f"from {len(sources)} videos"
            )

            return {
                "context": context,
                "sources": sources,
                "videos": list(seen_videos),
            }

        except Exception as e:
            self.logger.error(f"YouTube transcript search failed: {e}")
            return {
                "context": None,
                "sources": [],
                "videos": [],
            }

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        response = self.openai.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding


# =============================================================================
# Convenience Functions
# =============================================================================


async def search_youtube_transcripts(
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Search psychology YouTube transcripts for information.

    Use this when the user asks about psychological concepts
    that might be explained in educational videos (e.g., TED-Ed).

    Examples:
    - "How does critical thinking work?"
    - "What is narcissism?"
    - "Why do people procrastinate?"

    Args:
        query: What to search for
        top_k: Number of transcript snippets to return (1-5)

    Returns:
        Dict with context, sources, and video IDs
    """
    tool = YouTubeTranscriptTool()
    return tool.search_transcripts(query, top_k)
