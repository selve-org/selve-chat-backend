"""
SELVE Web Content Search Tool

Searches indexed SELVE web content (selve.me) for information about:
- How SELVE works
- Product features
- Privacy & terms
- Blog content
"""

import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

from app.services.base import Config

logger = logging.getLogger(__name__)


class SelveWebTool:
    """
    Tool for searching SELVE web content.

    Use this when users ask about:
    - How SELVE works
    - SELVE features
    - Privacy policy
    - Terms & conditions
    - General product information
    """

    COLLECTION_NAME = "selve_web_content"
    DEFAULT_TOP_K = 3
    SCORE_THRESHOLD = 0.4  # Balanced threshold

    def __init__(self):
        """Initialize SELVE web search tool."""
        self.qdrant = QdrantClient(url=Config.QDRANT_URL)
        self.openai = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.logger = logging.getLogger(self.__class__.__name__)

    def search_content(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search SELVE web content for relevant information.

        Args:
            query: What to search for
            top_k: Number of results to return (max 5)
            category_filter: Optional category filter (landing, product, blog, legal)

        Returns:
            Dict with:
                - context: Formatted context string
                - sources: List of source dicts with title, url, category
                - pages: List of unique page URLs
        """
        try:
            # Validate inputs
            top_k = min(max(top_k, 1), 5)

            # Generate query embedding
            embedding = self._generate_embedding(query)

            # Build filter (optional category filter)
            search_filter = None
            if category_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category_filter),
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

            # Filter by score threshold
            relevant_results = [r for r in results if r.score >= self.SCORE_THRESHOLD]

            if not relevant_results:
                self.logger.info(f"No SELVE web content found above threshold {self.SCORE_THRESHOLD}")
                return {
                    "context": None,
                    "sources": [],
                    "pages": [],
                }

            # Limit to top_k
            relevant_results = relevant_results[:top_k]

            # Format context
            context_parts = []
            sources = []
            seen_pages = set()

            for i, result in enumerate(relevant_results, 1):
                payload = result.payload
                url = payload.get("url", "")

                # Build context
                score_pct = int(result.score * 100)
                context_parts.append(
                    f"{i}. {payload.get('title', 'Unknown')} (relevance: {score_pct}%)\n"
                    f"   Category: {payload.get('category', 'unknown')}\n"
                    f"   {payload.get('text', '')[:400]}...\n"
                )

                # Add to sources (deduplicate by URL)
                if url not in seen_pages:
                    sources.append({
                        "title": payload.get("title", "Unknown"),
                        "url": url,
                        "category": payload.get("category", "unknown"),
                        "relevance": int(result.score * 100),
                    })
                    seen_pages.add(url)

            context = "\n".join(context_parts)

            self.logger.info(
                f"Found {len(relevant_results)} relevant SELVE web content chunks "
                f"from {len(sources)} pages"
            )

            return {
                "context": context,
                "sources": sources,
                "pages": list(seen_pages),
            }

        except Exception as e:
            self.logger.error(f"SELVE web content search failed: {e}")
            return {
                "context": None,
                "sources": [],
                "pages": [],
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


async def search_selve_web(
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Search SELVE web content for information about the product.

    Use this when users ask about:
    - How SELVE works
    - SELVE features and capabilities
    - Privacy policy or data handling
    - Terms & conditions
    - General product information

    Examples:
    - "How does SELVE work?"
    - "What is your privacy policy?"
    - "What features does SELVE have?"

    Args:
        query: What to search for
        top_k: Number of content snippets to return (1-5)

    Returns:
        Dict with context, sources, and page URLs
    """
    tool = SelveWebTool()
    return tool.search_content(query, top_k)
