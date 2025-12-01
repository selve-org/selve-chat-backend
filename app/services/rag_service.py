"""
RAG Service for SELVE Chatbot
Retrieves relevant context from Qdrant vector database
"""
import os
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient


class RAGService:
    """Service for RAG (Retrieval-Augmented Generation)"""

    def __init__(self):
        """Initialize OpenAI and Qdrant clients"""
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "selve_knowledge")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from Qdrant

        Args:
            query: User's query text
            top_k: Number of top results to retrieve
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Search in Qdrant
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        # Filter by score threshold and format results
        context_chunks = []
        for result in results:
            if result.score >= score_threshold:
                context_chunks.append({
                    "content": result.payload.get("content", ""),
                    "dimension": result.payload.get("dimension_name", ""),
                    "section": result.payload.get("section", ""),
                    "title": result.payload.get("title", ""),
                    "score": result.score,
                    "source": result.payload.get("source", "dimension")
                })

        return context_chunks

    def format_context_for_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context into a prompt string"""
        if not context_chunks:
            return "No relevant context found."

        formatted = "Relevant SELVE Framework Context:\n\n"
        for i, chunk in enumerate(context_chunks, 1):
            formatted += f"[{i}] {chunk['title']} (relevance: {chunk['score']:.2f})\n"
            formatted += f"{chunk['content']}\n\n"

        return formatted.strip()

    def get_context_for_query(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Main method to get formatted context for a query

        Returns:
            {
                "context": "formatted context string",
                "chunks": [list of chunk metadata],
                "retrieved_count": int
            }
        """
        chunks = self.retrieve_context(query, top_k)
        formatted_context = self.format_context_for_prompt(chunks)

        return {
            "context": formatted_context,
            "chunks": chunks,
            "retrieved_count": len(chunks)
        }
