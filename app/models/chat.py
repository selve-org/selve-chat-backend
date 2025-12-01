"""
Pydantic models for chat API
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint"""
    message: str = Field(..., description="User's current message", min_length=1)
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Previous conversation messages"
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG context retrieval"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is LUMEN and how does it affect my social life?",
                "conversation_history": [
                    {"role": "user", "content": "Tell me about my SELVE results"},
                    {"role": "assistant", "content": "I'd be happy to help! Could you share which dimension you'd like to explore?"}
                ],
                "use_rag": True
            }
        }


class ContextChunk(BaseModel):
    """Retrieved context chunk metadata"""
    content: str
    dimension: str
    section: str
    title: str
    score: float
    source: str


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str = Field(..., description="Assistant's response")
    context_used: bool = Field(..., description="Whether RAG context was used")
    retrieved_chunks: List[ContextChunk] = Field(
        default=[],
        description="Context chunks retrieved from knowledge base"
    )
    model: str = Field(..., description="OpenAI model used")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "LUMEN measures your social energy and how you recharge...",
                "context_used": True,
                "retrieved_chunks": [
                    {
                        "content": "Dimension: LUMEN\\n\\nLUMEN comes from the Latin word for 'light'...",
                        "dimension": "LUMEN",
                        "section": "Overview",
                        "title": "LUMEN - Overview",
                        "score": 0.85,
                        "source": "dimension"
                    }
                ],
                "model": "gpt-4o-mini"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    qdrant_connected: bool
    collection_points: int
    services: Dict[str, bool]
