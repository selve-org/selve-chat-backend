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
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID to save messages to"
    )
    clerk_user_id: Optional[str] = Field(
        default=None,
        description="Clerk user ID for personalization"
    )
    selve_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="User's SELVE personality scores for personalization"
    )
    assessment_url: Optional[str] = Field(
        default=None,
        description="URL for the SELVE assessment (used when scores are missing)"
    )

    # Regeneration context fields
    regeneration_type: Optional[str] = Field(
        default=None,
        description="Type of regeneration: 'regenerate' or 'edit'"
    )
    parent_message_id: Optional[str] = Field(
        default=None,
        description="ID of the parent message being regenerated/edited"
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Group ID for versioning related messages"
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
    compression_needed: bool = Field(default=False, description="Whether compression is recommended")
    total_tokens: Optional[int] = Field(default=None, description="Total conversation tokens")

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


class FeedbackRequest(BaseModel):
    """Request body for feedback endpoint"""
    message_id: str = Field(..., description="ID of the message to provide feedback for")
    feedback_type: str = Field(..., description="Type of feedback: 'helpful' or 'not_helpful'")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    clerk_user_id: Optional[str] = Field(default=None, description="Clerk user ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "clk123456",
                "feedback_type": "helpful",
                "session_id": "session_123",
                "clerk_user_id": "user_123"
            }
        }


class FeedbackResponse(BaseModel):
    """Response from feedback endpoint"""
    success: bool
    message: str
    collection_points: int = 0
    services: Dict[str, bool] = Field(default_factory=dict)
