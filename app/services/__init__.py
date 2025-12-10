"""
SELVE Chat Services - Refactored for Security & Robustness

This package provides production-grade services for the SELVE chatbot with:
- Comprehensive input validation
- Proper error handling with Result types
- Security hardening (user isolation, rate limiting ready)
- DRY principles (shared base utilities)
- No sensitive data in logs
- Clean separation of concerns

Services:
- RAGService: Vector search through knowledge base
- MemorySearchService: Episodic memory search
- SemanticMemoryService: Long-term pattern extraction
- ContentIngestionService: Knowledge base ingestion
- ContentValidationService: Content validation against SELVE
- ContextService: Context orchestration for chat
- ConversationStateService: Conversation tracking

Usage:
	from services import (
		RAGService,
		MemorySearchService,
		SemanticMemoryService,
		ContextService,
	)

	# Services use lazy-loaded shared clients
	rag_service = RAGService()
	result = rag_service.get_context_for_query("What careers fit my personality?")

Configuration (via environment variables):
	OPENAI_API_KEY: OpenAI API key (required)
	EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small)
	QDRANT_HOST: Qdrant host (default: localhost)
	QDRANT_PORT: Qdrant port (default: 6333)
	QDRANT_URL: Qdrant URL (for cloud deployments)
	QDRANT_COLLECTION_NAME: Collection name (default: selve_knowledge)
	RAG_TIMEOUT_SECONDS: RAG timeout (default: 5.0)
	MEMORY_TIMEOUT_SECONDS: Memory timeout (default: 3.0)
"""

from .base import (
	# Configuration
	Config,
	# Result types
	Result,
	ResultStatus,
	# Exceptions
	ServiceError,
	ValidationError,
	ConfigurationError,
	ExternalServiceError,
	# Utilities
	Validator,
	ClientManager,
	generate_content_hash,
	safe_json_parse,
	# Base class
	BaseService,
)

from .rag_service import (
	RAGService,
	RAGResult,
	RetrievedChunk,
)

from .memory_search_service import (
	MemorySearchService,
	MemorySearchResult,
)

from .semantic_memory_service import (
	SemanticMemoryService,
	SemanticPattern,
	ConfidenceLevel,
)

from .content_ingestion_service import (
	ContentIngestionService,
	IngestionResult,
	BatchIngestionResult,
	ChunkingConfig,
)

from .content_validation_service import (
	ContentValidationService,
	ValidationResult,
	ValidationStatus,
	ValidationScore,
)

from .context_service import (
	ContextService,
	ContextResult,
)

from .conversation_state_service import (
	ConversationStateService,
	ConversationState,
	EmotionalTone,
	ConversationIntent,
)

__all__ = [
	# Base
	"Config",
	"Result",
	"ResultStatus",
	"ServiceError",
	"ValidationError",
	"ConfigurationError",
	"ExternalServiceError",
	"Validator",
	"ClientManager",
	"BaseService",
	"generate_content_hash",
	"safe_json_parse",
	# RAG
	"RAGService",
	"RAGResult",
	"RetrievedChunk",
	# Memory Search
	"MemorySearchService",
	"MemorySearchResult",
	# Semantic Memory
	"SemanticMemoryService",
	"SemanticPattern",
	"ConfidenceLevel",
	# Content Ingestion
	"ContentIngestionService",
	"IngestionResult",
	"BatchIngestionResult",
	"ChunkingConfig",
	# Content Validation
	"ContentValidationService",
	"ValidationResult",
	"ValidationStatus",
	"ValidationScore",
	# Context
	"ContextService",
	"ContextResult",
	# Conversation State
	"ConversationStateService",
	"ConversationState",
	"EmotionalTone",
	"ConversationIntent",
]

__version__ = "2.0.0"
