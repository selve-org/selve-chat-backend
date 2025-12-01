# SELVE Chatbot Backend API

FastAPI-based backend with RAG (Retrieval-Augmented Generation) for the SELVE personality framework chatbot.

## Features

- **RAG-Powered Chat**: Retrieves relevant context from Qdrant vector database
- **OpenAI Integration**: Uses GPT-4o-mini for response generation
- **SELVE Framework**: Specialized in 8 personality dimensions
- **FastAPI**: Modern, fast, and well-documented API
- **CORS Support**: Ready for frontend integration

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Frontend (Next.js)              │
└───────────────┬─────────────────────────┘
                │ HTTP/REST
┌───────────────▼─────────────────────────┐
│       FastAPI Backend (Port 8000)       │
│  ┌──────────────────────────────────┐   │
│  │  /api/chat - Main chat endpoint  │   │
│  │  /api/health - Health check      │   │
│  │  /api/context - Test retrieval   │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌─────────────┐    ┌───────────────┐   │
│  │ RAG Service │    │ Chat Service  │   │
│  └──────┬──────┘    └───────┬───────┘   │
└─────────┼─────────────────── ┼──────────┘
          │                    │
    ┌─────▼──────┐      ┌─────▼──────┐
    │  Qdrant    │      │   OpenAI   │
    │ (Vector DB)│      │    API     │
    └────────────┘      └────────────┘
```

---

## Setup

### Prerequisites

- Python 3.10+
- Qdrant running on `localhost:6333` (see docker-compose.yml)
- OpenAI API key with access to embeddings and chat models

### Installation

```bash
# Navigate to backend directory
cd selve-chat-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Ensure your `.env` file contains:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-...your-key-here...
OPENAI_CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=selve_knowledge

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

---

## Running the Server

### Development Mode (with auto-reload)

```bash
cd selve-chat-backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

Or simply:

```bash
python app/main.py
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

---

## API Endpoints

### 1. Chat Endpoint

**POST** `/api/chat`

Generate a chatbot response with RAG context.

**Request Body**:
```json
{
  "message": "What is LUMEN and how does it affect my social life?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Tell me about my SELVE results"
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help! Could you share which dimension you'd like to explore?"
    }
  ],
  "use_rag": true
}
```

**Response**:
```json
{
  "response": "LUMEN measures your social energy...",
  "context_used": true,
  "retrieved_chunks": [
    {
      "content": "Dimension: LUMEN\n\nLUMEN comes from...",
      "dimension": "LUMEN",
      "section": "Overview",
      "title": "LUMEN - Overview",
      "score": 0.85,
      "source": "dimension"
    }
  ],
  "model": "gpt-4o-mini"
}
```

### 2. Health Check

**GET** `/api/health`

Check service health and connectivity.

**Response**:
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "collection_points": 91,
  "services": {
    "qdrant": true,
    "openai": true
  }
}
```

### 3. Test Context Retrieval

**GET** `/api/context?query=What is LUMEN?&top_k=3`

Test RAG retrieval without generating a response.

**Response**:
```json
{
  "context": "Relevant SELVE Framework Context:\n\n[1] LUMEN - Overview (relevance: 0.85)\n...",
  "chunks": [...],
  "retrieved_count": 3
}
```

---

## Project Structure

```
selve-chat-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── models/
│   │   ├── __init__.py
│   │   └── chat.py          # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   └── chat.py          # API endpoints
│   └── services/
│       ├── __init__.py
│       ├── rag_service.py   # Qdrant retrieval
│       └── chat_service.py  # OpenAI integration
├── requirements.txt
├── .env
└── README.md
```

---

## Testing

### Manual Testing with curl

**Health Check**:
```bash
curl http://localhost:8000/api/health
```

**Chat Request**:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is LUMEN?",
    "use_rag": true
  }'
```

**Context Retrieval**:
```bash
curl "http://localhost:8000/api/context?query=What%20is%20LUMEN?&top_k=3"
```

### Using FastAPI Docs

Visit http://localhost:8000/docs for interactive API documentation where you can:
- View all endpoints
- Test requests directly in the browser
- See request/response schemas
- Download OpenAPI spec

---

## Configuration

### RAG Parameters

In `app/services/rag_service.py`:

- `top_k`: Number of context chunks to retrieve (default: 3)
- `score_threshold`: Minimum similarity score (default: 0.3)
- `embedding_model`: OpenAI embedding model (default: text-embedding-3-small)

### Chat Parameters

In `app/services/chat_service.py`:

- `model`: OpenAI chat model (default: gpt-4o-mini)
- `temperature`: Response creativity (default: 0.7)
- `max_tokens`: Maximum response length (default: 500)

---

## Troubleshooting

### Issue: Connection to Qdrant Failed

**Error**: `Connection to Qdrant at localhost:6333 failed`

**Solution**:
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not, start it
cd /home/chris/selve-org
docker compose up -d qdrant
```

### Issue: OpenAI Authentication Error

**Error**: `401 Unauthorized - Incorrect API key`

**Solution**:
- Verify `.env` file has correct `OPENAI_API_KEY`
- Ensure you're using the production key (not dev key)
- Check API key has access to embeddings and chat models

### Issue: No Context Retrieved

**Error**: Retrieved chunks is empty

**Solution**:
```bash
# Check if Qdrant collection has data
curl http://localhost:6333/collections/selve_knowledge

# If points_count is 0, re-index dimensions
cd /home/chris/selve-org
OPENAI_API_KEY="..." scripts/venv/bin/python scripts/index_dimensions.py
```

---

## Development

### Adding New Endpoints

1. Create route function in `app/routers/chat.py`
2. Add Pydantic models in `app/models/chat.py` if needed
3. Update tests

### Modifying System Prompt

Edit the system prompt in `app/services/chat_service.py` → `_load_system_prompt()`

### Adding New Services

1. Create service file in `app/services/`
2. Import in router
3. Use dependency injection pattern

---

## Deployment

### Production Checklist

- [ ] Set `CORS_ORIGINS` to production domain
- [ ] Use production-grade ASGI server (uvicorn with workers)
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and logging
- [ ] Configure rate limiting
- [ ] Enable API authentication (if needed)
- [ ] Set up backup for Qdrant data

### Deployment Script

```bash
# Install dependencies
pip install -r requirements.txt

# Run with multiple workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Next Steps

- [ ] Add authentication (Clerk integration)
- [ ] Implement conversation memory (Redis)
- [ ] Add streaming responses
- [ ] Create user profile context injection
- [ ] Add rate limiting
- [ ] Set up logging and monitoring
- [ ] Write integration tests
- [ ] Deploy to production

---

**Version**: 1.0.0
**Last Updated**: 2025-12-01
**Maintained By**: Christopher (with Claude Code assistance)
