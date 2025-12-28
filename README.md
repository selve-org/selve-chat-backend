# SELVE Chatbot - Backend

**Live:** [api-chat.selve.me](https://api-chat.selve.me)  
**Status:** Production

FastAPI backend with RAG (Retrieval-Augmented Generation) for SELVE personality chatbot. Retrieves psychology content from Qdrant and generates responses using OpenAI/Claude.

## Features

- **RAG Pipeline**: Retrieves relevant psychology content from vector database
- **Multi-LLM Support**: OpenAI GPT-4o-mini + Claude Haiku 4.5
- **Observability**: Langfuse integration for LLM tracing and cost tracking
- **SELVE Knowledge**: Specialized in 8 personality dimensions
- **FastAPI**: Auto-documented REST API

## Quick Start

```bash
# Install dependencies
make install

# Set up environment
cp .env.example .env  # Then edit .env with your API keys

# Start Qdrant (vector database)
cd ..
docker compose up -d qdrant

# Run server (port 9000)
make dev  # or: source venv/bin/activate && uvicorn main:app --reload --port 9000
```

## Architecture

```
Frontend (Next.js) → FastAPI → Qdrant (retrieval) → OpenAI/Claude → Response
                              ↘ Langfuse (observability)
```

## API Endpoints

- `POST /api/chat` - Send message, get AI response
- `GET /api/health` - Health check
- `GET /api/context?query={text}` - Test RAG retrieval

## Tech Stack

**Framework:** FastAPI (Python)  
**LLMs:** OpenAI GPT-4o-mini, Claude Haiku 4.5  
**Vector DB:** Qdrant (local + Docker)  
**Embeddings:** OpenAI text-embedding-3-small  
**Observability:** Langfuse  
**Deployment:** AWS EC2

## Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

## Development

```bash
make dev      # Start server (port 9000)
make test     # Run tests

```

**Docs:** http://localhost:9000/docs

## Related Repos

- [selve](https://github.com/selve-org/selve) - Main assessment platform
- [selve-chat-frontend](https://github.com/selve-org/selve-chat-frontend) - Chat UI
