"""
SELVE Chatbot Backend API
FastAPI application with RAG-powered chat endpoint
"""
import os
import logging
import re
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# Load environment variables from repo-local .env before importing app modules
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from app.routers import chat, compression, ingestion, sessions, users, crawler
from app.db import connect_db, disconnect_db
from app.logging_config import setup_logging
from app.middleware.request_logging import RequestLoggingMiddleware
from app.services.crawler_scheduler import start_crawler_scheduler, stop_crawler_scheduler

# Setup production logging with PII scrubbing and file rotation
setup_logging(app_name="selve-chat-backend")
logger = logging.getLogger(__name__)


def scrub_pii_and_llm_data(event, hint):
    """
    Scrub PII and sensitive LLM data from Sentry events

    Scrubs:
    - User emails
    - Financial data (costs)
    - LLM prompts/responses (may contain user PII)
    """
    # Scrub user emails
    if "user" in event and "email" in event["user"]:
        event["user"]["email"] = "***@***.***"

    # Scrub costs from messages
    if "message" in event and isinstance(event["message"], str):
        event["message"] = re.sub(r'\$\d+\.\d+', '$X.XX', event["message"])

    # Scrub LLM prompts/responses from breadcrumbs (may contain user PII)
    if "breadcrumbs" in event:
        for crumb in event["breadcrumbs"]:
            if crumb.get("category") == "llm" and "data" in crumb:
                # Truncate long LLM responses to avoid sending user conversations
                if "response" in crumb["data"]:
                    response = str(crumb["data"]["response"])
                    if len(response) > 200:
                        crumb["data"]["response"] = response[:200] + "...[TRUNCATED]"
                # Remove full prompts
                if "prompt" in crumb["data"]:
                    crumb["data"]["prompt"] = "[REDACTED - contains user data]"

    return event

# Configure rate limiter
# Uses Redis if available (configured via REDIS_URL env var), falls back to in-memory
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"],  # Default per-IP limit
    storage_uri=os.getenv("REDIS_URL", "memory://"),
    strategy="fixed-window"
)

# ASCII Art Banner
SELVE_BANNER = """
███████╗███████╗██╗░░░░░██╗░░░██╗███████╗░░░░░░░█████╗░██╗░░██╗░█████╗░████████╗
██╔════╝██╔════╝██║░░░░░██║░░░██║██╔════╝░░░░░░██╔══██╗██║░░██║██╔══██╗╚══██╔══╝
███████╗█████╗░░██║░░░░░╚██╗░██╔╝█████╗░░█████╗██║░░╚═╝███████║███████║░░░██║░░░
╚════██║██╔══╝░░██║░░░░░░╚████╔╝░██╔══╝░░╚════╝██║░░██╗██╔══██║██╔══██║░░░██║░░░
███████║███████╗███████╗░░╚██╔╝░░███████╗░░░░░░╚█████╔╝██║░░██║██║░░██║░░░██║░░░
╚══════╝╚══════╝╚══════╝░░░╚═╝░░░╚══════╝░░░░░░░╚════╝░╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░

              ✨ Discover Your True Self ✨
     Self-Exploration • Learning • Validation • Evolution
                      https://selve.me
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    print(SELVE_BANNER)
    logger.info("🚀 Starting SELVE Chatbot Backend...")

    # Initialize Sentry for error tracking (production only)
    sentry_dsn = os.getenv("SENTRY_DSN")
    environment = os.getenv("ENVIRONMENT", "development")
    if sentry_dsn and environment == "production":
        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=environment,
            integrations=[FastApiIntegration(transaction_style="endpoint")],
            traces_sample_rate=0.1,  # 10% sampling
            before_send=scrub_pii_and_llm_data,
        )
        logger.info("✅ Sentry initialized for error tracking (with LLM data scrubbing)")
    else:
        logger.info(f"ℹ️ Sentry disabled ({environment} mode or missing DSN)")

    await connect_db()
    logger.info("✅ Database connected")

    # Start automated web crawler scheduler
    try:
        start_crawler_scheduler()
        logger.info("✅ Web crawler scheduler started (runs daily at 3AM UTC)")
    except Exception as e:
        logger.warning(f"Failed to start crawler scheduler: {e}")

    logger.info(f"📡 API running on port {os.getenv('PORT', '8000')}")
    yield
    # Shutdown
    logger.info("👋 Shutting down SELVE Chatbot Backend...")

    # Stop crawler scheduler
    try:
        stop_crawler_scheduler()
        logger.info("✅ Web crawler scheduler stopped")
    except Exception as e:
        logger.warning(f"Failed to stop crawler scheduler: {e}")

    # Flush Langfuse traces before shutdown
    try:
        from app.services.langfuse_service import get_langfuse_service
        langfuse_service = get_langfuse_service()
        logger.info("🔄 Flushing Langfuse traces...")
        langfuse_service.flush()
        logger.info("✅ Langfuse traces flushed")
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse traces: {e}")

    await disconnect_db()
    logger.info("✅ Database disconnected")


# Create FastAPI app
app = FastAPI(
    title="SELVE Chatbot API",
    description="RAG-powered chatbot for the SELVE personality framework with dual LLM support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware (configure based on your frontend domain)
# Restrict methods and headers to only what's needed for security (SEC-4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With", "X-User-ID"],
)

# Request logging middleware with tracing
app.add_middleware(RequestLoggingMiddleware)


# Sentry context middleware for LLM tracking
@app.middleware("http")
async def add_sentry_llm_context(request: Request, call_next):
    """Add LLM-specific context to Sentry events"""
    if sentry_sdk.Hub.current.client:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("service", "chat-backend")
            scope.set_tag("llm_provider", os.getenv("LLM_PROVIDER", "openai"))
            # Add user context if available
            user_id = request.headers.get("X-User-ID")
            if user_id:
                scope.set_user({"id": user_id[:8] + "***"})

    response = await call_next(request)
    return response


# Include routers
app.include_router(chat.router)
app.include_router(compression.router)
app.include_router(ingestion.router)
app.include_router(sessions.router)
app.include_router(users.router)
app.include_router(crawler.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "SELVE Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
