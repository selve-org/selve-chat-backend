"""
SELVE Chatbot Backend API
FastAPI application with RAG-powered chat endpoint
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging

from app.routers import chat, compression, ingestion, sessions, users
from app.db import connect_db, disconnect_db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    await connect_db()
    logger.info("✅ Database connected")
    logger.info(f"📡 API running on port {os.getenv('PORT', '8000')}")
    yield
    # Shutdown
    logger.info("👋 Shutting down SELVE Chatbot Backend...")
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

# CORS middleware (configure based on your frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(compression.router)
app.include_router(ingestion.router)
app.include_router(sessions.router)
app.include_router(users.router)


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
