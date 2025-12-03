"""
Database client for SELVE Chatbot
Connects to shared PostgreSQL database using Prisma Python Client
"""
from prisma import Prisma
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Global Prisma client instance
db = Prisma()


@asynccontextmanager
async def get_db() -> AsyncGenerator[Prisma, None]:
    """
    Async context manager for database connections

    Usage:
        async with get_db() as db:
            session = await db.chatsession.find_first(...)
    """
    if not db.is_connected():
        await db.connect()

    try:
        yield db
    finally:
        # Keep connection open for reuse
        pass


async def connect_db():
    """Connect to the database (call on startup)"""
    if not db.is_connected():
        await db.connect()
        print("✅ Database connected")


async def disconnect_db():
    """Disconnect from the database (call on shutdown)"""
    if db.is_connected():
        await db.disconnect()
        print("✅ Database disconnected")
