"""
Web Crawler API Router

Endpoints for managing automated web crawling.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.crawler_scheduler import get_crawler_scheduler
from app.services.web_crawler_service import WebCrawlerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/crawler", tags=["crawler"])


class CrawlerStatusResponse(BaseModel):
    """Crawler status response."""
    is_running: bool
    next_run_time: str
    job_count: int


class CrawlTriggerResponse(BaseModel):
    """Crawl trigger response."""
    status: str
    message: str


@router.get("/status", response_model=CrawlerStatusResponse)
async def get_crawler_status():
    """
    Get crawler scheduler status.

    Returns information about the automated crawler:
    - Whether it's running
    - Next scheduled run time
    - Number of scheduled jobs
    """
    try:
        scheduler = get_crawler_scheduler()
        status = scheduler.get_status()

        return CrawlerStatusResponse(**status)

    except Exception as e:
        logger.error(f"Error getting crawler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger", response_model=CrawlTriggerResponse)
async def trigger_crawl(background_tasks: BackgroundTasks):
    """
    Manually trigger a web crawl.

    Runs the crawler immediately in the background.
    Useful for:
    - Testing the crawler
    - Forcing an update after content changes
    - Initial setup

    The crawl runs in the background and doesn't block the response.
    """
    try:
        scheduler = get_crawler_scheduler()

        # Run crawl in background
        background_tasks.add_task(scheduler.run_now)

        return CrawlTriggerResponse(
            status="started",
            message="Web crawl started in background. Check logs for progress."
        )

    except Exception as e:
        logger.error(f"Error triggering crawl: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def crawler_health():
    """Health check for crawler service."""
    try:
        scheduler = get_crawler_scheduler()
        return {
            "status": "healthy",
            "scheduler_running": scheduler.is_running,
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
