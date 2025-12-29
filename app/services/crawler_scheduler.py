"""
Web Crawler Scheduler

Automatically runs web crawler daily to keep content up-to-date.

Schedule:
- Runs every day at 3:00 AM UTC
- Can be triggered manually via API endpoint

Production-ready with proper error handling and logging.
"""

import logging
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.web_crawler_service import WebCrawlerService

logger = logging.getLogger(__name__)


class CrawlerScheduler:
    """
    Scheduler for automated web crawling.

    Runs daily at 3:00 AM UTC to keep SELVE web content updated.
    """

    def __init__(self):
        """Initialize crawler scheduler."""
        self.scheduler = AsyncIOScheduler()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False

    async def _run_crawl_job(self):
        """Execute crawl job (called by scheduler)."""
        try:
            self.logger.info("🕷️  Starting scheduled web crawl...")

            async with WebCrawlerService() as crawler:
                await crawler.run_full_crawl()

            self.logger.info("✅ Scheduled web crawl completed successfully")

        except Exception as e:
            self.logger.error(f"❌ Scheduled web crawl failed: {e}", exc_info=True)

    def start(self):
        """Start the scheduler."""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return

        # Schedule daily crawl at 3:00 AM UTC
        self.scheduler.add_job(
            self._run_crawl_job,
            trigger=CronTrigger(hour=3, minute=0),  # 3:00 AM UTC daily
            id='daily_web_crawl',
            name='Daily SELVE Web Crawl',
            replace_existing=True,
        )

        self.scheduler.start()
        self.is_running = True

        self.logger.info("✅ Crawler scheduler started - will run daily at 3:00 AM UTC")

    def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            return

        self.scheduler.shutdown(wait=True)
        self.is_running = False

        self.logger.info("Crawler scheduler stopped")

    async def run_now(self):
        """Manually trigger a crawl (for testing or manual updates)."""
        self.logger.info("Manual crawl triggered")
        await self._run_crawl_job()

    def get_next_run_time(self) -> str:
        """Get the next scheduled run time."""
        job = self.scheduler.get_job('daily_web_crawl')
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return "Not scheduled"

    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "is_running": self.is_running,
            "next_run_time": self.get_next_run_time(),
            "job_count": len(self.scheduler.get_jobs()),
        }


# Global scheduler instance
_crawler_scheduler: CrawlerScheduler = None


def get_crawler_scheduler() -> CrawlerScheduler:
    """Get the global crawler scheduler instance."""
    global _crawler_scheduler
    if _crawler_scheduler is None:
        _crawler_scheduler = CrawlerScheduler()
    return _crawler_scheduler


def start_crawler_scheduler():
    """Start the crawler scheduler (call on app startup)."""
    scheduler = get_crawler_scheduler()
    scheduler.start()


def stop_crawler_scheduler():
    """Stop the crawler scheduler (call on app shutdown)."""
    scheduler = get_crawler_scheduler()
    scheduler.stop()
