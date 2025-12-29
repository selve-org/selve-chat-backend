"""
Automated Web Crawler Service for SELVE

Crawls selve.me website automatically:
- Discovers all pages via sitemap.xml
- Recursively follows internal links
- Detects content changes via hashing
- Updates Qdrant vector database
- Scheduled to run daily

Production-ready with proper error handling and logging.
"""

import hashlib
import logging
import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import OpenAI

from app.services.base import Config

logger = logging.getLogger(__name__)


class WebCrawlerService:
    """
    Automated web crawler for SELVE content.

    Features:
    - Sitemap discovery
    - Recursive crawling
    - Change detection via content hashing
    - Vector database updates
    - Stale content removal
    """

    COLLECTION_NAME = "selve_web_content"
    BASE_URL = "https://selve.me"
    SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

    # Crawl settings
    CHUNK_SIZE = 600  # words
    CHUNK_OVERLAP = 75  # words
    MAX_DEPTH = 3  # Maximum link depth
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536

    # Rate limiting
    REQUEST_DELAY = 0.5  # seconds between requests

    def __init__(self):
        """Initialize web crawler service."""
        self.qdrant = QdrantClient(url=Config.QDRANT_URL)
        self.openai = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[aiohttp.ClientSession] = None

        # Track crawled URLs and content hashes
        self.crawled_urls: Set[str] = set()
        self.url_hashes: Dict[str, str] = {}  # url -> content_hash

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_internal_url(self, url: str) -> bool:
        """Check if URL is internal to selve.me."""
        parsed = urlparse(url)
        return parsed.netloc == "" or "selve.me" in parsed.netloc

    def _should_crawl_url(self, url: str) -> bool:
        """Determine if URL should be crawled."""
        # Skip non-HTML resources
        skip_extensions = {'.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.ico', '.svg', '.woff', '.ttf'}
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False

        # Skip external URLs
        if not self._is_internal_url(url):
            return False

        # Skip already crawled
        if url in self.crawled_urls:
            return False

        return True

    async def discover_sitemap_urls(self) -> List[str]:
        """Discover URLs from sitemap.xml."""
        urls = []

        try:
            async with self.session.get(self.SITEMAP_URL, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()

                    # Parse sitemap XML
                    root = ET.fromstring(content)

                    # Handle both regular sitemap and sitemap index
                    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                    # Try regular sitemap first
                    url_elements = root.findall('.//ns:url/ns:loc', namespace)
                    if url_elements:
                        urls = [elem.text for elem in url_elements if elem.text]
                    else:
                        # Try sitemap index (references to other sitemaps)
                        sitemap_elements = root.findall('.//ns:sitemap/ns:loc', namespace)
                        for elem in sitemap_elements:
                            if elem.text:
                                # Recursively fetch sub-sitemaps
                                sub_urls = await self._fetch_subsitemap(elem.text)
                                urls.extend(sub_urls)

                    self.logger.info(f"Discovered {len(urls)} URLs from sitemap")
                else:
                    self.logger.warning(f"Failed to fetch sitemap: {response.status}")

        except Exception as e:
            self.logger.error(f"Error fetching sitemap: {e}")

        return urls

    async def _fetch_subsitemap(self, sitemap_url: str) -> List[str]:
        """Fetch URLs from a sub-sitemap."""
        urls = []

        try:
            async with self.session.get(sitemap_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    root = ET.fromstring(content)
                    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                    url_elements = root.findall('.//ns:url/ns:loc', namespace)
                    urls = [elem.text for elem in url_elements if elem.text]

        except Exception as e:
            self.logger.error(f"Error fetching sub-sitemap {sitemap_url}: {e}")

        return urls

    async def crawl_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Crawl a single page and extract content.

        Returns:
            Dict with url, title, content, content_hash, internal_links
        """
        try:
            await asyncio.sleep(self.REQUEST_DELAY)  # Rate limiting

            async with self.session.get(url, timeout=15) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch {url}: {response.status}")
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title
                title_elem = soup.find('title')
                title = title_elem.get_text().strip() if title_elem else url

                # Remove script, style, nav, footer
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()

                # Extract main content
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if not main_content:
                    self.logger.warning(f"No main content found for {url}")
                    return None

                # Get text content
                text = main_content.get_text(separator=' ', strip=True)

                # Clean up whitespace
                text = ' '.join(text.split())

                if len(text) < 50:  # Skip pages with too little content
                    self.logger.info(f"Skipping {url} - insufficient content ({len(text)} chars)")
                    return None

                # Compute content hash
                content_hash = self._compute_content_hash(text)

                # Extract internal links for recursive crawling
                internal_links = set()
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)

                    # Normalize URL (remove fragments, trailing slashes)
                    parsed = urlparse(absolute_url)
                    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')

                    if self._should_crawl_url(normalized):
                        internal_links.add(normalized)

                self.logger.info(f"Crawled {url} - {len(text)} chars, {len(internal_links)} links")

                return {
                    "url": url,
                    "title": title,
                    "content": text,
                    "content_hash": content_hash,
                    "internal_links": list(internal_links),
                    "crawled_at": datetime.utcnow().isoformat(),
                }

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout crawling {url}")
            return None

        except Exception as e:
            self.logger.error(f"Error crawling {url}: {e}")
            return None

    async def crawl_recursive(self, start_urls: List[str], max_depth: int = MAX_DEPTH) -> List[Dict[str, Any]]:
        """
        Recursively crawl pages starting from seed URLs.

        Args:
            start_urls: Initial URLs to crawl
            max_depth: Maximum link depth to follow

        Returns:
            List of crawled page data
        """
        pages = []
        to_crawl = [(url, 0) for url in start_urls]  # (url, depth)

        while to_crawl:
            url, depth = to_crawl.pop(0)

            if url in self.crawled_urls or depth > max_depth:
                continue

            self.crawled_urls.add(url)

            page_data = await self.crawl_page(url)
            if page_data:
                pages.append(page_data)

                # Add internal links to crawl queue (at next depth level)
                if depth < max_depth:
                    for link in page_data['internal_links']:
                        if link not in self.crawled_urls:
                            to_crawl.append((link, depth + 1))

        self.logger.info(f"Crawled {len(pages)} pages total")
        return pages

    def chunk_content(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk page content into smaller segments."""
        content = page_data['content']
        words = content.split()
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + self.CHUNK_SIZE
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Create unique ID
            chunk_id_str = f"{page_data['url']}_chunk_{chunk_index}_{page_data['content_hash'][:8]}"
            chunk_id = hashlib.md5(chunk_id_str.encode()).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "chunk_index": chunk_index,
                "url": page_data['url'],
                "title": page_data['title'],
                "content_hash": page_data['content_hash'],
                "crawled_at": page_data['crawled_at'],
            })

            chunk_index += 1
            start = end - self.CHUNK_OVERLAP

        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding for text."""
        response = self.openai.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    async def get_existing_hashes(self) -> Dict[str, str]:
        """
        Get existing content hashes from Qdrant.

        Returns:
            Dict mapping url -> content_hash
        """
        try:
            # Scroll through all points to get hashes
            result = self.qdrant.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=1000,
                with_payload=["url", "content_hash"],
                with_vectors=False,
            )

            url_hashes = {}
            for point in result[0]:
                url = point.payload.get("url")
                content_hash = point.payload.get("content_hash")
                if url and content_hash:
                    url_hashes[url] = content_hash

            return url_hashes

        except Exception as e:
            self.logger.error(f"Error getting existing hashes: {e}")
            return {}

    async def delete_stale_urls(self, current_urls: Set[str]):
        """Delete entries for URLs that no longer exist."""
        try:
            existing_hashes = await self.get_existing_hashes()
            stale_urls = set(existing_hashes.keys()) - current_urls

            if stale_urls:
                self.logger.info(f"Deleting {len(stale_urls)} stale URLs")

                for url in stale_urls:
                    # Delete all chunks for this URL
                    self.qdrant.delete(
                        collection_name=self.COLLECTION_NAME,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="url",
                                    match=MatchValue(value=url),
                                )
                            ]
                        ),
                    )
                    self.logger.info(f"Deleted stale URL: {url}")

        except Exception as e:
            self.logger.error(f"Error deleting stale URLs: {e}")

    async def update_vector_database(self, pages: List[Dict[str, Any]]):
        """
        Update Qdrant with crawled content.

        Only updates pages whose content has changed.
        """
        # Get existing content hashes
        existing_hashes = await self.get_existing_hashes()

        # Determine which pages need updating
        pages_to_update = []
        for page in pages:
            url = page['url']
            new_hash = page['content_hash']
            old_hash = existing_hashes.get(url)

            if old_hash != new_hash:
                if old_hash:
                    self.logger.info(f"Content changed for {url} - updating")
                else:
                    self.logger.info(f"New page {url} - adding")

                pages_to_update.append(page)
            else:
                self.logger.info(f"No changes for {url} - skipping")

        if not pages_to_update:
            self.logger.info("No pages to update")
            return

        # Delete old chunks for updated pages
        for page in pages_to_update:
            try:
                self.qdrant.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="url",
                                match=MatchValue(value=page['url']),
                            )
                        ]
                    ),
                )
            except Exception as e:
                self.logger.warning(f"Error deleting old chunks for {page['url']}: {e}")

        # Chunk and embed new content
        all_chunks = []
        for page in pages_to_update:
            chunks = self.chunk_content(page)
            all_chunks.extend(chunks)

        self.logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")

        points = []
        for i, chunk in enumerate(all_chunks):
            if i % 10 == 0:
                self.logger.info(f"Progress: {i}/{len(all_chunks)}")

            embedding = self.generate_embedding(chunk["text"])

            point = PointStruct(
                id=chunk["id"],
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "url": chunk["url"],
                    "title": chunk["title"],
                    "content_hash": chunk["content_hash"],
                    "category": self._categorize_url(chunk["url"]),
                    "priority": self._get_priority(chunk["url"]),
                    "crawled_at": chunk["crawled_at"],
                },
            )
            points.append(point)

        # Upload to Qdrant
        self.logger.info(f"Uploading {len(points)} points to Qdrant...")
        self.qdrant.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

        self.logger.info(f"✅ Updated {len(pages_to_update)} pages ({len(points)} chunks)")

    def _categorize_url(self, url: str) -> str:
        """Categorize URL based on path."""
        if '/blog/' in url:
            return 'blog'
        elif '/privacy' in url:
            return 'legal'
        elif '/terms' in url:
            return 'legal'
        elif '/how-it-works' in url or '/features' in url:
            return 'product'
        else:
            return 'landing'

    def _get_priority(self, url: str) -> int:
        """Get priority based on URL."""
        if url == self.BASE_URL or url == f"{self.BASE_URL}/":
            return 1  # Highest priority
        elif '/how-it-works' in url or '/features' in url:
            return 1
        elif '/blog/' in url:
            return 2
        elif '/privacy' in url or '/terms' in url:
            return 3
        else:
            return 2

    async def run_full_crawl(self):
        """
        Run complete crawl and update cycle.

        This is the main entry point for the crawler.
        """
        self.logger.info("=" * 60)
        self.logger.info("🕷️  Starting SELVE Web Crawler")
        self.logger.info("=" * 60)

        start_time = datetime.utcnow()

        # Ensure collection exists
        try:
            self.qdrant.get_collection(self.COLLECTION_NAME)
        except Exception:
            self.logger.info(f"Creating collection '{self.COLLECTION_NAME}'")
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )

        # Step 1: Discover URLs from sitemap
        sitemap_urls = await self.discover_sitemap_urls()

        # Step 2: Recursively crawl pages
        pages = await self.crawl_recursive(sitemap_urls or [self.BASE_URL])

        # Step 3: Update vector database (only changed content)
        await self.update_vector_database(pages)

        # Step 4: Delete stale URLs
        current_urls = {page['url'] for page in pages}
        await self.delete_stale_urls(current_urls)

        # Summary
        duration = (datetime.utcnow() - start_time).total_seconds()

        self.logger.info("=" * 60)
        self.logger.info(f"✅ Crawl Complete")
        self.logger.info(f"📄 Pages crawled: {len(pages)}")
        self.logger.info(f"🔗 URLs discovered: {len(self.crawled_urls)}")
        self.logger.info(f"⏱️  Duration: {duration:.1f}s")
        self.logger.info("=" * 60)


async def run_crawler():
    """Standalone function to run the crawler."""
    async with WebCrawlerService() as crawler:
        await crawler.run_full_crawl()


if __name__ == "__main__":
    # For testing
    asyncio.run(run_crawler())
