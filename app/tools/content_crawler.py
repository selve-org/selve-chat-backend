"""
Content crawling tools for external personality-related content.
Operates as MCP-like tools that the LLM can invoke.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import httpx
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrawledContent:
    """Represents crawled and validated content."""
    source: str
    url: str
    title: str
    content: str
    relevance_score: float
    is_valid: bool
    crawled_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "is_valid": self.is_valid,
            "crawled_at": self.crawled_at.isoformat(),
            "metadata": self.metadata,
        }


class ContentValidator:
    """Validates crawled content for personality relevance."""
    
    ALLOWED_TOPICS = [
        "personality", "temperament", "introvert", "extrovert",
        "mbti", "big five", "psychology", "self-improvement",
        "emotional intelligence", "self-awareness", "personal growth",
        "relationships", "communication style", "behavior",
        "motivation", "values", "strengths", "weaknesses",
        "career fit", "leadership", "team dynamics",
        "stress", "coping", "mindfulness", "wellbeing"
    ]
    
    BLOCKED_TOPICS = [
        "programming", "coding", "software", "politics",
        "religion", "medical", "legal", "financial",
        "cryptocurrency", "stocks", "gambling",
        "violence", "weapons", "drugs", "hate"
    ]
    
    def validate(self, content: str) -> Tuple[bool, float]:
        """
        Validate content for personality relevance.
        
        Returns:
            Tuple of (is_valid, relevance_score)
        """
        if not content or len(content.strip()) < 50:
            return False, 0.0
            
        content_lower = content.lower()
        
        # Check for blocked topics
        for topic in self.BLOCKED_TOPICS:
            if topic in content_lower:
                logger.debug(f"Content blocked due to topic: {topic}")
                return False, 0.0
        
        # Calculate relevance based on allowed topics
        matches = sum(1 for topic in self.ALLOWED_TOPICS if topic in content_lower)
        relevance = min(matches / 5, 1.0)  # Cap at 1.0, need 5 matches for full score
        
        is_valid = relevance > 0.2  # At least 1 topic match
        
        return is_valid, round(relevance, 2)


class YouTubeTool:
    """Extract and validate YouTube video transcripts."""
    
    def __init__(self):
        self.validator = ContentValidator()
        self._transcript_api = None
    
    def _get_transcript_api(self):
        """Lazy load YouTube transcript API."""
        if self._transcript_api is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                self._transcript_api = YouTubeTranscriptApi
            except ImportError:
                logger.warning("youtube_transcript_api not installed")
                return None
        return self._transcript_api
    
    async def get_transcript(self, video_url: str) -> Optional[CrawledContent]:
        """
        Extract transcript from YouTube video.
        
        Args:
            video_url: Full YouTube video URL
            
        Returns:
            CrawledContent if successful and valid, None otherwise
        """
        video_id = self._extract_video_id(video_url)
        if not video_id:
            logger.warning(f"Could not extract video ID from URL: {video_url}")
            return None
        
        api = self._get_transcript_api()
        if api is None:
            return None
        
        try:
            # Get transcript (this is sync, but we're wrapping in async)
            transcript = api.get_transcript(video_id)
            full_text = " ".join([t["text"] for t in transcript])
            
            is_valid, score = self.validator.validate(full_text)
            
            return CrawledContent(
                source="youtube",
                url=video_url,
                title=f"YouTube Video {video_id}",
                content=full_text[:5000],  # Limit length
                relevance_score=score,
                is_valid=is_valid,
                metadata={
                    "video_id": video_id,
                    "duration_segments": len(transcript),
                }
            )
        except Exception as e:
            logger.error(f"Failed to get YouTube transcript: {e}")
            return None
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
            r"(?:embed\/)([0-9A-Za-z_-]{11})",
            r"(?:shorts\/)([0-9A-Za-z_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None


class RedditTool:
    """Fetch personality-related Reddit discussions."""
    
    ALLOWED_SUBREDDITS = [
        "psychology", "selfimprovement", "mbti", 
        "intj", "infp", "enfp", "infj", "entp",
        "personality", "emotionalintelligence",
        "decidingtobebetter", "getdisciplined"
    ]
    
    def __init__(self):
        self.validator = ContentValidator()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": "SELVE-Bot/1.0 (Personality Research)"}
            )
        return self._client
    
    async def search(self, query: str, limit: int = 5) -> List[CrawledContent]:
        """
        Search Reddit for personality-related content.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of validated CrawledContent objects
        """
        results = []
        client = await self._get_client()
        
        # Search top 3 most relevant subreddits
        for subreddit in self.ALLOWED_SUBREDDITS[:3]:
            try:
                response = await client.get(
                    f"https://www.reddit.com/r/{subreddit}/search.json",
                    params={
                        "q": query,
                        "limit": limit,
                        "restrict_sr": True,
                        "sort": "relevance",
                        "t": "year"  # Last year
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for post in data.get("data", {}).get("children", []):
                        post_data = post.get("data", {})
                        
                        # Combine title and content
                        content = f"{post_data.get('title', '')}\n{post_data.get('selftext', '')}"
                        
                        is_valid, score = self.validator.validate(content)
                        if is_valid:
                            results.append(CrawledContent(
                                source="reddit",
                                url=f"https://reddit.com{post_data.get('permalink', '')}",
                                title=post_data.get('title', 'Untitled'),
                                content=content[:2000],  # Limit length
                                relevance_score=score,
                                is_valid=True,
                                metadata={
                                    "subreddit": subreddit,
                                    "score": post_data.get("score", 0),
                                    "num_comments": post_data.get("num_comments", 0),
                                }
                            ))
                elif response.status_code == 429:
                    logger.warning("Reddit rate limit hit")
                    break
                    
            except Exception as e:
                logger.error(f"Reddit search error for r/{subreddit}: {e}")
                continue
        
        # Sort by relevance and return top results
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:limit]
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class WebPageTool:
    """Fetch and validate web page content."""
    
    def __init__(self):
        self.validator = ContentValidator()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SELVE-Bot/1.0)",
                    "Accept": "text/html,application/xhtml+xml"
                }
            )
        return self._client
    
    async def fetch(self, url: str) -> Optional[CrawledContent]:
        """
        Fetch and validate a web page.
        
        Args:
            url: URL to fetch
            
        Returns:
            CrawledContent if successful and valid, None otherwise
        """
        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return None
        
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: status {response.status_code}")
                return None
            
            # Extract text content (basic HTML stripping)
            html = response.text
            text = self._extract_text(html)
            title = self._extract_title(html)
            
            is_valid, score = self.validator.validate(text)
            
            return CrawledContent(
                source="webpage",
                url=url,
                title=title or url,
                content=text[:5000],  # Limit length
                relevance_score=score,
                is_valid=is_valid,
                metadata={
                    "content_type": response.headers.get("content-type", ""),
                    "content_length": len(text),
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch webpage {url}: {e}")
            return None
    
    def _extract_text(self, html: str) -> str:
        """Extract text from HTML (basic implementation)."""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML."""
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# =============================================================================
# Tool Registry
# =============================================================================

# Singleton instances of tools
_youtube_tool: Optional[YouTubeTool] = None
_reddit_tool: Optional[RedditTool] = None
_webpage_tool: Optional[WebPageTool] = None


def _get_youtube_tool() -> YouTubeTool:
    global _youtube_tool
    if _youtube_tool is None:
        _youtube_tool = YouTubeTool()
    return _youtube_tool


def _get_reddit_tool() -> RedditTool:
    global _reddit_tool
    if _reddit_tool is None:
        _reddit_tool = RedditTool()
    return _reddit_tool


def _get_webpage_tool() -> WebPageTool:
    global _webpage_tool
    if _webpage_tool is None:
        _webpage_tool = WebPageTool()
    return _webpage_tool


# Tool registry for LLM function calling
CONTENT_TOOLS = {
    "youtube_transcript": {
        "name": "youtube_transcript",
        "description": "Extract transcript from a YouTube video about personality topics",
        "get_instance": _get_youtube_tool,
        "method": "get_transcript",
        "parameters": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "Full YouTube video URL"
                }
            },
            "required": ["video_url"]
        }
    },
    "reddit_search": {
        "name": "reddit_search", 
        "description": "Search Reddit for personality-related discussions",
        "get_instance": _get_reddit_tool,
        "method": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for personality topics"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    "webpage_fetch": {
        "name": "webpage_fetch",
        "description": "Fetch and validate content from a web page",
        "get_instance": _get_webpage_tool,
        "method": "fetch",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        }
    }
}


def get_tool(name: str) -> Optional[dict]:
    """Get a tool by name."""
    return CONTENT_TOOLS.get(name)


def list_tools() -> List[dict]:
    """List all available tools with their schemas."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"]
        }
        for tool in CONTENT_TOOLS.values()
    ]


async def invoke_tool(name: str, **kwargs) -> Optional[CrawledContent | List[CrawledContent]]:
    """
    Invoke a tool by name with the given arguments.
    
    Args:
        name: Tool name
        **kwargs: Tool arguments
        
    Returns:
        Tool result or None if tool not found
    """
    tool = CONTENT_TOOLS.get(name)
    if not tool:
        logger.error(f"Unknown tool: {name}")
        return None
    
    instance = tool["get_instance"]()
    method = getattr(instance, tool["method"])
    
    try:
        return await method(**kwargs)
    except Exception as e:
        logger.error(f"Error invoking tool {name}: {e}")
        return None
