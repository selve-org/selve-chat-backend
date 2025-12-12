"""
Thinking Engine - Agentic Reasoning for SELVE Chatbot.

This is the "brain" of the chatbot. Instead of a simple LLM call,
it implements a multi-step reasoning process:

1. ANALYZE: What does the user actually need?
2. PLAN: What tools/research are needed?
3. EXECUTE: Run RAG, web search, validation
4. SYNTHESIZE: Combine findings into response

Each step emits status events for the ThinkingIndicator UI.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class ThinkingConfig:
    """Thinking engine configuration."""
    
    # Analysis settings
    INTENT_CLASSIFICATION_ENABLED: bool = True
    
    # Tool settings
    RAG_ENABLED: bool = True
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
    YOUTUBE_SEARCH_ENABLED: bool = os.getenv("YOUTUBE_SEARCH_ENABLED", "false").lower() == "true"
    
    # Response settings
    MAX_THINKING_STEPS: int = 5
    THINKING_TIMEOUT_SECONDS: float = 30.0
    
    # Model selection
    THINKING_MODEL: str = "claude-haiku-4-5"  # For reasoning
    RESPONSE_MODEL: str = "claude-haiku-4-5"  # For final response


# =============================================================================
# DATA TYPES
# =============================================================================


class ThinkingPhase(str, Enum):
    """Phases of the thinking process."""
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    RESEARCHING = "researching"
    PERSONALIZING = "personalizing"
    SYNTHESIZING = "synthesizing"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETE = "complete"
    ERROR = "error"


class UserIntent(str, Enum):
    """Classified user intents."""
    GREETING = "greeting"
    QUESTION_ABOUT_SELF = "question_about_self"
    QUESTION_ABOUT_SELVE = "question_about_selve"
    QUESTION_ABOUT_DIMENSION = "question_about_dimension"
    SEEKING_ADVICE = "seeking_advice"
    EXPLORING_PERSONALITY = "exploring_personality"
    RELATIONSHIP_QUESTION = "relationship_question"
    CAREER_QUESTION = "career_question"
    EMOTIONAL_SUPPORT = "emotional_support"
    OFF_TOPIC = "off_topic"
    UNCLEAR = "unclear"
    FOLLOW_UP = "follow_up"


@dataclass
class ThinkingStep:
    """A single step in the thinking process."""
    
    phase: ThinkingPhase
    description: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0
    
    def complete(self, result: Any = None) -> None:
        self.completed_at = datetime.utcnow()
        self.result = result
    
    def fail(self, error: str) -> None:
        self.completed_at = datetime.utcnow()
        self.error = error


@dataclass
class ThinkingStatus:
    """Status event for UI display."""
    
    phase: ThinkingPhase
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "thinking_status",
            "phase": self.phase.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class AnalysisResult:
    """Result of analyzing user message."""
    
    intent: UserIntent
    confidence: float
    needs_rag: bool
    needs_personality_context: bool
    needs_web_research: bool
    key_topics: List[str]
    referenced_dimensions: List[str]
    is_follow_up: bool
    emotional_tone: Optional[str]
    

@dataclass
class PlanStep:
    """A planned action step."""
    
    action: str  # "rag_search", "fetch_personality", "web_search", etc.
    priority: int
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of executing the plan."""
    
    rag_context: Optional[str] = None
    rag_sources: List[Dict[str, str]] = field(default_factory=list)
    web_research: Optional[str] = None
    web_sources: List[Dict[str, str]] = field(default_factory=list)
    personality_insights: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class ThinkingResult:
    """Complete result of the thinking process."""
    
    response: str
    reasoning: str  # Internal reasoning (for logging)
    sources: List[Dict[str, str]]
    steps: List[ThinkingStep]
    user_intent: UserIntent
    confidence: float
    should_add_note: bool = False
    note_content: Optional[str] = None
    total_duration_ms: float = 0.0


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================


class IntentClassifier:
    """
    Classifies user intent without LLM call.
    
    Uses pattern matching and heuristics for speed.
    """
    
    # Greeting patterns
    GREETING_PATTERNS = [
        r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|what'?s\s*up)\s*[!.?]*\s*$",
        r"^\s*(yo|sup|hiya|howdy)\s*[!.?]*\s*$",
    ]
    
    # Dimension keywords
    DIMENSION_KEYWORDS = {
        "LUMEN": ["lumen", "social", "introvert", "extrovert", "energy", "recharge"],
        "AETHER": ["aether", "thinking", "information", "analytical", "intuitive"],
        "ORPHEUS": ["orpheus", "empathy", "emotional", "decisions", "feeling"],
        "ORIN": ["orin", "organization", "structure", "planning", "discipline"],
        "LYRA": ["lyra", "creative", "artistic", "imagination", "openness"],
        "VARA": ["vara", "stability", "emotional", "grounded", "steady"],
        "CHRONOS": ["chronos", "time", "spontaneous", "adaptable", "flexible"],
        "KAEL": ["kael", "resilience", "courage", "bold", "brave"],
    }
    
    # Career keywords
    CAREER_PATTERNS = [
        r"\b(career|job|work|profession|occupation|employ)\b",
        r"\b(interview|resume|cv|hiring|salary)\b",
        r"\b(boss|colleague|coworker|manager|team)\b",
    ]
    
    # Relationship keywords
    RELATIONSHIP_PATTERNS = [
        r"\b(relationship|partner|spouse|boyfriend|girlfriend|husband|wife)\b",
        r"\b(dating|marriage|divorce|breakup)\b",
        r"\b(friend|family|parent|sibling|child)\b",
    ]
    
    # Emotional support patterns
    EMOTIONAL_PATTERNS = [
        r"\b(feel|feeling|felt)\s+(sad|anxious|stressed|worried|overwhelmed)\b",
        r"\b(i'?m|i\s+am)\s+(struggling|having\s+a\s+hard\s+time|not\s+okay)\b",
        r"\b(help\s+me|need\s+support|can'?t\s+cope)\b",
    ]
    
    @classmethod
    def classify(
        cls,
        message: str,
        user_state: Any,  # UserState type
        conversation_history: List[Dict[str, str]],
    ) -> AnalysisResult:
        """
        Classify user intent and analyze message.
        
        Args:
            message: User's message
            user_state: Current user state
            conversation_history: Recent conversation
            
        Returns:
            AnalysisResult with intent and analysis
        """
        message_lower = message.lower().strip()
        
        # Check for follow-up
        is_follow_up = cls._is_follow_up(message, conversation_history)
        
        # Detect referenced dimensions
        referenced_dimensions = cls._detect_dimensions(message_lower)
        
        # Detect emotional tone
        emotional_tone = cls._detect_emotion(message_lower)
        
        # Check if user has assessment
        has_assessment = getattr(user_state, 'has_assessment', False)
        
        # === Classify Intent ===
        
        # Greeting
        for pattern in cls.GREETING_PATTERNS:
            if re.match(pattern, message_lower, re.I):
                return AnalysisResult(
                    intent=UserIntent.GREETING,
                    confidence=0.95,
                    needs_rag=False,
                    needs_personality_context=True,
                    needs_web_research=False,
                    key_topics=[],
                    referenced_dimensions=[],
                    is_follow_up=False,
                    emotional_tone=None,
                )
        
        # Question about specific dimension
        if referenced_dimensions:
            return AnalysisResult(
                intent=UserIntent.QUESTION_ABOUT_DIMENSION,
                confidence=0.85,
                needs_rag=True,
                needs_personality_context=has_assessment,
                needs_web_research=False,
                key_topics=referenced_dimensions,
                referenced_dimensions=referenced_dimensions,
                is_follow_up=is_follow_up,
                emotional_tone=emotional_tone,
            )
        
        # Career question
        for pattern in cls.CAREER_PATTERNS:
            if re.search(pattern, message_lower, re.I):
                return AnalysisResult(
                    intent=UserIntent.CAREER_QUESTION,
                    confidence=0.8,
                    needs_rag=True,
                    needs_personality_context=True,
                    needs_web_research=False,
                    key_topics=["career"],
                    referenced_dimensions=[],
                    is_follow_up=is_follow_up,
                    emotional_tone=emotional_tone,
                )
        
        # Relationship question
        for pattern in cls.RELATIONSHIP_PATTERNS:
            if re.search(pattern, message_lower, re.I):
                return AnalysisResult(
                    intent=UserIntent.RELATIONSHIP_QUESTION,
                    confidence=0.8,
                    needs_rag=True,
                    needs_personality_context=True,
                    needs_web_research=False,
                    key_topics=["relationships"],
                    referenced_dimensions=[],
                    is_follow_up=is_follow_up,
                    emotional_tone=emotional_tone,
                )
        
        # Emotional support
        for pattern in cls.EMOTIONAL_PATTERNS:
            if re.search(pattern, message_lower, re.I):
                return AnalysisResult(
                    intent=UserIntent.EMOTIONAL_SUPPORT,
                    confidence=0.85,
                    needs_rag=False,
                    needs_personality_context=True,
                    needs_web_research=False,
                    key_topics=["emotional support"],
                    referenced_dimensions=[],
                    is_follow_up=is_follow_up,
                    emotional_tone=emotional_tone or "distressed",
                )
        
        # Question about SELVE framework
        if "selve" in message_lower or "assessment" in message_lower:
            return AnalysisResult(
                intent=UserIntent.QUESTION_ABOUT_SELVE,
                confidence=0.8,
                needs_rag=True,
                needs_personality_context=True,
                needs_web_research=False,
                key_topics=["selve framework"],
                referenced_dimensions=[],
                is_follow_up=is_follow_up,
                emotional_tone=emotional_tone,
            )
        
        # Question about scores/results (highest priority for assessment users)
        if has_assessment:
            score_patterns = [
                r"\b(show|tell|what).{0,20}\b(my|me)\b.{0,20}\b(score|result|dimension|profile)s?\b",
                r"\b(my|me)\b.{0,20}\b(score|result|dimension|profile)s?\b",
                r"\bwhat\s+(are|is)\s+my\b.{0,20}\b(score|result|dimension)s?\b",
            ]
            for pattern in score_patterns:
                if re.search(pattern, message_lower, re.I):
                    return AnalysisResult(
                        intent=UserIntent.QUESTION_ABOUT_SELF,
                        confidence=0.95,
                        needs_rag=False,
                        needs_personality_context=True,
                        needs_web_research=False,
                        key_topics=["scores", "profile"],
                        referenced_dimensions=[],
                        is_follow_up=is_follow_up,
                        emotional_tone=emotional_tone,
                    )
        
        # Question about self (personality exploration)
        self_patterns = [
            r"\b(my|me|i|myself)\b.*\b(personality|type|trait|character)\b",
            r"\bwhat\s+(am|kind|type)\s+i\b",
            r"\bwho\s+am\s+i\b",
            r"\babout\s+me\b",
        ]
        for pattern in self_patterns:
            if re.search(pattern, message_lower, re.I):
                return AnalysisResult(
                    intent=UserIntent.EXPLORING_PERSONALITY,
                    confidence=0.8,
                    needs_rag=True,
                    needs_personality_context=True,
                    needs_web_research=False,
                    key_topics=["personality"],
                    referenced_dimensions=[],
                    is_follow_up=is_follow_up,
                    emotional_tone=emotional_tone,
                )
        
        # External references or research-heavy questions
        external_patterns = [
            r"\b(article|blog|study|research|paper|book|video|youtube|podcast|report)\b",
            r"(found|came across|saw|read|heard).{0,40}(about|that|saying)",
            r"\b(what|who|where|when|how).{0,50}(is|are|was|were).{0,50}(company|organization|person|technology|product)\b",
            r"(latest|recent|new|current).{0,30}(news|trends|updates|research|studies)\b",
        ]
        if any(re.search(pattern, message_lower) for pattern in external_patterns):
            return AnalysisResult(
                intent=UserIntent.OFF_TOPIC,
                confidence=0.7,
                needs_rag=False,
                needs_personality_context=False,
                needs_web_research=True,
                key_topics=["research"],
                referenced_dimensions=[],
                is_follow_up=is_follow_up,
                emotional_tone=emotional_tone,
            )
        
        # Follow-up or continuation
        if is_follow_up:
            return AnalysisResult(
                intent=UserIntent.FOLLOW_UP,
                confidence=0.7,
                needs_rag=True,
                needs_personality_context=True,
                needs_web_research=False,
                key_topics=[],
                referenced_dimensions=[],
                is_follow_up=True,
                emotional_tone=emotional_tone,
            )
        
        # Default: unclear, use RAG to help
        return AnalysisResult(
            intent=UserIntent.UNCLEAR,
            confidence=0.5,
            needs_rag=True,
            needs_personality_context=True,
            needs_web_research=False,
            key_topics=[],
            referenced_dimensions=[],
            is_follow_up=is_follow_up,
            emotional_tone=emotional_tone,
        )
    
    @classmethod
    def _is_follow_up(
        cls,
        message: str,
        history: List[Dict[str, str]],
    ) -> bool:
        """Check if message is a follow-up to previous conversation."""
        if not history:
            return False
        
        message_lower = message.lower().strip()
        
        # Short affirmative responses
        if message_lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "tell me", "1", "2", "3"]:
            return True
        
        # Pronouns without clear antecedent
        if re.match(r"^(it|that|this|they|them)\b", message_lower):
            return True
        
        # "More" requests
        if re.match(r"^(more|continue|go on|and|also)\b", message_lower):
            return True
        
        return False
    
    @classmethod
    def _detect_dimensions(cls, message_lower: str) -> List[str]:
        """Detect SELVE dimensions mentioned in message."""
        found = []
        for dim, keywords in cls.DIMENSION_KEYWORDS.items():
            for kw in keywords:
                if kw in message_lower:
                    found.append(dim)
                    break
        return found
    
    @classmethod
    def _detect_emotion(cls, message_lower: str) -> Optional[str]:
        """Detect emotional tone of message."""
        emotions = {
            "anxious": r"\b(anxious|anxiety|nervous|worried|worry)\b",
            "sad": r"\b(sad|depressed|down|unhappy|miserable)\b",
            "angry": r"\b(angry|frustrated|annoyed|irritated|mad)\b",
            "excited": r"\b(excited|thrilled|happy|joy|amazing)\b",
            "confused": r"\b(confused|unsure|don'?t\s+know|uncertain)\b",
            "curious": r"\b(curious|wondering|interested|want\s+to\s+know)\b",
        }
        
        for emotion, pattern in emotions.items():
            if re.search(pattern, message_lower, re.I):
                return emotion
        
        return None


# =============================================================================
# THINKING ENGINE
# =============================================================================


class ThinkingEngine:
    """
    Agentic reasoning engine for SELVE chatbot.
    
    Implements multi-step thinking process:
    1. ANALYZE: Classify intent, understand need
    2. PLAN: Determine what tools/data needed
    3. EXECUTE: Run RAG, fetch data
    4. SYNTHESIZE: Combine into coherent response
    """
    
    def __init__(
        self,
        llm_service=None,
        rag_service=None,
        compression_service=None,
    ):
        """
        Initialize thinking engine.
        
        Args:
            llm_service: LLM service for generation
            rag_service: RAG service for retrieval
            compression_service: Compression service for memories
        """
        self._llm_service = llm_service
        self._rag_service = rag_service
        self._compression_service = compression_service
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def llm_service(self):
        """Lazy-loaded LLM service."""
        if self._llm_service is None:
            from .llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service
    
    @property
    def rag_service(self):
        """Lazy-loaded RAG service."""
        if self._rag_service is None:
            from .rag_service import RAGService
            self._rag_service = RAGService()
        return self._rag_service
    
    @property
    def compression_service(self):
        """Lazy-loaded compression service."""
        if self._compression_service is None:
            from .compression_service import CompressionService
            self._compression_service = CompressionService()
        return self._compression_service
    
    # =========================================================================
    # Main Thinking Method (Streaming)
    # =========================================================================
    
    async def think_and_respond(
        self,
        message: str,
        user_state: Any,  # UserState type
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
        emit_status: bool = True,
    ) -> AsyncGenerator[Any, None]:
        """
        Main thinking method - analyzes, plans, executes, generates.
        
        Yields status events and response chunks.
        
        Args:
            message: User's message
            user_state: Complete user state
            conversation_history: Recent conversation
            system_prompt: Base system prompt
            emit_status: Whether to emit status events
            
        Yields:
            ThinkingStatus events and response text chunks
        """
        steps: List[ThinkingStep] = []
        start_time = datetime.utcnow()
        
        try:
            # === PHASE 1: ANALYZE ===
            if emit_status:
                yield ThinkingStatus(
                    phase=ThinkingPhase.ANALYZING,
                    message="Understanding your question...",
                    details={"step": 1, "total": 4},
                ).to_dict()
            
            step = ThinkingStep(
                phase=ThinkingPhase.ANALYZING,
                description="Analyzing user intent",
            )
            
            analysis = IntentClassifier.classify(
                message=message,
                user_state=user_state,
                conversation_history=conversation_history,
            )
            
            step.complete(analysis)
            steps.append(step)
            
            self.logger.info(
                f"Intent: {analysis.intent.value} "
                f"(confidence: {analysis.confidence:.2f})"
            )
            
            # === PHASE 2: PLAN ===
            if emit_status:
                yield ThinkingStatus(
                    phase=ThinkingPhase.PLANNING,
                    message="Planning response approach...",
                    details={"step": 2, "total": 4},
                ).to_dict()
            
            step = ThinkingStep(
                phase=ThinkingPhase.PLANNING,
                description="Creating execution plan",
            )
            
            plan = self._create_plan(analysis, user_state)
            step.complete(plan)
            steps.append(step)
            
            # === PHASE 3: EXECUTE ===
            if emit_status:
                yield ThinkingStatus(
                    phase=ThinkingPhase.RETRIEVING,
                    message="Gathering relevant information...",
                    details={"step": 3, "total": 4},
                ).to_dict()
            
            step = ThinkingStep(
                phase=ThinkingPhase.RETRIEVING,
                description="Executing plan",
            )
            
            execution_result = await self._execute_plan(plan, message, analysis)
            step.complete(execution_result)
            steps.append(step)
            
            # === PHASE 4: GENERATE ===
            if emit_status:
                yield ThinkingStatus(
                    phase=ThinkingPhase.GENERATING,
                    message="Crafting your response...",
                    details={"step": 4, "total": 4},
                ).to_dict()
            
            step = ThinkingStep(
                phase=ThinkingPhase.GENERATING,
                description="Generating response",
            )
            
            # Build enhanced system prompt
            enhanced_prompt = self._build_enhanced_prompt(
                base_prompt=system_prompt,
                user_state=user_state,
                analysis=analysis,
                execution_result=execution_result,
            )
            
            # Build messages
            messages = self._build_messages(
                system_prompt=enhanced_prompt,
                message=message,
                conversation_history=conversation_history,
                rag_context=execution_result.rag_context,
                web_research=execution_result.web_research,
            )
            
            # Stream response
            full_response_chunks = []
            
            async for chunk in self.llm_service.generate_response_stream(
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            ):
                if isinstance(chunk, str):
                    full_response_chunks.append(chunk)
                    yield chunk
                elif isinstance(chunk, dict) and chunk.get("__metadata__"):
                    # Capture metadata
                    pass
            
            full_response = "".join(full_response_chunks)
            step.complete({"response_length": len(full_response)})
            steps.append(step)
            
            # === COMPLETE ===
            if emit_status:
                yield ThinkingStatus(
                    phase=ThinkingPhase.COMPLETE,
                    message="Response complete",
                    details={
                        "sources": execution_result.rag_sources,
                        "intent": analysis.intent.value,
                    },
                ).to_dict()
            
            # Calculate total duration
            total_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"Thinking complete: {len(steps)} steps, "
                f"{total_duration:.0f}ms total"
            )
        
        except Exception as e:
            self.logger.error(f"Thinking error: {e}", exc_info=True)
            
            if emit_status:
                yield ThinkingStatus(
                    phase=ThinkingPhase.ERROR,
                    message="An error occurred",
                    details={"error": str(e)},
                ).to_dict()
            
            # Yield error message
            yield "I apologize, but I encountered an issue processing your request. Could you try rephrasing that?"
    
    # =========================================================================
    # Planning
    # =========================================================================
    
    def _create_plan(
        self,
        analysis: AnalysisResult,
        user_state: Any,
    ) -> List[PlanStep]:
        """Create execution plan based on analysis."""
        plan = []
        
        # Check if user has assessment
        has_assessment = getattr(user_state, 'has_assessment', False)
        
        # Always fetch personality context if user has assessment
        if has_assessment:
            plan.append(PlanStep(
                action="fetch_personality",
                priority=1,
                parameters={},
            ))
        
        # RAG search if needed
        if analysis.needs_rag:
            plan.append(PlanStep(
                action="rag_search",
                priority=2,
                parameters={
                    "topics": analysis.key_topics,
                    "dimensions": analysis.referenced_dimensions,
                },
            ))
        
        # Web research if needed (future)
        if analysis.needs_web_research and ThinkingConfig.WEB_SEARCH_ENABLED:
            plan.append(PlanStep(
                action="web_search",
                priority=3,
                parameters={"topics": analysis.key_topics},
            ))
        
        # Sort by priority
        plan.sort(key=lambda x: x.priority)
        
        return plan
    
    # =========================================================================
    # Execution
    # =========================================================================
    
    async def _execute_plan(
        self,
        plan: List[PlanStep],
        message: str,
        analysis: AnalysisResult,
    ) -> ExecutionResult:
        """Execute the plan and gather information."""
        result = ExecutionResult()
        
        for step in plan:
            try:
                if step.action == "rag_search":
                    rag_result = await self._execute_rag_search(
                        message, 
                        analysis.referenced_dimensions,
                    )
                    result.rag_context = rag_result.get("context")
                    result.rag_sources = rag_result.get("sources", [])
                
                elif step.action == "fetch_personality":
                    # Personality context is already in user_state
                    pass
                
                elif step.action == "web_search":
                    web_result = await self._execute_web_search(message)
                    result.web_research = web_result.get("context")
                    result.web_sources = web_result.get("sources", [])
            
            except Exception as e:
                result.errors.append(f"{step.action}: {str(e)}")
                self.logger.warning(f"Plan step failed: {step.action}: {e}")
        
        return result
    
    async def _execute_rag_search(
        self,
        message: str,
        dimensions: List[str],
    ) -> Dict[str, Any]:
        """Execute RAG search."""
        try:
            # Build search query - enhance with dimension names if relevant
            search_query = message
            if dimensions:
                search_query = f"{message} {' '.join(dimensions)}"
            
            # Run RAG service in executor (it's synchronous)
            loop = asyncio.get_event_loop()
            context_info = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.rag_service.get_context_for_query(search_query, top_k=3),
                ),
                timeout=5.0,
            )
            
            if context_info:
                sources = [
                    {
                        "title": chunk.get("title", "SELVE Knowledge"),
                        "source": chunk.get("source", "knowledge_base"),
                    }
                    for chunk in context_info.get("chunks", [])
                ]
                
                return {
                    "context": context_info.get("context", ""),
                    "sources": sources,
                }
            
            return {"context": None, "sources": []}
        
        except asyncio.TimeoutError:
            self.logger.warning("RAG search timed out")
            return {"context": None, "sources": []}
        
        except Exception as e:
            self.logger.error(f"RAG search failed: {e}")
            return {"context": None, "sources": []}
    
    async def _execute_web_search(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """Execute web search using crawler tools."""
        try:
            from app.tools.crawler.tools import YouTubeTool, RedditTool, WebPageTool
            from app.tools.crawler.core import YouTubeRequest, RedditRequest, WebpageRequest
            
            sources = []
            contexts = []
            
            # For now, just search Reddit and web pages (no YouTube needed without URLs)
            # In the future, could extract URLs from message and fetch those
            
            # Try Reddit search
            try:
                reddit_tool = RedditTool()
                reddit_request = RedditRequest(
                    query=message,
                    limit=5,
                    subreddits=["psychology", "selfimprovement", "mbti", "personalitytypes"],
                )
                reddit_results = await reddit_tool.search(reddit_request)
                await reddit_tool.close()
                
                for result in reddit_results[:3]:  # Limit to top 3
                    if result.is_valid:
                        sources.append({
                            "title": result.title,
                            "source": "reddit",
                            "url": result.url,
                        })
                        contexts.append(f"Reddit - {result.title}: {result.content[:500]}")
                
            except Exception as e:
                self.logger.warning(f"Reddit search failed: {e}")
            
            # Try web search for personality/psychology resources
            try:
                webpage_tool = WebPageTool()
                # Build a simple search query for personality resources
                search_query = f"personality psychology {message}"
                
                # Could fetch specific URLs if mentioned in message
                # For now, we'd need a proper web search engine integration
                # This is a placeholder for when that's implemented
                
                await webpage_tool.close()
                
            except Exception as e:
                self.logger.warning(f"Web search failed: {e}")
            
            if contexts:
                combined_context = "\n\n".join(contexts)
                return {
                    "context": combined_context[:2000],  # Limit context length
                    "sources": sources,
                }
            
            return {"context": None, "sources": []}
        
        except asyncio.TimeoutError:
            self.logger.warning("Web search timed out")
            return {"context": None, "sources": []}
        
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return {"context": None, "sources": []}
    
    # =========================================================================
    # Prompt Building
    # =========================================================================
    
    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        user_state: Any,
        analysis: AnalysisResult,
        execution_result: ExecutionResult,
    ) -> str:
        """Build enhanced system prompt with all context."""
        parts = [base_prompt]
        
        # Add user state context (if user_state has to_context_string method)
        if hasattr(user_state, 'to_context_string'):
            parts.append("\n\n" + user_state.to_context_string())
        
        # Add intent-specific guidance
        intent_guidance = self._get_intent_guidance(analysis, user_state)
        if intent_guidance:
            parts.append(f"\n\n### RESPONSE GUIDANCE ###\n{intent_guidance}")
        
        # Add emotional awareness if detected
        if analysis.emotional_tone:
            parts.append(
                f"\n\n### EMOTIONAL AWARENESS ###\n"
                f"The user seems to be feeling {analysis.emotional_tone}. "
                f"Respond with appropriate empathy and sensitivity."
            )
        
        return "".join(parts)
    
    def _get_intent_guidance(
        self,
        analysis: AnalysisResult,
        user_state: Any,
    ) -> str:
        """Get intent-specific response guidance."""
        has_assessment = getattr(user_state, 'has_assessment', False)
        
        guidance_map = {
            UserIntent.GREETING: (
                "Keep your greeting brief and warm. "
                f"{'Reference their SELVE profile naturally.' if has_assessment else 'Gently mention the assessment if appropriate.'}"
            ),
            
            UserIntent.QUESTION_ABOUT_SELF: (
                "When showing scores or results, ALWAYS list ALL 8 dimensions with their exact scores. "
                "Never omit dimensions. Show: LUMEN, AETHER, ORPHEUS, ORIN, LYRA, VARA, CHRONOS, KAEL. "
                "CRITICAL: A score of 0 means that dimension was not assessed, NOT that it's pending or missing. "
                "Say '0/100 (not yet assessed)' for 0 scores, not 'pending' or 'not shown'. "
                "List all 8 dimensions every time, including the 0s."
            ),
            
            UserIntent.QUESTION_ABOUT_DIMENSION: (
                "Explain the dimension clearly using the RAG context. "
                f"{'Connect it to their specific scores.' if has_assessment else 'Keep it general but invite them to take the assessment.'}"
            ),
            
            UserIntent.CAREER_QUESTION: (
                "Provide thoughtful career insights based on personality. "
                f"{'Use their SELVE scores to give personalized suggestions.' if has_assessment else 'Give general guidance and mention how SELVE assessment could help.'}"
            ),
            
            UserIntent.RELATIONSHIP_QUESTION: (
                "Be empathetic and thoughtful about relationship dynamics. "
                f"{'Connect advice to their personality dimensions.' if has_assessment else 'Provide general wisdom.'}"
            ),
            
            UserIntent.EMOTIONAL_SUPPORT: (
                "Lead with empathy. Validate their feelings first before offering any advice. "
                "Don't immediately try to solve - just listen and acknowledge."
            ),
            
            UserIntent.FOLLOW_UP: (
                "This is a follow-up to the previous message. "
                "Continue the conversation naturally without reintroducing yourself. "
                "If they said 'yes' or gave a number, deliver what you offered."
            ),
            
            UserIntent.EXPLORING_PERSONALITY: (
                f"{'Share insights from their SELVE profile engagingly.' if has_assessment else 'Encourage them to discover their personality through the SELVE assessment.'}"
            ),
        }
        
        return guidance_map.get(analysis.intent, "")
    
    def _build_messages(
        self,
        system_prompt: str,
        message: str,
        conversation_history: List[Dict[str, str]],
        rag_context: Optional[str],
        web_research: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build message list for LLM."""
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                if msg.get("role") in ("user", "assistant"):
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })
        
        # Build user message with context
        user_message = message
        context_parts = []
        
        if rag_context:
            context_parts.append(f"<knowledge_context>\n{rag_context}\n</knowledge_context>")
        
        if web_research:
            context_parts.append(f"<web_research>\n{web_research}\n</web_research>")
        
        if context_parts:
            user_message = "\n\n".join(context_parts) + f"\n\nUser Question: {message}"
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    # =========================================================================
    # Non-Streaming Method (for simple responses)
    # =========================================================================
    
    async def think_and_respond_sync(
        self,
        message: str,
        user_state: Any,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
    ) -> ThinkingResult:
        """
        Non-streaming version of think_and_respond.
        
        Returns complete ThinkingResult instead of streaming.
        """
        steps: List[ThinkingStep] = []
        start_time = datetime.utcnow()
        
        try:
            # === ANALYZE ===
            step = ThinkingStep(
                phase=ThinkingPhase.ANALYZING,
                description="Analyzing user intent",
            )
            
            analysis = IntentClassifier.classify(
                message=message,
                user_state=user_state,
                conversation_history=conversation_history,
            )
            
            step.complete(analysis)
            steps.append(step)
            
            # === PLAN ===
            step = ThinkingStep(
                phase=ThinkingPhase.PLANNING,
                description="Creating execution plan",
            )
            
            plan = self._create_plan(analysis, user_state)
            step.complete(plan)
            steps.append(step)
            
            # === EXECUTE ===
            step = ThinkingStep(
                phase=ThinkingPhase.RETRIEVING,
                description="Executing plan",
            )
            
            execution_result = await self._execute_plan(plan, message, analysis)
            step.complete(execution_result)
            steps.append(step)
            
            # === GENERATE ===
            step = ThinkingStep(
                phase=ThinkingPhase.GENERATING,
                description="Generating response",
            )
            
            enhanced_prompt = self._build_enhanced_prompt(
                base_prompt=system_prompt,
                user_state=user_state,
                analysis=analysis,
                execution_result=execution_result,
            )
            
            messages = self._build_messages(
                system_prompt=enhanced_prompt,
                message=message,
                conversation_history=conversation_history,
                rag_context=execution_result.rag_context,
                web_research=execution_result.web_research,
            )
            
            # Non-streaming generation
            llm_response = await asyncio.wait_for(
                self.llm_service.generate_response_async(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                ),
                timeout=30.0,
            )
            
            response = llm_response.get("content", "")
            step.complete({"response_length": len(response)})
            steps.append(step)
            
            # Calculate duration
            total_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Check if we should add a note about this interaction
            should_add_note = False
            note_content = None
            
            if analysis.emotional_tone in ["sad", "anxious", "angry"]:
                should_add_note = True
                note_content = f"User expressed {analysis.emotional_tone} feelings"
            
            return ThinkingResult(
                response=response,
                reasoning=f"Intent: {analysis.intent.value}, Confidence: {analysis.confidence}",
                sources=execution_result.rag_sources,
                steps=steps,
                user_intent=analysis.intent,
                confidence=analysis.confidence,
                should_add_note=should_add_note,
                note_content=note_content,
                total_duration_ms=total_duration,
            )
        
        except Exception as e:
            self.logger.error(f"Thinking error: {e}", exc_info=True)
            
            return ThinkingResult(
                response="I apologize, but I encountered an issue. Could you try rephrasing that?",
                reasoning=f"Error: {str(e)}",
                sources=[],
                steps=steps,
                user_intent=UserIntent.UNCLEAR,
                confidence=0.0,
                total_duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_thinking_engine: Optional[ThinkingEngine] = None


def get_thinking_engine() -> ThinkingEngine:
    """Get the global thinking engine instance."""
    global _thinking_engine
    if _thinking_engine is None:
        _thinking_engine = ThinkingEngine()
    return _thinking_engine
