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
    """
    Phases of the thinking process.
    Each phase represents actual work being done, not cosmetic loading states.
    """
    SECURITY_CHECK = "security_check"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    MEMORY_SEARCHING = "memory_searching"
    RAG_SEARCHING = "rag_searching"
    YOUTUBE_SEARCHING = "youtube_searching"
    YOUTUBE_FETCHING = "youtube_fetching"
    WEB_SEARCHING = "web_searching"
    SELVE_WEB_SEARCHING = "selve_web_searching"
    GENERATING = "generating"
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
    youtube_context: Optional[str] = None
    youtube_sources: List[Dict[str, str]] = field(default_factory=list)
    selve_web_context: Optional[str] = None
    selve_web_sources: List[Dict[str, str]] = field(default_factory=list)
    assessment_data: Optional[Dict[str, Any]] = None  # User's assessment scores and narrative
    assessment_comparison: Optional[Dict[str, Any]] = None  # Comparison of current vs archived assessments
    personality_insights: Optional[str] = None
    relevant_memories: List[Any] = field(default_factory=list)  # MemorySearchResult objects
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
            r"\b(reddit|twitter|facebook|instagram|social\s+media)\b",  # Social media references
            r"(what.*saying|discussions?|talking about|opinions?).{0,30}(reddit|online|internet|people)",  # Discussion queries
            r"https?://",  # URLs in message
            r"\b(mbti|big\s+five|enneagram|disc)\b",  # External frameworks trigger research to transform
            r"(analyze|explain|tell\s+me\s+about).{0,30}(this|that|the).{0,30}(video|post|article|link)",  # Content analysis requests
        ]

        # Check if message contains external references
        has_external_ref = any(re.search(pattern, message_lower) for pattern in external_patterns)

        if has_external_ref:
            # Determine if it's personality-related or truly off-topic
            personality_keywords = ["personality", "mbti", "type", "trait", "introvert", "extrovert",
                                   "psychology", "behavior", "temperament", "character"]
            is_personality_related = any(kw in message_lower for kw in personality_keywords)

            return AnalysisResult(
                intent=UserIntent.EXPLORING_PERSONALITY if is_personality_related else UserIntent.OFF_TOPIC,
                confidence=0.7,
                needs_rag=True if is_personality_related else False,
                needs_personality_context=is_personality_related,
                needs_web_research=True,  # Always enable web research for external refs
                key_topics=["research", "external content"],
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
    # Helper: Contextual Thinking Messages
    # =========================================================================

    def _generate_thinking_message(
        self,
        action: str,
        user_message: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a contextual thinking message for a tool action.

        Returns random static messages to feel natural without LLM cost.
        Multiple variations prevent robotic repetition.

        Args:
            action: The tool action being performed
            user_message: What the user asked
            context: Additional context (optional)

        Returns:
            Short, natural thinking message (randomly selected)
        """
        import random

        # Multiple variations for each action type (10+ variants to feel dynamic)
        message_variants = {
            "fetch_assessment": [
                "Let me pull up your results...",
                "Checking your personality profile...",
                "Looking at your scores now...",
                "Give me a moment to get your data...",
                "Pulling your assessment from the database...",
                "Hmm, let me see your profile...",
                "One sec, grabbing your results...",
                "Let me check your dimensions...",
                "Accessing your personality data...",
                "Retrieving your SELVE profile...",
                "Looking up your scores...",
                "Getting your assessment data...",
            ],
            "rag_search": [
                "Searching the knowledge base...",
                "Looking that up for you...",
                "Let me find information on that...",
                "Checking our personality insights...",
                "Searching through SELVE concepts...",
                "Digging into the research...",
                "Looking up personality info...",
                "Exploring the knowledge base...",
                "Searching for insights...",
                "Let me see what we have on that...",
                "Checking the research...",
                "Looking through personality concepts...",
            ],
            "youtube_search": [
                "Looking for insights...",
                "Searching for educational content...",
                "Finding relevant videos...",
                "Let me search YouTube for that...",
                "Looking for expert perspectives...",
                "Searching for psychology insights...",
                "Finding educational resources...",
                "Checking for video content...",
                "Looking up related videos...",
                "Searching for learning materials...",
                "Finding expert videos...",
                "Searching educational channels...",
            ],
            "youtube_live_fetch": [
                "Analyzing video transcript...",
                "Processing video content...",
                "Reading through the transcript...",
                "Let me check this video...",
                "Extracting insights from the video...",
                "Analyzing the content...",
                "Going through the transcript...",
                "Processing what they're saying...",
                "Reviewing the video details...",
                "Checking out this resource...",
                "Reading the video transcript...",
                "Analyzing what they said...",
            ],
            "selve_web_search": [
                "Checking the SELVE website...",
                "Looking up SELVE features...",
                "Let me see what SELVE offers...",
                "Searching SELVE documentation...",
                "Checking how SELVE works...",
                "Looking at SELVE pricing...",
                "Reviewing SELVE features...",
                "Let me check SELVE info...",
                "Exploring SELVE details...",
                "Searching SELVE resources...",
                "Looking up SELVE info...",
                "Checking SELVE docs...",
            ],
            "memory_search": [
                "Recalling our conversation...",
                "Let me remember what we discussed...",
                "Checking our chat history...",
                "Looking back at what you said...",
                "Refreshing my memory...",
                "Let me think back...",
                "Reviewing our conversation...",
                "Recalling previous messages...",
                "Checking what we talked about...",
                "Let me look back at that...",
                "Remembering our chat...",
                "Reviewing what you mentioned...",
            ],
            "web_search": [
                "Researching that for you...",
                "Searching the web...",
                "Let me look that up...",
                "Finding information online...",
                "Checking recent sources...",
                "Searching for answers...",
                "Looking up current info...",
                "Researching the topic...",
                "Searching online resources...",
                "Let me find that for you...",
                "Looking up information...",
                "Searching online...",
            ],
            "analyzing": [
                "Analyzing your question...",
                "Thinking about this...",
                "Let me process that...",
                "Considering your query...",
                "Working on your request...",
                "Processing your message...",
                "Analyzing what you asked...",
                "Let me think about this...",
                "Reviewing your question...",
                "Processing your request...",
                "Thinking this through...",
                "Considering this carefully...",
            ],
            "planning": [
                "Planning my response...",
                "Organizing my thoughts...",
                "Figuring out the best approach...",
                "Determining what to do...",
                "Deciding how to help...",
                "Planning the next steps...",
                "Thinking through this...",
                "Organizing information...",
                "Structuring my response...",
                "Planning how to answer...",
                "Working out the approach...",
                "Deciding the best way...",
            ],
            "generating": [
                "Crafting a response...",
                "Putting together my thoughts...",
                "Writing a response for you...",
                "Generating an answer...",
                "Composing a reply...",
                "Working on your answer...",
                "Creating a response...",
                "Formulating my thoughts...",
                "Preparing your answer...",
                "Writing this out...",
                "Composing my response...",
                "Putting this together...",
            ],
        }

        # Get variants for this action, or use generic messages
        variants = message_variants.get(
            action,
            [
                "One moment...",
                "Just a sec...",
                "Working on it...",
                "Give me a moment...",
                "Processing...",
                "Almost ready...",
                "Let me check...",
                "Thinking...",
                "Just a moment...",
                "On it...",
                "Hold on...",
                "Coming right up...",
            ]
        )

        # Return random variant
        return random.choice(variants)

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
            
            # === PHASE 2 & 3: PLAN & EXECUTE ===
            # Check if agentic RAG is enabled
            agentic_enabled = os.getenv("ENABLE_AGENTIC_RAG", "true").lower() == "true"

            if agentic_enabled:
                # NEW: Agentic function calling approach
                step = ThinkingStep(
                    phase=ThinkingPhase.PLANNING,
                    description="Agentic tool selection and execution",
                )

                # Iterate over agentic tool loop generator to get real-time status
                execution_result = None
                async for item in self._agentic_tool_loop(
                    message=message,
                    user_state=user_state,
                    session_id=getattr(user_state, 'session_id', 'unknown'),
                    emit_status=emit_status,
                ):
                    if item.get("type") == "status":
                        # Forward real-time status updates to user
                        if emit_status:
                            yield item
                    elif item.get("type") == "result":
                        # Final result from agentic loop
                        execution_result = item["execution_result"]

                step.complete(execution_result)
                steps.append(step)

            else:
                # LEGACY: Keyword-based planning approach
                if emit_status:
                    yield ThinkingStatus(
                        phase=ThinkingPhase.PLANNING,
                        message="Planning response approach...",
                        details={"step": 2, "total": 4, "mode": "keyword"},
                    ).to_dict()

                step = ThinkingStep(
                    phase=ThinkingPhase.PLANNING,
                    description="Creating execution plan",
                )

                plan = self._create_plan(analysis, user_state, message)
                step.complete(plan)
                steps.append(step)

                # === PHASE 3: EXECUTE (Keyword-based) ===
                # Emit contextual thinking messages for each tool action
                if plan and emit_status:
                    # Generate and emit contextual message for first action
                    first_action = plan[0].action
                    thinking_msg = self._generate_thinking_message(first_action, message)
                    yield ThinkingStatus(
                        phase=ThinkingPhase.PLANNING,
                        message=thinking_msg,
                        details={"step": 3, "total": 4, "action": first_action},
                    ).to_dict()
                elif emit_status:
                    yield ThinkingStatus(
                        phase=ThinkingPhase.PLANNING,
                        message="Gathering relevant information...",
                        details={"step": 3, "total": 4},
                    ).to_dict()

                step = ThinkingStep(
                    phase=ThinkingPhase.PLANNING,
                    description="Executing plan",
                )

                execution_result = await self._execute_plan(plan, message, analysis, user_state)
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
                youtube_context=execution_result.youtube_context,
                selve_web_context=execution_result.selve_web_context,
                assessment_data=execution_result.assessment_data,
                assessment_comparison=execution_result.assessment_comparison,
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
                    # Pass metadata upstream to AgenticChatService for Langfuse logging
                    yield chunk
            
            full_response = "".join(full_response_chunks)
            step.complete({"response_length": len(full_response)})
            steps.append(step)
            
            # === COMPLETE ===
            # Aggregate all sources for frontend display
            all_sources = []

            # Add RAG sources (knowledge base)
            for source in execution_result.rag_sources:
                all_sources.append({
                    "title": source.get("title", "SELVE Knowledge"),
                    "source": "rag",
                    "type": "rag",
                    "section": source.get("section"),
                    "score": source.get("score"),
                })

            # Add SELVE web sources
            for source in execution_result.selve_web_sources:
                all_sources.append({
                    "title": source.get("title", "SELVE Page"),
                    "source": "selve_web",
                    "type": "selve_web",
                    "url": source.get("url"),
                    "category": source.get("category"),
                    "relevance": source.get("relevance"),
                })

            # Add YouTube sources
            for source in execution_result.youtube_sources:
                all_sources.append({
                    "title": source.get("title", "YouTube Video"),
                    "source": "youtube",
                    "type": "youtube",
                    "url": source.get("url"),
                    "channel": source.get("channel"),
                    "video_id": source.get("video_id"),
                    "relevance": source.get("relevance"),
                })

            # Add web search sources
            for source in execution_result.web_sources:
                all_sources.append({
                    "title": source.get("title", "Web Source"),
                    "source": "web",
                    "type": "web",
                    "url": source.get("url"),
                    "relevance": source.get("relevance"),
                })

            if emit_status:
                yield ThinkingStatus(
                    phase=ThinkingPhase.COMPLETE,
                    message="Response complete",
                    details={
                        "sources": all_sources,
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
        message: str,
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

        # Search for relevant memories from older conversations
        # This implements "reactive memory" pattern from Google ADK
        user_id = getattr(user_state, 'user_id', None)
        if user_id:
            plan.append(PlanStep(
                action="memory_search",
                priority=1,  # High priority - run early
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

        # YouTube LIVE FETCH for specific video URLs
        # Triggers when user provides a YouTube URL or asks to fetch a specific video
        import re
        youtube_url_pattern = r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        youtube_url_match = re.search(youtube_url_pattern, message)
        fetch_keywords = ["fetch", "get", "retrieve", "analyze this video", "what does this video"]

        if youtube_url_match or any(keyword in message.lower() for keyword in fetch_keywords if "youtube" in message.lower() or "video" in message.lower()):
            # Extract video URL/ID if present
            video_url_or_id = youtube_url_match.group(0) if youtube_url_match else None
            plan.append(PlanStep(
                action="youtube_live_fetch",
                priority=1,  # High priority - user explicitly requesting specific content
                parameters={"url_or_id": video_url_or_id or message},
            ))

        # YouTube transcript search for psychology/educational content
        # Search when asking about psychological concepts, behaviors, or educational topics
        youtube_keywords = [
            "psychology", "cognitive", "behavior", "mental", "emotion", "social",
            "thinking", "learning", "development", "personality", "motivation",
            "consciousness", "perception", "memory", "decision", "bias", "anxiety",
            "depression", "therapy", "mindfulness", "self-improvement", "habit",
            "procrastination", "creativity", "intelligence", "narcissism", "empathy"
        ]
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in youtube_keywords):
            plan.append(PlanStep(
                action="youtube_search",
                priority=2,  # Same priority as RAG
                parameters={},
            ))

        # SELVE web content search for product information
        # Search when asking about how SELVE works, features, privacy, terms
        selve_keywords = [
            "selve", "how does selve", "how selve works", "features", "privacy",
            "terms", "conditions", "policy", "works", "assessment", "test",
            "dimensions", "profile", "results", "score", "framework"
        ]
        if any(keyword in message_lower for keyword in selve_keywords):
            plan.append(PlanStep(
                action="selve_web_search",
                priority=2,  # Same priority as RAG and YouTube
                parameters={},
            ))

        # User assessment data (scores and narrative)
        # ALWAYS fetch if user has assessment - needed for personality-aware responses
        # This ensures the AI has access to the user's full narrative for context
        if user_state and user_state.has_assessment:
            plan.append(PlanStep(
                action="fetch_assessment",
                priority=1,  # High priority - essential personality context
                parameters={},
            ))

        # Assessment history and comparison (archived results)
        # Fetch when user asks about changes, growth, or previous results
        comparison_keywords = [
            "how have i changed", "how i've changed", "have i changed",
            "my previous", "my old", "my past", "my earlier",
            "compare", "comparison", "difference", "changed",
            "growth", "progress", "evolution", "development",
            "before and after", "then and now", "used to be",
            "last time", "previous assessment", "old results",
            "archived", "history", "over time"
        ]
        if any(keyword in message_lower for keyword in comparison_keywords):
            plan.append(PlanStep(
                action="compare_assessments",
                priority=1,  # High priority - direct question about changes
                parameters={},
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
    # Execution - Agentic Tool Loop (New)
    # =========================================================================

    async def _agentic_tool_loop(
        self,
        message: str,
        user_state: Any,
        session_id: str,
        max_iterations: int = None,
        emit_status: bool = True,
    ):
        """
        Agentic tool execution loop using LLM function calling.

        The LLM decides which tools to call based on tool definitions,
        receives results, and can chain multiple tool calls.

        Args:
            message: User's message
            user_state: User context (auth, assessment, etc.)
            session_id: Session ID for memory search
            max_iterations: Maximum tool loop iterations (default from env)
            emit_status: Whether to yield status updates

        Yields:
            Status dicts with real-time progress updates
            Final result dict with ExecutionResult
        """
        from app.tools.function_definitions import get_tool_definitions
        from langfuse import get_client
        import json

        langfuse = get_client()
        max_iterations = max_iterations or int(os.getenv("MAX_TOOL_ITERATIONS", "5"))

        # Build conversation with system prompt
        messages = [
            {"role": "system", "content": self._get_agentic_system_prompt(user_state)},
            {"role": "user", "content": message}
        ]

        # Get tool definitions based on user context
        tools = get_tool_definitions(user_state)

        result = ExecutionResult()
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"Agentic tool loop iteration {iteration}/{max_iterations}")

            # REAL STATUS: Emit iteration start
            if emit_status:
                yield {
                    "type": "status",
                    "phase": "tool_iteration",
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "message": f"Analyzing tools ({iteration}/{max_iterations})...",
                }

            try:
                # LLM decides which tools to call (with Langfuse tracing)
                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name=f"llm-tool-decision-iter-{iteration}",
                    model=self.llm_service.model,
                    input=json.dumps({"iteration": iteration, "num_tools": len(tools)}),
                ) as llm_gen:
                    response = await self.llm_service.call_with_tools(
                        messages=messages,
                        tools=tools,
                        temperature=0.7,
                        max_tokens=1500,
                    )

                    # Update Langfuse with response metadata
                    llm_gen.update(
                        output=json.dumps({
                            "tool_calls": response.get("tool_calls"),
                            "content": response.get("content")
                        }),
                        metadata={
                            "provider": response.get("provider"),
                            "model": response.get("model"),
                        },
                        usage={
                            "input": response.get("usage", {}).get("input_tokens", 0),
                            "output": response.get("usage", {}).get("output_tokens", 0),
                            "total": response.get("usage", {}).get("total_tokens", 0),
                        },
                    )

                # Check if LLM wants to call tools
                tool_calls = response.get("tool_calls")

                if not tool_calls:
                    # LLM is done - no more tools to call
                    self.logger.info("LLM finished - no more tool calls requested")
                    if emit_status:
                        yield {
                            "type": "status",
                            "phase": "tool_complete",
                            "message": "Analysis complete, crafting response...",
                        }
                    break

                # REAL STATUS: Emit tool calls about to be executed
                if emit_status:
                    tool_names = [tc["function"]["name"] for tc in tool_calls]
                    yield {
                        "type": "status",
                        "phase": "calling_tools",
                        "tools": tool_names,
                        "message": f"Using {', '.join(tool_names)}...",
                    }

                # Execute each tool call (with Langfuse spans)
                for idx, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call["function"]["name"]
                    try:
                        # Parse arguments (they come as JSON string)
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}

                    self.logger.info(f"Executing tool: {tool_name} with args: {arguments}")

                    # REAL STATUS: Emit individual tool execution
                    if emit_status:
                        yield {
                            "type": "status",
                            "phase": "executing_tool",
                            "tool": tool_name,
                            "tool_index": idx,
                            "total_tools": len(tool_calls),
                            "message": f"Executing {tool_name}...",
                            "args": arguments,
                        }

                    # Execute the tool with Langfuse span
                    tool_start = datetime.utcnow()
                    with langfuse.start_as_current_observation(
                        as_type="span",
                        name=f"tool-{tool_name}",
                        input=json.dumps(arguments),
                        metadata={"tool_name": tool_name, "iteration": iteration},
                    ) as tool_span:
                        tool_result = await self._execute_tool(
                            tool_name=tool_name,
                            arguments=arguments,
                            user_state=user_state,
                            session_id=session_id,
                        )

                        # Update span with output
                        tool_span.update(
                            output=json.dumps(tool_result),
                            metadata={
                                "status": tool_result.get("status", "unknown"),
                                "has_context": bool(tool_result.get("context")),
                            },
                        )

                    tool_duration = (datetime.utcnow() - tool_start).total_seconds()

                    # REAL STATUS: Emit tool completion
                    if emit_status:
                        yield {
                            "type": "status",
                            "phase": "tool_executed",
                            "tool": tool_name,
                            "duration_seconds": round(tool_duration, 2),
                            "status": tool_result.get("status", "unknown"),
                            "message": f"Completed {tool_name} ({tool_duration:.1f}s)",
                        }

                    # Aggregate results into ExecutionResult
                    self._aggregate_tool_result(result, tool_name, tool_result)

                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result)
                    })

            except Exception as e:
                self.logger.error(f"Error in agentic tool loop iteration {iteration}: {e}")
                result.errors.append(f"Agentic loop error: {str(e)}")
                if emit_status:
                    yield {
                        "type": "status",
                        "phase": "error",
                        "message": f"Error in tool execution: {str(e)}",
                    }
                break

        # Yield final result
        yield {"type": "result", "execution_result": result}

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_state: Any,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Execute a single tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            user_state: User context
            session_id: Session ID

        Returns:
            Tool execution result as dict
        """
        try:
            if tool_name == "rag_search":
                query = arguments.get("query", "")
                dimensions = arguments.get("dimensions", [])
                return await self._execute_rag_search(query, dimensions)

            elif tool_name == "youtube_search":
                query = arguments.get("query", "")
                max_results = arguments.get("max_results", 5)
                result = await self._execute_youtube_search(query)
                return result

            elif tool_name == "youtube_fetch":
                video_id = arguments.get("video_id", "")
                include_transcript = arguments.get("include_transcript", True)
                return await self._execute_youtube_live_fetch(video_id)

            elif tool_name == "web_search":
                query = arguments.get("query", "")
                return await self._execute_web_search(query)

            elif tool_name == "selve_web_search":
                query = arguments.get("query", "")
                return await self._execute_selve_web_search(query)

            elif tool_name == "memory_search":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 10)
                if user_state and hasattr(user_state, 'user_id'):
                    return await self._execute_memory_search(query, user_state.user_id)
                return {"status": "error", "message": "User not logged in"}

            elif tool_name == "assessment_fetch":
                if not user_state or not hasattr(user_state, 'clerk_user_id'):
                    return {"status": "error", "message": "User not logged in"}
                include_narrative = arguments.get("include_narrative", True)
                include_scores = arguments.get("include_scores", True)
                return await self._execute_fetch_assessment(user_state.clerk_user_id, "")

            elif tool_name == "assessment_compare":
                archetype_a = arguments.get("archetype_a", "")
                archetype_b = arguments.get("archetype_b", "")
                # TODO: Implement comparison logic
                return {"status": "success", "comparison": f"Comparing {archetype_a} vs {archetype_b}"}

            else:
                return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        except Exception as e:
            self.logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"status": "error", "message": str(e)}

    def _aggregate_tool_result(
        self,
        result: ExecutionResult,
        tool_name: str,
        tool_result: Dict[str, Any]
    ):
        """
        Aggregate tool result into ExecutionResult.

        Args:
            result: ExecutionResult to update
            tool_name: Name of the tool
            tool_result: Tool execution result
        """
        if tool_name == "rag_search":
            result.rag_context = tool_result.get("context")
            result.rag_sources = tool_result.get("sources", [])

        elif tool_name in ["youtube_search", "youtube_fetch"]:
            result.youtube_context = tool_result.get("context")
            result.youtube_sources = tool_result.get("sources", [])

        elif tool_name == "web_search":
            result.web_research = tool_result.get("context")
            result.web_sources = tool_result.get("sources", [])

        elif tool_name == "selve_web_search":
            result.selve_web_context = tool_result.get("context")
            result.selve_web_sources = tool_result.get("sources", [])

        elif tool_name == "memory_search":
            result.relevant_memories = tool_result.get("memories", [])

        elif tool_name == "assessment_fetch":
            result.assessment_data = tool_result.get("data")

        elif tool_name == "assessment_compare":
            result.assessment_comparison = tool_result.get("comparison")

    def _get_agentic_system_prompt(self, user_state: Any) -> str:
        """
        Enhanced system prompt for agentic behavior with user context awareness.

        Args:
            user_state: User context

        Returns:
            System prompt string
        """
        prompt = """You are SELVE, an AI psychology assistant with access to specialized tools.

**Your Mission**: Help users understand themselves through dimensional psychology.

**Available Tools**: Use function calling to intelligently gather information:
- rag_search: SELVE psychology knowledge base (dimensional psychology, archetypes)
- youtube_search/youtube_fetch: Educational psychology videos
- web_search: Current events, recent research, general knowledge
- selve_web_search: Official SELVE content and information
- memory_search: User's conversation history
- assessment_fetch: User's personality assessment (only if logged in)
- assessment_compare: Compare personality archetypes

**User Context:**"""

        if user_state and hasattr(user_state, 'clerk_user_id') and user_state.clerk_user_id:
            prompt += f"""
- **Logged in**: YES
- **User ID**: {user_state.clerk_user_id}
- **Has Assessment**: {getattr(user_state, 'has_assessment', False)}
- **Archetype**: {getattr(user_state, 'archetype', 'Unknown')}"""
        else:
            prompt += "\n- **User**: Anonymous (not logged in)"

        prompt += """

**Critical Instructions**:
1. Use tools to gather accurate information - NEVER hallucinate data
2. ONLY use assessment tools if user is logged in (check user context above)
3. Call multiple tools if needed to fully answer the question
4. Be concise but insightful in your responses
5. Ground responses in psychology research and SELVE's dimensional framework

**Example Behavior**:
- User asks "What's my personality type?" → If logged in: call assessment_fetch, if not: explain they need to take assessment
- User asks "Tell me about the Explorer archetype" → call rag_search for archetype information
- User asks "Videos about CBT" → call youtube_search
- User asks "What did we discuss last time?" → call memory_search

Always use the right tool for the task. Never claim to have information you haven't retrieved."""

        return prompt

    # =========================================================================
    # Execution - Keyword-Based (Legacy Fallback)
    # =========================================================================

    async def _execute_plan(
        self,
        plan: List[PlanStep],
        message: str,
        analysis: AnalysisResult,
        user_state: Any = None,
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

                elif step.action == "memory_search":
                    # Search for semantically relevant memories from older conversations
                    if user_state and hasattr(user_state, 'user_id'):
                        memory_result = await self._execute_memory_search(
                            message,
                            user_state.user_id,
                        )
                        result.relevant_memories = memory_result.get("memories", [])

                elif step.action == "fetch_personality":
                    # Personality context is already in user_state
                    pass

                elif step.action == "youtube_live_fetch":
                    # Fetch, validate, and ingest a specific YouTube video
                    youtube_fetch_result = await self._execute_youtube_live_fetch(
                        step.parameters.get("url_or_id", message)
                    )
                    result.youtube_context = youtube_fetch_result.get("context")
                    result.youtube_sources = youtube_fetch_result.get("sources", [])

                elif step.action == "youtube_search":
                    youtube_result = await self._execute_youtube_search(message)
                    result.youtube_context = youtube_result.get("context")
                    result.youtube_sources = youtube_result.get("sources", [])

                elif step.action == "selve_web_search":
                    selve_web_result = await self._execute_selve_web_search(message)
                    result.selve_web_context = selve_web_result.get("context")
                    result.selve_web_sources = selve_web_result.get("sources", [])

                elif step.action == "fetch_assessment":
                    assessment_result = await self._execute_fetch_assessment(user_state.clerk_user_id, message)
                    result.assessment_data = assessment_result.get("data")

                elif step.action == "compare_assessments":
                    comparison_result = await self._execute_compare_assessments(user_state.clerk_user_id, message)
                    result.assessment_comparison = comparison_result.get("data")

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
        """
        Execute web search using crawler tools with content transformation.

        Searches Reddit, webpages, and YouTube for personality-related content,
        then transforms it to SELVE's framework and voice.

        Security:
        - Domain allowlisting for webpages
        - Subreddit allowlisting for Reddit
        - SSRF protection on all URLs
        - Content validation for personality relevance
        - Transformation to remove competing frameworks
        """
        try:
            from app.tools.crawler.tools import YouTubeTool, RedditTool, WebPageTool
            from app.tools.crawler.core import YouTubeRequest, RedditRequest, WebpageRequest
            from app.services.content_transformation_service import transform_crawled_content

            sources = []
            contexts = []

            # Extract URLs from message for direct fetching
            urls = self._extract_urls(message)
            youtube_urls = [url for url in urls if self._is_youtube_url(url)]
            web_urls = [url for url in urls if not self._is_youtube_url(url)]

            # 1. Try Reddit search (for general personality insights)
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
                        # Transform content to SELVE framework
                        transformation = transform_crawled_content(
                            content=result.content,
                            source_url=result.url,
                            add_citation=True
                        )

                        if transformation.is_valid:
                            sources.append({
                                "title": result.title,
                                "source": "reddit",
                                "url": result.url,
                                "relevance": result.relevance_score,
                            })
                            contexts.append(
                                f"From Reddit discussion: {result.title}\n"
                                f"{transformation.transformed_content[:600]}"
                            )
                            self.logger.info(
                                f"Transformed Reddit content: {transformation.transformation_notes}"
                            )
                        else:
                            self.logger.warning(
                                f"Reddit content transformation failed: {transformation.validation_message}"
                            )

            except Exception as e:
                self.logger.warning(f"Reddit search failed: {e}", exc_info=True)

            # 2. Try YouTube transcripts (if URLs mentioned in message)
            if youtube_urls and ThinkingConfig.YOUTUBE_SEARCH_ENABLED:
                try:
                    youtube_tool = YouTubeTool()

                    for yt_url in youtube_urls[:2]:  # Limit to 2 videos max
                        try:
                            youtube_request = YouTubeRequest(video_url=yt_url)
                            result = await youtube_tool.get_transcript(youtube_request)

                            if result.is_valid:
                                # Transform content to SELVE framework
                                transformation = transform_crawled_content(
                                    content=result.content,
                                    source_url=result.url,
                                    add_citation=True
                                )

                                if transformation.is_valid:
                                    sources.append({
                                        "title": result.title,
                                        "source": "youtube",
                                        "url": result.url,
                                        "relevance": result.relevance_score,
                                    })
                                    contexts.append(
                                        f"From video: {result.title}\n"
                                        f"{transformation.transformed_content[:800]}"
                                    )
                                    self.logger.info(
                                        f"Transformed YouTube content: {transformation.transformation_notes}"
                                    )
                                else:
                                    self.logger.warning(
                                        f"YouTube content transformation failed: {transformation.validation_message}"
                                    )
                            else:
                                self.logger.info(f"YouTube content not valid: {result.error}")

                        except Exception as e:
                            self.logger.warning(f"YouTube fetch failed for {yt_url}: {e}")

                    await youtube_tool.close()

                except Exception as e:
                    self.logger.warning(f"YouTube tool initialization failed: {e}", exc_info=True)

            # 3. Try webpage fetching (if URLs mentioned in message)
            if web_urls and ThinkingConfig.WEB_SEARCH_ENABLED:
                try:
                    webpage_tool = WebPageTool()

                    for web_url in web_urls[:2]:  # Limit to 2 pages max
                        try:
                            webpage_request = WebpageRequest(url=web_url)
                            result = await webpage_tool.fetch(webpage_request)

                            if result.is_valid:
                                # Transform content to SELVE framework
                                transformation = transform_crawled_content(
                                    content=result.content,
                                    source_url=result.url,
                                    add_citation=True
                                )

                                if transformation.is_valid:
                                    sources.append({
                                        "title": result.title,
                                        "source": "webpage",
                                        "url": result.url,
                                        "relevance": result.relevance_score,
                                    })
                                    contexts.append(
                                        f"From {result.title}:\n"
                                        f"{transformation.transformed_content[:800]}"
                                    )
                                    self.logger.info(
                                        f"Transformed webpage content: {transformation.transformation_notes}"
                                    )
                                else:
                                    self.logger.warning(
                                        f"Webpage content transformation failed: {transformation.validation_message}"
                                    )
                            else:
                                self.logger.info(f"Webpage content not valid: {result.error}")

                        except Exception as e:
                            self.logger.warning(f"Webpage fetch failed for {web_url}: {e}")

                    await webpage_tool.close()

                except Exception as e:
                    self.logger.warning(f"Webpage tool initialization failed: {e}", exc_info=True)

            # Combine all contexts
            if contexts:
                combined_context = "\n\n---\n\n".join(contexts)
                return {
                    "context": combined_context[:3000],  # Limit total context length
                    "sources": sources,
                }

            return {"context": None, "sources": []}

        except asyncio.TimeoutError:
            self.logger.warning("Web search timed out")
            return {"context": None, "sources": []}

        except Exception as e:
            self.logger.error(f"Web search failed: {e}", exc_info=True)
            return {"context": None, "sources": []}

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)

    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        return any(domain in url.lower() for domain in ['youtube.com', 'youtu.be', 'm.youtube.com'])

    async def _execute_memory_search(
        self,
        message: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Search for semantically relevant memories from older conversations.

        This implements the "reactive memory" pattern from Google ADK,
        where relevant past conversations are retrieved based on the current query.

        Args:
            message: Current user message
            user_id: User ID for filtering memories

        Returns:
            Dict with memories list
        """
        try:
            from .memory_search_service import MemorySearchService

            memory_service = MemorySearchService()

            # Search for relevant memories
            result = await memory_service.search_memories(
                query=message,
                user_id=user_id,
                top_k=3,
                score_threshold=0.6,  # Higher threshold for relevance
            )

            if result.success and result.data:
                self.logger.info(f"Found {len(result.data)} relevant memories for query")
                return {"memories": result.data}

            return {"memories": []}

        except Exception as e:
            self.logger.warning(f"Memory search failed: {e}")
            return {"memories": []}

    def _format_memory_context(self, memories: List[Any]) -> str:
        """
        Format memory search results as context for LLM.

        Args:
            memories: List of MemorySearchResult objects

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        parts = [
            "=" * 60,
            "RELEVANT PAST CONVERSATIONS",
            "=" * 60,
            "",
            "The following are relevant conversations from the past that may provide context:",
            "",
        ]

        for i, mem in enumerate(memories[:3], 1):
            relevance_pct = int(mem.relevance_score * 100)
            parts.append(f"{i}. {mem.title} (relevance: {relevance_pct}%)")
            parts.append(f"   Summary: {mem.summary}")

            if mem.key_insights:
                insights = ", ".join(mem.key_insights[:2])
                parts.append(f"   Key insights: {insights}")

            if mem.emotional_state:
                parts.append(f"   Emotional context: {mem.emotional_state}")

            parts.append("")

        parts.extend([
            "Use these memories to:",
            "- Maintain continuity across conversations",
            "- Reference past discussions naturally",
            "- Understand context and history",
            "- Avoid repeating information already covered",
            "=" * 60,
        ])

        return "\n".join(parts)

    async def _execute_youtube_live_fetch(
        self,
        url_or_id: str,
    ) -> Dict[str, Any]:
        """
        Fetch, validate, and ingest a specific YouTube video transcript.

        This implements the SELVE "initiation" process:
        1. Fetch raw transcript from youtube-transcript.io
        2. Validate content through SELVE framework (cleanse & purify)
        3. Ingest approved content into knowledge base
        4. Return validated content for immediate use

        Returns:
            Dict with context (transcript text) and sources
        """
        try:
            from app.tools.youtube_live_fetch_tool import fetch_youtube_transcript

            self.logger.info(f"📺 Fetching YouTube video: {url_or_id}")

            # Fetch, validate, and ingest
            result = await fetch_youtube_transcript(
                url_or_id=url_or_id,
                auto_ingest=True,
            )

            # Check for critical errors (API failure, no transcript, etc.)
            error = result.get("error")
            error_code = result.get("error_code")

            # If it's a critical error (not validation), return empty
            if error and error_code in ["FEATURE_DISABLED", "FETCH_ERROR", "NO_TRANSCRIPT"]:
                self.logger.warning(f"YouTube fetch failed: {error}")
                return {"context": None, "sources": []}

            # Format context from fetched transcript (even if validation failed)
            title = result.get("title", "Unknown Video")
            channel = result.get("channel", "Unknown Channel")
            validation_status = result.get("validation_status", "unknown")
            ingested = result.get("ingested", False)
            chunks_created = result.get("chunks_created", 0)
            transcript_text = result.get("transcript_text", "")
            validation_scores = result.get("validation_scores", {})

            # Create context string with validation feedback
            context_parts = [
                f"YouTube Video: {title}",
                f"Channel: {channel}",
            ]

            if validation_status == "approved" and ingested:
                context_parts.append(f"✅ Content validated and added to knowledge base ({chunks_created} chunks)")
                context_parts.append("\nTranscript:")
                context_parts.append(transcript_text[:2000] + "..." if len(transcript_text) > 2000 else transcript_text)

            elif validation_status == "needs_revision":
                context_parts.append("⚠️ Content fetched but needs revision before ingestion")
                if validation_scores:
                    context_parts.append(f"Validation scores: SELVE-aligned: {validation_scores.get('selve_aligned', 'N/A')}/10, "
                                       f"Accuracy: {validation_scores.get('factually_accurate', 'N/A')}/10, "
                                       f"Tone: {validation_scores.get('appropriate_tone', 'N/A')}/10")
                context_parts.append("\nNote: This content was reviewed but doesn't fully align with SELVE's psychology framework yet.")

            elif validation_status == "rejected":
                context_parts.append("❌ Content validation failed - not suitable for SELVE knowledge base")
                if validation_scores:
                    context_parts.append(f"Validation scores: SELVE-aligned: {validation_scores.get('selve_aligned', 'N/A')}/10, "
                                       f"Accuracy: {validation_scores.get('factually_accurate', 'N/A')}/10, "
                                       f"Tone: {validation_scores.get('appropriate_tone', 'N/A')}/10")
                context_parts.append("\nReason: This video doesn't contain psychology or personality-related content that aligns with the SELVE framework.")
                context_parts.append("I can only analyze educational psychology content, research, or personality-related videos.")

            else:
                context_parts.append(f"Status: {validation_status}")

            context = "\n".join(context_parts)

            # Create source entry
            video_id = result.get("video_id", "")
            sources = [{
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
                "channel": channel,
                "video_id": video_id,
                "validation_status": validation_status,
                "ingested": ingested,
            }]

            self.logger.info(
                f"✅ YouTube fetch complete: {title} | "
                f"Status: {validation_status} | Ingested: {ingested}"
            )

            return {
                "context": context,
                "sources": sources,
            }

        except Exception as e:
            self.logger.error(f"YouTube live fetch failed: {e}")
            return {"context": None, "sources": []}

    async def _execute_youtube_search(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Search YouTube psychology transcripts for relevant educational content.

        This searches pre-indexed YouTube transcripts (TED-Ed, Crash Course, etc.)
        stored in Qdrant for semantic similarity.

        Args:
            message: User's query

        Returns:
            Dict with context and sources
        """
        try:
            from app.tools.youtube_tool import search_youtube_transcripts

            # Search YouTube transcripts
            result = await search_youtube_transcripts(
                query=message,
                top_k=3,
            )

            if result.get("context"):
                self.logger.info(f"Found YouTube transcript content from {len(result.get('videos', []))} videos")
                return {
                    "context": result["context"],
                    "sources": result.get("sources", []),
                }

            return {"context": None, "sources": []}

        except Exception as e:
            self.logger.warning(f"YouTube transcript search failed: {e}")
            return {"context": None, "sources": []}

    async def _execute_selve_web_search(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Search SELVE web content for product information.

        This searches indexed content from selve.me including:
        - How it works
        - Features
        - Privacy policy
        - Terms & conditions

        Args:
            message: User's query

        Returns:
            Dict with context and sources
        """
        try:
            from app.tools.selve_web_tool import search_selve_web

            # Search SELVE web content
            result = await search_selve_web(
                query=message,
                top_k=3,
            )

            if result.get("context"):
                self.logger.info(f"Found SELVE web content from {len(result.get('pages', []))} pages")
                return {
                    "context": result["context"],
                    "sources": result.get("sources", []),
                }

            return {"context": None, "sources": []}

        except Exception as e:
            self.logger.warning(f"SELVE web content search failed: {e}")
            return {"context": None, "sources": []}

    async def _execute_fetch_assessment(
        self,
        user_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Fetch user's assessment scores and narrative.

        This fetches the user's complete assessment data including:
        - 8 dimension scores (LUMEN, AETHER, ORPHEUS, VARA, CHRONOS, KAEL, ORIN, LYRA)
        - Full personality narrative
        - Archetype and profile pattern

        Args:
            user_id: User's Clerk ID
            message: User's query (to determine what data to fetch)

        Returns:
            Dict with assessment data
        """
        try:
            from app.tools.assessment_tool import AssessmentTool

            tool = AssessmentTool()

            # Fetch full assessment data
            result = await tool.get_user_assessment(
                user_id=user_id,
                include_narrative=True,
                include_scores=True,
            )

            if result.get("status") == "success":
                self.logger.info(f"✅ Fetched assessment data for user {user_id[:8]}...")
                return {"data": result}
            else:
                self.logger.info(f"No assessment found for user {user_id[:8]}...")
                return {"data": result}

        except Exception as e:
            self.logger.warning(f"Assessment fetch failed: {e}")
            return {"data": None}

    async def _execute_compare_assessments(
        self,
        user_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Compare user's current and archived assessments to track personality changes.

        This fetches:
        - Current assessment (most recent)
        - Most recent archived assessment (previous)
        - Score changes for all 8 dimensions
        - Biggest increases and decreases
        - Archetype changes

        Args:
            user_id: User's Clerk ID
            message: User's query

        Returns:
            Dict with comparison data
        """
        try:
            from app.tools.assessment_tool import AssessmentTool

            tool = AssessmentTool()

            # Compare assessments
            result = await tool.compare_assessments(user_id=user_id)

            if result.get("status") == "success":
                self.logger.info(f"✅ Compared assessments for user {user_id[:8]}...")
                return {"data": result}
            elif result.get("status") == "no_comparison":
                self.logger.info(f"No archived assessments for user {user_id[:8]}...")
                return {"data": result}
            else:
                self.logger.info(f"Assessment comparison failed for user {user_id[:8]}...")
                return {"data": result}

        except Exception as e:
            self.logger.warning(f"Assessment comparison failed: {e}")
            return {"data": None}

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

        # Add semantically relevant memories from older conversations
        if execution_result.relevant_memories:
            memory_context = self._format_memory_context(execution_result.relevant_memories)
            if memory_context:
                parts.append(f"\n\n{memory_context}")

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
        youtube_context: Optional[str] = None,
        selve_web_context: Optional[str] = None,
        assessment_data: Optional[Dict[str, Any]] = None,
        assessment_comparison: Optional[Dict[str, Any]] = None,
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

        if selve_web_context:
            context_parts.append(f"<selve_content>\n{selve_web_context}\n</selve_content>")

        if youtube_context:
            context_parts.append(f"<youtube_transcripts>\n{youtube_context}\n</youtube_transcripts>")

        if web_research:
            context_parts.append(f"<web_research>\n{web_research}\n</web_research>")

        # Add assessment data if fetched
        if assessment_data and assessment_data.get("status") == "success":
            assessment_parts = []

            # Add scores
            if "scores" in assessment_data:
                scores = assessment_data["scores"]
                scores_text = "DIMENSION SCORES:\n"
                for dim, score in scores.items():
                    scores_text += f"  {dim}: {score:.1f}/100\n"
                assessment_parts.append(scores_text)

            # Add archetype and profile
            if assessment_data.get("archetype"):
                assessment_parts.append(f"ARCHETYPE: {assessment_data['archetype']}")
            if assessment_data.get("profile_pattern"):
                assessment_parts.append(f"PROFILE PATTERN: {assessment_data['profile_pattern']}")

            # Add quality metrics if available
            if assessment_data.get("quality_info"):
                assessment_parts.append(f"ASSESSMENT QUALITY: {assessment_data['quality_info']}")

            # Add narrative sections
            if "narrative" in assessment_data:
                narrative = assessment_data["narrative"]
                if isinstance(narrative, dict):
                    assessment_parts.append("\nPERSONALITY NARRATIVE:")
                    for key in ["overview", "core_traits", "strengths", "growth_areas"]:
                        if key in narrative and narrative[key]:
                            formatted_key = key.replace("_", " ").title()
                            assessment_parts.append(f"\n{formatted_key}: {narrative[key]}")

            if assessment_parts:
                context_parts.append(f"<assessment_results>\n{chr(10).join(assessment_parts)}\n</assessment_results>")

        # Add assessment comparison data if fetched
        if assessment_comparison and assessment_comparison.get("status") == "success":
            comparison_parts = []

            comparison_parts.append("PERSONALITY CHANGES OVER TIME:")
            comparison_parts.append(f"\nCurrent Archetype: {assessment_comparison.get('current_archetype')}")
            comparison_parts.append(f"Previous Archetype: {assessment_comparison.get('previous_archetype')}")

            if assessment_comparison.get("archetype_changed"):
                comparison_parts.append("⚠️ ARCHETYPE HAS CHANGED - Significant personality shift detected!")

            comparison_parts.append(f"\nAssessment completed: {assessment_comparison.get('current_completed_at')}")
            comparison_parts.append(f"Previous assessment: {assessment_comparison.get('previous_completed_at')}")

            # Add score changes
            if "score_changes" in assessment_comparison:
                comparison_parts.append("\nDIMENSION SCORE CHANGES:")
                for dim, data in assessment_comparison["score_changes"].items():
                    change = data["change"]
                    arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                    comparison_parts.append(
                        f"  {dim}: {data['previous']:.1f} → {data['current']:.1f} "
                        f"({arrow} {abs(change):.1f}, {data['percent_change']:+.1f}%)"
                    )

            # Highlight biggest changes
            if assessment_comparison.get("biggest_increase"):
                inc = assessment_comparison["biggest_increase"]
                comparison_parts.append(
                    f"\nBiggest Increase: {inc['dimension']} "
                    f"({inc['previous']:.1f} → {inc['current']:.1f}, +{inc['change']:.1f})"
                )

            if assessment_comparison.get("biggest_decrease"):
                dec = assessment_comparison["biggest_decrease"]
                comparison_parts.append(
                    f"Biggest Decrease: {dec['dimension']} "
                    f"({dec['previous']:.1f} → {dec['current']:.1f}, {dec['change']:.1f})"
                )

            comparison_parts.append(f"\nTotal archived assessments: {assessment_comparison.get('total_archived', 0)}")

            if comparison_parts:
                context_parts.append(f"<assessment_comparison>\n{chr(10).join(comparison_parts)}\n</assessment_comparison>")

        elif assessment_comparison and assessment_comparison.get("status") == "no_comparison":
            # User has only taken the assessment once
            context_parts.append(
                "<assessment_comparison>\n"
                "No previous assessments found. This is the user's first assessment.\n"
                "</assessment_comparison>"
            )

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
            
            plan = self._create_plan(analysis, user_state, message)
            step.complete(plan)
            steps.append(step)
            
            # === EXECUTE ===
            step = ThinkingStep(
                phase=ThinkingPhase.PLANNING,
                description="Executing plan",
            )
            
            execution_result = await self._execute_plan(plan, message, analysis, user_state)
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
                youtube_context=execution_result.youtube_context,
                selve_web_context=execution_result.selve_web_context,
                assessment_data=execution_result.assessment_data,
                assessment_comparison=execution_result.assessment_comparison,
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
