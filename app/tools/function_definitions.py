"""
Function Calling Definitions for Agentic RAG

This module defines function calling schemas for all available tools in OpenAI format.
These schemas are used by LLMs to understand when and how to call tools.
"""

from typing import List, Dict, Optional, Any


def get_tool_definitions(user_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Get function calling tool definitions based on user context.

    Args:
        user_state: Optional user state dict containing userId, hasAssessment, archetype, etc.

    Returns:
        List of tool definitions in OpenAI function calling format
    """
    # Base tools available to all users
    tools = [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "Search the SELVE psychology knowledge base for information about dimensional psychology, personality archetypes, psychological concepts, and self-understanding frameworks. ALWAYS use this tool when users ask about: personality theory, dimensional psychology concepts (LUMEN, AETHER, ORPHEUS, VARA, CHRONOS, KAEL, ORIN, LYRA), specific archetypes (Explorer, Architect, Guardian, Harmonizer, Visionary, Catalyst, Protector, Sage, etc.), psychological insights, self-understanding frameworks, how dimensions work, what archetypes mean, archetype characteristics, dimension definitions, personality science, or any SELVE psychology concepts. This is your PRIMARY tool for answering psychology-related questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for the knowledge base"
                        },
                        "dimensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of specific psychological dimensions to focus on (e.g., 'freedom', 'structure', 'energy', 'reflection')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "youtube_search",
                "description": "Search for educational psychology videos on YouTube. Returns a list of relevant videos with titles, descriptions, and video IDs. Use this when users want to learn through video content or when recommending educational resources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for YouTube videos (e.g., 'cognitive behavioral therapy', 'personality psychology')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of video results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "youtube_fetch",
                "description": "Fetch the full transcript and metadata for a specific YouTube video. Use this when you need the detailed content of a video, such as after finding it via youtube_search. Requires a valid YouTube video ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "YouTube video ID (e.g., 'dQw4w9WgXcQ')"
                        },
                        "include_transcript": {
                            "type": "boolean",
                            "description": "Whether to include the full video transcript (default: true)",
                            "default": True
                        }
                    },
                    "required": ["video_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, recent research, news, or topics not covered in the SELVE knowledge base. ALWAYS use this tool when users ask about: recent psychology research, current events, news, statistics, studies published after 2024, non-SELVE psychology topics, general knowledge questions, scientific findings, research papers, external psychology frameworks (Big Five, MBTI, Enneagram), therapists or psychologists, therapy techniques, mental health topics, or any information not in the SELVE knowledge base. Use this to supplement SELVE knowledge with broader psychological context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The web search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of search results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "selve_web_search",
                "description": "Search specifically for SELVE-related content on the official SELVE website (selve.me). ALWAYS use this tool when users ask about: SELVE company information, SELVE's mission or approach, SELVE pricing or subscription plans, SELVE features, how to use SELVE, SELVE blog posts, SELVE team, SELVE partnerships, SELVE news or announcements, account or billing questions, privacy policy, terms of service, or any official SELVE business/product information not covered in the psychology knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SELVE-specific search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search the user's conversation history for previous discussions, questions, or topics. Use this to recall past conversations, reference earlier topics, or provide context-aware responses based on conversation history.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for conversation history"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of past messages to retrieve (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        },
    ]

    # Add assessment tools only if user is logged in
    # Handle both dict and object user_state
    user_id = None
    if user_state:
        if hasattr(user_state, 'get'):
            # Dict-like object
            user_id = user_state.get("userId") or user_state.get("clerk_user_id")
        else:
            # Object with attributes
            user_id = getattr(user_state, 'userId', None) or getattr(user_state, 'clerk_user_id', None)

    if user_id:
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "assessment_fetch",
                    "description": "Fetch the logged-in user's complete personality assessment including archetype, 8-dimensional scores (LUMEN, AETHER, ORPHEUS, VARA, CHRONOS, KAEL, ORIN, LYRA), and full narrative profile. ONLY use when user is logged in. ALWAYS call this tool when user asks about: their personality, assessment, profile, results, archetype, scores, dimensions, numbers, ratings, how they scored, what they are, their type, comparing their results ('compared to mine', 'vs my', 'against mine', 'my own result'), showing their data ('show me mine', 'pull up my', 'grab my', 'get my'), or any personal assessment query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_narrative": {
                                "type": "boolean",
                                "description": "Include the detailed personality narrative (default: true)",
                                "default": True
                            },
                            "include_scores": {
                                "type": "boolean",
                                "description": "Include dimensional scores (default: true)",
                                "default": True
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assessment_compare",
                    "description": "Compare two personality archetypes to show similarities, differences, and relationship dynamics. Use this when users want to compare their archetype with another, or understand archetype compatibility. Requires user to be logged in if comparing against their own archetype.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "archetype_a": {
                                "type": "string",
                                "description": "First archetype to compare (e.g., 'Explorer', 'Architect', 'Guardian')"
                            },
                            "archetype_b": {
                                "type": "string",
                                "description": "Second archetype to compare (e.g., 'Explorer', 'Architect', 'Guardian')"
                            }
                        },
                        "required": ["archetype_a", "archetype_b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "friend_insights_fetch",
                    "description": "Fetch the logged-in user's friend insights including blind spots, friend names who responded, invite limits, and comparison data showing how friends perceive them vs self-perception. ONLY use when user is logged in. ALWAYS call this tool when user asks about: blind spots, how friends see them, friend perception, what friends think/said, external perception, friend feedback, friend responses, who responded to assessment, how many friends responded, friend invites, invite limits, max invites, invites remaining, friend comparison, friend vs self scores, or any query about friends' views.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_narrative": {
                                "type": "boolean",
                                "description": "Include the friend insights narrative summary (default: true)",
                                "default": True
                            },
                            "include_individual_responses": {
                                "type": "boolean",
                                "description": "Include individual friend response details (default: false)",
                                "default": False
                            }
                        },
                        "required": []
                    }
                }
            },
        ])

    return tools


def convert_to_anthropic_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI function calling format to Anthropic tool use format.

    Args:
        tools: List of tools in OpenAI format

    Returns:
        List of tools in Anthropic format
    """
    anthropic_tools = []

    for tool in tools:
        function = tool["function"]
        anthropic_tools.append({
            "name": function["name"],
            "description": function["description"],
            "input_schema": function["parameters"]
        })

    return anthropic_tools


def convert_to_gemini_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI function calling format to Gemini function calling format.

    Args:
        tools: List of tools in OpenAI format

    Returns:
        List of tools in Gemini format
    """
    gemini_tools = []

    for tool in tools:
        function = tool["function"]
        gemini_tools.append({
            "function_declarations": [{
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"]
            }]
        })

    return gemini_tools


# Archetype list for reference (used in validation and comparison)
VALID_ARCHETYPES = [
    "Explorer", "Architect", "Guardian", "Harmonizer",
    "Visionary", "Catalyst", "Protector", "Sage",
    "Pioneer", "Innovator", "Nurturer", "Strategist"
]
