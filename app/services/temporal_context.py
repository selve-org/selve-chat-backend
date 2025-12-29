"""
Temporal Context Service - Provides time/situational awareness to chatbot.

Helps chatbot understand:
- Time of day (morning, afternoon, evening, late night)
- Day of week (weekday vs weekend)
- Timezone-aware context for better personalization

This makes the chatbot more contextually aware and empathetic.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any


class TemporalContext:
    """Provides temporal and situational context for the chatbot."""

    @staticmethod
    def get_time_of_day(hour: int) -> str:
        """
        Categorize time of day based on hour (0-23).

        Args:
            hour: Hour in 24-hour format (0-23)

        Returns:
            Time period: "early morning", "morning", "afternoon", "evening", or "late night"
        """
        if 0 <= hour < 6:
            return "late night"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late night"

    @staticmethod
    def get_day_context(weekday: int) -> Dict[str, Any]:
        """
        Get context about the day of week.

        Args:
            weekday: Day of week (0=Monday, 6=Sunday)

        Returns:
            Dict with day_name, is_weekend, is_monday, is_friday
        """
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return {
            "day_name": day_names[weekday],
            "is_weekend": weekday >= 5,  # Saturday or Sunday
            "is_monday": weekday == 0,
            "is_friday": weekday == 4,
        }

    @classmethod
    def get_context(cls, user_timezone: str = "UTC") -> str:
        """
        Get complete temporal context as a formatted string for system prompt.

        Args:
            user_timezone: User's timezone (e.g., "America/New_York", "Europe/London")

        Returns:
            Formatted context string to inject into system prompt
        """
        try:
            tz = ZoneInfo(user_timezone)
        except Exception:
            # Fallback to UTC if timezone is invalid
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        hour = now.hour
        weekday = now.weekday()

        time_of_day = cls.get_time_of_day(hour)
        day_context = cls.get_day_context(weekday)

        # Build context string
        parts = [
            f"Current time context: It's {time_of_day} on {day_context['day_name']}.",
        ]

        # Add relevant situational insights
        if time_of_day == "late night" and hour >= 22:
            parts.append(
                "Late night conversations may indicate stress, insomnia, or deep reflection. "
                "Be extra empathetic and considerate."
            )
        elif time_of_day == "late night" and hour < 6:
            parts.append(
                "Very late/early hour - user may be sleep-deprived or dealing with urgent concerns. "
                "Show extra care and understanding."
            )

        if day_context["is_monday"]:
            parts.append(
                "Monday can be stressful for many (back-to-work anxiety). "
                "Be understanding if they mention stress or overwhelm."
            )
        elif day_context["is_friday"]:
            parts.append(
                "Friday often brings relief as the work week ends. "
                "User may be more relaxed or planning for the weekend."
            )
        elif day_context["is_weekend"]:
            parts.append(
                "Weekend - user may have more time for reflection and self-discovery. "
                "Or they might be dealing with loneliness if socially isolated."
            )

        return "\n".join(parts)

    @classmethod
    def get_appropriate_greeting(cls, user_timezone: str = "UTC") -> str:
        """
        Get contextually appropriate greeting based on time of day.

        Args:
            user_timezone: User's timezone

        Returns:
            Greeting like "Good morning!", "Good evening!", etc.
        """
        try:
            tz = ZoneInfo(user_timezone)
        except Exception:
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        hour = now.hour
        time_of_day = cls.get_time_of_day(hour)

        greetings = {
            "late night": "Hey there!",  # Neutral, since "good night" sounds like goodbye
            "morning": "Good morning!",
            "afternoon": "Good afternoon!",
            "evening": "Good evening!",
        }

        return greetings.get(time_of_day, "Hey!")
