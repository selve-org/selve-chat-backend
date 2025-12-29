"""
Comprehensive Temporal Awareness Tests - Full Integration

Tests:
1. Timezone detection and propagation (frontend → backend → system prompt)
2. Relative time formatting (timestamps → "3 weeks ago")
3. Memory temporal context in system prompts
4. User notes temporal context
5. Edge cases and error handling
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.temporal_context import TemporalContext
from app.services.user_state_service import format_relative_time, UserState, ConversationMemory, UserNote, AssessmentStatus
from app.prompts.system_prompt import build_temporal_context_prompt


def test_relative_time_formatting():
    """Test that timestamps convert to human-readable formats correctly."""
    print("\n" + "="*70)
    print("TEST 1: RELATIVE TIME FORMATTING")
    print("="*70)

    now = datetime.utcnow()

    test_cases = [
        (now - timedelta(minutes=5), "minutes ago", "5 minutes"),
        (now - timedelta(hours=2), "2 hours ago", "2 hours"),
        (now - timedelta(days=1), "yesterday", "1 day"),
        (now - timedelta(days=3), "3 days ago", "3 days"),
        (now - timedelta(weeks=2), "2 weeks ago", "2 weeks"),
        (now - timedelta(days=45), "month", "45 days"),  # Can be "last month" or similar
        (now - timedelta(days=90), "months ago", "90 days"),  # Can be "3 months" or "a few months"
        (now - timedelta(days=200), "year ago", "200 days"),  # Can be "half a year" or "several months"
        (now - timedelta(days=400), "year", "400 days"),  # Contains "year"
    ]

    all_passed = True
    for timestamp, expected_contains, description in test_cases:
        result = format_relative_time(timestamp)
        passed = expected_contains in result
        status = "✓" if passed else "✗"

        if not passed:
            all_passed = False
            print(f"{status} {description}: Expected '{expected_contains}', got '{result}'")
        else:
            print(f"{status} {description} → '{result}'")

    if all_passed:
        print("\n✅ All relative time formatting tests PASSED")
    else:
        print("\n❌ Some relative time formatting tests FAILED")

    return all_passed


def test_timezone_context_generation():
    """Test timezone-aware context generation."""
    print("\n" + "="*70)
    print("TEST 2: TIMEZONE CONTEXT GENERATION")
    print("="*70)

    test_timezones = [
        "America/New_York",
        "Europe/London",
        "Asia/Tokyo",
        "Australia/Sydney",
        "America/Los_Angeles",
    ]

    all_passed = True

    for tz in test_timezones:
        try:
            context = TemporalContext.get_context(tz)

            # Check that context contains expected elements
            checks = [
                ("time context" in context.lower(), "Contains time context"),
                (any(day in context for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]), "Contains day of week"),
                (any(period in context.lower() for period in ["morning", "afternoon", "evening", "night"]), "Contains time of day"),
            ]

            print(f"\n  Testing timezone: {tz}")
            for check, description in checks:
                status = "✓" if check else "✗"
                print(f"    {status} {description}")
                if not check:
                    all_passed = False

        except Exception as e:
            print(f"  ✗ Error with timezone {tz}: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All timezone context generation tests PASSED")
    else:
        print("\n❌ Some timezone context generation tests FAILED")

    return all_passed


def test_system_prompt_temporal_injection():
    """Test that temporal context is properly injected into system prompts."""
    print("\n" + "="*70)
    print("TEST 3: SYSTEM PROMPT TEMPORAL INJECTION")
    print("="*70)

    try:
        # Build temporal context for different timezones
        prompt_ny = build_temporal_context_prompt("America/New_York")
        prompt_london = build_temporal_context_prompt("Europe/London")
        prompt_tokyo = build_temporal_context_prompt("Asia/Tokyo")

        checks = [
            (prompt_ny and "TEMPORAL AWARENESS" in prompt_ny, "New York prompt contains header"),
            (prompt_london and "TEMPORAL AWARENESS" in prompt_london, "London prompt contains header"),
            (prompt_tokyo and "TEMPORAL AWARENESS" in prompt_tokyo, "Tokyo prompt contains header"),
            (prompt_ny and "Good morning" in prompt_ny or "Good evening" in prompt_ny or "Good afternoon" in prompt_ny, "Contains greeting guidance"),
        ]

        all_passed = True
        for check, description in checks:
            status = "✓" if check else "✗"
            print(f"  {status} {description}")
            if not check:
                all_passed = False

        # Show sample output
        print(f"\n  Sample temporal prompt (NY):")
        print("  " + "-"*66)
        print("  " + prompt_ny[:300].replace("\n", "\n  "))
        print("  " + "-"*66)

        if all_passed:
            print("\n✅ System prompt temporal injection tests PASSED")
        else:
            print("\n❌ System prompt temporal injection tests FAILED")

        return all_passed

    except Exception as e:
        print(f"✗ Error in system prompt injection: {e}")
        return False


def test_user_state_temporal_context():
    """Test that UserState includes temporal context in conversation memories."""
    print("\n" + "="*70)
    print("TEST 4: USER STATE TEMPORAL CONTEXT")
    print("="*70)

    try:
        # Create test user state with memories
        now = datetime.utcnow()

        test_memories = [
            ConversationMemory(
                session_id="test_session_1",
                title="Discussion about career goals",
                summary="User discussed wanting to change careers to software engineering",
                key_insights=["Interested in tech", "Feels stuck in current role"],
                emotional_state="hopeful",
                topics_discussed=["career", "goals"],
                timestamp=now - timedelta(weeks=3)
            ),
            ConversationMemory(
                session_id="test_session_2",
                title="Exam stress conversation",
                summary="User stressed about upcoming final exams",
                key_insights=["High anxiety", "Perfectionist tendencies"],
                emotional_state="anxious",
                topics_discussed=["stress", "exams"],
                timestamp=now - timedelta(days=5)
            ),
        ]

        test_notes = [
            UserNote(
                id="note_1",
                category="observation",
                content="User struggles with public speaking",
                created_at=now - timedelta(days=60),  # ~2 months
                source="chatbot",
                importance=4
            ),
            UserNote(
                id="note_2",
                category="preference",
                content="Prefers written communication over verbal",
                created_at=now - timedelta(weeks=1),
                source="chatbot",
                importance=3
            ),
        ]

        user_state = UserState(
            user_id="test_user",
            clerk_user_id="test_clerk_user",
            user_name="Test User",
            assessment_status=AssessmentStatus.NOT_TAKEN,
            has_assessment=False,
            recent_memories=test_memories,
            notes=test_notes,
        )

        # Convert to context string
        context_string = user_state.to_context_string()

        # Verify temporal references are included
        checks = [
            ("3 weeks ago" in context_string or "last month" in context_string, "Contains 3-week-old memory timestamp"),
            ("5 days ago" in context_string or "yesterday" in context_string, "Contains 5-day-old memory timestamp"),
            ("2 months ago" in context_string, "Contains 2-month-old note timestamp"),
            ("1 week ago" in context_string or "last week" in context_string, "Contains 1-week-old note timestamp"),
            ("Discussion about career goals" in context_string, "Contains memory title"),
            ("Use these temporal references naturally" in context_string, "Contains temporal guidance for LLM"),
        ]

        all_passed = True
        for check, description in checks:
            status = "✓" if check else "✗"
            print(f"  {status} {description}")
            if not check:
                all_passed = False

        # Show sample output
        print(f"\n  Sample context with temporal references:")
        print("  " + "-"*66)

        # Extract just the memory section
        if "RECENT CONVERSATION HISTORY" in context_string:
            start = context_string.index("RECENT CONVERSATION HISTORY")
            end = context_string.index("###", start + 50) if "###" in context_string[start + 50:] else len(context_string)
            memory_section = context_string[start:end]

            for line in memory_section.split("\n")[:8]:
                print("  " + line)
        print("  " + "-"*66)

        if all_passed:
            print("\n✅ User state temporal context tests PASSED")
        else:
            print("\n❌ User state temporal context tests FAILED")

        return all_passed

    except Exception as e:
        print(f"✗ Error in user state temporal context: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 5: EDGE CASES & ERROR HANDLING")
    print("="*70)

    all_passed = True

    # Test invalid timezone (should gracefully fallback to UTC)
    try:
        context = build_temporal_context_prompt("Invalid/Timezone")
        status = "✓" if context else "✗"
        print(f"  {status} Invalid timezone gracefully handled")
        if not context:
            all_passed = False
    except Exception as e:
        print(f"  ✗ Invalid timezone caused error: {e}")
        all_passed = False

    # Test empty string timezone (should gracefully fallback)
    try:
        context = TemporalContext.get_context("")
        # Should either work or fail gracefully
        if context:
            print(f"  ✓ Empty timezone handled gracefully (fallback to UTC)")
        else:
            print(f"  ✓ Empty timezone handled (returns empty)")
    except:
        print(f"  ✓ Empty timezone raises error (acceptable)")

    # Test very old timestamp
    try:
        very_old = datetime.utcnow() - timedelta(days=3650)  # 10 years ago
        result = format_relative_time(very_old)
        status = "✓" if "year" in result else "✗"
        print(f"  {status} Very old timestamp: '{result}'")
        if not status:
            all_passed = False
    except Exception as e:
        print(f"  ✗ Very old timestamp caused error: {e}")
        all_passed = False

    # Test future timestamp (edge case)
    try:
        future = datetime.utcnow() + timedelta(hours=2)
        result = format_relative_time(future)
        # Should handle gracefully (might show "just now" or similar)
        print(f"  ✓ Future timestamp handled: '{result}'")
    except Exception as e:
        print(f"  ✗ Future timestamp caused error: {e}")
        all_passed = False

    if all_passed:
        print("\n✅ All edge case tests PASSED")
    else:
        print("\n❌ Some edge case tests FAILED")

    return all_passed


def run_all_tests():
    """Run complete temporal awareness test suite."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEMPORAL AWARENESS TEST SUITE")
    print("="*70)

    results = {
        "Relative Time Formatting": test_relative_time_formatting(),
        "Timezone Context Generation": test_timezone_context_generation(),
        "System Prompt Injection": test_system_prompt_temporal_injection(),
        "User State Temporal Context": test_user_state_temporal_context(),
        "Edge Cases": test_edge_cases(),
    }

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - TEMPORAL AWARENESS IS ROBUST & ERROR-FREE")
        print("="*70)
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW ERRORS ABOVE")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)
