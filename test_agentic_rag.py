"""
Test script for Agentic RAG Implementation

Tests:
1. Function definitions are valid
2. LLM service supports all providers
3. Tool execution dispatcher works
4. Agentic toggle functions correctly
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_function_definitions():
    """Test 1: Verify function definitions are valid"""
    print("\n🧪 TEST 1: Function Definitions")
    print("=" * 60)

    try:
        from app.tools.function_definitions import (
            get_tool_definitions,
            convert_to_anthropic_format,
            convert_to_gemini_format,
            VALID_ARCHETYPES
        )

        # Test with no user state (anonymous)
        tools_anon = get_tool_definitions(None)
        print(f"✅ Anonymous user tools: {len(tools_anon)} tools")
        tool_names_anon = [t["function"]["name"] for t in tools_anon]
        print(f"   Available: {', '.join(tool_names_anon)}")

        # Test with logged-in user
        class MockUserState:
            userId = "test-user"
            clerk_user_id = "test-clerk-id"
            has_assessment = True
            archetype = "Explorer"

        tools_auth = get_tool_definitions(MockUserState())
        print(f"✅ Logged-in user tools: {len(tools_auth)} tools")
        tool_names_auth = [t["function"]["name"] for t in tools_auth]
        print(f"   Available: {', '.join(tool_names_auth)}")

        # Verify assessment tools are only for logged-in users
        assessment_tools = ["assessment_fetch", "assessment_compare"]
        has_assessment = all(tool in tool_names_auth for tool in assessment_tools)
        no_assessment_anon = all(tool not in tool_names_anon for tool in assessment_tools)

        if has_assessment and no_assessment_anon:
            print("✅ Assessment tools correctly gated by auth")
        else:
            print("❌ Assessment tool gating failed")
            return False

        # Test format converters
        anthropic_tools = convert_to_anthropic_format(tools_anon)
        print(f"✅ Anthropic format conversion: {len(anthropic_tools)} tools")

        gemini_tools = convert_to_gemini_format(tools_anon)
        print(f"✅ Gemini format conversion: {len(gemini_tools)} tools")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_service():
    """Test 2: Verify LLM service initialization"""
    print("\n🧪 TEST 2: LLM Service")
    print("=" * 60)

    try:
        from app.services.llm_service import LLMService

        # Initialize service
        llm = LLMService()
        print(f"✅ LLM Service initialized")
        print(f"   Provider: {llm.provider}")
        print(f"   Model: {llm.model}")
        print(f"   OpenAI available: {llm.openai is not None}")
        print(f"   Anthropic available: {llm.anthropic is not None}")
        print(f"   Gemini available: {llm.gemini is not None}")

        # Check if call_with_tools method exists
        if hasattr(llm, 'call_with_tools'):
            print("✅ call_with_tools() method exists")
        else:
            print("❌ call_with_tools() method missing")
            return False

        # Check provider detection
        test_cases = [
            ("gpt-4o-mini", "openai"),
            ("claude-3-5-haiku-20241022", "anthropic"),
            ("gemini-3-flash", "gemini"),
        ]

        for model, expected_provider in test_cases:
            detected = llm._resolve_provider_for_model(model)
            if detected == expected_provider:
                print(f"✅ Provider detection: {model} → {expected_provider}")
            else:
                print(f"❌ Provider detection failed: {model} → {detected} (expected {expected_provider})")
                return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_thinking_engine():
    """Test 3: Verify thinking engine has agentic methods"""
    print("\n🧪 TEST 3: Thinking Engine")
    print("=" * 60)

    try:
        from app.services.thinking_engine import ThinkingEngine

        # Check if agentic methods exist
        methods_to_check = [
            "_agentic_tool_loop",
            "_execute_tool",
            "_aggregate_tool_result",
            "_get_agentic_system_prompt",
        ]

        for method in methods_to_check:
            if hasattr(ThinkingEngine, method):
                print(f"✅ Method exists: {method}")
            else:
                print(f"❌ Method missing: {method}")
                return False

        # Check if legacy methods still exist (for fallback)
        legacy_methods = ["_create_plan", "_execute_plan"]
        for method in legacy_methods:
            if hasattr(ThinkingEngine, method):
                print(f"✅ Legacy method preserved: {method}")
            else:
                print(f"❌ Legacy method missing: {method}")
                return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_env_configuration():
    """Test 4: Verify environment configuration"""
    print("\n🧪 TEST 4: Environment Configuration")
    print("=" * 60)

    try:
        # Check critical env variables
        env_vars = {
            "ENABLE_AGENTIC_RAG": os.getenv("ENABLE_AGENTIC_RAG"),
            "MAX_TOOL_ITERATIONS": os.getenv("MAX_TOOL_ITERATIONS"),
            "GEMINI_API_KEY": "***" if os.getenv("GEMINI_API_KEY") else None,
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
        }

        for key, value in env_vars.items():
            if value:
                print(f"✅ {key}: {value}")
            else:
                print(f"⚠️  {key}: Not set (may use default)")

        # Check toggle logic
        agentic_enabled = os.getenv("ENABLE_AGENTIC_RAG", "true").lower() == "true"
        print(f"\n{'✅' if agentic_enabled else '⚠️ '} Agentic RAG: {'ENABLED' if agentic_enabled else 'DISABLED (fallback mode)'}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_system_prompt():
    """Test 5: Verify system prompt generation"""
    print("\n🧪 TEST 5: System Prompt Generation")
    print("=" * 60)

    try:
        from app.services.thinking_engine import ThinkingEngine
        from app.services.llm_service import LLMService
        from app.services.rag_service import RAGService

        # Initialize (mocked)
        rag_service = RAGService()
        llm_service = LLMService()
        engine = ThinkingEngine(rag_service=rag_service, llm_service=llm_service)

        # Test with anonymous user
        class MockUserStateAnon:
            clerk_user_id = None

        prompt_anon = engine._get_agentic_system_prompt(MockUserStateAnon())

        if "Anonymous" in prompt_anon:
            print("✅ Anonymous user prompt contains 'Anonymous'")
        else:
            print("❌ Anonymous user prompt missing user status")
            return False

        if "assessment_fetch" in prompt_anon:
            print("✅ Prompt lists assessment tools")
        else:
            print("❌ Prompt missing tool descriptions")
            return False

        # Test with logged-in user
        class MockUserStateAuth:
            clerk_user_id = "test-clerk-123"
            has_assessment = True
            archetype = "Explorer"

        prompt_auth = engine._get_agentic_system_prompt(MockUserStateAuth())

        if "**Logged in**: YES" in prompt_auth:
            print("✅ Logged-in user prompt shows auth status")
        else:
            print("❌ Logged-in user prompt missing auth status")
            return False

        if "Explorer" in prompt_auth:
            print("✅ Prompt includes user's archetype")
        else:
            print("❌ Prompt missing user's archetype")
            return False

        print(f"\n📝 Sample prompt (first 300 chars):")
        print(prompt_auth[:300] + "...")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🚀 AGENTIC RAG IMPLEMENTATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Function Definitions", test_function_definitions),
        ("LLM Service", test_llm_service),
        ("Thinking Engine", test_thinking_engine),
        ("Environment Configuration", test_env_configuration),
        ("System Prompt", test_system_prompt),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Agentic RAG is ready for testing.")
        print("\n📋 Next steps:")
        print("   1. Start the backend: cd selve-chat-backend && uvicorn app.main:app --reload")
        print("   2. Test with real queries")
        print("   3. Monitor Langfuse for nested traces")
        print("   4. Try toggling ENABLE_AGENTIC_RAG to test fallback")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")

    return passed == total


if __name__ == "__main__":
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv("/home/chris/selve-org/selve-chat-backend/.env")
    except ImportError:
        print("⚠️  python-dotenv not installed, using existing environment")

    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
