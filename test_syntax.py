"""
Syntax and Structure Validation for Agentic RAG Implementation
Tests code structure without requiring dependencies
"""

import ast
import os
import sys

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_file_structure(filepath, required_items):
    """Check if file contains required classes/functions"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)

        found_items = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                found_items.add(node.name)
            elif isinstance(node, ast.ClassDef):
                found_items.add(node.name)

        missing = set(required_items) - found_items
        return missing
    except Exception as e:
        return None

print("=" * 70)
print("🔍 AGENTIC RAG SYNTAX & STRUCTURE VALIDATION")
print("=" * 70)

tests_passed = 0
tests_total = 0

# Test 1: function_definitions.py
print("\n📄 Testing: app/tools/function_definitions.py")
print("-" * 70)
tests_total += 1

filepath = "/home/chris/selve-org/selve-chat-backend/app/tools/function_definitions.py"
valid, error = check_python_syntax(filepath)

if valid:
    print("✅ Syntax valid")

    required = [
        "get_tool_definitions",
        "convert_to_anthropic_format",
        "convert_to_gemini_format",
        "VALID_ARCHETYPES"
    ]
    missing = check_file_structure(filepath, required)

    if not missing:
        print("✅ All required items present:")
        for item in required:
            print(f"   - {item}")
        tests_passed += 1
    else:
        print(f"❌ Missing items: {missing}")
else:
    print(f"❌ Syntax error: {error}")

# Test 2: llm_service.py
print("\n📄 Testing: app/services/llm_service.py")
print("-" * 70)
tests_total += 1

filepath = "/home/chris/selve-org/selve-chat-backend/app/services/llm_service.py"
valid, error = check_python_syntax(filepath)

if valid:
    print("✅ Syntax valid")

    required = [
        "LLMService",
        "call_with_tools",
        "_call_openai_with_tools",
        "_call_anthropic_with_tools",
        "_call_gemini_with_tools",
        "_generate_gemini",
        "_generate_gemini_stream"
    ]
    missing = check_file_structure(filepath, required)

    if not missing:
        print("✅ All required items present:")
        for item in required:
            print(f"   - {item}")
        tests_passed += 1
    else:
        print(f"❌ Missing items: {missing}")
else:
    print(f"❌ Syntax error: {error}")

# Test 3: thinking_engine.py
print("\n📄 Testing: app/services/thinking_engine.py")
print("-" * 70)
tests_total += 1

filepath = "/home/chris/selve-org/selve-chat-backend/app/services/thinking_engine.py"
valid, error = check_python_syntax(filepath)

if valid:
    print("✅ Syntax valid")

    required = [
        "ThinkingEngine",
        "_agentic_tool_loop",
        "_execute_tool",
        "_aggregate_tool_result",
        "_get_agentic_system_prompt",
        "_create_plan",  # Legacy method preserved
        "_execute_plan"  # Legacy method preserved
    ]
    missing = check_file_structure(filepath, required)

    if not missing:
        print("✅ All required items present:")
        for item in required:
            print(f"   - {item}")
        tests_passed += 1
    else:
        print(f"❌ Missing items: {missing}")
else:
    print(f"❌ Syntax error: {error}")

# Test 4: Check .env configuration
print("\n📄 Testing: .env configuration")
print("-" * 70)
tests_total += 1

env_file = "/home/chris/selve-org/selve-chat-backend/.env"
try:
    with open(env_file, 'r') as f:
        env_content = f.read()

    required_vars = [
        "ENABLE_AGENTIC_RAG",
        "MAX_TOOL_ITERATIONS",
        "GEMINI_API_KEY_DEV",
        "GEMINI_MODEL"
    ]

    found_vars = []
    for var in required_vars:
        if var in env_content:
            found_vars.append(var)

    if len(found_vars) == len(required_vars):
        print("✅ All required environment variables present:")
        for var in found_vars:
            print(f"   - {var}")
        tests_passed += 1
    else:
        missing_vars = set(required_vars) - set(found_vars)
        print(f"❌ Missing environment variables: {missing_vars}")
except Exception as e:
    print(f"❌ Error reading .env: {e}")

# Test 5: Check requirements.txt
print("\n📄 Testing: requirements.txt")
print("-" * 70)
tests_total += 1

req_file = "/home/chris/selve-org/selve-chat-backend/requirements.txt"
try:
    with open(req_file, 'r') as f:
        req_content = f.read()

    if "google-generativeai" in req_content:
        print("✅ google-generativeai package added to requirements")
        tests_passed += 1
    else:
        print("❌ google-generativeai package missing from requirements")
except Exception as e:
    print(f"❌ Error reading requirements.txt: {e}")

# Test 6: Code quality checks
print("\n📄 Testing: Code integration")
print("-" * 70)
tests_total += 1

integration_checks = []

# Check thinking_engine.py has the toggle logic
thinking_file = "/home/chris/selve-org/selve-chat-backend/app/services/thinking_engine.py"
with open(thinking_file, 'r') as f:
    thinking_content = f.read()

if 'ENABLE_AGENTIC_RAG' in thinking_content:
    integration_checks.append("✅ Toggle logic present in thinking_engine.py")
else:
    integration_checks.append("❌ Toggle logic missing")

if '_agentic_tool_loop' in thinking_content and 'langfuse' in thinking_content:
    integration_checks.append("✅ Langfuse tracing in agentic loop")
else:
    integration_checks.append("❌ Langfuse tracing missing")

# Check llm_service has Gemini support
llm_file = "/home/chris/selve-org/selve-chat-backend/app/services/llm_service.py"
with open(llm_file, 'r') as f:
    llm_content = f.read()

if 'google.generativeai' in llm_content:
    integration_checks.append("✅ Gemini import present")
else:
    integration_checks.append("❌ Gemini import missing")

if 'GEMINI_PREFIXES' in llm_content:
    integration_checks.append("✅ Gemini provider detection")
else:
    integration_checks.append("❌ Gemini provider detection missing")

all_passed = all("✅" in check for check in integration_checks)
for check in integration_checks:
    print(check)

if all_passed:
    tests_passed += 1

# Summary
print("\n" + "=" * 70)
print("📊 VALIDATION SUMMARY")
print("=" * 70)
print(f"\n✅ Passed: {tests_passed}/{tests_total} tests")

if tests_passed == tests_total:
    print("\n🎉 All syntax and structure checks passed!")
    print("\n✨ Agentic RAG implementation is structurally sound")
    print("\n📋 Ready for runtime testing:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Start backend: uvicorn app.main:app --reload")
    print("   3. Test with queries:")
    print("      - Anonymous: 'Tell me about the Explorer archetype'")
    print("      - Logged in: 'What's my personality type?'")
    print("   4. Monitor Langfuse for nested traces")
    print("   5. Toggle ENABLE_AGENTIC_RAG to test fallback")
else:
    print(f"\n⚠️  {tests_total - tests_passed} validation(s) failed")

sys.exit(0 if tests_passed == tests_total else 1)
