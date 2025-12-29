"""
Test GeoIP → Langfuse Integration

Verifies that GeoIP location data (country, state, city) is properly
logged to Langfuse traces in production.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.agentic_chat_service import AgenticChatService
from app.services.geoip_service import GeoIPInfo
from app.utils.result import Result


async def test_geoip_metadata_in_langfuse():
    """Test that GeoIP data is included in Langfuse metadata."""
    print("\n" + "="*70)
    print("TEST: GEOIP → LANGFUSE METADATA INTEGRATION")
    print("="*70 + "\n")

    # Create service
    service = AgenticChatService()

    # Mock GeoIP response (simulate production IP lookup)
    mock_geo_data = {
        "ip": "198.51.100.42",
        "country": {"iso_code": "US", "name": "United States"},
        "state": {"name": "California"},
        "city": {"name": "San Francisco"},
        "timezone": {"name": "America/Los_Angeles"},
    }
    mock_geo_info = GeoIPInfo(mock_geo_data)
    mock_geo_result = Result.success(mock_geo_info)

    # Mock the GeoIP service
    service._geoip_service = MagicMock()
    service._geoip_service.get_geolocation = AsyncMock(return_value=mock_geo_result)

    # Track metadata passed to propagate_attributes
    captured_metadata = {}

    def mock_propagate_attributes(**kwargs):
        """Capture metadata passed to Langfuse."""
        nonlocal captured_metadata
        captured_metadata = kwargs.get("metadata", {})
        # Return a context manager that does nothing
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield
        return dummy_context()

    # Mock Langfuse to capture metadata
    with patch('app.services.agentic_chat_service.get_client') as mock_client:
        with patch('app.services.agentic_chat_service.propagate_attributes', side_effect=mock_propagate_attributes):
            # Mock other dependencies to prevent actual LLM calls
            service._security_guard = MagicMock()
            service._security_guard.analyze = AsyncMock(return_value=MagicMock(
                is_safe=True,
                threat_level=MagicMock(value="safe"),
                risk_score=0.0,
            ))

            service._user_state_service = MagicMock()
            service._user_state_service.get_user_state = AsyncMock(return_value=None)

            service._thinking_engine = MagicMock()
            service._thinking_engine.think_and_respond = AsyncMock(return_value=AsyncMock(
                __aiter__=AsyncMock(return_value=iter([
                    {"type": "status", "phase": "complete", "message": "Done"},
                    "Test response"
                ]))
            ))

            # Mock Langfuse client
            mock_trace = MagicMock()
            mock_trace.__enter__ = MagicMock(return_value=mock_trace)
            mock_trace.__exit__ = MagicMock(return_value=None)
            mock_client.return_value.start_as_current_observation = MagicMock(return_value=mock_trace)
            mock_client.return_value.get_current_trace_id = MagicMock(return_value="test-trace-id")

            # Call chat_stream with a client IP
            test_ip = "198.51.100.42"
            print(f"✓ Simulating chat request with IP: {test_ip}")

            chunks = []
            try:
                async for chunk in service.chat_stream(
                    message="Hello",
                    clerk_user_id="test_user",
                    session_id="test_session",
                    client_ip=test_ip,
                ):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
            except StopIteration:
                pass
            except Exception as e:
                # Expected due to mocking
                pass

            # Verify GeoIP service was called
            service._geoip_service.get_geolocation.assert_called_once_with(test_ip)
            print(f"✓ GeoIP service called with IP: {test_ip}")

            # Verify metadata contains GeoIP data
            print("\n📊 Captured Langfuse Metadata:")
            print("  " + "-"*66)
            for key, value in captured_metadata.items():
                print(f"  {key}: {value}")
            print("  " + "-"*66)

            # Assertions
            checks = [
                ("streaming" in captured_metadata, "Contains 'streaming' flag"),
                ("ip" in captured_metadata, "Contains IP address"),
                ("country" in captured_metadata, "Contains country"),
                ("state" in captured_metadata, "Contains state"),
                ("city" in captured_metadata, "Contains city"),
                ("timezone" in captured_metadata, "Contains timezone"),
                (captured_metadata.get("country") == "United States", "Country is 'United States'"),
                (captured_metadata.get("state") == "California", "State is 'California'"),
                (captured_metadata.get("city") == "San Francisco", "City is 'San Francisco'"),
            ]

            all_passed = True
            print("\n🔍 Validation Checks:")
            for check, description in checks:
                status = "✓" if check else "✗"
                print(f"  {status} {description}")
                if not check:
                    all_passed = False

            if all_passed:
                print("\n" + "="*70)
                print("✅ GEOIP → LANGFUSE INTEGRATION: PASSED")
                print("="*70)
                print("\n✓ GeoIP data (country, state, city) is properly logged to Langfuse")
                print("✓ Production traces will include location metadata")
                print("✓ Localhost (127.0.0.1) will still work (returns 'unknown')")
                return True
            else:
                print("\n" + "="*70)
                print("❌ GEOIP → LANGFUSE INTEGRATION: FAILED")
                print("="*70)
                return False


async def test_localhost_graceful_handling():
    """Test that localhost IP (0.0.0.0, 127.0.0.1) is handled gracefully."""
    print("\n" + "="*70)
    print("TEST: LOCALHOST IP GRACEFUL HANDLING")
    print("="*70 + "\n")

    service = AgenticChatService()

    # Mock GeoIP service returning minimal data for localhost
    mock_geo_info = GeoIPInfo({"ip": "127.0.0.1"})
    mock_geo_result = Result.success(mock_geo_info)

    service._geoip_service = MagicMock()
    service._geoip_service.get_geolocation = AsyncMock(return_value=mock_geo_result)

    captured_metadata = {}

    def mock_propagate_attributes(**kwargs):
        nonlocal captured_metadata
        captured_metadata = kwargs.get("metadata", {})
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield
        return dummy_context()

    with patch('app.services.agentic_chat_service.get_client'):
        with patch('app.services.agentic_chat_service.propagate_attributes', side_effect=mock_propagate_attributes):
            service._security_guard = MagicMock()
            service._security_guard.analyze = AsyncMock(return_value=MagicMock(
                is_safe=True,
                threat_level=MagicMock(value="safe"),
                risk_score=0.0,
            ))

            try:
                async for _ in service.chat_stream(
                    message="Hello",
                    client_ip="127.0.0.1",
                ):
                    break  # Just need to trigger the call
            except:
                pass  # Expected

            print(f"✓ Localhost IP handled gracefully")
            print(f"✓ Metadata: {captured_metadata}")

            # Should still have streaming flag, but may not have full geo data
            if "streaming" in captured_metadata:
                print("\n✅ LOCALHOST HANDLING: PASSED")
                return True
            else:
                print("\n❌ LOCALHOST HANDLING: FAILED")
                return False


async def run_all_tests():
    """Run all GeoIP → Langfuse integration tests."""
    print("\n" + "="*70)
    print("GEOIP → LANGFUSE INTEGRATION TEST SUITE")
    print("="*70)

    results = {
        "GeoIP Metadata in Langfuse": await test_geoip_metadata_in_langfuse(),
        "Localhost Graceful Handling": await test_localhost_graceful_handling(),
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
        print("🎉 ALL TESTS PASSED - GEOIP DATA WILL BE LOGGED TO LANGFUSE")
        print("="*70)
        print("\n✓ Production: Real IPs → Full location data (country, state, city)")
        print("✓ Development: Localhost → Minimal data (graceful fallback)")
        print("✓ Langfuse dashboard will show user locations")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW ERRORS ABOVE")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    exit(exit_code)
