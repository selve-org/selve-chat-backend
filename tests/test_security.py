"""
Security Test Suite
Tests for critical security vulnerabilities in usage tracking and authentication

Run with: pytest tests/test_security.py -v
"""

import asyncio
import pytest
import httpx
from datetime import datetime


# Configuration
API_BASE_URL = "http://localhost:8001"  # Adjust to your chat backend URL


class TestUsageTracking:
    """Test usage tracking security"""

    @pytest.mark.asyncio
    async def test_race_condition_prevention(self):
        """
        Test that concurrent requests cannot bypass usage limit

        This simulates 5 concurrent requests at $0.30 each ($1.50 total).
        With proper transaction locking, only 3 should succeed (total $0.90),
        and 2 should be rejected to prevent exceeding the $1.00 limit.
        """
        test_user_id = "test_user_race_condition"
        cost_per_request = 0.30

        async with httpx.AsyncClient() as client:
            # Reset user to free plan with $0 usage (setup)
            # TODO: Add reset endpoint for testing

            # Create 5 concurrent requests
            tasks = [
                client.post(
                    f"{API_BASE_URL}/api/chat",
                    json={
                        "message": "test",
                        "user_id": test_user_id,
                        "cost": cost_per_request
                    },
                    headers={"X-User-ID": test_user_id}
                )
                for _ in range(5)
            ]

            # Execute concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful vs rejected requests
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            rejected = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 429)

            print(f"Successful requests: {successful}")
            print(f"Rejected requests: {rejected}")
            print(f"Total cost would be: ${successful * cost_per_request:.2f}")

            # Assertions
            assert successful <= 3, "Too many requests succeeded - race condition detected!"
            assert successful * cost_per_request <= 1.0, "Total cost exceeded $1.00 limit!"
            assert rejected >= 2, "Not enough requests were rejected"

    @pytest.mark.asyncio
    async def test_negative_cost_rejection(self):
        """Test that negative costs are rejected"""
        test_user_id = "test_user_negative_cost"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/chat",
                json={
                    "message": "test",
                    "user_id": test_user_id,
                    "cost": -1.0,  # Negative cost - should be rejected
                    "tokens": 100
                },
                headers={"X-User-ID": test_user_id}
            )

            # Should return 400 Bad Request for invalid input
            assert response.status_code == 400, f"Expected 400, got {response.status_code}"
            assert "non-negative" in response.json().get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_excessive_cost_rejection(self):
        """Test that suspiciously high costs are rejected"""
        test_user_id = "test_user_excessive_cost"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/chat",
                json={
                    "message": "test",
                    "user_id": test_user_id,
                    "cost": 99.99,  # Excessive cost - should be rejected
                    "tokens": 100000
                },
                headers={"X-User-ID": test_user_id}
            )

            # Should return 400 Bad Request for suspicious cost
            assert response.status_code == 400
            assert "suspicious" in response.json().get("detail", "").lower()


class TestAuthentication:
    """Test authentication security"""

    @pytest.mark.asyncio
    async def test_missing_auth_header(self):
        """Test that requests without X-User-ID header are rejected"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/users/user_123/usage"
                # No X-User-ID header
            )

            assert response.status_code == 401, f"Expected 401, got {response.status_code}"
            assert "authentication required" in response.json().get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_unauthorized_access(self):
        """Test that users cannot access other users' data"""
        async with httpx.AsyncClient() as client:
            # User A trying to access User B's data
            response = await client.get(
                f"{API_BASE_URL}/api/users/user_b/usage",
                headers={"X-User-ID": "user_a"}  # Different user ID
            )

            assert response.status_code == 403, f"Expected 403, got {response.status_code}"
            assert "cannot access" in response.json().get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_authorized_access(self):
        """Test that users CAN access their own data"""
        test_user_id = "user_authorized_test"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/users/{test_user_id}/usage",
                headers={"X-User-ID": test_user_id}  # Same user ID
            )

            # Should succeed (or 404 if user doesn't exist, but not 401/403)
            assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"


class TestUsageLimit:
    """Test usage limit enforcement"""

    @pytest.mark.asyncio
    async def test_strict_limit_enforcement(self):
        """Test that free users are strictly limited to $1.00 (no 5% buffer)"""
        test_user_id = "test_user_strict_limit"

        async with httpx.AsyncClient() as client:
            # TODO: Reset user to $0.95 usage

            # Try to add $0.10 usage (would exceed $1.00)
            response = await client.post(
                f"{API_BASE_URL}/api/chat",
                json={
                    "message": "test",
                    "user_id": test_user_id,
                    "cost": 0.10,
                    "tokens": 100
                },
                headers={"X-User-ID": test_user_id}
            )

            # Should be rejected (429 Too Many Requests)
            assert response.status_code == 429, "Limit should be strict - no buffer allowed"
            assert "limit exceeded" in response.json().get("detail", "").lower()


class TestWebhookSecurity:
    """Test webhook security"""

    def test_startup_validation(self):
        """
        Test that application fails to start without required environment variables

        This test should be run manually by:
        1. Unsetting CLERK_WEBHOOK_SECRET environment variable
        2. Attempting to start the backend
        3. Verifying it fails with clear error message

        Manual test command:
        $ unset CLERK_WEBHOOK_SECRET && python -m uvicorn app.main:app
        """
        # This is a documentation test - actual testing done manually
        # or via integration test framework
        pass


# Utility Functions for Test Setup
async def reset_user_usage(user_id: str, current_cost: float = 0.0):
    """
    Helper function to reset a user's usage for testing

    TODO: Implement this as a test-only endpoint or direct database access
    """
    pass


if __name__ == "__main__":
    print("Running security tests...")
    print("\nTo run tests:")
    print("  pytest tests/test_security.py -v")
    print("\nTo run specific test:")
    print("  pytest tests/test_security.py::TestUsageTracking::test_race_condition_prevention -v")
