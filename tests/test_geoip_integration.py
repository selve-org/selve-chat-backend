"""
Test suite for GeoIP Service integration

Tests the Geoapify IP geolocation service functionality including:
- IP extraction from headers
- Geolocation API calls (mocked)
- Integration with Langfuse metadata
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.geoip_service import GeoIPService, GeoIPInfo


@pytest.fixture
def geoip_service():
    """Create a GeoIPService instance with mocked API key."""
    with patch.dict('os.environ', {
        'GEOAPIFY_API_KEY': 'test_api_key',
        'GEOAPIFY_ENABLED': 'true',
        'GEOAPIFY_TIMEOUT': '5'
    }):
        return GeoIPService()


@pytest.fixture
def sample_geoapify_response():
    """Sample response from Geoapify API."""
    return {
        "ip": "8.8.8.8",
        "country": {
            "iso_code": "US",
            "name": "United States"
        },
        "state": {
            "iso_code": "CA",
            "name": "California"
        },
        "city": {
            "name": "Mountain View"
        },
        "location": {
            "latitude": 37.386051,
            "longitude": -122.083855
        },
        "timezone": {
            "name": "America/Los_Angeles"
        },
        "organization": "Google LLC"
    }


class TestGeoIPInfo:
    """Test GeoIPInfo data class."""
    
    def test_geoip_info_parsing(self, sample_geoapify_response):
        """Test parsing of Geoapify response into GeoIPInfo."""
        info = GeoIPInfo(sample_geoapify_response)
        
        assert info.ip == "8.8.8.8"
        assert info.country == "US"
        assert info.country_name == "United States"
        assert info.state == "CA"
        assert info.state_name == "California"
        assert info.city == "Mountain View"
        assert info.latitude == 37.386051
        assert info.longitude == -122.083855
        assert info.timezone == "America/Los_Angeles"
        assert info.organization == "Google LLC"
    
    def test_geoip_info_to_dict(self, sample_geoapify_response):
        """Test conversion to Langfuse-safe string dict."""
        info = GeoIPInfo(sample_geoapify_response)
        result = info.to_dict()
        
        # All values should be strings
        assert all(isinstance(v, str) for v in result.values())
        
        # Check expected keys
        assert result["ip"] == "8.8.8.8"
        assert result["country"] == "United States"
        assert result["state"] == "California"
        assert result["city"] == "Mountain View"
        assert result["timezone"] == "America/Los_Angeles"
        assert result["location"] == "37.386051,-122.083855"
        assert result["org"] == "Google LLC"


class TestGeoIPService:
    """Test GeoIPService functionality."""
    
    def test_service_disabled_when_no_api_key(self):
        """Test service is disabled when API key is missing."""
        with patch.dict('os.environ', {}, clear=True):
            service = GeoIPService()
            assert service.enabled is False
    
    def test_service_enabled_with_api_key(self, geoip_service):
        """Test service is enabled when API key is present."""
        assert geoip_service.enabled is True
        assert geoip_service.api_key == 'test_api_key'
    
    @pytest.mark.asyncio
    async def test_geolocation_success(self, geoip_service, sample_geoapify_response):
        """Test successful geolocation lookup."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock the response
            mock_response = AsyncMock()
            mock_response.json.return_value = sample_geoapify_response
            mock_response.raise_for_status = MagicMock()
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            result = await geoip_service.get_geolocation("8.8.8.8")
            
            assert result.status == "success"
            assert result.data is not None
            assert result.data.ip == "8.8.8.8"
            assert result.data.city == "Mountain View"
    
    @pytest.mark.asyncio
    async def test_geolocation_disabled_service(self, geoip_service):
        """Test geolocation when service is disabled."""
        geoip_service.enabled = False
        
        result = await geoip_service.get_geolocation("8.8.8.8")
        
        assert result.status == "success"
        assert result.data.ip == "8.8.8.8"
    
    @pytest.mark.asyncio
    async def test_geolocation_timeout(self, geoip_service):
        """Test geolocation timeout handling."""
        import httpx
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timeout")
            )
            
            result = await geoip_service.get_geolocation("8.8.8.8")
            
            assert result.status == "timeout"
            assert "timeout" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_geolocation_api_error(self, geoip_service):
        """Test geolocation API error handling."""
        import httpx
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Forbidden", request=MagicMock(), response=mock_response
            )
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            result = await geoip_service.get_geolocation("8.8.8.8")
            
            assert result.status == "external_service_error"
            assert "403" in result.message
    
    def test_extract_client_ip_from_forwarded_for(self, geoip_service):
        """Test IP extraction from X-Forwarded-For header."""
        headers = {
            "x-forwarded-for": "203.0.113.1, 198.51.100.1, 192.0.2.1"
        }
        
        ip = geoip_service.extract_client_ip(headers)
        
        # Should extract the first IP (client's real IP)
        assert ip == "203.0.113.1"
    
    def test_extract_client_ip_from_cf_connecting(self, geoip_service):
        """Test IP extraction from CF-Connecting-IP header (Cloudflare)."""
        headers = {
            "cf-connecting-ip": "203.0.113.1"
        }
        
        ip = geoip_service.extract_client_ip(headers)
        
        assert ip == "203.0.113.1"
    
    def test_extract_client_ip_priority(self, geoip_service):
        """Test header priority: x-forwarded-for should come before x-real-ip."""
        headers = {
            "x-real-ip": "192.0.2.1",
            "x-forwarded-for": "203.0.113.1"
        }
        
        ip = geoip_service.extract_client_ip(headers)
        
        # x-forwarded-for has higher priority
        assert ip == "203.0.113.1"
    
    def test_extract_client_ip_no_headers(self, geoip_service):
        """Test IP extraction with no proxy headers."""
        headers = {}
        
        ip = geoip_service.extract_client_ip(headers)
        
        # Should return None (fallback to request.client.host)
        assert ip is None


@pytest.mark.skip(reason="Legacy test - needs update for AgenticChatService streaming API")
@pytest.mark.asyncio
async def test_geoip_integration_with_chat_service():
    """
    Integration test: verify GeoIP metadata flows into Langfuse traces.
    
    This test ensures that:
    1. GeoIP service is called when client_ip is provided
    2. Geolocation data is converted to string dict
    3. Metadata is passed to propagate_attributes for Langfuse
    
    NOTE: This test needs to be updated to work with AgenticChatService.chat_stream()
    which uses a streaming API instead of generate_response().
    """
    with patch.dict('os.environ', {
        'GEOAPIFY_API_KEY': 'test_key',
        'GEOAPIFY_ENABLED': 'true'
    }):
        from app.services.agentic_chat_service import AgenticChatService
        from app.services.geoip_service import GeoIPInfo
        
        service = AgenticChatService()
        
        # Mock the GeoIP service method
        mock_geo_result = MagicMock()
        mock_geo_result.status = "success"
        mock_geo_result.data = GeoIPInfo({
            "ip": "8.8.8.8",
            "country": {"name": "United States"},
            "state": {"name": "California"},
            "city": {"name": "Mountain View"},
            "location": {"latitude": 37.386051, "longitude": -122.083855},
            "timezone": {"name": "America/Los_Angeles"}
        })
        
        service.geoip_service.get_geolocation = AsyncMock(return_value=mock_geo_result)
        
        # Mock Langfuse to capture propagated metadata
        with patch('app.services.chat_service.propagate_attributes') as mock_propagate:
            with patch('app.services.chat_service.get_client'):
                with patch.object(service.llm_service, 'generate_response_async', 
                                  return_value={
                                      "content": "Test response",
                                      "usage": {"input_tokens": 10, "output_tokens": 20},
                                      "model": "test-model",
                                      "provider": "test",
                                      "cost": 0.001
                                  }):
                    with patch.object(service.context_service, 'build_context',
                                      return_value=MagicMock(
                                          context_info=None,
                                          system_content="test",
                                          sources_used=[]
                                      )):
                        with patch.object(service.context_service, 'build_messages',
                                          return_value=[{"role": "user", "content": "test"}]):
                            
                            # Call generate_response with client_ip
                            await service.generate_response(
                                message="Test message",
                                client_ip="8.8.8.8"
                            )
                            
                            # Verify propagate_attributes was called with geo metadata
                            assert mock_propagate.called
                            call_kwargs = mock_propagate.call_args[1]
                            
                            # Check that geo metadata is present
                            metadata = call_kwargs['metadata']
                            assert 'ip' in metadata
                            assert metadata['ip'] == "8.8.8.8"
                            assert metadata['country'] == "United States"
                            assert metadata['city'] == "Mountain View"
