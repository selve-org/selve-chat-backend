"""
GeoIP Service for IP-based geolocation via Geoapify API.

Provides user location data (IP, city, state, country) for enriching
observability traces and user context.
"""

import os
import logging
from typing import Optional, Dict, Any
import httpx

from .base import BaseService, Config, Result, ExternalServiceError, Validator

logger = logging.getLogger(__name__)


class GeoIPInfo:
    """Data class for parsed geolocation information."""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize from Geoapify API response.
        
        Args:
            data: Response dict from Geoapify IP Info API
        """
        self.raw = data
        self.ip = data.get("ip", "unknown")
        self.country = data.get("country", {}).get("iso_code", "")
        self.country_name = data.get("country", {}).get("name", "")
        self.state = data.get("state", {}).get("iso_code", "")
        self.state_name = data.get("state", {}).get("name", "")
        self.city = data.get("city", {}).get("name", "")
        self.latitude = data.get("location", {}).get("latitude")
        self.longitude = data.get("location", {}).get("longitude")
        self.timezone = data.get("timezone", {}).get("name", "")
        self.organization = data.get("organization", "")
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert to string-keyed dict safe for Langfuse metadata.
        
        Returns:
            Dict with string values suitable for tracing
        """
        result = {
            "ip": str(self.ip),
            "country": str(self.country_name),
            "state": str(self.state_name),
            "city": str(self.city),
            "timezone": str(self.timezone),
        }
        
        # Include coordinates as strings if available
        if self.latitude is not None and self.longitude is not None:
            result["location"] = f"{self.latitude},{self.longitude}"
        
        if self.organization:
            result["org"] = str(self.organization)
        
        return result


class GeoIPService(BaseService):
    """
    Service for looking up user geolocation by IP address via Geoapify.
    
    Uses the Geoapify IP Geolocation API (https://api.geoapify.com/v1/ipinfo)
    to enrich user sessions with geographic context.
    
    Environment variables:
    - GEOAPIFY_API_KEY: API key for Geoapify (required)
    - GEOAPIFY_ENABLED: Enable/disable service (default: true)
    - GEOAPIFY_TIMEOUT: Request timeout in seconds (default: 5)
    """
    
    def __init__(self):
        super().__init__()
        
        self.api_key = os.getenv("GEOAPIFY_API_KEY", "")
        self.enabled = os.getenv("GEOAPIFY_ENABLED", "true").lower() == "true"
        self.timeout = int(os.getenv("GEOAPIFY_TIMEOUT", "5"))
        self.base_url = "https://api.geoapify.com/v1/ipinfo"
        
        if not self.api_key:
            logger.warning("GEOAPIFY_API_KEY not configured, GeoIP service disabled")
            self.enabled = False
        elif self.enabled:
            logger.info("GeoIP service initialized with Geoapify API")
    
    async def get_geolocation(
        self,
        ip_address: Optional[str] = None
    ) -> Result[GeoIPInfo]:
        """
        Look up geolocation for an IP address.
        
        If ip_address is None, Geoapify will detect the requester's IP.
        
        Args:
            ip_address: IPv4 or IPv6 address (optional; auto-detect if None)
            
        Returns:
            Result[GeoIPInfo] with location data or error
        """
        if not self.enabled:
            return Result.success(
                GeoIPInfo({"ip": ip_address or "unknown"}),
                "GeoIP service disabled"
            )
        
        # Validate IP if provided
        if ip_address:
            validator = Validator()
            try:
                validator.validate_string(ip_address, "ip_address", max_length=45)
            except Exception as e:
                return Result.validation_error(f"Invalid IP: {str(e)}")
        
        try:
            params = {"apiKey": self.api_key}
            if ip_address:
                params["ip"] = ip_address
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                geo_info = GeoIPInfo(data)
                
                logger.debug(
                    f"GeoIP lookup: {geo_info.ip} → "
                    f"{geo_info.city}, {geo_info.state_name}, {geo_info.country_name}"
                )
                
                return Result.success(geo_info)
        
        except httpx.HTTPStatusError as e:
            error_msg = f"Geoapify API error: {e.status_code}"
            logger.error(error_msg, exc_info=True)
            return Result.external_service_error(error_msg)
        
        except httpx.TimeoutException:
            error_msg = f"Geoapify request timeout (>{self.timeout}s)"
            logger.error(error_msg, exc_info=True)
            return Result.timeout(error_msg)
        
        except Exception as e:
            error_msg = f"GeoIP lookup failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.error(error_msg)
    
    def extract_client_ip(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Extract client IP from request headers.
        
        Checks for common proxy headers first (X-Forwarded-For, CF-Connecting-IP),
        falls back to direct connection IP if available.
        
        Args:
            headers: HTTP request headers dict
            
        Returns:
            IP address string or None
        """
        # Common proxy headers (order matters - most specific first)
        proxy_headers = [
            "x-forwarded-for",
            "cf-connecting-ip",  # Cloudflare
            "x-real-ip",
            "x-client-ip",
        ]
        
        for header in proxy_headers:
            if header in headers:
                # X-Forwarded-For can contain multiple IPs; use the first (client's real IP)
                ips = headers[header].split(",")
                if ips:
                    return ips[0].strip()
        
        # Fallback: try to get from request.client
        return None
