#!/usr/bin/env python3
"""
Production Health Monitor - Runs comprehensive checks on deployed services.

Usage:
    python scripts/health_monitor.py
    python scripts/health_monitor.py --check qdrant
    python scripts/health_monitor.py --check all --alert
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

try:
    import httpx
    from qdrant_client import QdrantClient
    from openai import OpenAI
except ImportError:
    print("❌ Missing dependencies. Install: pip install httpx qdrant-client openai")
    sys.exit(1)


@dataclass
class HealthStatus:
    """Health check result."""
    service: str
    healthy: bool
    message: str
    details: Dict = None
    
    def __str__(self):
        icon = "✅" if self.healthy else "❌"
        return f"{icon} {self.service}: {self.message}"


class HealthMonitor:
    """Comprehensive health monitoring for SELVE services."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[HealthStatus] = []
    
    async def check_backend(self, url: str = "https://api.selve.me") -> HealthStatus:
        """Check main backend health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/health")
                data = response.json()
                
                if data.get("status") == "healthy":
                    db_status = data.get("database", "unknown")
                    return HealthStatus(
                        service="selve-backend",
                        healthy=True,
                        message=f"Healthy (DB: {db_status})",
                        details=data
                    )
                else:
                    return HealthStatus(
                        service="selve-backend",
                        healthy=False,
                        message=f"Degraded: {data}",
                        details=data
                    )
        except Exception as e:
            return HealthStatus(
                service="selve-backend",
                healthy=False,
                message=f"Connection failed: {e}"
            )
    
    async def check_chat_backend(self, url: str = "https://api-chat.selve.me") -> HealthStatus:
        """Check chat backend health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/api/health")
                data = response.json()
                
                if data.get("status") == "healthy":
                    qdrant = data.get("qdrant_connected", False)
                    points = data.get("collection_points", 0)
                    
                    if not qdrant:
                        return HealthStatus(
                            service="selve-chat-backend",
                            healthy=False,
                            message="Healthy but Qdrant disconnected!",
                            details=data
                        )
                    
                    return HealthStatus(
                        service="selve-chat-backend",
                        healthy=True,
                        message=f"Healthy (Qdrant: {points} points)",
                        details=data
                    )
                else:
                    return HealthStatus(
                        service="selve-chat-backend",
                        healthy=False,
                        message=f"Degraded: {data.get('status')}",
                        details=data
                    )
        except Exception as e:
            return HealthStatus(
                service="selve-chat-backend",
                healthy=False,
                message=f"Connection failed: {e}"
            )
    
    def check_qdrant(self, host: str = "localhost", port: int = 6333) -> HealthStatus:
        """Check Qdrant vector database."""
        try:
            client = QdrantClient(host=host, port=port, timeout=5.0)
            collections = client.get_collections()
            
            collection_info = {}
            for col in collections.collections:
                info = client.get_collection(col.name)
                collection_info[col.name] = info.points_count
            
            total_points = sum(collection_info.values())
            
            return HealthStatus(
                service="qdrant",
                healthy=True,
                message=f"{len(collections.collections)} collections, {total_points} total points",
                details=collection_info
            )
        except Exception as e:
            return HealthStatus(
                service="qdrant",
                healthy=False,
                message=f"Connection failed: {e}"
            )
    
    def check_openai(self, api_key: Optional[str] = None) -> HealthStatus:
        """Check OpenAI API connectivity."""
        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return HealthStatus(
                service="openai",
                healthy=False,
                message="API key not set"
            )
        
        try:
            client = OpenAI(api_key=api_key, timeout=10.0)
            models = client.models.list()
            
            return HealthStatus(
                service="openai",
                healthy=True,
                message=f"Connected ({len(list(models))} models available)"
            )
        except Exception as e:
            return HealthStatus(
                service="openai",
                healthy=False,
                message=f"API call failed: {e}"
            )
    
    async def check_all(self) -> List[HealthStatus]:
        """Run all health checks."""
        print("🔍 Running comprehensive health checks...\n")
        
        # Production services
        self.results.append(await self.check_backend())
        self.results.append(await self.check_chat_backend())
        
        # Infrastructure (if accessible)
        try:
            self.results.append(self.check_qdrant())
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Skipping Qdrant check (not accessible): {e}")
        
        # External APIs
        self.results.append(self.check_openai())
        
        return self.results
    
    def print_summary(self):
        """Print health check summary."""
        print("\n" + "="*60)
        print("HEALTH CHECK SUMMARY")
        print("="*60 + "\n")
        
        for result in self.results:
            print(result)
            
            if self.verbose and result.details:
                print(f"   Details: {json.dumps(result.details, indent=2)}")
        
        print("\n" + "="*60)
        
        healthy_count = sum(1 for r in self.results if r.healthy)
        total_count = len(self.results)
        
        print(f"Status: {healthy_count}/{total_count} services healthy")
        
        if healthy_count < total_count:
            print("❌ Some services are unhealthy!")
            return 1
        else:
            print("✅ All services healthy!")
            return 0


async def main():
    parser = argparse.ArgumentParser(description="SELVE Production Health Monitor")
    parser.add_argument("--check", choices=["all", "backend", "chat", "qdrant", "openai"],
                       default="all", help="Which service to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--alert", action="store_true", help="Exit with error on failures")
    
    args = parser.parse_args()
    
    monitor = HealthMonitor(verbose=args.verbose)
    
    if args.check == "all":
        await monitor.check_all()
    elif args.check == "backend":
        monitor.results.append(await monitor.check_backend())
    elif args.check == "chat":
        monitor.results.append(await monitor.check_chat_backend())
    elif args.check == "qdrant":
        monitor.results.append(monitor.check_qdrant())
    elif args.check == "openai":
        monitor.results.append(monitor.check_openai())
    
    exit_code = monitor.print_summary()
    
    if args.alert and exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
