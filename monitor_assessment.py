#!/usr/bin/env python3
"""
Real-time monitoring of user's assessment status.
Checks database every 2 seconds to see when assessment completes.
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import db


async def monitor_assessment(clerk_user_id: str):
    """Monitor user's assessment status in real-time."""
    await db.connect()
    
    print(f"\n{'='*80}")
    print(f"🔍 MONITORING USER: {clerk_user_id}")
    print(f"{'='*80}\n")
    
    last_state = None
    check_count = 0
    
    try:
        while True:
            check_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Check for assessment sessions (in progress)
            sessions = await db.assessmentsession.find_many(
                where={"clerkUserId": clerk_user_id},
                order={"createdAt": "desc"},
                take=3
            )
            
            # Check for assessment results (completed)
            results = await db.assessmentresult.find_many(
                where={"clerkUserId": clerk_user_id},
                order={"createdAt": "desc"},
                take=3
            )
            
            current_state = {
                "sessions_count": len(sessions),
                "results_count": len(results),
                "latest_session": sessions[0].id if sessions else None,
                "latest_result": results[0].id if results else None,
                "result_is_current": results[0].isCurrent if results else None,
            }
            
            # Only print when state changes
            if current_state != last_state:
                print(f"\n[{timestamp}] CHECK #{check_count} - STATE CHANGE DETECTED!")
                print(f"{'─'*80}")
                
                if sessions:
                    latest_session = sessions[0]
                    print(f"📝 LATEST SESSION:")
                    print(f"   ID: {latest_session.id}")
                    print(f"   isCurrent: {latest_session.isCurrent}")
                    print(f"   Created: {latest_session.createdAt}")
                    print(f"   Completed: {latest_session.completedAt}")
                    if latest_session.completedAt:
                        print(f"   ✅ SESSION IS COMPLETED!")
                else:
                    print(f"📝 No assessment sessions found")
                
                if results:
                    latest_result = results[0]
                    print(f"\n✨ LATEST RESULT:")
                    print(f"   ID: {latest_result.id}")
                    print(f"   Session ID: {latest_result.sessionId}")
                    print(f"   isCurrent: {latest_result.isCurrent}")
                    print(f"   Archetype: {latest_result.archetype}")
                    print(f"   Profile: {latest_result.profilePattern}")
                    print(f"   Created: {latest_result.createdAt}")
                    print(f"   🎉 ASSESSMENT COMPLETED! Chatbot should now see it!")
                else:
                    print(f"\n✨ No assessment results found yet")
                
                if results and results[0].isCurrent:
                    print(f"\n{'='*80}")
                    print(f"✅ CURRENT ASSESSMENT DETECTED!")
                    print(f"   User can now query their scores in the chat")
                    print(f"{'='*80}\n")
                    break
                
                last_state = current_state
            else:
                print(f"[{timestamp}] CHECK #{check_count} - No changes...", end='\r')
            
            await asyncio.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n\n⏸️  Monitoring stopped by user")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    clerk_user_id = "user_37i1kCPu5ZPiiVisskWBrzwL809"  # gihoh24354@24faw.com
    
    if len(sys.argv) > 1:
        clerk_user_id = sys.argv[1]
    
    print(f"\n🚀 Starting real-time assessment monitor...")
    print(f"   Press Ctrl+C to stop\n")
    
    asyncio.run(monitor_assessment(clerk_user_id))
