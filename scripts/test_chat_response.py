#!/usr/bin/env python3
"""
Test chat response with ingested content
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.agentic_chat_service import AgenticChatService


async def test_chat():
    chat = AgenticChatService()

    print('\n' + '='*80)
    print('🤖 Testing Full RAG-Powered Chat Response (Streaming)')
    print('='*80)

    query = 'I scored high on ORIN. What does that mean for my work style?'

    print(f'\n👤 User: {query}')
    print('\n🤖 Assistant:')
    print('-'*80)

    # Consume the stream and collect the response
    full_response = ""
    async for event in chat.chat_stream(message=query, emit_status=False):
        if isinstance(event, str):
            full_response += event
            print(event, end='', flush=True)
        elif isinstance(event, dict) and 'chunk' in event:
            chunk = event['chunk']
            full_response += chunk
            print(chunk, end='', flush=True)

    print()
    print('-'*80)
    print(f'\n📊 Stats:')
    print(f'   Response Length: {len(full_response)} characters')

    print('\n' + '='*80)
    print('✅ Full pipeline test complete')
    print('='*80)


if __name__ == "__main__":
    asyncio.run(test_chat())
