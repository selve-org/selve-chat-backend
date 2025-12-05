#!/usr/bin/env python3
"""
Test chat response with ingested content
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.chat_service import ChatService


async def test_chat():
    chat = ChatService()

    print('\n' + '='*80)
    print('🤖 Testing Full RAG-Powered Chat Response')
    print('='*80)

    query = 'I scored high on ORIN. What does that mean for my work style?'

    print(f'\n👤 User: {query}')
    print('\n🤖 Assistant:')
    print('-'*80)

    result = await chat.generate_response(
        message=query,
        use_rag=True
    )

    # Word wrap the response
    response = result['response']
    words = response.split()
    line = ''
    for word in words:
        if len(line) + len(word) + 1 > 76:
            print(line)
            line = word
        else:
            line += (' ' + word) if line else word
    if line:
        print(line)

    print('-'*80)
    print(f'\n📊 Stats:')
    print(f'   Model: {result["model"]}')
    print(f'   Context Used: {result["context_used"]}')
    print(f'   Retrieved Chunks: {len(result["retrieved_chunks"])}')
    print(f'   Cost: ${result["cost"]:.6f}')

    if result['retrieved_chunks']:
        print(f'\n   Retrieved from:')
        for chunk in result['retrieved_chunks'][:2]:
            print(f'      • Score: {chunk["score"]:.4f} - {chunk["text"][:60]}...')

    print('\n' + '='*80)
    print('✅ Full pipeline test complete')
    print('='*80)


if __name__ == "__main__":
    asyncio.run(test_chat())
