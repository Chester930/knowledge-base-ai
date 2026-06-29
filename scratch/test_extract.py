import asyncio
import sys
import os

sys.path.append(os.getcwd())

from core.database import connect, disconnect
from core.providers.factory import init_providers
from services.concept_engine import build_query_concepts

async def main():
    await connect()
    init_providers()
    # 從參數取得問題，若無則預設
    query_text = sys.argv[1] if len(sys.argv) > 1 else "遊戲化 八角框架是什麼"
    try:
        concepts = await build_query_concepts(query_text)
        print(f"Query: {query_text}")
        for c in concepts:
            print(f"Concept: {c['name']}")
    except Exception as e:
        print(f"Error: {e}")
    await disconnect()

if __name__ == '__main__':
    asyncio.run(main())
