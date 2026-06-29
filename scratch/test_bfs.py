import asyncio
import sys
import os

sys.path.append(os.getcwd())

from core.database import connect, disconnect
from services.svo_service import query_svo_facts
from uuid import UUID

async def main():
    await connect()
    kg_id = UUID("3de0f63b-b7b3-46ed-8c52-603766752fd0")
    terms = ["遊戲化", "八角框架"]
    try:
        facts, docs, cids = await query_svo_facts(kg_id, terms)
        print(f"Facts count: {len(facts)}")
        print(f"Docs count: {len(docs)}")
        print(f"Cids count: {len(cids)}")
        print(f"Facts: {facts[:10]}")
    except Exception as e:
        print(f"Error: {e}")
    await disconnect()

if __name__ == '__main__':
    asyncio.run(main())
