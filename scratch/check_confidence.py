import asyncio
import sys
import os

sys.path.append(os.getcwd())

from core.database import connect, disconnect, get_driver

async def main():
    await connect()
    driver = get_driver()
    query = "MATCH ()-[r]->() RETURN r.confidence LIMIT 30"
    try:
        records, _, _ = await driver.execute_query(query, database_="kgai901d32e7")
        confidences = [r['r.confidence'] for r in records]
        print(f"Confidences: {confidences}")
    except Exception as e:
        print(f"Error: {e}")
    await disconnect()

if __name__ == '__main__':
    asyncio.run(main())
