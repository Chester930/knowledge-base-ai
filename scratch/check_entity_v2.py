import asyncio
import sys
import os

sys.path.append(os.getcwd())

from core.database import connect, disconnect, get_driver

async def main():
    await connect()
    driver = get_driver()
    query = "MATCH (e:Entity) WHERE e.name CONTAINS '八' RETURN e.name LIMIT 30"
    try:
        records, _, _ = await driver.execute_query(query, database_="kgai901d32e7")
        for r in records:
            name = r['e.name']
            hex_chars = [hex(ord(c)) for c in name]
            print(f"Name: {name} | Hex: {hex_chars}")
    except Exception as e:
        print(f"Error: {e}")
    await disconnect()

if __name__ == '__main__':
    asyncio.run(main())
