import asyncio
import sys
import os

# 把工作區加入 path
sys.path.append(os.getcwd())

from core.database import connect, disconnect, get_driver

async def main():
    await connect()
    driver = get_driver()
    query = "MATCH (e:Entity) WHERE e.name CONTAINS '八' RETURN e.name, e.kg_id LIMIT 30"
    records, _, _ = await driver.execute_query(query)
    for r in records:
        name = r['e.name']
        hex_chars = [hex(ord(c)) for c in name]
        print(f"Name: {name} | Hex: {hex_chars} | KG: {r['e.kg_id']}")
    await disconnect()

if __name__ == '__main__':
    asyncio.run(main())
