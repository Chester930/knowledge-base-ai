import asyncio
from core.database import connect, disconnect, get_driver

async def main():
    await connect()
    driver = get_driver()
    
    # 1. 查詢所有 KG
    print("--- 1. 所有 KG 列表 ---")
    res = await driver.execute_query("MATCH (g:KGInfo) RETURN g.id AS id, g.name AS name, g.is_public AS is_public")
    for r in res.records:
        print(f"KG ID: {r['id']}, Name: {r['name']}, Public: {r['is_public']}")
        
    # 2. 查詢檔名包含 "龍蝦" 的 Document
    print("\n--- 2. 搜尋包含 '龍蝦' 的 Document 節點 ---")
    res_doc = await driver.execute_query(
        "MATCH (d:Document) WHERE toLower(d.title) CONTAINS '龍蝦' OR toLower(d.name) CONTAINS '龍蝦' RETURN d.name AS name, d.kg_id AS kg_id"
    )
    if not res_doc.records:
        print("未找到任何檔名含有 '龍蝦' 的 Document")
    for r in res_doc.records:
        print(f"Document: {r['name']}, Belonging to KG: {r['kg_id']}")

    # 3. 查詢名稱包含 "龍蝦" 的 Entity
    print("\n--- 3. 搜尋包含 '龍蝦' 的 Entity 節點 ---")
    res_ent = await driver.execute_query(
        "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS '龍蝦' RETURN e.name AS name, e.type AS type, e.kg_id AS kg_id LIMIT 10"
    )
    if not res_ent.records:
        print("未找到任何名稱含有 '龍蝦' 的 Entity")
    for r in res_ent.records:
        print(f"Entity: {r['name']}, Type: {r['type']}, KG: {r['kg_id']}")

    await disconnect()

if __name__ == "__main__":
    asyncio.run(main())
