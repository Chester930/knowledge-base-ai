import asyncio
from core.database import connect, disconnect, get_driver

async def main():
    await connect()
    driver = get_driver()
    
    print("--- 查詢所有 Document 節點屬性 ---")
    res = await driver.execute_query("MATCH (d:Document) RETURN d, id(d) AS element_id")
    for r in res.records:
        node = r["d"]
        print(f"Node ID: {r['element_id']}")
        print(f"  Properties: {dict(node)}")
        
    print("\n--- 查詢資料庫中的 Entity 數量與標籤 ---")
    res_labels = await driver.execute_query("CALL db.labels()")
    labels = [rec["label"] for rec in res_labels.records]
    print(f"Labels: {labels}")

    await disconnect()

if __name__ == "__main__":
    asyncio.run(main())
