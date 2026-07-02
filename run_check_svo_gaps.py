"""
掃描所有 KG，找出 SVO 抽取不完整的文件與段落（chunk）。

判斷邏輯：
  1. Document.svo_processed_at IS NULL → 此文件尚未 100% 完成 SVO 抽取
  2. 對照本地進度檔 chunk_store/{kg_id}/{doc_id}/svo_progress.json，
     比對句子切分後的總 chunk 數，找出哪些 chunk idx 尚未標記為 processed=true
  3. 若進度檔中該 chunk 有記錄（processed=false + error），會顯示失敗原因
     （此欄位為修復後新增，舊資料可能沒有記錄，僅顯示「未知（無記錄）」）

用法：
  python run_check_svo_gaps.py                # 檢查所有 KG
  python run_check_svo_gaps.py --kg <kg_id>   # 只檢查指定 KG
"""
import argparse
import asyncio
import json
from uuid import UUID


async def main(target_kg_id: str | None):
    from core.database import connect, disconnect, get_driver
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    from repositories.document_repo import DocumentRepository
    from services.chunk_store import get_chunk_store, sentence_chunk

    await connect()
    kg_repo = KnowledgeGraphRepository(get_driver())
    doc_repo = DocumentRepository(get_driver())
    chunk_store = get_chunk_store()

    if target_kg_id:
        kg = await kg_repo.get_by_id(UUID(target_kg_id))
        kgs = [kg] if kg else []
    else:
        kgs = await kg_repo.list_all(include_private=True)

    print(f"共 {len(kgs)} 個 KG 待檢查\n")

    grand_total_incomplete_docs = 0
    grand_total_missing_chunks = 0

    for kg in kgs:
        raw = await kg_repo.get_documents(kg.id)
        all_doc_ids = [r["id"] for r in raw]
        if not all_doc_ids:
            continue

        result = await get_driver().execute_query(
            """
            UNWIND $ids AS doc_id
            MATCH (d:Document {id: doc_id})
            WHERE d.svo_processed_at IS NULL
            RETURN d.id AS id, d.title AS title
            """,
            ids=all_doc_ids,
        )
        incomplete = result.records
        if not incomplete:
            continue

        print(f"=== KG：{kg.name}（{kg.id}）\u2500 {len(incomplete)} 份文件未完成 ===")
        grand_total_incomplete_docs += len(incomplete)

        for rec in incomplete:
            doc_id = rec["id"]
            title = rec["title"]
            doc = await doc_repo.get_by_id(UUID(doc_id))
            if doc is None:
                print(f"  \u26a0\ufe0f  {title}（{doc_id}）\u2014 文件已不存在，略過")
                continue

            sent_chunks = sentence_chunk(doc_id, doc.content or "")
            total_chunks = len(sent_chunks)

            progress_file = chunk_store._base / str(kg.id) / doc_id / "svo_progress.json"
            progress_data = {}
            if progress_file.exists():
                try:
                    progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"  \u26a0\ufe0f  進度檔讀取失敗（{progress_file}）：{e}")

            done_count = sum(1 for v in progress_data.values() if v.get("processed"))
            missing = []
            for sc in sent_chunks:
                st = progress_data.get(str(sc.idx))
                if not st or not st.get("processed"):
                    reason = st.get("error") if st else None
                    missing.append((sc.idx, reason))

            grand_total_missing_chunks += len(missing)
            print(f"  \U0001F4C4 {title}（{doc_id}）：{done_count}/{total_chunks} 已完成，{len(missing)} 個待補")
            for idx, reason in missing[:20]:
                reason_str = reason if reason else "未知（無記錄，可能是修復前的舊失敗）"
                print(f"      - chunk {idx}: {reason_str}")
            if len(missing) > 20:
                print(f"      ...（其餘 {len(missing) - 20} 個省略）")
        print()

    print("=" * 50)
    print(f"總計：{grand_total_incomplete_docs} 份文件不完整，{grand_total_missing_chunks} 個 chunk 待補跑")
    print("修復方式：直接執行 `python run_build_kg.py`（不加 --force）即可增量補跑，")
    print("已完成的 chunk 會被跳過，只重新抽取上面列出的段落。")

    await disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="檢查 SVO 抽取缺口")
    parser.add_argument("--kg", type=str, default=None, help="只檢查指定 kg_id")
    args = parser.parse_args()
    asyncio.run(main(args.kg))
