# Ingestion 執行結果

**執行時間**：2026-06-29 22:55:59 → 2026-06-29 23:04:35
**耗時**：8.6 分鐘
**退出碼**：0
**KG ID**：839aa61d-8d97-4e2a-8c74-10fa111c3f38

## 結果

| 項目 | 數值 |
|------|------|
| 成功 | 42 |
| 失敗 | 197 |
| Neo4j Document | 321 |
| Neo4j ConceptNode | 1846 |
| RAG 驗收 | ⚠️ 無召回（記憶未進 KG 或 index 未建） |
| RAG 來源 |  |

## 下一步

- [ ] 確認 Document 節點數正確（應為 39）
- [ ] 若 RAG 無召回，檢查 ConceptNode 向量索引
- [ ] 執行 `run_build_kg.py` 更新 SVO 圖譜

---
*Log 檔：.\ingest_run_20260629_225558.log*
