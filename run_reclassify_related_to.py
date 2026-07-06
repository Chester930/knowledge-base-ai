"""
run_reclassify_related_to.py
針對所有 KG 中的 RELATED_TO 邊，用 LLM 重新分類為更精確的關係類型。
執行：docker exec kg-api python run_reclassify_related_to.py [--kg-id <uuid>] [--dry-run]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import re
import sys
from uuid import UUID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reclassify_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_BATCH_SIZE = 10  # 每批送 LLM 的三元組數

_VALID_REL_TYPES = {
    "IS_A", "PART_OF", "CONTAINS", "INSTANCE_OF",
    "CAUSES", "PREVENTS", "ENABLES", "IMPROVES", "INHIBITS",
    "USES", "REQUIRES", "PRODUCES", "IMPLEMENTS", "REPLACES", "EXTENDS",
    "CONTRASTS", "SIMILAR_TO", "OUTPERFORMS",
    "DEFINED_AS", "HAS_PROPERTY", "MEASURED_BY", "APPLIES_TO",
    "PRECEDES", "FOLLOWS", "CO_OCCURS",
    "INPUTS", "TRANSFORMS",
    "CREATED_BY", "SOLVES", "VIOLATES", "RELATED_TO",
}

_RECLASSIFY_PROMPT = """\
你是知識圖譜關係重分類器。以下三元組目前標記為 RELATED_TO（語意太模糊），
請依據主詞、動詞、受詞的語意，為每條選擇更精確的關係類型。

【可用類型（30種，不含 RELATED_TO）】
層級/組成: IS_A, PART_OF, CONTAINS, INSTANCE_OF
因果/效應: CAUSES, PREVENTS, ENABLES, IMPROVES, INHIBITS
功能/操作: USES, REQUIRES, PRODUCES, IMPLEMENTS, REPLACES, EXTENDS
比較:       CONTRASTS, SIMILAR_TO, OUTPERFORMS
描述/定義: DEFINED_AS, HAS_PROPERTY, MEASURED_BY, APPLIES_TO
時序:       PRECEDES, FOLLOWS, CO_OCCURS
資料流:     INPUTS, TRANSFORMS
歸屬/解決: CREATED_BY, SOLVES
規範/合規: VIOLATES（違反、觸犯、不符合規定 —— 優先於 PREVENTS/INHIBITS/IMPROVES 使用）

【規則】
- 若有更精確類型，務必選用
- 若真的無法確定比 RELATED_TO 更好的分類，回傳 RELATED_TO 維持原樣
- 只能從上方 31 種選一，不可自造

【三元組列表】
{triples_text}

請回傳純 JSON（不加說明）：
{{"results": [{{"id": 1, "new_type": "IS_A"}}, ...]}}
"""


async def _get_related_to_edges(driver, db_name: str, kg_name: str) -> list[dict]:
    """查詢指定 KG 所有 RELATED_TO 邊。"""
    db_kw = {"database_": db_name} if db_name else {}
    result = await driver.execute_query(
        """
        MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
        RETURN s.name AS subject, s.type AS s_type,
               r.verb AS verb,
               o.name AS object, o.type AS o_type,
               r.source_doc_id AS doc_id,
               r.confidence AS confidence,
               r.created_at AS created_at
        ORDER BY r.confidence DESC
        """,
        **db_kw,
    )
    rows = []
    for rec in result.records:
        rows.append({
            "subject":    rec["subject"],
            "s_type":     rec["s_type"] or "概念",
            "verb":       rec["verb"] or "相關",
            "object":     rec["object"],
            "o_type":     rec["o_type"] or "概念",
            "doc_id":     rec["doc_id"] or "",
            "confidence": rec["confidence"] or 1,
            "created_at": rec["created_at"],
        })
    logger.info(f"[{kg_name}] 找到 {len(rows)} 條 RELATED_TO 邊")
    return rows


def _build_reclassify_prompt(batch: list[dict]) -> str:
    lines = []
    for i, row in enumerate(batch, 1):
        lines.append(
            f'{i}. 主詞={row["subject"]}({row["s_type"]}) '
            f'動詞={row["verb"]} '
            f'受詞={row["object"]}({row["o_type"]})'
        )
    return _RECLASSIFY_PROMPT.format(triples_text="\n".join(lines))


async def _reclassify_batch(llm, batch: list[dict]) -> list[dict]:
    """對一批三元組呼叫 LLM，回傳 [{id, new_type}, ...] 已驗證的清單。"""
    prompt = _build_reclassify_prompt(batch)
    try:
        raw = await llm.generate_json(prompt)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("找不到 JSON 物件")
        data = json.loads(match.group())
        results = data.get("results", [])
        out = []
        for item in results:
            idx = int(item.get("id", 0)) - 1
            new_type = str(item.get("new_type", "RELATED_TO")).strip().upper()
            if new_type not in _VALID_REL_TYPES:
                new_type = "RELATED_TO"
            if 0 <= idx < len(batch):
                out.append({"row": batch[idx], "new_type": new_type})
        return out
    except Exception as e:
        logger.warning(f"批次重分類失敗，全部維持 RELATED_TO：{e}")
        return [{"row": r, "new_type": "RELATED_TO"} for r in batch]


async def _apply_reclassification(
    driver, db_name: str, row: dict, new_type: str, dry_run: bool
) -> bool:
    """刪除 RELATED_TO 邊，建立新類型邊。返回是否實際執行。"""
    if new_type == "RELATED_TO":
        return False  # 不需要變更

    db_kw = {"database_": db_name} if db_name else {}

    if dry_run:
        logger.info(
            f"[DRY-RUN] {row['subject']} -[RELATED_TO|{row['verb']}]→ {row['object']}"
            f"  →  {new_type}"
        )
        return True

    try:
        cypher = f"""
        MATCH (s:Entity {{name: $subject}})-[r:RELATED_TO]->(o:Entity {{name: $object}})
        WHERE ($doc_id = '' OR r.source_doc_id = $doc_id)
        WITH s, r, o,
             coalesce(r.verb, $verb)        AS verb,
             coalesce(r.confidence, 1)      AS conf,
             coalesce(r.created_at, datetime()) AS cat,
             r.source_doc_id               AS doc_id
        DELETE r
        MERGE (s)-[newR:{new_type} {{source_doc_id: doc_id}}]->(o)
        ON CREATE SET newR.verb = verb, newR.confidence = conf,
                      newR.created_at = cat, newR.reclassified = true
        ON MATCH  SET newR.confidence = newR.confidence + conf,
                      newR.reclassified = true, newR.updated_at = datetime()
        RETURN count(newR) AS changed
        """
        result = await driver.execute_query(
            cypher,
            subject=row["subject"],
            object=row["object"],
            doc_id=row["doc_id"],
            verb=row["verb"],
            **db_kw,
        )
        changed = result.records[0]["changed"] if result.records else 0
        if changed:
            logger.info(
                f"✅ {row['subject']} -[{new_type}|{row['verb']}]→ {row['object']}"
            )
        return bool(changed)
    except Exception as e:
        logger.warning(
            f"❌ 更新失敗 [{row['subject']}|{row['object']}]: {e}"
        )
        return False


async def reclassify_kg(
    driver, llm, kg_name: str, db_name: str, dry_run: bool
) -> dict:
    edges = await _get_related_to_edges(driver, db_name, kg_name)
    if not edges:
        return {"kg": kg_name, "total": 0, "reclassified": 0, "kept": 0}

    total = len(edges)
    reclassified = 0
    kept = 0

    batches = [edges[i:i + _BATCH_SIZE] for i in range(0, len(edges), _BATCH_SIZE)]
    logger.info(f"[{kg_name}] 分成 {len(batches)} 批，每批 {_BATCH_SIZE} 條")

    for batch_idx, batch in enumerate(batches, 1):
        logger.info(f"[{kg_name}] 批次 {batch_idx}/{len(batches)}…")
        results = await _reclassify_batch(llm, batch)
        for item in results:
            changed = await _apply_reclassification(
                driver, db_name, item["row"], item["new_type"], dry_run
            )
            if changed:
                reclassified += 1
            else:
                kept += 1

    logger.info(
        f"[{kg_name}] 完成：{reclassified}/{total} 重分類，{kept} 維持 RELATED_TO"
    )
    return {"kg": kg_name, "total": total, "reclassified": reclassified, "kept": kept}


async def main(target_kg_id: str | None, dry_run: bool):
    from core.database import connect, disconnect, get_driver
    from core.providers.factory import init_providers, get_llm_provider
    from services.svo_service import create_entity_index

    init_providers()
    await connect()
    await create_entity_index()
    driver = get_driver()
    llm = get_llm_provider()

    # 取得所有 KG（或指定 KG）
    if target_kg_id:
        result = await driver.execute_query(
            "MATCH (kg:KnowledgeGraph {id: $id}) RETURN kg.name AS name, kg.db_name AS db",
            id=target_kg_id,
        )
    else:
        result = await driver.execute_query(
            "MATCH (kg:KnowledgeGraph) RETURN kg.name AS name, kg.db_name AS db ORDER BY kg.name"
        )

    kgs = [(r["name"], r["db"] or "") for r in result.records]
    if not kgs:
        logger.error("找不到 KG，請確認資料庫連線")
        await disconnect()
        return

    logger.info(f"共 {len(kgs)} 個 KG：{[k for k, _ in kgs]}")
    if dry_run:
        logger.info("★ DRY-RUN 模式：只顯示，不實際修改 Neo4j")

    summary = []
    for kg_name, db_name in kgs:
        stat = await reclassify_kg(driver, llm, kg_name, db_name, dry_run)
        summary.append(stat)

    logger.info("\n=== 重分類彙總 ===")
    total_r = total_k = 0
    for s in summary:
        logger.info(
            f"  {s['kg']}: 共 {s['total']} 條，重分類 {s['reclassified']} 條，"
            f"維持 {s['kept']} 條"
        )
        total_r += s["reclassified"]
        total_k += s["kept"]
    logger.info(f"  合計重分類：{total_r}，維持 RELATED_TO：{total_k}")

    await disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RELATED_TO 邊重分類")
    parser.add_argument("--kg-id", default=None, help="指定 KG UUID（不填 = 全部）")
    parser.add_argument("--dry-run", action="store_true", help="只顯示，不修改 Neo4j")
    args = parser.parse_args()

    asyncio.run(main(args.kg_id, args.dry_run))
