from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator
from uuid import UUID, uuid4

from core.database import get_driver
from core.providers.factory import get_llm_provider
from models.knowledge_graph import SVOTriple
from repositories.document_repo import DocumentRepository
from repositories.knowledge_graph_repo import KnowledgeGraphRepository

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1000   # 每段文字上限（字元）
_CHUNK_OVERLAP = 100 # 段落重疊（避免三元組跨段斷裂）


# ── 進度事件（供 SSE 串流使用）────────────────────────────────────────────────

@dataclass
class BuildProgress:
    event: str          # chunk_start | chunk_done | merge_done | error | done
    chunk_idx: int = 0
    total_chunks: int = 0
    triples_extracted: int = 0
    triples_merged: int = 0
    message: str = ""


# ── 公開 API ──────────────────────────────────────────────────────────────────

async def _get_kg_db(kg_id: UUID) -> str:
    """取得 KG 的專用資料庫名稱（空白 = 主資料庫）。"""
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    return await KnowledgeGraphRepository(get_driver()).get_db_name(kg_id)


async def build_graph_for_kg(
    kg_id: UUID,
    doc_ids: list[UUID] | None = None,
    force_rebuild: bool = False,
) -> AsyncIterator[BuildProgress]:
    """
    對 KG 下的所有（或指定）文件逐一執行 SVO 提取，直接 MERGE 進 Neo4j。
    以 AsyncIterator[BuildProgress] 回報進度，供 SSE 端點消費。
    """
    kg_repo = KnowledgeGraphRepository(get_driver())
    doc_repo = DocumentRepository(get_driver())

    kg = await kg_repo.get_by_id(kg_id)
    if kg is None:
        yield BuildProgress(event="error", message=f"KG 不存在：{kg_id}")
        return

    db_name = kg.db_name  # "" = 使用主資料庫 + kg_id 隔離

    # 決定要處理的文件清單
    if doc_ids:
        docs = [d for d in [await doc_repo.get_by_id(did) for did in doc_ids] if d]
    else:
        raw = await kg_repo.get_documents(kg_id)
        docs = [await doc_repo.get_by_id(UUID(r["id"])) for r in raw]
        docs = [d for d in docs if d]

    if not docs:
        yield BuildProgress(event="done", message="此 KG 下沒有文件")
        return

    if force_rebuild:
        await _clear_kg_entities(kg_id, db_name)

    total_merged = 0
    for doc in docs:
        text = doc.content or ""
        chunks = _chunk_text(text)
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            yield BuildProgress(
                event="chunk_start",
                chunk_idx=i, total_chunks=total_chunks,
                message=f"[{doc.title}] 提取第 {i}/{total_chunks} 段…",
            )
            try:
                triples = await extract_svo_from_text(chunk)
            except Exception as e:
                logger.warning(f"SVO 提取失敗 [{doc.title} chunk {i}]: {e}")
                yield BuildProgress(event="error", chunk_idx=i, message=str(e))
                continue

            merged = await merge_triples_to_neo4j(triples, kg_id, doc.id, db_name)
            total_merged += merged
            yield BuildProgress(
                event="chunk_done",
                chunk_idx=i, total_chunks=total_chunks,
                triples_extracted=len(triples), triples_merged=merged,
                message=f"[{doc.title}] 第 {i} 段完成：{len(triples)} 組三元組",
            )

    await kg_repo.refresh_counts(kg_id)
    yield BuildProgress(
        event="done",
        triples_merged=total_merged,
        message=f"圖譜建立完成，共合併 {total_merged} 組三元組",
    )


async def extract_svo_from_text(text: str) -> list[SVOTriple]:
    """呼叫 LLM 從單段文字提取本體論知識三元組（6欄格式）。"""
    if not text.strip():
        return []

    prompt = (
        "請從以下文字中提取知識關係，以六欄格式輸出，每行一組：\n"
        "主詞|主詞類型|關係類別|動詞|受詞|受詞類型\n\n"
        "【實體類型】選最接近：概念、算法、技術、方法、工具、框架、模型、系統、人物、組織、資料集、指標、其他\n\n"
        "【關係類別】必須從以下 8 種選一，不可自造：\n"
        "  IS_A       → 階層歸屬（是一種、屬於、屬於類別）\n"
        "  PART_OF    → 組成關係（是...的部分、包含於）\n"
        "  USES       → 功能依賴（使用、調用、依賴、需要、基於）\n"
        "  ENABLES    → 賦能關係（使...成為可能、支援、允許、提供）\n"
        "  CAUSES     → 因果關係（導致、引起、造成、觸發）\n"
        "  HAS_PROPERTY → 屬性描述（具有特性、是...的特點）\n"
        "  PRECEDES   → 時序關係（先於、之前執行、觸發後才有）\n"
        "  RELATED_TO → 其他相關（無法歸入以上類別時使用）\n\n"
        "規則：\n"
        "- 主詞與受詞為名詞或名詞短語（2-15字）\n"
        "- 動詞欄位直接從原文提取最能代表關係的核心動詞或動詞短語（2-6字），"
        "如：使用、導致、屬於、包含、提供、改善、基於、需要、支援、觸發\n"
        "- 只輸出六欄格式，不加說明、序號、標點\n"
        "- 行數上限 30 行，優先抽取最重要的知識關係\n\n"
        "範例：\n"
        "Q-Learning|算法|IS_A|屬於|強化學習|概念\n"
        "工具呼叫|方法|PART_OF|包含於|代理迴圈|概念\n"
        "提示快取|技術|ENABLES|使能|效能優化|指標\n"
        "Context 超限|概念|CAUSES|導致|回應延遲|指標\n"
        "Transformer|模型|USES|使用|注意力機制|技術\n"
        "並行執行|技術|HAS_PROPERTY|具有|非阻塞特性|概念\n"
        "路由層|系統|PRECEDES|先於|SVO 查詢|方法\n\n"
        f"文字：\n{text}"
    )
    raw = await get_llm_provider().generate(prompt)
    return _parse_svo_lines(raw)


async def merge_triples_to_neo4j(
    triples: list[SVOTriple],
    kg_id: UUID,
    doc_id: UUID,
    db_name: str = "",
) -> int:
    """
    將帶型別的三元組以 rel_type 作為真正的 Neo4j relationship type 寫入。
    每個 rel_type 一批 UNWIND（最多 8 批），確保邊標籤有語意意義。
    db_name 不為空 → 寫入 KG 專用資料庫
    db_name 為空  → 寫入主資料庫並以 kg_id 隔離
    """
    if not triples:
        return 0

    from collections import defaultdict
    driver = get_driver()
    doc_id_str = str(doc_id)
    kg_id_str = str(kg_id)

    # 按 rel_type 分組，rel_type 已通過 _VALID_REL_TYPES 驗證，可安全嵌入 Cypher
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        groups[t.rel_type].append({
            "subject": t.subject,
            "s_type":  t.subject_type,
            "object":  t.object,
            "o_type":  t.object_type,
            "verb":    t.verb,
            "s_id":    str(uuid4()),
            "o_id":    str(uuid4()),
        })

    total_merged = 0
    for rel_type, rows in groups.items():
        try:
            if db_name:
                result = await driver.execute_query(
                    f"""
                    UNWIND $rows AS r
                    MERGE (s:Entity {{name: r.subject}})
                    ON CREATE SET s.id = r.s_id, s.type = r.s_type, s.created_at = datetime()
                    ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN r.s_type ELSE s.type END
                    MERGE (o:Entity {{name: r.object}})
                    ON CREATE SET o.id = r.o_id, o.type = r.o_type, o.created_at = datetime()
                    ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN r.o_type ELSE o.type END
                    MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                    ON CREATE SET rel.verb = r.verb, rel.confidence = 1, rel.created_at = datetime()
                    ON MATCH SET  rel.confidence = rel.confidence + 1,
                                  rel.verb = r.verb, rel.updated_at = datetime()
                    RETURN count(rel) AS merged
                    """,
                    rows=rows, doc_id=doc_id_str,
                    database_=db_name,
                )
            else:
                result = await driver.execute_query(
                    f"""
                    UNWIND $rows AS r
                    MERGE (s:Entity {{name: r.subject, kg_id: $kg_id}})
                    ON CREATE SET s.id = r.s_id, s.type = r.s_type, s.created_at = datetime()
                    ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN r.s_type ELSE s.type END
                    MERGE (o:Entity {{name: r.object, kg_id: $kg_id}})
                    ON CREATE SET o.id = r.o_id, o.type = r.o_type, o.created_at = datetime()
                    ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN r.o_type ELSE o.type END
                    MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                    ON CREATE SET rel.verb = r.verb, rel.confidence = 1, rel.created_at = datetime()
                    ON MATCH SET  rel.confidence = rel.confidence + 1,
                                  rel.verb = r.verb, rel.updated_at = datetime()
                    RETURN count(rel) AS merged
                    """,
                    rows=rows, doc_id=doc_id_str, kg_id=kg_id_str,
                )
            total_merged += result.records[0]["merged"] if result.records else len(rows)
        except Exception as e:
            logger.warning(f"批次 MERGE 失敗 [{rel_type}]，回退逐條模式：{e}")
            for row in rows:
                try:
                    if db_name:
                        await driver.execute_query(
                            f"""
                            MERGE (s:Entity {{name: $subject}})
                            ON CREATE SET s.id = $s_id, s.type = $s_type, s.created_at = datetime()
                            ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN $s_type ELSE s.type END
                            MERGE (o:Entity {{name: $object}})
                            ON CREATE SET o.id = $o_id, o.type = $o_type, o.created_at = datetime()
                            ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN $o_type ELSE o.type END
                            MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                            ON CREATE SET rel.verb = $verb, rel.confidence = 1, rel.created_at = datetime()
                            ON MATCH SET  rel.confidence = rel.confidence + 1,
                                          rel.verb = $verb, rel.updated_at = datetime()
                            """,
                            subject=row["subject"], object=row["object"],
                            verb=row["verb"], s_type=row["s_type"], o_type=row["o_type"],
                            doc_id=doc_id_str, s_id=row["s_id"], o_id=row["o_id"],
                            database_=db_name,
                        )
                    else:
                        await driver.execute_query(
                            f"""
                            MERGE (s:Entity {{name: $subject, kg_id: $kg_id}})
                            ON CREATE SET s.id = $s_id, s.type = $s_type, s.created_at = datetime()
                            ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN $s_type ELSE s.type END
                            MERGE (o:Entity {{name: $object, kg_id: $kg_id}})
                            ON CREATE SET o.id = $o_id, o.type = $o_type, o.created_at = datetime()
                            ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN $o_type ELSE o.type END
                            MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                            ON CREATE SET rel.verb = $verb, rel.confidence = 1, rel.created_at = datetime()
                            ON MATCH SET  rel.confidence = rel.confidence + 1,
                                          rel.verb = $verb, rel.updated_at = datetime()
                            """,
                            subject=row["subject"], object=row["object"],
                            verb=row["verb"], s_type=row["s_type"], o_type=row["o_type"],
                            kg_id=kg_id_str, doc_id=doc_id_str,
                            s_id=row["s_id"], o_id=row["o_id"],
                        )
                    total_merged += 1
                except Exception as inner_e:
                    logger.warning(f"MERGE 失敗 [{row['subject']}|{rel_type}|{row['object']}]: {inner_e}")

    return total_merged


async def get_kg_graph(
    kg_id: UUID,
    limit: int = 200,
    min_confidence: int = 1,
) -> dict:
    """取得 KG 的 Entity 節點與 RELATION 邊清單，供前端視覺化或 API 輸出。"""
    import asyncio

    driver = get_driver()
    db_name = await _get_kg_db(kg_id)
    db_kw = {"database_": db_name} if db_name else {}
    kg_filter = "" if db_name else "{kg_id: $kg_id}"
    params = {"kg_id": str(kg_id), "limit": limit, "min_conf": min_confidence}

    entities_q = driver.execute_query(
        f"""
        MATCH (e:Entity {kg_filter})
        OPTIONAL MATCH (e)-[r:{_ALL_REL_PATTERN}]->()
        WITH e, count(r) AS out_degree
        RETURN e.id AS id, e.name AS name, e.type AS type, out_degree
        ORDER BY out_degree DESC LIMIT $limit
        """,
        **params, **db_kw,
    )
    relations_q = driver.execute_query(
        f"""
        MATCH (s:Entity {kg_filter})-[r:{_ALL_REL_PATTERN}]->(o:Entity {kg_filter})
        WHERE r.confidence >= $min_conf
        RETURN s.name AS subject, s.type AS subject_type,
               type(r) AS rel_type, r.verb AS verb,
               o.name AS object, o.type AS object_type,
               r.confidence AS confidence, r.source_doc_id AS source_doc_id
        ORDER BY r.confidence DESC LIMIT $limit
        """,
        **params, **db_kw,
    )

    entities_result, relations_result = await asyncio.gather(entities_q, relations_q)

    entities = [dict(r) for r in entities_result.records]
    relations = [dict(r) for r in relations_result.records]

    return {
        "kg_id": str(kg_id),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "entities": entities,
        "relations": relations,
    }


def _build_ft_query(terms: list[str]) -> str:
    """將 terms 轉成 Lucene OR 查詢字串，特殊字元逸出。"""
    escape = str.maketrans({c: f"\\{c}" for c in r'+-&|!(){}[]^"~*?:\/'})
    return " OR ".join(f'"{t.translate(escape)}"' for t in terms)


async def query_svo_facts(
    kg_id: UUID,
    terms: list[str],
    hops: int = 2,
    limit: int = 50,
    db_name: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    BFS 遍歷 SVO 圖，回傳：
    - facts:       知識事實字串清單（供 LLM prompt 使用）
    - source_docs: 這些事實來源的 doc_id 字串清單（供圖譜驅動文件選取）
    seed 搜尋優先使用 fulltext index（entity_name_ft），失敗時回退 CONTAINS。
    """
    if not terms:
        return [], []

    if db_name is None:
        db_name = await _get_kg_db(kg_id)

    driver = get_driver()
    db_kw = {"database_": db_name} if db_name else {}
    kg_id_str = str(kg_id)
    ft_query_str = _build_ft_query(terms)

    # ── 嘗試 fulltext seed 搜尋 ───────────────────────────────────────────────
    seed_cypher_ft: str
    seed_params_ft: dict

    if db_name:
        seed_cypher_ft = (
            "CALL db.index.fulltext.queryNodes('entity_name_ft', $ft_q) "
            "YIELD node AS e RETURN e"
        )
        seed_params_ft = {"ft_q": ft_query_str}
    else:
        seed_cypher_ft = (
            "CALL db.index.fulltext.queryNodes('entity_name_ft', $ft_q) "
            "YIELD node AS e WHERE e.kg_id = $kg_id RETURN e"
        )
        seed_params_ft = {"ft_q": ft_query_str, "kg_id": kg_id_str}

    try:
        seed_result = await driver.execute_query(seed_cypher_ft, **seed_params_ft, **db_kw)
        use_ft = True
    except Exception:
        use_ft = False

    # ── BFS 展開（共用，seed 來源不同）──────────────────────────────────────────
    kg_filter = "" if db_name else "{kg_id: $kg_id}"

    if use_ft and seed_result.records:
        seed_ids = [r["e"].element_id for r in seed_result.records]
        bfs_params: dict = {"seed_ids": seed_ids, "limit": limit}
        if not db_name:
            bfs_params["kg_id"] = kg_id_str
        result = await driver.execute_query(
            f"""
            MATCH (seed)
            WHERE elementId(seed) IN $seed_ids
            WITH collect(seed) AS seeds
            UNWIND seeds AS seed
            MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(neighbor:Entity {kg_filter})
            UNWIND relationships(path) AS r
            WITH startNode(r) AS s, r, endNode(r) AS o
            RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                   type(r) AS rel_type, r.verb AS verb,
                   o.name AS object, o.type AS object_type,
                   r.confidence AS confidence, r.source_doc_id AS source_doc_id
            ORDER BY confidence DESC LIMIT $limit
            """,
            **bfs_params, **db_kw,
        )
    else:
        # fallback：CONTAINS 全表掃描
        where_clauses = " OR ".join(
            f"toLower(e.name) CONTAINS toLower($term{i})" for i in range(len(terms))
        )
        fallback_params: dict = {f"term{i}": t for i, t in enumerate(terms)}
        fallback_params["limit"] = limit
        if db_name:
            result = await driver.execute_query(
                f"""
                MATCH (e:Entity)
                WHERE {where_clauses}
                WITH collect(e) AS seeds
                UNWIND seeds AS seed
                MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(neighbor:Entity)
                UNWIND relationships(path) AS r
                WITH startNode(r) AS s, r, endNode(r) AS o
                RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                       type(r) AS rel_type, r.verb AS verb,
                       o.name AS object, o.type AS object_type,
                       r.confidence AS confidence, r.source_doc_id AS source_doc_id
                ORDER BY confidence DESC LIMIT $limit
                """,
                database_=db_name, **fallback_params,
            )
        else:
            fallback_params["kg_id"] = kg_id_str
            result = await driver.execute_query(
                f"""
                MATCH (e:Entity {{kg_id: $kg_id}})
                WHERE {where_clauses}
                WITH collect(e) AS seeds
                UNWIND seeds AS seed
                MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(neighbor:Entity {{kg_id: $kg_id}})
                UNWIND relationships(path) AS r
                WITH startNode(r) AS s, r, endNode(r) AS o
                RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                       type(r) AS rel_type, r.verb AS verb,
                       o.name AS object, o.type AS object_type,
                       r.confidence AS confidence, r.source_doc_id AS source_doc_id
                ORDER BY confidence DESC LIMIT $limit
                """,
                **fallback_params,
            )

    facts = []
    source_docs: list[str] = []
    seen_docs: set[str] = set()

    for r in result.records:
        rel_type = r.get("rel_type") or "RELATED_TO"
        verb = r.get("verb") or rel_type
        label = REL_TYPE_LABELS.get(rel_type, rel_type)
        facts.append(
            f"[{label}] {r['subject']}({r.get('subject_type') or '概念'})"
            f" {verb} "
            f"{r['object']}({r.get('object_type') or '概念'})"
        )
        doc_id = r.get("source_doc_id")
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            source_docs.append(doc_id)

    return facts, source_docs


async def create_entity_index() -> None:
    """在 lifespan 啟動時建立 Entity 的 Neo4j 索引，加速查詢。"""
    driver = get_driver()
    await driver.execute_query(
        "CREATE INDEX entity_kg_name IF NOT EXISTS FOR (e:Entity) ON (e.kg_id, e.name)"
    )
    # fulltext index：加速 query_svo_facts 的關鍵字模糊搜尋
    try:
        await driver.execute_query(
            "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name]"
        )
    except Exception as e:
        logger.debug(f"fulltext index 建立跳過（可能已存在）：{e}")


# type 屬性 → Neo4j 附加標籤的映射（保留 Entity 主標籤，加上語義標籤）
_TYPE_LABEL_MAP: dict[str, str] = {
    "概念":  "Concept",
    "算法":  "Algorithm",
    "技術":  "Technology",
    "方法":  "Method",
    "工具":  "Tool",
    "框架":  "Framework",
    "模型":  "Model",
    "系統":  "System",
    "人物":  "Person",
    "組織":  "Organization",
    "資料集": "Dataset",
    "指標":  "Metric",
    "其他":  "Other",
}


async def apply_type_labels(kg_id: UUID, db_name: str = "") -> dict[str, int]:
    """
    依 Entity.type 屬性為節點附加語義標籤（e.g. :Algorithm, :Model）。
    不移除 :Entity 主標籤，只疊加。
    回傳各標籤打上的節點數 {label: count}。
    """
    driver = get_driver()
    db_kw = {"database_": db_name} if db_name else {}
    kg_filter = "" if db_name else "{kg_id: $kg_id}"
    params_base: dict = {} if db_name else {"kg_id": str(kg_id)}

    stats: dict[str, int] = {}
    for type_name, label in _TYPE_LABEL_MAP.items():
        result = await driver.execute_query(
            f"MATCH (e:Entity {kg_filter}) WHERE e.type = $type "
            f"SET e:`{label}` RETURN count(e) AS n",
            type=type_name, **params_base, **db_kw,
        )
        n = result.records[0]["n"] if result.records else 0
        if n:
            stats[label] = n

    # 為 rel_type 也加關係類型索引標籤（RELATION 本身已有，這裡確保完整性）
    logger.info(f"KG {kg_id} 標籤完成：{stats}")
    return stats


# ── 內部工具 ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """將長文本分成大小接近 _CHUNK_SIZE 的段落，盡量在句尾斷開。"""
    if len(text) <= _CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:])
            break
        # 往回找句尾標點（。？！.!?）
        cut = end
        for punct in ("。", "？", "！", ".", "!", "?", "\n"):
            pos = text.rfind(punct, start + _CHUNK_SIZE // 2, end)
            if pos != -1:
                cut = pos + 1
                break
        chunks.append(text[start:cut])
        start = cut - _CHUNK_OVERLAP  # 重疊部分
        if start < 0:
            start = 0
    return chunks


_VALID_TYPES = {
    "概念", "算法", "技術", "方法", "工具", "框架",
    "模型", "系統", "人物", "組織", "資料集", "指標", "其他",
}

_VALID_REL_TYPES = {
    "IS_A", "PART_OF", "USES", "ENABLES",
    "CAUSES", "HAS_PROPERTY", "PRECEDES", "RELATED_TO",
}

# Cypher relationship type pattern（供 MATCH 使用）
_ALL_REL_PATTERN = "IS_A|PART_OF|USES|ENABLES|CAUSES|HAS_PROPERTY|PRECEDES|RELATED_TO"

# 關係類別的中文顯示名稱（供 UI 顯示）
REL_TYPE_LABELS = {
    "IS_A":         "階層",
    "PART_OF":      "組成",
    "USES":         "依賴",
    "ENABLES":      "賦能",
    "CAUSES":       "因果",
    "HAS_PROPERTY": "屬性",
    "PRECEDES":     "時序",
    "RELATED_TO":   "相關",
}


def _parse_svo_lines(raw: str) -> list[SVOTriple]:
    """解析 LLM 回傳的知識三元組行。

    6欄（新）：主詞|主詞類型|關係類別|關係描述|受詞|受詞類型
    5欄（舊）：主詞|主詞類型|關係描述|受詞|受詞類型
    3欄（舊）：主詞|關係描述|受詞
    """
    triples: list[SVOTriple] = []
    seen: set[tuple[str, str, str]] = set()

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[|｜]", line)]

        if len(parts) == 6:
            s, s_type, rel_type, v, o, o_type = parts
            if s_type not in _VALID_TYPES:
                s_type = "其他"
            if o_type not in _VALID_TYPES:
                o_type = "其他"
            if rel_type not in _VALID_REL_TYPES:
                rel_type = "RELATED_TO"
        elif len(parts) == 5:
            s, s_type, v, o, o_type = parts
            rel_type = "RELATED_TO"
            if s_type not in _VALID_TYPES:
                s_type = "其他"
            if o_type not in _VALID_TYPES:
                o_type = "其他"
        elif len(parts) == 3:
            s, v, o = parts
            s_type = o_type = "概念"
            rel_type = "RELATED_TO"
        else:
            continue

        if not s or not v or not o:
            continue
        if len(s) > 50 or len(v) > 15 or len(o) > 50:
            continue
        key = (s, rel_type, o)   # 用 rel_type 去重（同類別的同主受詞只保留一條）
        if key in seen:
            continue
        seen.add(key)
        triples.append(SVOTriple(
            subject=s, subject_type=s_type,
            rel_type=rel_type,
            verb=v,
            object=o, object_type=o_type,
        ))

    return triples


async def _clear_kg_entities(kg_id: UUID, db_name: str = "") -> None:
    """force_rebuild 時清除此 KG 的所有 Entity 與 RELATION（路由層 ConceptNode 不動）。"""
    driver = get_driver()
    if db_name:
        await driver.execute_query(
            "MATCH (e:Entity) DETACH DELETE e",
            database_=db_name,
        )
    else:
        await driver.execute_query(
            "MATCH (e:Entity {kg_id: $kg_id}) DETACH DELETE e",
            kg_id=str(kg_id),
        )
    logger.info(f"KG {kg_id} Entity 清除完成（force_rebuild）")
