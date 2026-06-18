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
        "主詞|主詞類型|關係類別|關係描述|受詞|受詞類型\n\n"
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
        "- 關係描述為自然語言動詞短語（2-8字），清楚說明關係內容\n"
        "- 只輸出六欄格式，不加說明、序號、標點\n"
        "- 行數上限 30 行，優先抽取最重要的知識關係\n\n"
        "範例：\n"
        "Q-Learning|算法|IS_A|屬於強化學習的一種算法|強化學習|概念\n"
        "工具呼叫|方法|PART_OF|是代理迴圈的執行步驟|代理迴圈|概念\n"
        "提示快取|技術|ENABLES|使跨請求重用 context 成為可能|效能優化|指標\n"
        "Context 超限|概念|CAUSES|導致回應延遲增加|回應延遲|指標\n"
        "Transformer|模型|USES|使用多頭注意力機制進行特徵提取|注意力機制|技術\n"
        "並行執行|技術|HAS_PROPERTY|具有非阻塞的執行特性|非阻塞|概念\n"
        "路由層|系統|PRECEDES|先完成路由才進行 SVO 查詢|SVO 查詢|方法\n\n"
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
    將帶型別的三元組批次 MERGE 進 Neo4j。
    db_name 不為空 → 寫入 KG 專用資料庫（Entity 不需要 kg_id）
    db_name 為空  → 寫入主資料庫並以 kg_id 隔離（向下相容）
    """
    if not triples:
        return 0

    driver = get_driver()
    merged = 0

    if db_name:
        for triple in triples:
            try:
                await driver.execute_query(
                    """
                    MERGE (s:Entity {name: $subject})
                    ON CREATE SET s.id = $s_id, s.type = $s_type, s.created_at = datetime()
                    ON MATCH SET s.type = CASE WHEN s.type IS NULL THEN $s_type ELSE s.type END
                    MERGE (o:Entity {name: $object})
                    ON CREATE SET o.id = $o_id, o.type = $o_type, o.created_at = datetime()
                    ON MATCH SET o.type = CASE WHEN o.type IS NULL THEN $o_type ELSE o.type END
                    MERGE (s)-[r:RELATION {rel_type: $rel_type, source_doc_id: $doc_id}]->(o)
                    ON CREATE SET r.verb = $verb, r.confidence = 1, r.created_at = datetime()
                    ON MATCH  SET r.confidence = r.confidence + 1,
                                  r.verb = $verb, r.updated_at = datetime()
                    """,
                    subject=triple.subject, object=triple.object,
                    rel_type=triple.rel_type, verb=triple.verb,
                    s_type=triple.subject_type, o_type=triple.object_type,
                    doc_id=str(doc_id), s_id=str(uuid4()), o_id=str(uuid4()),
                    database_=db_name,
                )
                merged += 1
            except Exception as e:
                logger.warning(f"MERGE 失敗 [{triple.subject}|{triple.rel_type}|{triple.object}]: {e}")
    else:
        for triple in triples:
            try:
                await driver.execute_query(
                    """
                    MERGE (s:Entity {name: $subject, kg_id: $kg_id})
                    ON CREATE SET s.id = $s_id, s.type = $s_type, s.created_at = datetime()
                    ON MATCH SET s.type = CASE WHEN s.type IS NULL THEN $s_type ELSE s.type END
                    MERGE (o:Entity {name: $object, kg_id: $kg_id})
                    ON CREATE SET o.id = $o_id, o.type = $o_type, o.created_at = datetime()
                    ON MATCH SET o.type = CASE WHEN o.type IS NULL THEN $o_type ELSE o.type END
                    MERGE (s)-[r:RELATION {rel_type: $rel_type, source_doc_id: $doc_id}]->(o)
                    ON CREATE SET r.verb = $verb, r.confidence = 1, r.created_at = datetime()
                    ON MATCH  SET r.confidence = r.confidence + 1,
                                  r.verb = $verb, r.updated_at = datetime()
                    """,
                    subject=triple.subject, object=triple.object,
                    rel_type=triple.rel_type, verb=triple.verb,
                    s_type=triple.subject_type, o_type=triple.object_type,
                    kg_id=str(kg_id), doc_id=str(doc_id),
                    s_id=str(uuid4()), o_id=str(uuid4()),
                )
                merged += 1
            except Exception as e:
                logger.warning(f"MERGE 失敗 [{triple.subject}|{triple.rel_type}|{triple.object}]: {e}")

    return merged


async def get_kg_graph(
    kg_id: UUID,
    limit: int = 200,
    min_confidence: int = 1,
) -> dict:
    """取得 KG 的 Entity 節點與 RELATION 邊清單，供前端視覺化或 API 輸出。"""
    driver = get_driver()
    db_name = await _get_kg_db(kg_id)
    db_kw = {"database_": db_name} if db_name else {}
    kg_filter = "" if db_name else "{kg_id: $kg_id}"

    entities_result = await driver.execute_query(
        f"""
        MATCH (e:Entity {kg_filter})
        OPTIONAL MATCH (e)-[r:RELATION]->()
        WITH e, count(r) AS out_degree
        RETURN e.id AS id, e.name AS name, e.type AS type, out_degree
        ORDER BY out_degree DESC LIMIT $limit
        """,
        kg_id=str(kg_id), limit=limit, **db_kw,
    )

    relations_result = await driver.execute_query(
        f"""
        MATCH (s:Entity {kg_filter})-[r:RELATION]->(o:Entity {kg_filter})
        WHERE r.confidence >= $min_conf
        RETURN s.name AS subject, s.type AS subject_type,
               coalesce(r.rel_type, 'RELATED_TO') AS rel_type, r.verb AS verb,
               o.name AS object, o.type AS object_type,
               r.confidence AS confidence, r.source_doc_id AS source_doc_id
        ORDER BY r.confidence DESC LIMIT $limit
        """,
        kg_id=str(kg_id), min_conf=min_confidence, limit=limit, **db_kw,
    )

    entities = [dict(r) for r in entities_result.records]
    relations = [dict(r) for r in relations_result.records]

    return {
        "kg_id": str(kg_id),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "entities": entities,
        "relations": relations,
    }


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
    """
    if not terms:
        return [], []

    if db_name is None:
        db_name = await _get_kg_db(kg_id)

    driver = get_driver()
    where_clauses = " OR ".join(f"toLower(e.name) CONTAINS toLower($term{i})" for i in range(len(terms)))
    params: dict = {f"term{i}": t for i, t in enumerate(terms)}
    params["limit"] = limit

    if db_name:
        result = await driver.execute_query(
            f"""
            MATCH (e:Entity)
            WHERE {where_clauses}
            WITH collect(e) AS seeds
            UNWIND seeds AS seed
            MATCH path = (seed)-[:RELATION*1..{hops}]-(neighbor:Entity)
            UNWIND relationships(path) AS r
            WITH startNode(r) AS s, r, endNode(r) AS o
            RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                   coalesce(r.rel_type, 'RELATED_TO') AS rel_type, r.verb AS verb,
                   o.name AS object, o.type AS object_type,
                   r.confidence AS confidence,
                   r.source_doc_id AS source_doc_id
            ORDER BY confidence DESC LIMIT $limit
            """,
            database_=db_name, **params,
        )
    else:
        params["kg_id"] = str(kg_id)
        result = await driver.execute_query(
            f"""
            MATCH (e:Entity {{kg_id: $kg_id}})
            WHERE {where_clauses}
            WITH collect(e) AS seeds
            UNWIND seeds AS seed
            MATCH path = (seed)-[:RELATION*1..{hops}]-(neighbor:Entity {{kg_id: $kg_id}})
            UNWIND relationships(path) AS r
            WITH startNode(r) AS s, r, endNode(r) AS o
            RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                   coalesce(r.rel_type, 'RELATED_TO') AS rel_type, r.verb AS verb,
                   o.name AS object, o.type AS object_type,
                   r.confidence AS confidence,
                   r.source_doc_id AS source_doc_id
            ORDER BY confidence DESC LIMIT $limit
            """,
            **params,
        )

    facts = []
    source_docs: list[str] = []
    seen_docs: set[str] = set()

    for r in result.records:
        rel_type = r.get("rel_type") or "RELATED_TO"
        label = REL_TYPE_LABELS.get(rel_type, rel_type)
        facts.append(
            f"[{label}] {r['subject']}({r.get('subject_type') or '概念'})"
            f" {r['verb']} "
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
        if len(s) > 50 or len(v) > 30 or len(o) > 50:
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
