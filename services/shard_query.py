"""
Shard Query Engine — Phase 2c

asyncio.gather() 同時查詢所有分片（本機 + 遠端 AuraDB）：
- 每個分片獨立超時（預設 5 秒），超時後靜默跳過並標記離線
- 同名實體保留多個 instance 來源前綴
- 結果去重合併後回傳
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from neo4j import AsyncDriver

from models.kb_skill import KBSkill

logger = logging.getLogger(__name__)

SHARD_TIMEOUT = 5.0  # 每個分片的最長等待秒數

_ALL_REL = (
    "IS_A|PART_OF|CONTAINS|INSTANCE_OF|CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|"
    "USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|CONTRASTS|SIMILAR_TO|OUTPERFORMS|"
    "DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|PRECEDES|FOLLOWS|CO_OCCURS|"
    "INPUTS|TRANSFORMS|CREATED_BY|SOLVES|RELATED_TO"
)

_REL_LABELS: dict[str, str] = {
    "IS_A": "是", "PART_OF": "是...的一部分", "CONTAINS": "包含",
    "INSTANCE_OF": "是...的實例", "CAUSES": "導致", "PREVENTS": "防止",
    "ENABLES": "使能", "IMPROVES": "改善", "INHIBITS": "抑制",
    "USES": "使用", "REQUIRES": "需要", "PRODUCES": "產生",
    "IMPLEMENTS": "實作", "REPLACES": "取代", "EXTENDS": "擴展",
    "CONTRASTS": "對比", "SIMILAR_TO": "相似於", "OUTPERFORMS": "優於",
    "DEFINED_AS": "定義為", "HAS_PROPERTY": "具有屬性", "MEASURED_BY": "以...衡量",
    "APPLIES_TO": "適用於", "PRECEDES": "先於", "FOLLOWS": "後於",
    "CO_OCCURS": "共現", "INPUTS": "輸入", "TRANSFORMS": "轉換",
    "CREATED_BY": "由...創建", "SOLVES": "解決", "RELATED_TO": "相關於",
}


@dataclass
class ShardResult:
    shard_id: str        # kb_id
    shard_name: str
    instance_id: str
    status: str          # "ok" | "timeout" | "offline" | "error"
    facts: list[str] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    sourced_facts: list = field(default_factory=list)   # list[SourcedFact] Phase 3a


# ── 遠端 AuraDB BFS 查詢 ──────────────────────────────────────────────────────

async def _bfs_remote(
    driver: AsyncDriver,
    terms: list[str],
    hops: int,
    limit: int,
    instance_id: str,
) -> tuple[list[str], list[str]]:
    """
    在遠端 AuraDB 執行 BFS（不依賴 FTS，用 CONTAINS 比對種子節點）。
    事實字串加上 instance 前綴，方便溯源。
    """
    where_parts = " OR ".join(
        f"toLower(e.name) CONTAINS toLower($t{i})" for i in range(len(terms))
    )
    params: dict = {f"t{i}": t for i, t in enumerate(terms)}
    params["lim"] = limit

    cypher = f"""
    MATCH (e:Entity)
    WHERE {where_parts}
    WITH collect(e) AS seeds
    UNWIND seeds AS seed
    MATCH path = (seed)-[:{_ALL_REL}*1..{min(hops, 1)}]-(nb:Entity)
    UNWIND relationships(path) AS r
    WITH startNode(r) AS s, r, endNode(r) AS o
    RETURN DISTINCT
        s.name AS subject, s.type AS subject_type,
        type(r) AS rel_type, r.verb AS verb,
        o.name AS object, o.type AS object_type,
        r.source_doc_id AS src_doc
    ORDER BY s.name LIMIT $lim
    """

    result = await driver.execute_query(cypher, **params)

    facts: list[str] = []
    source_docs: list[str] = []
    seen_docs: set[str] = set()
    prefix = f"[{instance_id}] " if instance_id else ""

    for rec in result.records:
        rel_type = rec.get("rel_type") or "RELATED_TO"
        verb = rec.get("verb") or rel_type
        label = _REL_LABELS.get(rel_type, rel_type)
        s = rec["subject"]
        o = rec["object"]
        st = rec.get("subject_type") or "概念"
        ot = rec.get("object_type") or "概念"
        facts.append(f"{prefix}{s}({st}) -[{label}:{verb}]→ {o}({ot})")
        doc_id = rec.get("src_doc")
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            source_docs.append(doc_id)

    return facts, source_docs


# ── 單一分片查詢 ──────────────────────────────────────────────────────────────

async def _query_one_shard(
    skill: KBSkill,
    terms: list[str],
    hops: int,
    limit: int,
) -> ShardResult:
    """查詢單一分片（不含超時包裝，由呼叫方設定）。"""
    t0 = time.monotonic()
    result = ShardResult(
        shard_id=skill.kb_id,
        shard_name=skill.name,
        instance_id=skill.instance_id,
        status="ok",
    )

    try:
        if skill.is_local:
            from uuid import UUID
            from services.svo_service import query_svo_facts_with_provenance
            sourced = await query_svo_facts_with_provenance(
                UUID(skill.kb_id), terms,
                hops=hops, limit=limit, db_name=skill.db_name,
                instance_id=skill.instance_id,
            )
            result.sourced_facts = sourced
            result.facts = [sf.fact_str for sf in sourced]
            result.source_docs = list({sf.source_doc_id for sf in sourced if sf.source_doc_id})
        else:
            from services.federation_service import get_federation_cache
            cache = get_federation_cache()
            driver: AsyncDriver | None = await cache.get_shard_driver(skill)
            if driver is None:
                result.status = "offline"
                return result
            facts, src_docs = await _bfs_remote(
                driver, terms, hops=hops, limit=limit,
                instance_id=skill.instance_id,
            )
            result.facts = facts
            result.source_docs = src_docs

    except Exception as e:
        logger.warning(f"分片查詢失敗 [{skill.name}]：{e}")
        result.status = "error"

    result.elapsed_ms = int((time.monotonic() - t0) * 1000)
    return result


# ── 路由輔助 ──────────────────────────────────────────────────────────────────

def route_skill_score(skill: KBSkill, query_terms: list[str]) -> float:
    """
    用 top_concepts 名稱做關鍵字比對，回傳 0-1 路由分數。
    空 top_concepts 回傳 0（不路由）。
    """
    if not skill.top_concepts or not query_terms:
        return 0.0
    concept_names = [c.name.lower() for c in skill.top_concepts]
    matched = sum(
        1 for t in query_terms
        if any(t.lower() in cn or cn in t.lower() for cn in concept_names)
    )
    return matched / len(query_terms)


# ── 並行查詢主函式 ────────────────────────────────────────────────────────────

async def query_shards_parallel(
    skills: list[KBSkill],
    terms: list[str],
    hops: int = 1,
    limit_per_shard: int = 30,
    timeout: float = SHARD_TIMEOUT,
    expand_synonyms: bool = True,
) -> tuple[list[str], list[str], list[ShardResult]]:
    """
    並行查詢所有分片，回傳：
    - merged_facts   : 去重後的知識事實清單（遠端事實帶 instance 前綴）
    - merged_docs    : 本機 source_doc_id 清單（跨分片去重）
    - shard_results  : 每個分片的詳細結果（狀態、耗時、facts 數量）

    expand_synonyms=True（Phase 2d）：查詢前自動展開同義詞（zh↔en 術語對照）。
    """
    if not skills or not terms:
        return [], [], []

    if expand_synonyms:
        from services.entity_alignment import expand_terms
        terms = expand_terms(terms)

    async def _run(skill: KBSkill) -> ShardResult:
        try:
            return await asyncio.wait_for(
                _query_one_shard(skill, terms, hops, limit_per_shard),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"分片超時（{timeout:.0f}s）：{skill.name} [{skill.instance_id}]")
            if not skill.is_local:
                from services.federation_service import get_federation_cache
                get_federation_cache().mark_shard_offline(skill.kb_id)
            return ShardResult(
                shard_id=skill.kb_id,
                shard_name=skill.name,
                instance_id=skill.instance_id,
                status="timeout",
                elapsed_ms=int(timeout * 1000),
            )

    shard_results: list[ShardResult] = list(
        await asyncio.gather(*[_run(s) for s in skills])
    )

    # 合併去重（facts 字串 + sourced_facts）
    seen_facts: set[str] = set()
    merged_facts: list[str] = []
    seen_docs: set[str] = set()
    merged_docs: list[str] = []
    merged_sourced: list = []   # list[SourcedFact]

    for sr in shard_results:
        if sr.status != "ok":
            continue
        for f in sr.facts:
            if f not in seen_facts:
                seen_facts.add(f)
                merged_facts.append(f)
        for d in sr.source_docs:
            if d not in seen_docs:
                seen_docs.add(d)
                merged_docs.append(d)
        merged_sourced.extend(sr.sourced_facts)

    return merged_facts, merged_docs, shard_results, merged_sourced
