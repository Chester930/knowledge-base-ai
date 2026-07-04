"""
真實 Neo4j 整合測試。

現有測試套件（tests/routers, tests/services）全用 MagicMock 驅動 Neo4j，
Cypher 語法錯誤（索引建立語句、BFS pattern 等）不會被任何測試抓到。
本檔案改連線真實 Neo4j（見 .env 的 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD），
驗證索引建立與 BFS 查詢的 Cypher 語法在真實資料庫上可正確執行。

連線失敗時自動 skip（本機未啟動 Neo4j 時不影響其餘測試套件）。
CI 已在 .github/workflows/ci.yml 啟動 Neo4j service container 讓本檔案實際執行。
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from core import database
from core.database import get_driver
from models.knowledge_graph import SVOTriple
from repositories.concept_repo import ConceptRepository
from services.svo_service import (
    cleanup_orphan_entities,
    create_entity_index,
    merge_triples_to_neo4j,
    query_svo_facts,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def neo4j_conn():
    try:
        await database.connect()
    except Exception as e:
        pytest.skip(f"無法連線到測試用 Neo4j，略過整合測試：{e}")
    yield
    await database.disconnect()


async def test_create_entity_index_executes_on_real_db(neo4j_conn):
    """驗證 create_entity_index() 的複合索引 + 全文索引 Cypher 語法可在真實 DB 執行。"""
    await create_entity_index()

    driver = get_driver()
    result = await driver.execute_query("SHOW INDEXES YIELD name RETURN name")
    names = {r["name"] for r in result.records}
    assert "entity_kg_name" in names


async def test_create_vector_index_executes_on_real_db(neo4j_conn):
    """驗證 ConceptRepository.create_vector_index() 的 Cypher 語法可在真實 DB 執行。"""
    driver = get_driver()
    await ConceptRepository(driver).create_vector_index(dim=384)

    result = await driver.execute_query("SHOW INDEXES YIELD name RETURN name")
    names = {r["name"] for r in result.records}
    assert "concept_q_vector" in names


async def test_bfs_query_finds_merged_triples(neo4j_conn):
    """寫入一組 SVO 三元組後，驗證 BFS 查詢語法能正確找回對應事實與來源文件。"""
    await create_entity_index()
    driver = get_driver()
    kg_id = uuid4()
    doc_id = uuid4()

    triples = [
        SVOTriple(
            subject="Transformer", subject_type="模型", rel_type="USES",
            verb="使用", object="多頭注意力", object_type="技術",
            confidence=2, source_doc_id=doc_id,
        ),
        SVOTriple(
            subject="多頭注意力", subject_type="技術", rel_type="IMPROVES",
            verb="提升", object="長距離依賴建模", object_type="概念",
            confidence=2, source_doc_id=doc_id,
        ),
    ]
    try:
        written = await merge_triples_to_neo4j(triples, kg_id, doc_id, db_name="")
        assert written == len(triples)

        facts, source_docs, _chunk_ids = await query_svo_facts(
            kg_id, ["Transformer"], hops=2, limit=10
        )
        assert any("Transformer" in f for f in facts)
        assert str(doc_id) in source_docs
    finally:
        await driver.execute_query(
            "MATCH (e:Entity {kg_id: $kg_id}) DETACH DELETE e", kg_id=str(kg_id)
        )


async def test_cleanup_orphan_entities_removes_only_degree_zero_nodes(neo4j_conn):
    """驗證孤兒節點清理只刪除無任何關係邊的 Entity，不影響有邊的節點。"""
    await create_entity_index()
    driver = get_driver()
    kg_id = uuid4()
    doc_id = uuid4()

    triples = [
        SVOTriple(
            subject="A", subject_type="概念", rel_type="RELATED_TO",
            verb="相關", object="B", object_type="概念",
            confidence=1, source_doc_id=doc_id,
        ),
    ]
    try:
        await merge_triples_to_neo4j(triples, kg_id, doc_id, db_name="")
        # 手動插入一個無任何關係邊的孤兒節點
        await driver.execute_query(
            "CREATE (:Entity {id: $id, name: '孤兒節點', kg_id: $kg_id})",
            id=str(uuid4()), kg_id=str(kg_id),
        )

        removed = await cleanup_orphan_entities(kg_id, db_name="")
        assert removed == 1

        result = await driver.execute_query(
            "MATCH (e:Entity {kg_id: $kg_id}) RETURN e.name AS name",
            kg_id=str(kg_id),
        )
        names = {r["name"] for r in result.records}
        assert names == {"A", "B"}
    finally:
        await driver.execute_query(
            "MATCH (e:Entity {kg_id: $kg_id}) DETACH DELETE e", kg_id=str(kg_id)
        )
