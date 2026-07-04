"""
Community Service — 多層次社群摘要檢索 (THEORETICAL_ARCHITECTURE.md 第9節⑤)

對 KG 的 Entity 關係圖做社群偵測（networkx Louvain），為每個社群生成 LLM 摘要並持久化，
供全域性/宏觀查詢（例如「總結這個知識庫的技術演進」）路由使用，補足 BFS 1-2 跳遍歷
只能匹配局部實體、無法回答全局性問題的缺口。
"""
from __future__ import annotations
import logging
from uuid import UUID

from core.database import get_driver
from core.providers.factory import get_llm_provider
from services.svo_service import _ALL_REL_PATTERN

logger = logging.getLogger(__name__)

_MAX_MEMBER_NAMES_IN_PROMPT = 15
_MAX_FACTS_IN_PROMPT = 30


async def build_communities_for_kg(
    kg_id: UUID,
    db_name: str = "",
    min_size: int = 3,
    max_communities: int = 20,
) -> int:
    """
    對 KG 的 Entity 圖譜做社群偵測 + LLM 摘要，持久化為 :Community 節點。

    每次呼叫會先清除該 KG 既有的 :Community 節點再重建（語意會隨圖譜演進而變，
    沿用 run_build_kg.py --force 的「先清後建」慣例，避免陳舊摘要殘留）。
    回傳成功建立摘要的社群數。
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx 未安裝，跳過社群偵測（此為選用相依套件）")
        return 0

    driver = get_driver()
    db_kw = {"database_": db_name} if db_name else {}
    kg_filter = "" if db_name else "{kg_id: $kg_id}"
    params_base: dict = {} if db_name else {"kg_id": str(kg_id)}

    edge_result = await driver.execute_query(
        f"""
        MATCH (s:Entity {kg_filter})-[r:{_ALL_REL_PATTERN}]-(o:Entity {kg_filter})
        WHERE s.name <> o.name
        RETURN DISTINCT s.name AS a, o.name AS b
        """,
        **params_base, **db_kw,
    )
    if not edge_result.records:
        logger.info(f"KG {kg_id} 無關係邊，跳過社群偵測")
        return 0

    graph = nx.Graph()
    for r in edge_result.records:
        graph.add_edge(r["a"], r["b"])

    if graph.number_of_nodes() < min_size:
        logger.info(f"KG {kg_id} 節點數過少（{graph.number_of_nodes()}），跳過社群偵測")
        return 0

    # networkx 內建 Louvain（非文件原提案的 Leiden，見 THEORETICAL_ARCHITECTURE.md 第9節⑤說明）
    communities = nx.algorithms.community.louvain_communities(graph, seed=42)
    qualifying = sorted((c for c in communities if len(c) >= min_size), key=len, reverse=True)
    qualifying = qualifying[:max_communities]

    if not qualifying:
        logger.info(f"KG {kg_id} 無符合最小規模（min_size={min_size}）的社群")
        return 0

    await driver.execute_query(
        f"MATCH (c:Community {kg_filter}) DETACH DELETE c",
        **params_base, **db_kw,
    )

    built = 0
    for idx, members in enumerate(qualifying):
        member_names = list(members)
        facts = await _sample_community_facts(
            driver, member_names, kg_filter, params_base, db_kw,
        )
        summary = await _summarize_community(member_names, facts)
        if not summary:
            continue

        await driver.execute_query(
            f"""
            CREATE (c:Community {{
                kg_id: $kg_id_str, community_key: $key, summary: $summary,
                member_count: $count, top_entities: $top_entities, updated_at: datetime()
            }})
            WITH c
            UNWIND $member_names AS n
            MATCH (e:Entity {kg_filter}) WHERE e.name = n
            MERGE (e)-[:IN_COMMUNITY]->(c)
            """,
            kg_id_str=str(kg_id), key=f"{kg_id}_{idx}", summary=summary,
            count=len(member_names), top_entities=member_names[:10],
            member_names=member_names,
            **params_base, **db_kw,
        )
        built += 1

    logger.info(f"KG {kg_id} 社群摘要建立完成：{built}/{len(qualifying)} 個")
    return built


async def _sample_community_facts(
    driver, member_names: list[str], kg_filter: str, params_base: dict, db_kw: dict,
    limit: int = _MAX_FACTS_IN_PROMPT,
) -> list[str]:
    """取樣社群成員之間的關係事實，供 LLM 摘要用。"""
    result = await driver.execute_query(
        f"""
        MATCH (s:Entity {kg_filter})-[r:{_ALL_REL_PATTERN}]->(o:Entity {kg_filter})
        WHERE s.name IN $names AND o.name IN $names
        RETURN s.name AS s, type(r) AS rel_type, r.verb AS verb, o.name AS o
        LIMIT $limit
        """,
        names=member_names, limit=limit, **params_base, **db_kw,
    )
    return [f"{r['s']} -[{r['verb'] or r['rel_type']}]→ {r['o']}" for r in result.records]


async def _summarize_community(member_names: list[str], facts: list[str]) -> str:
    """呼叫 LLM 為一個社群生成 1-3 句摘要；無事實時退回僅用成員名稱概括。"""
    if facts:
        facts_text = "\n".join(facts)
        prompt = (
            f"以下是知識圖譜中一組相關概念之間的關係事實：\n{facts_text}\n\n"
            "請用 2-3 句繁體中文摘要這個知識群組的核心主題與關聯，不需逐條複述。只回傳摘要文字。"
        )
    else:
        names_preview = "、".join(member_names[:_MAX_MEMBER_NAMES_IN_PROMPT])
        prompt = (
            f"以下是知識圖譜中偵測到的一組相關概念：{names_preview}\n"
            "請用 1-2 句繁體中文摘要這組概念的共同主題，不需列舉每一項。只回傳摘要文字。"
        )
    try:
        summary = await get_llm_provider().generate(prompt)
        return summary.strip()
    except Exception as e:
        logger.warning(f"社群摘要生成失敗：{e}")
        return ""


async def get_community_summaries(kg_id: UUID, db_name: str = "", limit: int = 5) -> list[dict]:
    """讀取 KG 已建立的社群摘要，依成員數降冪排序（供全域性查詢路由使用）。"""
    driver = get_driver()
    db_kw = {"database_": db_name} if db_name else {}
    result = await driver.execute_query(
        """
        MATCH (c:Community {kg_id: $kg_id})
        RETURN c.summary AS summary, c.member_count AS member_count,
               c.top_entities AS top_entities
        ORDER BY c.member_count DESC LIMIT $limit
        """,
        kg_id=str(kg_id), limit=limit, **db_kw,
    )
    return [dict(r) for r in result.records]
