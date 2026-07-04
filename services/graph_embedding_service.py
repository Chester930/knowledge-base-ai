"""
Graph-Aware Co-embedding — 圖拓撲感知共嵌入空間 (THEORETICAL_ARCHITECTURE.md 第9節①)

Phase 1 落地：對 (Document|KnowledgeGraph)-[:EFFECTIVE]->(ConceptNode) 二分圖跑 node2vec，
產生「感知圖結構」的 ConceptNode 向量（`q_vector_graph`），與既有純文字 embedding
（`q_vector`）並存，由 `repositories/concept_repo.py` 在讀取 KG 路由概念時做加權融合。

技術選型：node2vec（biased 2nd-order random walk + skip-gram），複用專案已引入的
`networkx`（圖結構、隨機遊走）與 `gensim`（skip-gram 訓練），不需要 PyTorch/GraphSAGE
這類重型 GNN 框架——這是刻意的風險控制，詳見文件第9節①「技術選型」。
"""
from __future__ import annotations
import logging
import random
from uuid import UUID

logger = logging.getLogger(__name__)

_CONCEPT_PREFIX = "concept:"
_DOC_PREFIX = "doc:"
_KG_PREFIX = "kg:"


async def build_bipartite_graph(driver):
    """抓取 (Document|KnowledgeGraph)-[:EFFECTIVE]->(ConceptNode) 邊，建二分圖。

    節點 id 以前綴區分類型（`doc:`/`kg:`/`concept:`）避免命名空間碰撞
    （Document/KG 用 UUID、ConceptNode 用 name，理論上不會撞，但前綴讓意圖明確）。
    """
    import networkx as nx

    graph = nx.Graph()

    doc_result = await driver.execute_query(
        """
        MATCH (d:Document)-[:EFFECTIVE]->(c:ConceptNode)
        RETURN d.id AS doc_id, c.name AS concept_name
        """
    )
    for r in doc_result.records:
        graph.add_edge(f"{_DOC_PREFIX}{r['doc_id']}", f"{_CONCEPT_PREFIX}{r['concept_name']}")

    kg_result = await driver.execute_query(
        """
        MATCH (kg:KnowledgeGraph)-[:EFFECTIVE]->(c:ConceptNode)
        RETURN kg.id AS kg_id, c.name AS concept_name
        """
    )
    for r in kg_result.records:
        graph.add_edge(f"{_KG_PREFIX}{r['kg_id']}", f"{_CONCEPT_PREFIX}{r['concept_name']}")

    return graph


def _node2vec_walk(graph, start: str, walk_length: int, p: float, q: float) -> list[str]:
    """單次 2nd-order biased random walk（node2vec 核心演算法）。"""
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            nxt = random.choice(neighbors)
        else:
            prev = walk[-2]
            prev_neighbors = set(graph.neighbors(prev))
            weights = []
            for x in neighbors:
                if x == prev:
                    weights.append(1.0 / p)          # 回頭：由 p 控制
                elif x in prev_neighbors:
                    weights.append(1.0)               # 近鄰（BFS 傾向）
                else:
                    weights.append(1.0 / q)           # 遠離（DFS 傾向）：由 q 控制
            nxt = random.choices(neighbors, weights=weights, k=1)[0]
        walk.append(nxt)
    return walk


def generate_node2vec_walks(
    graph, num_walks: int = 10, walk_length: int = 40,
    p: float = 1.0, q: float = 1.0, seed: int | None = None,
) -> list[list[str]]:
    """對圖中每個節點跑 `num_walks` 次隨機遊走，回傳供 skip-gram 訓練用的 walk 語料。"""
    if seed is not None:
        random.seed(seed)
    nodes = list(graph.nodes())
    walks: list[list[str]] = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for n in nodes:
            walks.append(_node2vec_walk(graph, n, walk_length, p, q))
    return walks


def train_concept_vectors(
    walks: list[list[str]], vector_size: int = 384, window: int = 5,
    epochs: int = 5, seed: int = 42,
) -> dict[str, list[float]]:
    """對 walk 語料跑 skip-gram（gensim Word2Vec），回傳 {concept_name: graph_vector}。

    只抽取 `concept:` 前綴的節點向量（Document/KG 節點只是圖結構的橋接點，不需要它們的向量）。
    """
    from gensim.models import Word2Vec

    if not any(walks):
        return {}

    model = Word2Vec(
        walks, vector_size=vector_size, window=window, min_count=1,
        sg=1, workers=1, epochs=epochs, seed=seed,
    )
    result: dict[str, list[float]] = {}
    for node_id in model.wv.index_to_key:
        if node_id.startswith(_CONCEPT_PREFIX):
            name = node_id[len(_CONCEPT_PREFIX):]
            result[name] = model.wv[node_id].tolist()
    return result


async def build_graph_embeddings(
    driver, num_walks: int = 10, walk_length: int = 40,
    p: float = 1.0, q: float = 1.0, vector_size: int = 384, epochs: int = 5,
) -> int:
    """完整流程：抓二分圖 → node2vec 隨機遊走 → skip-gram 訓練 → 寫回 Neo4j。回傳更新的概念數。"""
    from repositories.concept_repo import ConceptRepository

    graph = await build_bipartite_graph(driver)
    if graph.number_of_nodes() == 0:
        logger.info("二分圖無節點（尚無 Document/KG 與 ConceptNode 的 EFFECTIVE 關聯），跳過")
        return 0

    walks = generate_node2vec_walks(graph, num_walks=num_walks, walk_length=walk_length, p=p, q=q)
    vectors = train_concept_vectors(walks, vector_size=vector_size, epochs=epochs)

    if not vectors:
        logger.info("未產生任何 ConceptNode 圖向量，跳過寫回")
        return 0

    await ConceptRepository(driver).set_concept_graph_vectors(vectors)
    logger.info(f"圖結構共嵌入完成，共更新 {len(vectors)} 個 ConceptNode 的 q_vector_graph")
    return len(vectors)
