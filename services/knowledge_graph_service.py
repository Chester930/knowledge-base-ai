from __future__ import annotations
import asyncio
import logging
import re
from pathlib import Path
from uuid import UUID, uuid4

from core.config import settings
from core.database import get_driver, create_kg_database, drop_kg_database
from models.knowledge_graph import KnowledgeGraph, KnowledgeGraphDetail
from repositories.knowledge_graph_repo import KnowledgeGraphRepository
from repositories.concept_repo import ConceptRepository
from services.file_watcher_service import add_watch_dir

logger = logging.getLogger(__name__)


def _kg_folder(kg_name: str) -> Path:
    """KG 資料夾命名：workspace/kg_{slug}/"""
    slug = kg_name.lower().replace(" ", "_").replace("/", "_")
    return Path(settings.workspace_dir) / f"kg_{slug}"


def _make_db_name(kg_name: str) -> str:
    """
    生成合法的 Neo4j 資料庫名稱。
    規則：字母開頭，只含字母/數字/底線/連字號，加短 UUID 避免碰撞。
    範例：「強化學習」→ 'kg_a3f2b1c9'
    """
    safe = re.sub(r'[^a-z0-9]', '', kg_name.lower())[:8]
    suffix = uuid4().hex[:8]
    return f"kg{safe}{suffix}" if safe else f"kg{suffix}"


async def create_kg(
    name: str,
    description: str = "",
    owner_id: str = "default",
    is_public: bool = True,
) -> KnowledgeGraph:
    """
    建立 KnowledgeGraph 節點：
    1. 在 workspace 建立資料夾
    2. 在 Neo4j 建立專用資料庫（Enterprise 功能）
    3. 在主資料庫建立 KG 節點
    4. 建立 Entity 索引於新資料庫
    """
    kg_repo = KnowledgeGraphRepository(get_driver())

    existing = await kg_repo.get_by_name(name, owner_id)
    if existing:
        raise ValueError(f"KG 名稱已存在：{name}")

    # workspace 資料夾
    folder = _kg_folder(name)
    source_dir = folder / "_source"
    text_dir = folder / "_text"
    source_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    # Neo4j 專用資料庫
    db_name = _make_db_name(name)
    try:
        await create_kg_database(db_name)
        # 建立 Entity 索引與全文索引於新資料庫
        await get_driver().execute_query(
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            database_=db_name,
        )
        try:
            await get_driver().execute_query(
                "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name]",
                database_=db_name,
            )
        except Exception as _ft_err:
            logger.debug(f"KG 全文索引建立跳過：{_ft_err}")
        logger.info(f"KG 資料庫已建立：{db_name}")
    except Exception as e:
        logger.warning(f"Neo4j 資料庫建立失敗（可能非 Enterprise 版）：{e}，改用主資料庫模式")
        db_name = ""

    kg = await kg_repo.create(
        name=name,
        description=description,
        folder_path=str(folder),
        owner_id=owner_id,
        is_public=is_public,
        db_name=db_name,
    )

    add_watch_dir(source_dir)
    logger.info(f"KG 建立完成：{name}（db={db_name or '主資料庫'}）")
    return kg


async def delete_kg(kg_id: UUID, delete_files: bool = False) -> bool:
    """
    刪除 KG 節點，並同步刪除其專用 Neo4j 資料庫。
    delete_files=True 時同時刪除 workspace 資料夾。
    """
    kg_repo = KnowledgeGraphRepository(get_driver())
    kg = await kg_repo.get_by_id(kg_id)
    if kg is None:
        return False

    # 刪除 Neo4j 專用資料庫
    if kg.db_name:
        try:
            await drop_kg_database(kg.db_name)
        except Exception as e:
            logger.warning(f"刪除 KG 資料庫失敗：{e}")

    ok = await kg_repo.delete(kg_id)
    if ok:
        from services.chunk_store import get_chunk_store
        get_chunk_store().delete_kg(kg_id)
    if ok and delete_files:
        import shutil
        folder = Path(kg.folder_path)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            logger.info(f"已刪除 KG 資料夾：{folder}")
    return ok


async def auto_cluster_kgs(
    min_docs: int = 2,
    max_docs_preview: int = 300,
) -> list[dict]:
    """
    分群來源（優先順序）：
    1. _staging/ 下的 .txt 檔（未匯入的原始文件）
    2. Neo4j 中尚未關聯任何 KG 的孤立 Document 節點
    兩種來源合併後交給 LLM 分群命名，回傳方案但不建立 KG。
    """
    from core.providers.factory import get_llm_provider
    from repositories.document_repo import DocumentRepository
    import json, re

    # ── 來源 1：_staging/ .txt 檔 ──
    staging = Path(settings.workspace_dir) / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    staging_files = sorted(staging.glob("*.txt"))

    staging_items: list[dict] = []
    for f in staging_files:
        try:
            preview = f.read_text(encoding="utf-8")[:max_docs_preview].replace("\n", " ")
        except Exception:
            preview = "(無法讀取)"
        staging_items.append({"key": f.name, "title": f.stem, "preview": preview, "source": "staging"})

    # ── 來源 2：Neo4j 孤立文件 ──
    doc_repo = DocumentRepository(get_driver())
    orphan_docs = await doc_repo.get_orphan_documents(preview_chars=max_docs_preview)
    db_items: list[dict] = [
        {"key": d["id"], "title": d["title"], "preview": d["preview"], "source": "db"}
        for d in orphan_docs
    ]

    all_items = staging_items + db_items
    if len(all_items) < min_docs:
        raise ValueError(
            f"只找到 {len(all_items)} 份可分群文件（暫存區 {len(staging_items)} 份 + "
            f"未分配文件 {len(db_items)} 份），至少需要 {min_docs} 份"
        )

    # 用短索引取代 UUID/filename，LLM 容易準確回傳
    idx_map: dict[str, dict] = {}  # "d1" -> item
    for i, item in enumerate(all_items):
        idx_map[f"d{i+1}"] = item

    doc_list_text = "\n".join(
        f"- id={idx}\n  標題：{item['title']}\n  摘要：{item['preview']}"
        for idx, item in idx_map.items()
    )

    prompt = (
        "你是知識庫架構師。以下是一批文件的標題與摘要，請將它們分群，"
        "為每個群組命名一個簡潔有意義的知識圖譜名稱（2–8個中文字或英文單詞），"
        "並附上一句話描述。相同系列或主題的文件應放在同一群。\n\n"
        f"文件清單：\n{doc_list_text}\n\n"
        "請以純 JSON 格式回答（不要有任何 markdown 標記），格式如下：\n"
        '[\n  {"name": "KG名稱", "description": "描述", "ids": ["d1", "d2"]}\n]\n'
        "ids 的值必須完全對應上方文件清單的 id 欄位（d1, d2...）。只輸出 JSON 陣列，不要有其他文字。"
    )

    full = ""
    async for token in get_llm_provider().stream(prompt):
        full += token

    match = re.search(r"\[[\s\S]*\]", full)
    if not match:
        raise ValueError(f"LLM 未回傳有效 JSON，原始回應：{full[:300]}")

    clusters: list[dict] = json.loads(match.group())

    result = []
    assigned_idxs: set[str] = set()

    for c in clusters:
        name = str(c.get("name", "")).strip()
        description = str(c.get("description", "")).strip()
        raw_ids = c.get("ids", [])
        valid_items = [idx_map[idx] for idx in raw_ids if idx in idx_map]
        if name and valid_items:
            assigned_idxs.update(idx for idx in raw_ids if idx in idx_map)
            result.append({
                "name": name,
                "description": description,
                "files": [i["key"] for i in valid_items if i["source"] == "staging"],
                "doc_ids": [i["key"] for i in valid_items if i["source"] == "db"],
            })

    # 未分配的放入「其他」
    leftover_items = [item for idx, item in idx_map.items() if idx not in assigned_idxs]
    if leftover_items:
        result.append({
            "name": "其他",
            "description": "未自動分類的文件",
            "files": [i["key"] for i in leftover_items if i["source"] == "staging"],
            "doc_ids": [i["key"] for i in leftover_items if i["source"] == "db"],
        })

    logger.info(f"自動分群完成：{len(result)} 個 KG，共 {len(all_items)} 份文件")
    return result


async def confirm_auto_cluster(clusters: list[dict]) -> list[dict]:
    """
    根據使用者確認的分群方案，建立 KG 並分配文件。
    clusters 格式：
      {"name": str, "description": str,
       "files": [staging_filename, ...],    # _staging/ 中的 .txt
       "doc_ids": [neo4j_doc_id, ...]}      # 已在 DB 的孤立文件
    """
    from services.classify_service import assign_document_to_kg
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    from uuid import UUID as _UUID

    kg_repo = KnowledgeGraphRepository(get_driver())

    results = []
    for cluster in clusters:
        name = cluster.get("name", "").strip()
        description = cluster.get("description", "").strip()
        files = cluster.get("files", [])
        doc_ids = cluster.get("doc_ids", [])
        if not name or (not files and not doc_ids):
            continue
        try:
            kg = await create_kg(name=name, description=description)
            assigned_count = 0
            errors = []

            # 1. 處理 staging 檔案
            for filename in files:
                try:
                    await assign_document_to_kg(filename, kg.id)
                    assigned_count += 1
                except FileNotFoundError:
                    errors.append(f"{filename}（找不到檔案）")
                except Exception as e:
                    errors.append(f"{filename}（{e}）")

            # 2. 處理已在 DB 的孤立文件 → 直接建立 KG↔Doc 關聯並刷新概念
            for doc_id_str in doc_ids:
                try:
                    await kg_repo.add_document(kg.id, _UUID(str(doc_id_str)))
                    assigned_count += 1
                except Exception as e:
                    errors.append(f"{doc_id_str}（{e}）")

            # 刷新 KG 路由層概念（db 文件不經 assign_document_to_kg，需手動刷新）
            if doc_ids:
                await refresh_kg_concepts(kg.id)

            results.append({
                "kg_id": str(kg.id),
                "name": name,
                "db_name": kg.db_name,
                "assigned": assigned_count,
                "errors": errors,
            })
        except ValueError as e:
            results.append({"name": name, "error": str(e)})
        except Exception as e:
            logger.exception(f"建立 KG 失敗：{name}")
            results.append({"name": name, "error": str(e)})

    return results


async def refresh_kg_concepts(kg_id: UUID, text: str | None = None) -> None:
    """
    重新計算 KG 路由層概念。
    來源優先順序：
      1. 此 KG 下所有 Document 的現有 EFFECTIVE ConceptNode
      2. fallback：此 KG SVO 圖中出現頻率最高的 Entity name（top-50）
      3. 可選：傳入 text 時，額外用 LLM 提取概念
    """
    from core.providers.factory import get_embedding_provider
    from services.concept_engine import extract_concepts
    from core.constants import INTEREST_INIT, PROFESSIONAL_INIT

    embedding = get_embedding_provider()
    concept_repo = ConceptRepository(get_driver())
    kg_repo = KnowledgeGraphRepository(get_driver())

    # ── 來源 1：Document EFFECTIVE ConceptNode ────────────────────────────────
    doc_concepts_map = await concept_repo.get_all_documents_concepts()
    kg_docs = await kg_repo.get_documents(kg_id)
    kg_doc_ids = {d["id"] for d in kg_docs}

    concept_names: set[str] = set()
    for doc_id_str, concepts in doc_concepts_map.items():
        if str(doc_id_str) in kg_doc_ids or doc_id_str in kg_doc_ids:
            for c in concepts:
                concept_names.add(c["name"])

    # ── 來源 2 fallback：從 SVO Entity 取高頻名詞 ────────────────────────────
    if not concept_names:
        logger.info(f"KG {kg_id} 無 Document 概念，從 Entity 節點取 top-50 作 fallback")
        db_name = await kg_repo.get_db_name(kg_id)
        driver = get_driver()
        db_kw = {"database_": db_name} if db_name else {}
        kg_filter = "" if db_name else "{kg_id: $kg_id}"
        try:
            result = await driver.execute_query(
                f"""
                MATCH (e:Entity {kg_filter})
                WHERE e.type IN ['概念', '技術', '算法', '方法', '框架', '模型', '工具']
                WITH e.name AS name,
                     size([(e)-[:IS_A|PART_OF|CONTAINS|INSTANCE_OF|CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|CONTRASTS|SIMILAR_TO|OUTPERFORMS|DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|PRECEDES|FOLLOWS|CO_OCCURS|INPUTS|TRANSFORMS|CREATED_BY|SOLVES|RELATED_TO]-() | 1]) AS degree
                ORDER BY degree DESC LIMIT 50
                RETURN name
                """,
                kg_id=str(kg_id), **db_kw,
            )
            concept_names = {r["name"] for r in result.records if r["name"]}
        except Exception as e:
            logger.warning(f"Entity fallback 失敗：{e}")

    # ── 來源 3：text LLM 提取 ─────────────────────────────────────────────────
    if text:
        extra = await extract_concepts(text)
        concept_names.update(extra)

    for name in concept_names:
        vec = await asyncio.to_thread(embedding.encode, name)
        await concept_repo.get_or_create(name, "general", vec)
        await concept_repo.init_kg_concept(kg_id, name, INTEREST_INIT, PROFESSIONAL_INIT)

    await concept_repo.sync_kg_effective(kg_id)
    await kg_repo.refresh_counts(kg_id)
    logger.info(f"KG {kg_id} 路由層概念刷新完成，共 {len(concept_names)} 個")
