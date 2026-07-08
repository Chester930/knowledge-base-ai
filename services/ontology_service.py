"""
本體動態擴充服務（Dynamic Ontology Extension）。

背景：SVO 抽取使用固定的實體類型（`_VALID_TYPES`）與關係類型（`_VALID_REL_TYPES`）清單，
但某些領域文件的真實語意可能超出這個固定清單。當抽取結果連續被驗證模型拒絕
（見 `services/svo_service.py::verify_svo_extraction`）時，由「本體擴充模型」
（`propose_ontology_extension`）提出目前清單缺少的新類別，寫入本檔案管理的
`ontology_extensions.json` 持久化。

依使用者明確指示：
- 預設僅供「該 KG」使用（scope="kg"），不影響其他 KG 的抽取行為。
- 若模型判斷該類別具有跨領域普適性，可標記 scope="global"，直接併入全域清單供所有 KG 共用。
- 不需要人工審核關卡——模型的擴充決定直接生效（與第9節②「動態本體對齊」草案要求人工審核的
  立場不同，此為使用者針對本機制的明確取捨）。

檔案格式：
{
  "global": {"entity_types": [...], "rel_types": [...]},
  "kg": {"<kg_id>": {"entity_types": [...], "rel_types": [...]}}
}
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_EXT_FILE = Path("ontology_extensions.json")
_lock = asyncio.Lock()
_cache: dict | None = None

# 單次擴充的類型數量上限，避免模型單次呼叫失控地新增大量類型
_MAX_NEW_TYPES_PER_CALL = 3


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    data: dict = {}
    if _EXT_FILE.exists():
        try:
            with open(_EXT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"讀取 ontology_extensions.json 失敗，視為空：{e}")
            data = {}
    data.setdefault("global", {"entity_types": [], "rel_types": []})
    data.setdefault("kg", {})
    _cache = data
    return _cache


def _save_locked() -> None:
    """呼叫端須已持有 _lock。"""
    if _cache is None:
        return
    try:
        tmp = _EXT_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
        tmp.replace(_EXT_FILE)
    except Exception as e:
        logger.warning(f"寫入 ontology_extensions.json 失敗：{e}")


def get_extra_entity_types(kg_id: str) -> list[str]:
    """回傳此 KG 有效的擴充實體類型（全域 + 該 KG 專屬，去重）。"""
    data = _load()
    kg_types = data["kg"].get(str(kg_id), {}).get("entity_types", [])
    return list(dict.fromkeys([*data["global"]["entity_types"], *kg_types]))


def get_extra_rel_types(kg_id: str) -> list[str]:
    """回傳此 KG 有效的擴充關係類型（全域 + 該 KG 專屬，去重）。"""
    data = _load()
    kg_types = data["kg"].get(str(kg_id), {}).get("rel_types", [])
    return list(dict.fromkeys([*data["global"]["rel_types"], *kg_types]))


def get_effective_rel_pattern(kg_id: str, base_pattern: str) -> str:
    """組出這個 KG 實際查詢時該用的 Cypher relationship pattern
    （base_pattern | 該 KG 的擴充關係類型），供 BFS/社群偵測等 Cypher 使用。"""
    extra = get_extra_rel_types(kg_id)
    if not extra:
        return base_pattern
    return base_pattern + "|" + "|".join(extra)


async def add_extension(
    kg_id: str,
    entity_types: list[str],
    rel_types: list[str],
    scope: str = "kg",
) -> dict:
    """新增擴充類型並持久化。scope 為 "kg"（預設）或 "global"。
    回傳實際新增的類型（已去重、已套用數量上限）供呼叫端記錄。
    """
    if scope not in ("kg", "global"):
        scope = "kg"

    entity_types = [t.strip() for t in entity_types if t and t.strip()][:_MAX_NEW_TYPES_PER_CALL]
    rel_types = [t.strip().upper() for t in rel_types if t and t.strip()][:_MAX_NEW_TYPES_PER_CALL]

    added = {"entity_types": [], "rel_types": [], "scope": scope}
    async with _lock:
        data = _load()
        bucket = (
            data["global"] if scope == "global"
            else data["kg"].setdefault(str(kg_id), {"entity_types": [], "rel_types": []})
        )
        for t in entity_types:
            if t not in bucket["entity_types"]:
                bucket["entity_types"].append(t)
                added["entity_types"].append(t)
                logger.info(f"[ontology] 新增實體類型 '{t}'（scope={scope}, kg={kg_id}）")
        for t in rel_types:
            if t not in bucket["rel_types"]:
                bucket["rel_types"].append(t)
                added["rel_types"].append(t)
                logger.info(f"[ontology] 新增關係類型 '{t}'（scope={scope}, kg={kg_id}）")
        _save_locked()
    return added


def _reset_cache_for_tests() -> None:
    """僅供測試使用：清空記憶體快取，強制下次讀取重新載入檔案。"""
    global _cache
    _cache = None
