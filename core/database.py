from __future__ import annotations
import asyncio
import logging

from neo4j import AsyncGraphDatabase
from core.config import settings

_driver = None
logger = logging.getLogger(__name__)


async def connect():
    global _driver
    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    await _driver.verify_connectivity()


async def disconnect():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


def get_driver():
    if _driver is None:
        raise RuntimeError("資料庫未連線")
    return _driver


# ── 多資料庫管理（需要 Neo4j Enterprise）─────────────────────────────────────

async def create_kg_database(db_name: str) -> None:
    """在 system 資料庫建立新的 KG 專用資料庫，並等待上線。"""
    await _driver.execute_query(
        f"CREATE DATABASE `{db_name}` IF NOT EXISTS",
        database_="system",
    )
    # 等待資料庫上線（最多 20 秒）
    for _ in range(40):
        try:
            await _driver.execute_query("RETURN 1", database_=db_name)
            logger.info(f"KG 資料庫已上線：{db_name}")
            return
        except Exception:
            await asyncio.sleep(0.5)
    logger.warning(f"KG 資料庫啟動逾時：{db_name}（仍可能在背景啟動）")


async def drop_kg_database(db_name: str) -> None:
    """刪除 KG 專用資料庫並清除所有資料。"""
    await _driver.execute_query(
        f"DROP DATABASE `{db_name}` IF EXISTS DESTROY DATA",
        database_="system",
    )
    logger.info(f"KG 資料庫已刪除：{db_name}")


async def list_kg_databases() -> list[str]:
    """列出所有 kg 開頭的 KG 專用資料庫。"""
    result = await _driver.execute_query(
        "SHOW DATABASES YIELD name, currentStatus RETURN name, currentStatus",
        database_="system",
    )
    return [
        r["name"] for r in result.records
        if r["name"].startswith("kg") and r["currentStatus"] == "online"
    ]
