from neo4j import AsyncGraphDatabase
from core.config import settings

_driver = None


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
