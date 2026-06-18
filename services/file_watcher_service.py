from __future__ import annotations
import asyncio
import logging
import threading
from pathlib import Path

from services.transcribe_service import SUPPORTED_EXTENSIONS, transcribe_file

logger = logging.getLogger(__name__)

_observer = None        # watchdog Observer（全域單例）
_watch_handles: list = []


# ── Public API ────────────────────────────────────────────────────────────────

def start_watcher(watch_dirs: list[str | Path] | None = None) -> None:
    """
    啟動 File Watcher。watch_dirs 預設監控 workspace/_source/ 和 workspace/kg_*//_source/。
    需在 FastAPI lifespan startup 中呼叫。
    """
    global _observer
    try:
        from watchdog.observers import Observer
    except ImportError:
        logger.warning("watchdog 未安裝，File Watcher 停用。執行 pip install watchdog 可啟用。")
        return

    if _observer is not None and _observer.is_alive():
        logger.warning("File Watcher 已在執行中，略過重複啟動")
        return

    dirs_to_watch = _resolve_watch_dirs(watch_dirs)
    if not dirs_to_watch:
        logger.info("沒有找到需要監控的資料夾，File Watcher 不啟動")
        return

    _observer = Observer()
    handler = _TranscribeHandler()
    for d in dirs_to_watch:
        _observer.schedule(handler, str(d), recursive=False)
        logger.info(f"File Watcher 監控：{d}")

    _observer.start()
    logger.info(f"File Watcher 已啟動（監控 {len(dirs_to_watch)} 個資料夾）")


def stop_watcher() -> None:
    """停止 File Watcher，在 FastAPI lifespan shutdown 中呼叫。"""
    global _observer
    if _observer is not None and _observer.is_alive():
        _observer.stop()
        _observer.join(timeout=5)
        logger.info("File Watcher 已停止")
    _observer = None


def add_watch_dir(path: str | Path) -> bool:
    """執行期動態新增監控資料夾（建立 KG 後呼叫）。回傳是否成功加入。"""
    if _observer is None or not _observer.is_alive():
        logger.warning("File Watcher 未執行，無法動態新增監控目錄")
        return False
    try:
        from watchdog.observers import Observer
    except ImportError:
        return False

    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    _observer.schedule(_TranscribeHandler(), str(d), recursive=False)
    logger.info(f"動態新增監控：{d}")
    return True


def get_status() -> dict:
    """回傳目前 Watcher 狀態，供 API 查詢。"""
    if _observer is None:
        return {"running": False, "watched_dirs": []}
    watched = [str(w.path) for w in _observer.emitters] if _observer.is_alive() else []
    return {"running": _observer.is_alive(), "watched_dirs": watched}


# ── 事件處理器 ────────────────────────────────────────────────────────────────

class _TranscribeHandler:
    """watchdog event handler — 偵測新增或移入的支援格式檔案，自動觸發轉譯。"""

    # watchdog 會傳 FileCreatedEvent / FileMovedEvent；使用 dispatch 介面。
    def __init__(self):
        try:
            from watchdog.events import FileSystemEventHandler
            self._base = FileSystemEventHandler
        except ImportError:
            self._base = object

    def dispatch(self, event):
        try:
            from watchdog.events import (
                FileCreatedEvent, FileMovedEvent, EVENT_TYPE_CREATED, EVENT_TYPE_MOVED
            )
        except ImportError:
            return

        if event.is_directory:
            return

        src = None
        if isinstance(event, FileCreatedEvent):
            src = Path(event.src_path)
        elif isinstance(event, FileMovedEvent):
            src = Path(event.dest_path)

        if src is None:
            return
        if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        if src.name.startswith("."):
            return

        logger.info(f"偵測到新檔案：{src}")
        # 避免 watchdog 執行緒阻塞，用 threading 背景執行轉譯
        threading.Thread(target=self._run_transcribe, args=(src,), daemon=True).start()

    def _run_transcribe(self, src: Path) -> None:
        result = transcribe_file(src)
        if result.success:
            logger.info(f"自動轉譯成功：{src.name} → {result.txt_path}")
        else:
            logger.warning(f"自動轉譯失敗：{src.name}：{result.error}")


# ── 內部輔助 ──────────────────────────────────────────────────────────────────

def _resolve_watch_dirs(extra: list[str | Path] | None) -> list[Path]:
    """
    預設監控策略：
    1. workspace/_source/（全域收件匣的原始檔備份目錄）
    2. workspace/kg_*//_source/（各 KG 自己的原始檔資料夾）
    3. extra 中額外指定的目錄
    """
    from core.config import settings
    workspace = Path(settings.workspace_dir)
    dirs: list[Path] = []

    # 全域 _source/
    global_src = workspace / "_source"
    global_src.mkdir(parents=True, exist_ok=True)
    dirs.append(global_src)

    # KG _source/ 資料夾
    for kg_dir in workspace.glob("kg_*"):
        if kg_dir.is_dir():
            src_dir = kg_dir / "_source"
            src_dir.mkdir(parents=True, exist_ok=True)
            dirs.append(src_dir)

    # 額外指定
    if extra:
        for d in extra:
            p = Path(d)
            if p.is_dir():
                dirs.append(p)
            else:
                logger.warning(f"指定的監控目錄不存在，略過：{d}")

    return dirs
