from __future__ import annotations
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import services.file_watcher_service as fw


# ── _resolve_watch_dirs ───────────────────────────────────────────────────────

class TestResolveWatchDirs:
    def test_creates_global_source_dir(self, tmp_path):
        with patch("core.config.settings") as mock_settings:
            mock_settings.workspace_dir = str(tmp_path)
            dirs = fw._resolve_watch_dirs(None)
        assert (tmp_path / "_source") in dirs
        assert (tmp_path / "_source").exists()

    def test_includes_kg_source_dirs(self, tmp_path):
        (tmp_path / "kg_test").mkdir()
        with patch("core.config.settings") as mock_settings:
            mock_settings.workspace_dir = str(tmp_path)
            dirs = fw._resolve_watch_dirs(None)
        assert (tmp_path / "kg_test" / "_source") in dirs

    def test_ignores_non_kg_prefixed_dirs(self, tmp_path):
        (tmp_path / "other_folder").mkdir()
        with patch("core.config.settings") as mock_settings:
            mock_settings.workspace_dir = str(tmp_path)
            dirs = fw._resolve_watch_dirs(None)
        assert not any("other_folder" in str(d) for d in dirs)

    def test_extra_dirs_included_when_exist(self, tmp_path):
        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()
        with patch("core.config.settings") as mock_settings:
            mock_settings.workspace_dir = str(tmp_path)
            dirs = fw._resolve_watch_dirs([str(extra_dir)])
        assert extra_dir in dirs

    def test_extra_nonexistent_dir_skipped(self, tmp_path):
        with patch("core.config.settings") as mock_settings:
            mock_settings.workspace_dir = str(tmp_path)
            dirs = fw._resolve_watch_dirs([str(tmp_path / "ghost")])
        assert not any("ghost" in str(d) for d in dirs)


# ── start_watcher / stop_watcher without watchdog installed ─────────────────

class TestStartWatcherWithoutWatchdog:
    def test_logs_warning_and_noops_when_watchdog_missing(self, monkeypatch):
        monkeypatch.setattr(fw, "_observer", None)
        with patch.dict(sys.modules, {"watchdog.observers": None, "watchdog": None}):
            fw.start_watcher([])
        assert fw._observer is None


class TestStopWatcher:
    def test_noop_when_no_observer(self, monkeypatch):
        monkeypatch.setattr(fw, "_observer", None)
        fw.stop_watcher()
        assert fw._observer is None

    def test_stops_running_observer(self, monkeypatch):
        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = True
        monkeypatch.setattr(fw, "_observer", mock_observer)
        fw.stop_watcher()
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()
        assert fw._observer is None


# ── add_watch_dir ─────────────────────────────────────────────────────────────

class TestAddWatchDir:
    def test_returns_false_when_observer_not_running(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fw, "_observer", None)
        assert fw.add_watch_dir(tmp_path) is False

    def test_returns_false_when_observer_not_alive(self, monkeypatch, tmp_path):
        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = False
        monkeypatch.setattr(fw, "_observer", mock_observer)
        assert fw.add_watch_dir(tmp_path) is False


# ── get_status ────────────────────────────────────────────────────────────────

class TestGetStatus:
    def test_returns_not_running_when_no_observer(self, monkeypatch):
        monkeypatch.setattr(fw, "_observer", None)
        status = fw.get_status()
        assert status == {"running": False, "watched_dirs": []}

    def test_returns_running_true_with_watched_dirs(self, monkeypatch):
        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = True
        emitter = MagicMock()
        emitter.path = "/some/dir"
        mock_observer.emitters = [emitter]
        monkeypatch.setattr(fw, "_observer", mock_observer)
        status = fw.get_status()
        assert status["running"] is True
        assert status["watched_dirs"] == ["/some/dir"]

    def test_returns_empty_watched_dirs_when_not_alive(self, monkeypatch):
        mock_observer = MagicMock()
        mock_observer.is_alive.return_value = False
        monkeypatch.setattr(fw, "_observer", mock_observer)
        status = fw.get_status()
        assert status["running"] is False
        assert status["watched_dirs"] == []


# ── _TranscribeHandler._wait_until_stable / _run_transcribe ──────────────────

class TestWaitUntilStable:
    def test_missing_file_returns_false(self, tmp_path):
        handler = fw._TranscribeHandler()
        assert handler._wait_until_stable(tmp_path / "ghost.txt", timeout=1.0) is False

    def test_stable_file_returns_true(self, tmp_path):
        f = tmp_path / "stable.txt"
        f.write_text("內容")
        handler = fw._TranscribeHandler()
        assert handler._wait_until_stable(f, timeout=5.0) is True


class TestRunTranscribe:
    def test_unstable_file_skips_transcription(self, tmp_path):
        handler = fw._TranscribeHandler()
        with patch.object(handler, "_wait_until_stable", return_value=False), \
             patch("services.file_watcher_service.transcribe_file") as mock_transcribe:
            handler._run_transcribe(tmp_path / "unstable.mp3")
        mock_transcribe.assert_not_called()

    def test_stable_file_triggers_transcription(self, tmp_path):
        handler = fw._TranscribeHandler()
        mock_result = MagicMock(success=True, txt_path="out.txt")
        with patch.object(handler, "_wait_until_stable", return_value=True), \
             patch("services.file_watcher_service.transcribe_file", return_value=mock_result) as mock_transcribe:
            handler._run_transcribe(tmp_path / "a.txt")
        mock_transcribe.assert_called_once()

    def test_transcription_failure_does_not_raise(self, tmp_path):
        handler = fw._TranscribeHandler()
        mock_result = MagicMock(success=False, error="失敗原因")
        with patch.object(handler, "_wait_until_stable", return_value=True), \
             patch("services.file_watcher_service.transcribe_file", return_value=mock_result):
            handler._run_transcribe(tmp_path / "a.txt")  # 不應拋例外


# ── _TranscribeHandler.dispatch ───────────────────────────────────────────────

def _install_fake_watchdog_events():
    """
    dispatch() 內部使用 `from watchdog.events import FileCreatedEvent, FileMovedEvent`，
    測試環境未安裝 watchdog，用假模組注入 sys.modules 讓分派邏輯可被驗證。
    """
    events_mod = types.ModuleType("watchdog.events")

    class FileCreatedEvent:
        def __init__(self, src_path, is_directory=False):
            self.src_path = src_path
            self.is_directory = is_directory

    class FileMovedEvent:
        def __init__(self, dest_path, is_directory=False):
            self.dest_path = dest_path
            self.is_directory = is_directory

    events_mod.FileCreatedEvent = FileCreatedEvent
    events_mod.FileMovedEvent = FileMovedEvent
    watchdog_mod = types.ModuleType("watchdog")
    watchdog_mod.events = events_mod
    return watchdog_mod, events_mod


class TestDispatch:
    def test_directory_event_ignored(self, tmp_path):
        watchdog_mod, events_mod = _install_fake_watchdog_events()
        handler = fw._TranscribeHandler()
        event = events_mod.FileCreatedEvent(str(tmp_path / "dir"), is_directory=True)
        with patch.dict(sys.modules, {"watchdog": watchdog_mod, "watchdog.events": events_mod}), \
             patch("threading.Thread") as mock_thread:
            handler.dispatch(event)
        mock_thread.assert_not_called()

    def test_unsupported_extension_ignored(self, tmp_path):
        watchdog_mod, events_mod = _install_fake_watchdog_events()
        handler = fw._TranscribeHandler()
        event = events_mod.FileCreatedEvent(str(tmp_path / "a.xyz"))
        with patch.dict(sys.modules, {"watchdog": watchdog_mod, "watchdog.events": events_mod}), \
             patch("threading.Thread") as mock_thread:
            handler.dispatch(event)
        mock_thread.assert_not_called()

    def test_dotfile_ignored(self, tmp_path):
        watchdog_mod, events_mod = _install_fake_watchdog_events()
        handler = fw._TranscribeHandler()
        event = events_mod.FileCreatedEvent(str(tmp_path / ".hidden.txt"))
        with patch.dict(sys.modules, {"watchdog": watchdog_mod, "watchdog.events": events_mod}), \
             patch("threading.Thread") as mock_thread:
            handler.dispatch(event)
        mock_thread.assert_not_called()

    def test_supported_created_file_starts_thread(self, tmp_path):
        watchdog_mod, events_mod = _install_fake_watchdog_events()
        handler = fw._TranscribeHandler()
        event = events_mod.FileCreatedEvent(str(tmp_path / "a.txt"))
        with patch.dict(sys.modules, {"watchdog": watchdog_mod, "watchdog.events": events_mod}), \
             patch("threading.Thread") as mock_thread:
            handler.dispatch(event)
        mock_thread.assert_called_once()
        assert mock_thread.call_args.kwargs["daemon"] is True

    def test_supported_moved_file_uses_dest_path(self, tmp_path):
        watchdog_mod, events_mod = _install_fake_watchdog_events()
        handler = fw._TranscribeHandler()
        event = events_mod.FileMovedEvent(str(tmp_path / "b.pdf"))
        with patch.dict(sys.modules, {"watchdog": watchdog_mod, "watchdog.events": events_mod}), \
             patch("threading.Thread") as mock_thread:
            handler.dispatch(event)
        mock_thread.assert_called_once()
        _, kwargs = mock_thread.call_args
        assert kwargs["args"][0] == Path(str(tmp_path / "b.pdf"))
