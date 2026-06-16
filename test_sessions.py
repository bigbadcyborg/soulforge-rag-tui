"""Smoke tests for iteration 11 session persistence."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.core.chat_controller import ChatController
from app.core.config import load_config
from app.sessions.session_manager import (
    SavedSession,
    SessionManager,
    filter_conversation_messages,
    title_from_messages,
)


def _make_config(tmp_path):
    config = MagicMock()
    config.sessions.store_path = str(tmp_path / "sessions")
    config.sessions.max_saved_sessions = 50
    return config


def test_filter_conversation_messages_excludes_system() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    filtered = filter_conversation_messages(messages)
    assert len(filtered) == 2
    assert filtered[0]["role"] == "user"


def test_title_from_messages_uses_first_user_line() -> None:
    messages = [
        {"role": "user", "content": "USER MESSAGE:\nFix CUDA setup script"},
        {"role": "assistant", "content": "Sure"},
    ]
    assert title_from_messages(messages) == "Fix CUDA setup script"


def test_save_and_load_roundtrip(tmp_path) -> None:
    mgr = SessionManager(_make_config(tmp_path))
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    saved = mgr.save_session(messages, title="Test chat", turn_count=1)
    assert saved is not None
    loaded = mgr.get_session(saved.id)
    assert loaded is not None
    assert loaded.title == "Test chat"
    assert loaded.turn_count == 1
    assert len(loaded.messages) == 2


def test_list_sessions_sorted_by_updated_at(tmp_path) -> None:
    mgr = SessionManager(_make_config(tmp_path))
    first = mgr.save_session([{"role": "user", "content": "a"}], title="First")
    second = mgr.save_session([{"role": "user", "content": "b"}], title="Second")
    assert first is not None and second is not None
    sessions = mgr.list_sessions()
    assert len(sessions) == 2
    assert sessions[0].updated_at >= sessions[1].updated_at


def test_delete_session(tmp_path) -> None:
    mgr = SessionManager(_make_config(tmp_path))
    saved = mgr.save_session([{"role": "user", "content": "bye"}], title="Gone")
    assert saved is not None
    assert mgr.delete_session(saved.id) is True
    assert mgr.get_session(saved.id) is None


def test_corrupt_session_file_skipped(tmp_path) -> None:
    store = tmp_path / "sessions"
    store.mkdir()
    (store / "bad.json").write_text("{not json", encoding="utf-8")
    mgr = SessionManager(_make_config(tmp_path))
    assert mgr.list_sessions() == []


def test_get_session_prefix_match(tmp_path) -> None:
    mgr = SessionManager(_make_config(tmp_path))
    saved = mgr.save_session([{"role": "user", "content": "x"}], title="Prefix")
    assert saved is not None
    assert mgr.get_session(saved.id[:10]) is not None


def test_save_empty_conversation_returns_none(tmp_path) -> None:
    mgr = SessionManager(_make_config(tmp_path))
    assert mgr.save_session([{"role": "system", "content": "only"}]) is None


def test_update_summary(tmp_path) -> None:
    mgr = SessionManager(_make_config(tmp_path))
    saved = mgr.save_session([{"role": "user", "content": "work"}], title="Work")
    assert saved is not None
    assert mgr.update_summary(saved.id, "Did some work") is True
    loaded = mgr.get_session(saved.id)
    assert loaded is not None
    assert loaded.summary == "Did some work"


def _make_controller(tmp_path) -> ChatController:
    config = load_config()
    config.sessions.store_path = str(tmp_path / "sessions")
    controller = ChatController(config)
    controller.session_manager = SessionManager(config)
    controller.messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "USER MESSAGE:\nHello world"},
        {"role": "assistant", "content": "Hi"},
    ]
    controller.turn_count = 3
    return controller


def test_controller_save_and_load(tmp_path) -> None:
    controller = _make_controller(tmp_path)
    save_outcome = controller.save_session_direct("My chat")
    assert save_outcome.success is True

    controller.messages.append({"role": "user", "content": "more"})
    load_outcome = controller.load_session_direct(save_outcome.session_id or "")
    assert load_outcome.success is True
    assert controller.turn_count == 3
    assert len(controller.get_conversation_messages()) == 2


def test_message_for_display_strips_wrapper() -> None:
    controller = ChatController(load_config())
    text = controller.message_for_display(
        {"role": "user", "content": "USER MESSAGE:\nActual question"}
    )
    assert text == "Actual question"


@patch("app.core.chat_controller.generate_summary")
def test_run_session_summary_writes_session_md(mock_summary, tmp_path) -> None:
    config = load_config()
    config.sessions.store_path = str(tmp_path / "sessions")
    config.memory.session_file = str(tmp_path / "session.md")
    controller = ChatController(config)
    controller.session_manager = SessionManager(config)
    controller.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    mock_summary.return_value = "Summary bullet points"
    result = controller.run_session_summary()
    assert result.success is True
    assert "session.md" in result.message.lower() or "summary" in result.message.lower()
    snapshot = controller.memory_manager.load()
    assert "Summary bullet points" in snapshot.session
