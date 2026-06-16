"""Smoke tests for iteration 8 skill crystallization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.chat_controller import ChatController
from app.core.config import load_config
from app.skills.skill_crystallizer import (
    SkillSuggestion,
    _build_local_fallback,
    _interpret_crystallize_data,
    render_skill_markdown,
    resolve_unique_name,
    validate_skill_content,
)
from app.skills.skill_manager import SkillManager
from app.skills.workflow_observer import (
    WorkflowEntry,
    WorkflowObserver,
    fingerprint_workflow,
)


def _sample_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": f"ACTIVE MEMORY\n\nUSER MESSAGE:\n{text}",
        },
        {"role": "assistant", "content": "done"},
    ]


def test_fingerprint_stable() -> None:
    statements = ["rebuild cuda llama-cpp", "export CUDA paths"]
    assert fingerprint_workflow(statements) == fingerprint_workflow(statements)
    assert fingerprint_workflow(statements) != fingerprint_workflow(["other workflow"])


def test_mark_success_increments(tmp_path) -> None:
    config = MagicMock()
    config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")
    config.skills.success_window_turns = 3
    config.skills.min_successful_repeats = 3

    observer = WorkflowObserver(config)
    messages = _sample_messages("rebuild cuda llama-cpp in wsl")

    first = observer.mark_success(messages, note="cuda rebuild", turn_count=1)
    second = observer.mark_success(messages, note="cuda rebuild", turn_count=2)
    third = observer.mark_success(messages, note="cuda rebuild", turn_count=3)

    assert first.success_count == 1
    assert second.success_count == 2
    assert third.success_count == 3
    assert third.threshold_reached is True
    assert first.fingerprint == third.fingerprint


def test_mark_success_idempotent_same_turn(tmp_path) -> None:
    config = MagicMock()
    config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")
    config.skills.success_window_turns = 3
    config.skills.min_successful_repeats = 3

    observer = WorkflowObserver(config)
    messages = _sample_messages("rebuild cuda llama-cpp")

    first = observer.mark_success(messages, turn_count=5)
    second = observer.mark_success(messages, turn_count=5)

    assert first.success_count == 1
    assert second.success_count == 1


def test_render_and_validate_skill_markdown() -> None:
    content = render_skill_markdown(
        {
            "name": "rebuild_cuda",
            "description": "Rebuild CUDA llama-cpp",
            "trigger": "CUDA build broken",
            "procedure": "1. Activate venv\n2. Build",
            "validation": "ldd shows CUDA",
            "tags": ["cuda"],
        },
        success_count=3,
    )
    assert validate_skill_content(content) == []
    assert "## Trigger" in content
    assert "## Procedure" in content
    assert "## Validation" in content


def test_validate_skill_content_missing_sections() -> None:
    errors = validate_skill_content("just some text")
    assert len(errors) == 3


def test_interpret_crystallize_data() -> None:
    workflow = WorkflowEntry(
        fingerprint="abc123",
        summary="cuda rebuild",
        success_count=3,
        user_statements=["rebuild cuda llama-cpp", "set CMAKE_CUDA_ARCHITECTURES=120"],
    )
    data = {
        "name": "rebuild_cuda",
        "description": "Rebuild CUDA llama-cpp",
        "rationale": "Repeated workflow",
        "trigger": "CUDA build broken",
        "procedure": "1. Activate venv\n2. Build with CUDA",
        "validation": "GPU test passes",
        "tags": ["cuda"],
    }
    suggestion, error = _interpret_crystallize_data(
        data,
        workflow,
        workflow.user_statements,
        "{}",
    )
    assert error is None
    assert suggestion is not None
    assert suggestion.name == "rebuild_cuda"
    assert validate_skill_content(suggestion.proposed_content) == []


def test_local_fallback() -> None:
    fallback = _build_local_fallback(
        ["rebuild cuda llama-cpp", "set gpu arch 120"],
        "cuda rebuild",
        3,
    )
    assert validate_skill_content(fallback.proposed_content) == []
    assert fallback.success_count == 3


def test_resolve_unique_name() -> None:
    existing = {"rebuild_cuda", "rebuild_cuda-2"}
    assert resolve_unique_name("rebuild_cuda", existing) == "rebuild_cuda-3"


def test_controller_mark_and_crystallize_local_fallback(tmp_path) -> None:
    controller = ChatController(load_config())
    controller.config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")
    controller.config.skills.success_window_turns = 3
    controller.config.skills.min_successful_repeats = 3
    controller.workflow_observer = WorkflowObserver(controller.config)
    controller.skill_manager = SkillManager(controller.config)
    controller.skill_manager.active_path = tmp_path / "active"
    controller.skill_manager.archived_path = tmp_path / "archived"
    controller.skill_manager.registry_path = tmp_path / "registry.json"
    controller.skill_manager.ensure_dirs()

    text = "rebuild cuda llama-cpp with ggml cuda enabled"
    controller.messages = _sample_messages(text)

    for turn in (1, 2, 3):
        controller.turn_count = turn
        result = controller.mark_workflow_success("cuda rebuild")
        assert result.mark.success_count == turn

    fallback = _build_local_fallback([text], "cuda rebuild", 3)
    with patch(
        "app.core.chat_controller.generate_skill_suggestion",
        return_value=(fallback, None),
    ):
        crystallize = controller.crystallize_workflow()
    assert crystallize.has_suggestion is True
    assert controller.pending_skill_suggestion is not None

    skill_name = controller.pending_skill_suggestion.name
    controller.set_feature("skills", True)
    assert controller.accept_skill_suggestion() is True
    assert controller.pending_skill_suggestion is None
    assert controller.skill_manager.skill_exists(skill_name)


def test_controller_reject_clears_pending() -> None:
    controller = ChatController(load_config())
    controller.pending_skill_suggestion = SkillSuggestion(
        name="test_skill",
        description="test",
        rationale="test",
        proposed_content="## Trigger\n\nx\n\n## Procedure\n\ny\n\n## Validation\n\nz\n",
        fingerprint="fp1",
        success_count=3,
    )
    controller.reject_skill_suggestion()
    assert controller.pending_skill_suggestion is None


def test_mark_crystallized_blocks_resuggest(tmp_path) -> None:
    config = MagicMock()
    config.skills.workflow_log_path = str(tmp_path / "workflow_log.json")
    config.skills.success_window_turns = 3
    config.skills.min_successful_repeats = 3

    observer = WorkflowObserver(config)
    messages = _sample_messages("rebuild cuda llama-cpp")
    mark = observer.mark_success(messages, turn_count=1)
    observer.mark_crystallized(mark.fingerprint, "rebuild_cuda")

    again = observer.mark_success(messages, turn_count=2)
    assert again.already_crystallized is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
