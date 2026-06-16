"""Shell command tool handler."""

from __future__ import annotations

import subprocess

from app.core.config import PROJECT_ROOT, AppConfig
from app.tools.permissions import check_shell_command


def run_command(config: AppConfig, args: dict) -> str:
    command = str(args.get("command", "")).strip()
    argv = check_shell_command(config, command)
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
        shell=False,
    )
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")
    parts.append(f"[exit code {result.returncode}]")
    return "\n".join(parts)
