"""One sandboxed command and the short text the model is shown."""

from __future__ import annotations

from pathlib import Path

from cs329z_hw1 import sandbox
from cs329z_hw1.tokens import count_tokens

# Two of these still fit beside the system prompt inside a 6,000-token context.
SHOW_TOKENS = 800


class TerminalObservation(Exception):
    """A command that did not exit 0. The message is what the model sees."""


def observe(config, cmd: str) -> str:
    """Run ``cmd`` in the session workspace. Raise unless it exits 0."""
    workspace = (
        sandbox.make_workspace(config.workspace)
        if config.workspace is not None
        else sandbox.default_workspace()
    )
    result = sandbox.run_terminal(cmd, workspace=workspace, timeout=config.terminal_timeout)
    shown = _format(workspace, result)
    if result.rejected or result.timed_out or result.exit_code != 0:
        raise TerminalObservation(shown)
    return shown


def _format(workspace: Path, result: sandbox.TerminalResult) -> str:
    parts: list[str] = []
    if result.reason:
        parts.append(result.reason)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if count_tokens(stdout) + count_tokens(stderr) > SHOW_TOKENS:
        if stdout:
            path = _save(workspace, stdout)
            parts.append(
                f"stdout is {len(stdout.splitlines())} lines and was saved to {path.name}. "
                "Read one line with python3 -c or sed. First part:\n" + _head(stdout, SHOW_TOKENS // 2)
            )
        if stderr:
            parts.append("stderr:\n" + _head(stderr, SHOW_TOKENS // 4))
    else:
        parts.append("stdout:\n" + stdout)
        parts.append("stderr:\n" + stderr)
    code = "none" if result.exit_code is None else str(result.exit_code)
    parts.append(f"exit_code: {code}")
    return "\n".join(parts)


def _save(workspace: Path, stdout: str) -> Path:
    number = 1
    while (workspace / f"terminal-output-{number}.txt").exists():
        number += 1
    path = workspace / f"terminal-output-{number}.txt"
    path.write_text(stdout, encoding="utf-8")
    return path


def _head(text: str, limit: int) -> str:
    if count_tokens(text) <= limit:
        return text
    piece = text[: limit * 4]
    while piece and count_tokens(piece) > limit:
        piece = piece[: len(piece) - 4]
    return piece
