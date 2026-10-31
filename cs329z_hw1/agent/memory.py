"""File-backed facts for Problem 12 (memory).

Each fact is one JSON line in ``<memory_dir>/facts.jsonl``. A later session
reads the same file. Nothing is kept only in this process.
"""

from __future__ import annotations

import json
from pathlib import Path

from cs329z_hw1.tokenizer import tokenize

FACTS_NAME = "facts.jsonl"
EMPTY = "Nothing is stored yet."


def save_fact(memory_dir: Path, fact: str) -> str:
    """Append ``fact`` exactly as given and report that it was saved."""
    directory = Path(memory_dir)
    directory.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"fact": fact}, ensure_ascii=False)
    with (directory / FACTS_NAME).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return "Saved."


def recall(memory_dir: Path, query: str) -> str:
    """Return every stored fact, those that share a query token first.

    An empty store returns a non-empty sentence. A query that shares no
    tokens still returns the facts, after any that did match.
    """
    facts = _load(memory_dir)
    if not facts:
        return EMPTY
    wanted = set(tokenize(query))
    matched = [fact for fact in facts if wanted and wanted.intersection(tokenize(fact))]
    rest = [fact for fact in facts if fact not in matched]
    return "\n".join(matched + rest)


def _load(memory_dir: Path) -> list[str]:
    path = Path(memory_dir) / FACTS_NAME
    if not path.is_file():
        return []
    facts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        fact = item.get("fact") if isinstance(item, dict) else None
        if isinstance(fact, str):
            facts.append(fact)
    return facts
