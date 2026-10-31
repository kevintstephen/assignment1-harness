"""Context compaction for Problem 11.

`fit` shortens the message list sent to the model until it is within the
context budget. The transcript is a separate log and is never passed here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from cs329z_hw1.tokens import CHARS_PER_TOKEN, MESSAGE_OVERHEAD, count_message_tokens, count_tokens

# Fixed first line so a later summary replaces this message instead of stacking another one.
NOTES_HEADER = "Notes from earlier in this conversation:"

_SENTENCE_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+")
_FACT = re.compile(r"\d[\d,]*(?:\.\d+)?|\b[A-Z][a-z]{2,}\b")

_AUX_SYSTEM = (
    "You update notes that replace messages leaving a conversation window. "
    "Copy every number, dollar amount, date, city, and constraint exactly as written. "
    "Do not round, abbreviate, or paraphrase them. Put those exact facts first, "
    "then anything else worth keeping, in a few sentences."
)


def fit(
    messages: Sequence[dict],
    budget: int,
    current_user: str,
    aux_lm: Callable[[list[dict]], str],
    user_requests: Sequence[str] = (),
) -> list[dict]:
    """Return a copy of `messages` that fits in `budget` tokens.

    The system prompt, the notes message, and the current user message stay.
    A previous user request is summarized with `aux_lm` before it is dropped.
    Tool results and assistant replies are dropped with no model call. A single
    message that is still too large is replaced by a head that fits in the
    room left over.
    """
    fitted = [{"role": m["role"], "content": m["content"]} for m in messages]
    if count_message_tokens(fitted) <= budget:
        return fitted

    requests = set(user_requests)
    for _ in range(len(fitted) + len(requests) + 32):
        if count_message_tokens(fitted) <= budget:
            return fitted
        idx = _oldest_unprotected(fitted, current_user)
        if idx is None:
            if not _shrink_largest(fitted, budget, current_user):
                break
            continue
        target = fitted[idx]
        # A message bigger than the whole budget cannot be dropped usefully:
        # keep a head that fits beside everything else we are still holding.
        if count_message_tokens([target]) > budget:
            if not _shrink(fitted, idx, budget):
                _drop(fitted, target)
            continue
        if _is_previous_request(target["content"], current_user, requests):
            _remember(fitted, aux_lm, target["content"])
        _drop(fitted, target)
    return fitted


def _oldest_unprotected(messages: list[dict], current_user: str) -> int | None:
    for i, message in enumerate(messages):
        if i == 0 or _is_notes(message) or message["content"] == current_user:
            continue
        return i
    return None


def _is_notes(message: dict) -> bool:
    return message["content"].startswith(NOTES_HEADER)


def _is_previous_request(content: str, current_user: str, requests: set[str]) -> bool:
    return content in requests and content != current_user and not content.startswith(NOTES_HEADER)


def _notes_body(messages: list[dict]) -> str:
    if len(messages) > 1 and _is_notes(messages[1]):
        return messages[1]["content"][len(NOTES_HEADER) :].strip()
    return ""


def _set_notes(messages: list[dict], body: str) -> None:
    content = NOTES_HEADER if not body else f"{NOTES_HEADER}\n{body}"
    note = {"role": "user", "content": content}
    if len(messages) > 1 and _is_notes(messages[1]):
        messages[1] = note
    else:
        messages.insert(1, note)


def _remember(messages: list[dict], aux_lm: Callable[[list[dict]], str], dropped: str) -> None:
    old = _notes_body(messages)
    reply = aux_lm(
        [
            {"role": "system", "content": _AUX_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Existing notes:\n{old or '(none)'}\n\n"
                    f"Message leaving the window:\n{dropped}"
                ),
            },
        ]
    )
    reply = (reply or "").strip()
    # Keep any exact number, amount, city, or constraint the summary dropped.
    kept = _missing_sentences(reply, f"{old}\n{dropped}" if old else dropped)
    parts = kept + ([reply] if reply else [])
    _set_notes(messages, "\n".join(parts).strip())


def _missing_sentences(reply: str, source: str) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(source.strip()):
        sentence = sentence.strip()
        if not sentence or sentence in reply or sentence in seen:
            continue
        facts = _FACT.findall(sentence)
        if facts and any(fact not in reply for fact in facts):
            kept.append(sentence)
            seen.add(sentence)
    return kept


def _drop(messages: list[dict], target: dict) -> None:
    for i, message in enumerate(messages):
        if message is target:
            del messages[i]
            return


def _shrink(messages: list[dict], index: int, budget: int) -> bool:
    """Replace one message with the longest head that lets the list fit."""
    content = messages[index]["content"]
    others = count_message_tokens(messages) - count_tokens(content) - MESSAGE_OVERHEAD
    room = budget - others - MESSAGE_OVERHEAD
    head = content[: max(room, 0) * CHARS_PER_TOKEN]
    if head == content:
        return False
    messages[index]["content"] = head
    return True


def _shrink_largest(messages: list[dict], budget: int, current_user: str) -> bool:
    order = sorted(
        (i for i in range(len(messages)) if i != 0 and messages[i]["content"] != current_user),
        key=lambda i: count_tokens(messages[i]["content"]),
        reverse=True,
    )
    for index in order:
        if _shrink(messages, index, budget):
            return True
    return False
