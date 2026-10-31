"""Archive lookup for the email tools.

The index is built on the first search or read, not when a session is created.
"""

from __future__ import annotations

from cs329z_hw1.types import Email

_emails: list[Email] | None = None
_by_id: dict[str, Email] | None = None
_index = None

_QUOTE_CHARS = 400


def search_email_records(query: str, k: int = 8) -> list[Email]:
    """The top ``k`` emails for ``query``, in rank order."""
    emails, index = _archive()
    from cs329z_hw1.pipelines.bm25 import search

    return [emails[position] for position, _score in search(index, query, k)]


def search_messages(query: str, k: int = 5) -> str:
    """Up to ``k`` hits, each with id, subject, sender, date, and a short quote."""
    emails, index = _archive()
    from cs329z_hw1.pipelines.bm25 import search

    hits = search(index, query, k)
    if not hits:
        return "No messages matched."
    blocks = []
    for rank, (position, _score) in enumerate(hits, 1):
        email = emails[position]
        blocks.append(
            f"{rank}. id={email['id']} subject={email['subject']} "
            f"from={email['from']} date={email['date']}\n{_quote(email['body'])}"
        )
    return "\n\n".join(blocks)


def read_message(email_id: str) -> str:
    """One message. An unknown id raises ``KeyError``."""
    _archive()
    email = _by_id.get(email_id) if _by_id is not None else None
    if email is None:
        raise KeyError(email_id)
    return (
        f"id={email['id']} subject={email['subject']} from={email['from']} "
        f"date={email['date']}\n{email['body']}"
    )


def _archive():
    global _emails, _by_id, _index
    if _emails is None:
        from cs329z_hw1.data import email_text, load_emails
        from cs329z_hw1.pipelines.bm25 import build_index

        _emails = load_emails()
        _by_id = {email["id"]: email for email in _emails}
        _index = build_index([email_text(email) for email in _emails])
    return _emails, _index


def _quote(body: str) -> str:
    text = " ".join(body.split())
    if len(text) <= _QUOTE_CHARS:
        return text
    return text[:_QUOTE_CHARS].rstrip() + "..."
