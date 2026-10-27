"""Multi-call daily digest pipeline for Problem 2 (daily_digest)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from cs329z_hw1.pipelines.priority import classify_priority
from cs329z_hw1.types import Email, LMCallable

EMPTY_DAY = "No emails today."
MAX_BODY_CHARS = 1200
MAX_WORKERS = 8

WRITER_PROMPT = """\
You write a morning briefing for a busy employee from a pre-filtered pile of emails.
Ignore-pile mail has already been removed. Treat every listed email as real work or personal mail.

Rules:
- Write at most 250 words. Prefer 150-220. Do not pad.
- Reply with the briefing only. No title, no preamble, no word count.
- Open with the urgent items. For each urgent email you MUST use the proper name of the company, project, site, or person the email is about. A paraphrase without that name is wrong.
- Then cover remaining mail briefly: requests, deadlines, meetings, personal notes worth seeing.
- Collapse follow-ups in the same thread into one item.
- Do not invent senders, companies, or facts that are not in the listed emails.
"""


def _format_email_block(index: int, email: Email, category: str, reason: str) -> str:
    body = email.get("body") or ""
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n[truncated]"
    to_list = email.get("to") or []
    return (
        f"## Item {index}\n"
        f"Category: {category}\n"
        f"From: {email.get('from', '')}\n"
        f"To: {', '.join(to_list)}\n"
        f"Date: {email.get('date', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Classifier reason: {reason}\n"
        f"Body:\n{body}"
    )


def _classify_one(email: Email, lm: LMCallable) -> dict:
    result = classify_priority(email, lm)
    category = result.get("category") or "unknown"
    if category not in ("urgent", "normal", "ignore"):
        category = "normal"
    elif category == "unknown":
        category = "normal"
    reason = result.get("reason") or ""
    return {"email": email, "category": category, "reason": reason}


def _keep_one_per_thread(items: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    order: list[str] = []
    for position, item in enumerate(items):
        email = item["email"]
        thread_id = email.get("thread_id") or email.get("id") or str(position)
        candidate = {**item, "_position": position}
        if thread_id not in best:
            best[thread_id] = candidate
            order.append(thread_id)
            continue
        current = best[thread_id]
        urgent_new = candidate["category"] == "urgent"
        urgent_old = current["category"] == "urgent"
        if urgent_new and not urgent_old:
            best[thread_id] = candidate
            continue
        if urgent_new == urgent_old:
            if (candidate["email"].get("date") or "") > (current["email"].get("date") or ""):
                best[thread_id] = candidate
            elif (candidate["email"].get("date") or "") == (current["email"].get("date") or ""):
                if candidate["_position"] > current["_position"]:
                    best[thread_id] = candidate
    return [best[thread_id] for thread_id in order]


def write_daily_digest(emails: list[Email], lm: LMCallable) -> str:
    if not emails:
        return EMPTY_DAY

    workers = min(MAX_WORKERS, len(emails))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        classified = list(pool.map(lambda email: _classify_one(email, lm), emails))

    kept = [item for item in classified if item["category"] != "ignore"]
    if not kept:
        return "Nothing that needs your attention today."

    collapsed = _keep_one_per_thread(kept)
    collapsed.sort(
        key=lambda item: (0 if item["category"] == "urgent" else 1, item.get("_position", 0))
    )

    blocks = [
        _format_email_block(i + 1, item["email"], item["category"], item["reason"])
        for i, item in enumerate(collapsed)
    ]
    user_content = (
        f"{len(collapsed)} emails after dropping ignore and collapsing threads.\n\n"
        + "\n\n".join(blocks)
    )
    reply = lm(
        [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    if isinstance(reply, str):
        return reply
    return "" if reply is None else str(reply)
