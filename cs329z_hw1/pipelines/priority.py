"""One-call email priority classifier for Problem 1 (priority)."""

from __future__ import annotations

import re

from cs329z_hw1.types import Email, LMCallable

LABELS = ("urgent", "normal", "ignore")
LABEL_RE = re.compile(r"\b(urgent|normal|ignore)\b", re.IGNORECASE)
CATEGORY_LINE_RE = re.compile(
    r"^\s*CATEGORY\s*:\s*(urgent|normal|ignore)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
REASON_LINE_RE = re.compile(r"^\s*REASON\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
UNKNOWN_REASON = "The model reply did not state a label."

SYSTEM_PROMPT = """\
You classify one email for the person it is addressed to (the To field), at the time it was sent.
The question is what that person should do with the message, not how important the topic is to the company.

Each email gets exactly one label: urgent, normal, or ignore.

## urgent

The recipient has to act or decide soon. Two things must both be true:

1. The email asks the recipient to do something, or reports a live problem that the recipient is responsible for (an outage, a failed payment, a customer or manager escalating an issue).
2. There is time pressure: the email states or clearly implies that the action is needed the same day, the next working day, or by a stated deadline no more than two working days away.
   A working day is Monday to Friday, counted from the email's date.

Examples:
- "The bank rejected this morning's wire. I need you to approve the corrected one by 1:00 PM or we pay a late penalty."
- "The customer's CFO is threatening to cancel. Can you call him before 4:00 today?"

## normal

Legitimate work or personal mail that was written for this recipient or for a small working group, with no deadline in the next two working days. This covers status updates and FYIs, ordinary questions, documents sent for review without a near deadline, meeting scheduling for later dates, replies in an ongoing discussion, and personal notes from friends or family.

Examples:
- "Attached is the draft term sheet. Let me know if you have comments before our meeting the week after next."
- "Thanks for lunch yesterday. Are you still coming to the lake in August?"

## ignore

Mail the recipient can leave unread at no cost: advertising and spam from vendors, newsletters and news digests, announcements sent to large employee lists (events, training offerings, facilities notices), and automated notifications that do not require the recipient to do anything.

Examples:
- "This week only: 40% off all standing desks. Reply REMOVE to unsubscribe."
- "To all employees: the annual bake sale is in the lobby on Friday from 11:00 to 1:00."

## Deciding between two labels

- The wording does not decide the label. An email with "URGENT" in the subject that asks nothing of the recipient is not urgent. A calm email that says "I need your signature by 3:00 PM today" is urgent.
- A request with a deadline more than two working days away, or with no deadline, is normal.
- Mass mailings and advertising are ignore even when they use words such as "act now" or "important", because nothing is asked of this recipient in particular.
- An automated notice that asks this specific recipient to act (for example, an expense report waiting for their approval) is normal, or urgent if it states a deadline within two working days.
- A short personal or social message from someone the recipient knows is normal, not ignore.

Reply with exactly two lines and no other text:

CATEGORY: urgent
REASON: <one sentence>

CATEGORY must be exactly one of: urgent, normal, ignore.
"""


def _format_email(email: Email) -> str:
    to_list = email.get("to") or []
    to_line = ", ".join(to_list)
    return (
        f"From: {email.get('from', '')}\n"
        f"To: {to_line}\n"
        f"Recipient count: {len(to_list)}\n"
        f"Date: {email.get('date', '')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Body:\n{email.get('body', '')}"
    )


def _parse_reply(text: str) -> dict:
    raw = text if isinstance(text, str) else ""
    stripped = raw.strip()

    category_match = CATEGORY_LINE_RE.search(raw)
    found = [m.group(1).lower() for m in LABEL_RE.finditer(raw)]
    unique = []
    for label in found:
        if label not in unique:
            unique.append(label)

    if category_match:
        category = category_match.group(1).lower()
    elif len(unique) == 1:
        category = unique[0]
    else:
        category = "unknown"

    reason_match = REASON_LINE_RE.search(raw)
    if reason_match and reason_match.group(1).strip():
        reason = reason_match.group(1).strip()
    elif stripped:
        reason = " ".join(stripped.split())
    else:
        reason = UNKNOWN_REASON

    return {"category": category, "reason": reason}


def classify_priority(email: Email, lm: LMCallable) -> dict:
    reply = lm(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_email(email)},
        ]
    )
    return _parse_reply(reply)
