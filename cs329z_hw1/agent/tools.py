"""Cardinal tools for the agent session.

The email archive is read when a tool runs, not when a session is created.
"""

from __future__ import annotations

import contextvars

from cs329z_hw1.cardinal import CARDINAL_TOOLS, cardinal_spec
from cs329z_hw1.types import AgentConfig, ToolSpec

# One value per conversation. A module global is overwritten when several
# conversations create sessions at the same time, and a tool then reads
# another conversation's memory directory.
_config_var: contextvars.ContextVar[AgentConfig | None] = contextvars.ContextVar(
    "cs329z_agent_config",
    default=None,
)

_DESCRIPTIONS = {
    "search_emails": "Find messages in the archive for a question about an email or a thread. Pass the question as query. Each hit has an id, subject, sender, date, and a short quote.",
    "read_email": "Return one archive message by its id, including the subject, sender, date, and body.",
    "email_qa": "Answer one question from the archive.",
    "daily_digest": "Summarize the messages sent on one calendar date.",
    "search_docs": "Search company documents for a question about a policy or a company document. Pass the question as query. Each hit gives a doc_id, a title, a line number, and a quote. When hits disagree, trust the latest date in the id or title.",
    "read_doc": "Read one document from start_line, the line on a hit or 1. Returns that block and the line number to pass in order to continue.",
    "run_terminal": "Run one shell command. emails.jsonl is in the workspace; count, filter, or list its records with python3 -c '...'. Here-documents are rejected. A large result names a file in the workspace that a later command can read.",
    "remember": "Save one fact for a later session.",
    "recall": "Retrieve facts saved in a previous session.",
    "ask_user": "Ask the user a question and return their answer.",
}


def search_emails(query: str, k: int = 5) -> str:
    from cs329z_hw1.agent.emails import search_messages
    return search_messages(query, k)


def read_email(email_id: str) -> str:
    from cs329z_hw1.agent.emails import read_message
    return read_message(email_id)


def email_qa(question: str) -> str:
    from cs329z_hw1.agent.emails import search_email_records
    from cs329z_hw1.pipelines.email_qa import answer_question

    config = _require_config("email_qa")
    if config.aux_lm is None:
        raise RuntimeError("email_qa was called before a session was created.")
    result = answer_question(question, search_email_records, config.aux_lm)
    answer = result.get("answer")
    if not answer:
        return "No answer was found in the archive."
    support = [item for item in result.get("support") or [] if item]
    if support:
        return f"{answer}\nSupport: {', '.join(support)}"
    return str(answer)


def daily_digest(date: str) -> str:
    from cs329z_hw1.data import emails_on
    from cs329z_hw1.pipelines.digest import write_daily_digest

    config = _require_config("daily_digest")
    if config.aux_lm is None:
        raise RuntimeError("daily_digest was called before a session was created.")
    summary = write_daily_digest(emails_on(date), config.aux_lm)
    return f"Summary of the emails sent on {date}:\n{summary}"

def search_docs(query: str, k: int = 5) -> str:
    from cs329z_hw1.pipelines.docs import format_search, search_documents, shared_doc_index
    return format_search(search_documents(shared_doc_index(), query, k))

def read_doc(doc_id: str, start_line: int = 1) -> str:
    from cs329z_hw1.pipelines.docs import format_window, read_document, shared_doc_index
    return format_window(read_document(shared_doc_index(), doc_id, start_line))

def _require_config(name: str) -> AgentConfig:
    config = _config_var.get()
    if config is None:
        raise RuntimeError(f"{name} was called before a session was created.")
    return config


def run_terminal(cmd: str) -> str:
    from cs329z_hw1.agent.terminal import observe
    return observe(_require_config("run_terminal"), cmd)

def remember(fact: str) -> str:
    from cs329z_hw1.agent.memory import save_fact
    config = _require_config("remember")
    if config.memory_dir is None:
        raise RuntimeError("remember was called before a session was created.")
    return save_fact(config.memory_dir, fact)


def recall(query: str) -> str:
    from cs329z_hw1.agent.memory import recall
    config = _require_config("recall")
    if config.memory_dir is None:
        raise RuntimeError("recall was called before a session was created.")
    return recall(config.memory_dir, query)


def ask_user(question: str) -> str:
    config = _require_config("ask_user")
    if config.user is None:
        raise RuntimeError("ask_user was called before a session was created.")
    return config.user.ask(question)


_FUNCTIONS = {
    "search_emails": search_emails,
    "read_email": read_email,
    "email_qa": email_qa,
    "daily_digest": daily_digest,
    "search_docs": search_docs,
    "read_doc": read_doc,
    "run_terminal": run_terminal,
    "remember": remember,
    "recall": recall,
    "ask_user": ask_user,
}


def _spec(name: str) -> ToolSpec:
    return cardinal_spec(name, _DESCRIPTIONS[name], _FUNCTIONS[name])


def tools_for_session(tools: list[ToolSpec] | None, config: AgentConfig) -> list[ToolSpec]:
    _config_var.set(config)
    if tools is None:
        return [_spec(name) for name in CARDINAL_TOOLS]

    chosen = list(tools)
    names = {tool.name for tool in chosen}
    if config.user is not None and "ask_user" not in names:
        chosen.append(_spec("ask_user"))
        names.add("ask_user")
    if config.memory_dir is not None:
        for name in ("remember", "recall"):
            if name not in names:
                chosen.append(_spec(name))
                names.add(name)
    return chosen
