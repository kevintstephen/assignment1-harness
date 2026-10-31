"""Cardinal tool stubs for Problem 8.

The functions are registered so the model can see them. Later problems
replace the bodies. Nothing here reads the email archive.
"""

from __future__ import annotations

from cs329z_hw1.cardinal import CARDINAL_TOOLS, cardinal_spec
from cs329z_hw1.types import AgentConfig, ToolSpec

_DESCRIPTIONS = {
    "search_emails": "Find messages in the archive that match a query.",
    "read_email": "Return one message by its id.",
    "email_qa": "Answer one question from the archive.",
    "daily_digest": "Summarize the messages sent on one calendar date.",
    "search_docs": "Search company documents with the question as the query. Each hit gives a doc_id, a title, a line number, and a quote. When hits disagree, trust the latest date in the id or title.",
    "read_doc": "Read one document from start_line, the line on a hit or 1. Returns that block and the line number to pass in order to continue.",
    "run_terminal": "Run one shell command. emails.jsonl is in the workspace; count, filter, or list its records with python3 -c '...'. Here-documents are rejected. A large result names a file in the workspace that a later command can read.",
    "remember": "Save one fact for a later session.",
    "recall": "Retrieve facts saved in a previous session.",
    "ask_user": "Ask the user a question and return their answer.",
}


def search_emails(query: str, k: int = 5) -> str:
    return "Email search is not available yet."


def read_email(email_id: str) -> str:
    return "Reading a message is not available yet."


def email_qa(question: str) -> str:
    return "Question answering is not available yet."


def daily_digest(date: str) -> str:
    return "The daily summary is not available yet."

def search_docs(query: str, k: int = 5) -> str:
    from cs329z_hw1.pipelines.docs import format_search, search_documents, shared_doc_index
    return format_search(search_documents(shared_doc_index(), query, k))

def read_doc(doc_id: str, start_line: int = 1) -> str:
    from cs329z_hw1.pipelines.docs import format_window, read_document, shared_doc_index
    return format_window(read_document(shared_doc_index(), doc_id, start_line))

_config: AgentConfig | None = None


def run_terminal(cmd: str) -> str:
    from cs329z_hw1.agent.terminal import observe
    if _config is None:
        raise RuntimeError("run_terminal was called before a session was created.")
    return observe(_config, cmd) 

def remember(fact: str) -> str:
    from cs329z_hw1.agent.memory import save_fact
    if _config is None or _config.memory_dir is None:
        raise RuntimeError("remember was called before a session was created.")
    return save_fact(_config.memory_dir, fact)


def recall(query: str) -> str:
    from cs329z_hw1.agent.memory import recall
    if _config is None or _config.memory_dir is None:
        raise RuntimeError("recall was called before a session was created.")
    return recall(_config.memory_dir, query)


def ask_user(question: str) -> str:
    if _config is None or _config.user is None:
        raise RuntimeError("ask_user was called before a session was created.")
    return _config.user.ask(question)


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
    global _config
    _config = config
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
