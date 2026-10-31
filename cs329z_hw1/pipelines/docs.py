"""Document search and bounded reads for Problem 9 (search_docs)."""

from __future__ import annotations

from dataclasses import dataclass

from cs329z_hw1.pipelines.bm25 import build_index, search as search_passages
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens

SNIPPET_MAX_TOKENS = 150
WINDOW_MAX_TOKENS = 1500

_shared: DocIndex | None = None


@dataclass
class DocIndex:
    """Line-level BM25 index plus the original lines read_document slices."""

    docs: dict[str, dict]
    passages: list[tuple[str, int, str]]
    bm25: object


def build_doc_index(docs: list[dict]) -> DocIndex:
    """Index documents shaped like data.load_docs(): doc_id, title, text."""
    stored: dict[str, dict] = {}
    passages: list[tuple[str, int, str]] = []
    for doc in docs:
        lines = doc["text"].splitlines()
        stored[doc["doc_id"]] = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "lines": lines,
        }
        for number, line in enumerate(lines, 1):
            if tokenize(line):
                passages.append((doc["doc_id"], number, line))
    return DocIndex(
        docs=stored,
        passages=passages,
        bm25=build_index([text for _doc_id, _number, text in passages]),
    )


def search_documents(index: DocIndex, query: str, k: int = 5) -> list[dict]:
    """At most k passage hits. Each hit quotes the matching line and the lines after it."""
    if k <= 0 or not tokenize(query or ""):
        return []
    rare = _rare_terms(index, query)
    pool = search_passages(index.bm25, query, max(k * 8, k))
    picked = _prefer_rare(index, pool, rare, k) or pool[:k]
    results = []
    for passage_id, _score in picked:
        doc_id, line_number, _line = index.passages[passage_id]
        doc = index.docs[doc_id]
        results.append(
            {
                "doc_id": doc_id,
                "title": doc["title"],
                "snippet": _snippet(doc["lines"], line_number, query),
                "line": line_number,
            }
        )
    return results


def _rare_terms(index: DocIndex, query: str) -> set[str]:
    """The rarer half of the query terms that actually occur in the index."""
    idf = getattr(index.bm25, "idf", {})
    terms = [term for term in dict.fromkeys(tokenize(query)) if term in idf]
    if not terms:
        return set()
    ranked = sorted(terms, key=lambda term: idf[term], reverse=True)
    return set(ranked[: max(1, (len(ranked) + 1) // 2)])


def _prefer_rare(index: DocIndex, hits: list[tuple[int, float]], rare: set[str], k: int) -> list[tuple[int, float]]:
    if not rare:
        return []
    chosen = []
    for passage_id, score in hits:
        _doc_id, _number, line = index.passages[passage_id]
        if rare.intersection(tokenize(line)):
            chosen.append((passage_id, score))
            if len(chosen) == k:
                break
    return chosen


def read_document(index: DocIndex, doc_id: str, start_line: int = 1) -> dict:
    """Return exact lines from start_line, packed to at most 1,500 tokens."""
    doc = index.docs.get(doc_id)
    if doc is None:
        raise KeyError(f"Unknown document: {doc_id!r}.")
    lines = doc["lines"]
    total = len(lines)
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1 or start_line > total:
        raise ValueError(f"start_line {start_line} is outside 1..{total} for {doc_id}.")

    chosen: list[str] = []
    end_line = start_line
    for offset, line in enumerate(lines[start_line - 1 :], start_line):
        candidate = chosen + [line]
        if chosen and count_tokens("\n".join(candidate)) > WINDOW_MAX_TOKENS:
            break
        chosen.append(line)
        end_line = offset
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total,
        "text": "\n".join(chosen),
    }


def shared_doc_index() -> DocIndex:
    """Build the company-document index on first use and reuse it."""
    global _shared
    if _shared is None:
        from cs329z_hw1.data import load_docs

        _shared = build_doc_index(load_docs())
    return _shared


def format_search(results: list[dict]) -> str:
    """The string search_docs shows the model."""
    if not results:
        return "No documents matched."
    blocks = []
    for rank, hit in enumerate(results, 1):
        blocks.append(
            f"{rank}. doc_id={hit['doc_id']} title={hit['title']} line={hit['line']}\n{hit['snippet']}"
        )
    blocks.append(
        "If a quote does not state the answer, call read_doc with that doc_id and start_line. "
        "If documents disagree, use the one with the latest date in its id or title."
    )
    return "\n\n".join(blocks)


def format_window(window: dict) -> str:
    """The string read_doc shows the model."""
    start = window["start_line"]
    end = window["end_line"]
    total = window["total_lines"]
    if end < total:
        follow = f"Next window starts at line {end + 1}."
    else:
        follow = "End of document."
    return f"doc_id={window['doc_id']} lines {start}-{end} of {total}\n{window['text']}\n{follow}"


def _snippet(lines: list[str], line_number: int, query: str) -> str:
    """The matching line, widened until it has four words, or clipped if it is too long."""
    line = lines[line_number - 1]
    if count_tokens(line) > SNIPPET_MAX_TOKENS:
        return _clip(line, query)
    start = line_number - 1
    end = line_number
    while len(tokenize("\n".join(lines[start:end]))) < 4:
        if end < len(lines):
            end += 1
        elif start > 0:
            start -= 1
        else:
            break
        if count_tokens("\n".join(lines[start:end])) > SNIPPET_MAX_TOKENS:
            if end > line_number:
                end -= 1
            else:
                start += 1
            break
    while end < len(lines) and count_tokens("\n".join(lines[start : end + 1])) <= SNIPPET_MAX_TOKENS:
        end += 1
    return "\n".join(lines[start:end])


def _clip(line: str, query: str) -> str:
    lower = line.lower()
    anchor = 0
    for term in tokenize(query):
        at = lower.find(term)
        if at >= 0:
            anchor = at
            break
    begin = anchor
    end = min(len(line), anchor + 1)
    while True:
        grew = False
        if end < len(line) and count_tokens(line[begin : end + 1]) <= SNIPPET_MAX_TOKENS:
            end += 1
            grew = True
        if begin > 0 and count_tokens(line[begin - 1 : end]) <= SNIPPET_MAX_TOKENS:
            begin -= 1
            grew = True
        if not grew:
            break
    return line[begin:end]
