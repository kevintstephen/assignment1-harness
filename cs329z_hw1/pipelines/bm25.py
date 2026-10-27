"""From-scratch BM25 index for Problem 3 (bm25)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75


@dataclass
class BM25Index:
    """Postings and length stats. Search reads only the postings of query terms."""

    postings: dict[str, list[tuple[int, int]]]
    doc_lengths: list[int]
    n_docs: int
    avgdl: float
    idf: dict[str, float]


def build_index(docs: list[str]) -> BM25Index:
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_lengths: list[int] = []
    total_tokens = 0
    for doc_id, text in enumerate(docs):
        counts = Counter(tokenize(text or ""))
        length = sum(counts.values())
        doc_lengths.append(length)
        total_tokens += length
        for term, tf in counts.items():
            postings.setdefault(term, []).append((doc_id, tf))

    n_docs = len(docs)
    avgdl = total_tokens / n_docs if n_docs else 0.0
    idf = {
        term: math.log((n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1)
        for term, posting in postings.items()
    }
    return BM25Index(
        postings=postings,
        doc_lengths=doc_lengths,
        n_docs=n_docs,
        avgdl=avgdl,
        idf=idf,
    )


def search(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    if k <= 0 or index.n_docs == 0 or index.avgdl == 0.0:
        return []

    scores: dict[int, float] = {}
    lengths = index.doc_lengths
    avgdl = index.avgdl
    for term in dict.fromkeys(tokenize(query)):
        posting = index.postings.get(term)
        if not posting:
            continue
        idf = index.idf[term]
        weight = idf * (K1 + 1)
        for doc_id, tf in posting:
            norm = K1 * (1 - B + B * lengths[doc_id] / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * tf / (tf + norm)

    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(doc_id, float(score)) for doc_id, score in ranked[:k]]
