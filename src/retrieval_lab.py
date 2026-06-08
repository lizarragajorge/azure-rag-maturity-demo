"""Retrieval lab: compare keyword / vector / hybrid / hybrid+semantic.

Importable as a library by the Streamlit UI, and runnable from the command
line for a quick CLI demo.

    python src/retrieval_lab.py "what PPE is required for an arc flash above 25 cal/cm2?"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizableTextQuery,
)

from common import Settings, get_credential


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    id: str
    parent_id: str
    title: str
    document_type: str
    source: str
    chunk_index: int
    content: str
    search_score: Optional[float]
    reranker_score: Optional[float]
    captions: list[str]


@dataclass
class RunResult:
    mode: str
    query: str
    hits: list[Hit]
    answer: Optional[str] = None  # only populated for semantic with answers
    note: str = ""


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


_SELECT = ["id", "parent_id", "title", "document_type", "source", "chunk_index", "content"]


def _client(settings: Settings) -> SearchClient:
    return SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.search_index_name,
        credential=get_credential(),
    )


def _to_hits(results) -> list[Hit]:
    hits: list[Hit] = []
    for r in results:
        captions_raw = r.get("@search.captions") or []
        captions: list[str] = []
        for c in captions_raw:
            text = getattr(c, "text", None) or getattr(c, "highlights", None)
            if text:
                captions.append(text)
        hits.append(
            Hit(
                id=r["id"],
                parent_id=r["parent_id"],
                title=r["title"],
                document_type=r.get("document_type", ""),
                source=r.get("source", ""),
                chunk_index=r.get("chunk_index", 0),
                content=r["content"],
                search_score=r.get("@search.score"),
                reranker_score=r.get("@search.reranker_score"),
                captions=captions,
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_keyword(settings: Settings, query: str, top: int = 5) -> RunResult:
    """Plain BM25 keyword search. Strong on exact terms, weak on paraphrase."""
    client = _client(settings)
    results = client.search(
        search_text=query,
        top=top,
        select=_SELECT,
    )
    return RunResult(
        mode="keyword (BM25)",
        query=query,
        hits=_to_hits(results),
        note=(
            "Lexical match only. Misses synonyms ('PPE' vs 'protective gear'), "
            "abbreviations, and conceptual paraphrases."
        ),
    )


def run_vector(settings: Settings, query: str, top: int = 5) -> RunResult:
    """Pure vector search via the integrated vectorizer."""
    client = _client(settings)
    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=top,
        fields="content_vector",
    )
    results = client.search(
        search_text=None,  # vector only
        vector_queries=[vector_query],
        top=top,
        select=_SELECT,
    )
    return RunResult(
        mode="vector (embeddings only)",
        query=query,
        hits=_to_hits(results),
        note=(
            "Recall on paraphrase and concept overlap. Can drift on rare proper "
            "nouns or numeric thresholds that have no semantic neighborhood."
        ),
    )


def run_hybrid(settings: Settings, query: str, top: int = 5) -> RunResult:
    """Hybrid: BM25 + vector fused with Reciprocal Rank Fusion (RRF)."""
    client = _client(settings)
    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=50,
        fields="content_vector",
    )
    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=top,
        select=_SELECT,
    )
    return RunResult(
        mode="hybrid (BM25 + vector, RRF)",
        query=query,
        hits=_to_hits(results),
        note=(
            "Reciprocal Rank Fusion combines both rankings. This is the default "
            "reliable baseline for RAG retrieval."
        ),
    )


def run_semantic(
    settings: Settings,
    query: str,
    top: int = 5,
    semantic_configuration: str = "default-balanced",
    with_answer: bool = True,
) -> RunResult:
    """Hybrid retrieval + semantic ranker (L2) on the chosen config.

    The semantic ranker rescores the top-50 hybrid hits with a deep model and
    returns `@search.rerankerScore` plus extractive captions and (optionally)
    an extractive answer.  This is what makes retrieval "reliable" in practice.
    """
    client = _client(settings)
    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=50,
        fields="content_vector",
    )
    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=top,
        select=_SELECT,
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name=semantic_configuration,
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE if with_answer else None,
    )

    # `get_answers()` returns extractive answers if any were produced
    answer_text: Optional[str] = None
    if with_answer:
        answers = results.get_answers() or []
        if answers:
            answer_text = answers[0].text

    hits = _to_hits(results)
    return RunResult(
        mode=f"hybrid + semantic ranker  ·  config = {semantic_configuration!r}",
        query=query,
        hits=hits,
        answer=answer_text,
        note=(
            "Hybrid candidates are rescored by the semantic ranker, which uses a "
            "deep model trained on relevance judgments. Output includes "
            "`@search.rerankerScore` (a 0–4 relevance score) and extractive "
            "captions/answers that you can use as grounding for the LLM."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print(result: RunResult) -> None:
    bar = "─" * 78
    print(f"\n{bar}\n{result.mode}\n{bar}")
    print(f"Query: {result.query}")
    print(f"Note : {result.note}")
    if result.answer:
        print(f"\nExtractive answer:\n  {result.answer}\n")
    for i, h in enumerate(result.hits, 1):
        rs = f"{h.reranker_score:.3f}" if h.reranker_score is not None else "  -  "
        ss = f"{h.search_score:.3f}" if h.search_score is not None else "  -  "
        print(f"\n  [{i}] reranker={rs}  search={ss}")
        print(f"      title  : {h.title}")
        print(f"      source : {h.source}  (chunk {h.chunk_index})")
        if h.captions:
            print(f"      caption: {h.captions[0][:160]}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="RAG retrieval comparison lab")
    parser.add_argument("query", help="Question to evaluate")
    parser.add_argument(
        "--config",
        default="default-balanced",
        choices=["default-balanced", "content-only", "title-and-keywords"],
        help="Semantic configuration to use for the semantic mode",
    )
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    _print(run_keyword(settings, args.query, top=args.top))
    _print(run_vector(settings, args.query, top=args.top))
    _print(run_hybrid(settings, args.query, top=args.top))
    _print(run_semantic(settings, args.query, top=args.top, semantic_configuration=args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
