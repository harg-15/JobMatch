"""The full pipeline, shared by the API and the eval script.

    resume text
        -> embed
        -> vector search in Postgres (with hard filters)   ~20 candidates
        -> LLM re-rank + reasons                            top 5
"""
from typing import Literal

import psycopg

from app.config import settings
from app.embeddings import Embedder
from app.rerank import LLMCall, RerankError, rerank
from app.search import Filters, keyword_search, vector_search

Mode = Literal["keyword", "vector", "rerank"]


def match_resume(
    conn: psycopg.Connection,
    embedder: Embedder,
    llm: LLMCall | None,
    resume_text: str,
    mode: Mode = "rerank",
    wants: str | None = None,
    filters: Filters | None = None,
    top_k: int = 5,
) -> tuple[list[dict], bool]:
    """Returns (matches, reranked). reranked is False if we fell back to plain search."""
    if mode == "keyword":
        jobs = keyword_search(conn, resume_text, limit=top_k, filters=filters)
        return [_plain(j) for j in jobs], False

    # what the person wants is part of the query, so it also steers retrieval
    query = f"{wants}\n\n{resume_text}" if wants else resume_text
    query_vec = embedder.embed_query(query)

    if mode == "vector" or llm is None:
        jobs = vector_search(conn, query_vec, limit=top_k, filters=filters)
        return [_plain(j) for j in jobs], False

    candidates = vector_search(
        conn, query_vec, limit=settings.candidates_for_rerank, filters=filters
    )
    try:
        return rerank(llm, resume_text, candidates, wants=wants, top_k=top_k), True
    except RerankError:
        # LLM down or returned garbage: still give the user something useful
        return [_plain(j) for j in candidates[:top_k]], False


def _plain(job: dict) -> dict:
    return {**job, "score": round(float(job["similarity"]) * 100), "reason": None}
