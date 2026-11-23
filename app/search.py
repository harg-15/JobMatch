"""SQL for finding candidate jobs.

Two ways to search:
  - vector_search:  cosine distance between embeddings (pgvector, HNSW index)
  - keyword_search: Postgres full text search, used as a baseline in the eval

Hard filters (exclude words, location) are applied in SQL, before anything
reaches the LLM. A "never show me unpaid roles" rule should be guaranteed,
not something the model might forget.
"""
from dataclasses import dataclass, field

import numpy as np
import psycopg

JOB_COLUMNS = "j.id, j.external_id, j.title, j.company, j.location, j.url, j.description"


@dataclass
class Filters:
    exclude_terms: list[str] = field(default_factory=list)
    location: str | None = None

    def to_sql(self) -> tuple[str, dict]:
        parts, params = [], {}
        terms = [t.strip().lower() for t in self.exclude_terms if t.strip()]
        if terms:
            parts.append("NOT (lower(j.title || ' ' || j.description) LIKE ANY(%(exclude)s))")
            params["exclude"] = [f"%{_escape_like(t)}%" for t in terms]
        if self.location:
            parts.append("j.location ILIKE %(location)s")
            params["location"] = f"%{_escape_like(self.location.strip())}%"
        sql = "".join(f" AND {p}" for p in parts)
        return sql, params


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def vector_search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    limit: int = 20,
    filters: Filters | None = None,
) -> list[dict]:
    where, params = (filters or Filters()).to_sql()
    sql = f"""
        SELECT {JOB_COLUMNS},
               1 - (j.embedding <=> %(vec)s) AS similarity
        FROM jobs j
        WHERE j.embedding IS NOT NULL {where}
        ORDER BY j.embedding <=> %(vec)s
        LIMIT %(limit)s
    """
    return conn.execute(sql, {"vec": query_vec, "limit": limit, **params}).fetchall()


def keyword_search(
    conn: psycopg.Connection,
    text: str,
    limit: int = 20,
    filters: Filters | None = None,
) -> list[dict]:
    # A resume has hundreds of words; AND-ing them all would match nothing.
    # So: stem the resume into lexemes and OR them together.
    # 'simple' is used for the query because the lexemes are already stemmed.
    where, params = (filters or Filters()).to_sql()
    sql = f"""
        WITH q AS (
            SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS query
            FROM unnest(tsvector_to_array(to_tsvector('english', %(text)s))) AS lexeme
        )
        SELECT {JOB_COLUMNS},
               ts_rank(j.search_tsv, q.query) AS similarity
        FROM jobs j, q
        WHERE q.query IS NOT NULL AND j.search_tsv @@ q.query {where}
        ORDER BY similarity DESC
        LIMIT %(limit)s
    """
    return conn.execute(sql, {"text": text, "limit": limit, **params}).fetchall()
