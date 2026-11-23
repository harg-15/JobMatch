from contextlib import asynccontextmanager
from typing import Iterator

import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import db
from app.embeddings import Embedder, get_embedder
from app.matcher import Mode, match_resume
from app.rerank import LLMCall, openai_compatible_llm
from app.resume import ResumeError, clean, extract_text
from app.search import Filters, vector_search

MAX_PDF_BYTES = 5 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="JobMatch", lifespan=lifespan)


# ---- dependencies (overridden in tests) ----

def get_conn() -> Iterator[psycopg.Connection]:
    with db.connect() as conn:
        yield conn


_llm = openai_compatible_llm()


def get_llm() -> LLMCall | None:
    return _llm


# ---- response models ----

class JobOut(BaseModel):
    id: int
    title: str
    company: str
    location: str | None
    url: str | None
    score: int
    reason: str | None = None


class MatchOut(BaseModel):
    reranked: bool
    matches: list[JobOut]


class SearchIn(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=10, ge=1, le=50)
    exclude: list[str] = []
    location: str | None = None


# ---- routes ----

@app.get("/health")
def health(conn: psycopg.Connection = Depends(get_conn)):
    jobs = conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"]
    return {"status": "ok", "jobs": jobs, "llm_configured": get_llm() is not None}


@app.post("/search", response_model=list[JobOut])
def search(
    body: SearchIn,
    conn: psycopg.Connection = Depends(get_conn),
    embedder: Embedder = Depends(get_embedder),
):
    """Free text semantic search, e.g. 'backend intern python postgres'."""
    filters = Filters(exclude_terms=body.exclude, location=body.location)
    rows = vector_search(conn, embedder.embed_query(body.query), body.limit, filters)
    return [{**r, "score": round(float(r["similarity"]) * 100)} for r in rows]


@app.post("/match", response_model=MatchOut)
async def match(
    resume: UploadFile | None = File(None, description="resume as PDF"),
    resume_text: str | None = Form(None, description="or paste the resume as text"),
    wants: str | None = Form(None, description="e.g. 'backend internship, remote ok'"),
    exclude: str | None = Form(None, description="comma separated, e.g. 'unpaid,sales'"),
    location: str | None = Form(None),
    mode: Mode = Form("rerank"),
    top_k: int = Form(5, ge=1, le=20),
    conn: psycopg.Connection = Depends(get_conn),
    embedder: Embedder = Depends(get_embedder),
    llm: LLMCall | None = Depends(get_llm),
):
    text = await _read_resume(resume, resume_text)
    filters = Filters(
        exclude_terms=(exclude or "").split(","),
        location=location or None,
    )
    matches, reranked = match_resume(
        conn, embedder, llm, text, mode=mode, wants=wants, filters=filters, top_k=top_k
    )
    return {"reranked": reranked, "matches": matches}


async def _read_resume(resume: UploadFile | None, resume_text: str | None) -> str:
    if resume is not None:
        data = await resume.read()
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(413, "PDF is larger than 5 MB")
        try:
            return extract_text(data)
        except ResumeError as exc:
            raise HTTPException(422, str(exc))
    if resume_text and len(resume_text.strip()) >= 50:
        return clean(resume_text)
    raise HTTPException(422, "send a PDF in 'resume' or at least 50 characters in 'resume_text'")
