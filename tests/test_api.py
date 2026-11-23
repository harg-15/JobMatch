import io
import json

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app import db, main

RESUME = (
    "Second year student. Built a Spring Boot and PostgreSQL backend for loans, "
    "used Redis for locking and Docker for deployment. Java, SQL, data structures."
)


def make_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in [text[i:i + 90] for i in range(0, len(text), 90)]:
        c.drawString(40, y, line)
        y -= 15
    c.save()
    return buf.getvalue()


@pytest.fixture
def client(loaded, embedder, monkeypatch):
    # schema already exists in the test db; don't let startup touch the real one
    monkeypatch.setattr(db, "init_db", lambda *args, **kwargs: None)

    def conn_override():
        yield loaded

    main.app.dependency_overrides[main.get_conn] = conn_override
    main.app.dependency_overrides[main.get_embedder] = lambda: embedder
    main.app.dependency_overrides[main.get_llm] = lambda: None
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


def use_llm(response):
    main.app.dependency_overrides[main.get_llm] = lambda: (lambda system, user: response)


def test_health(client):
    body = client.get("/health").json()
    assert body["jobs"] == 30
    assert body["llm_configured"] is False


def test_search(client):
    r = client.post("/search", json={"query": "kotlin android app", "limit": 3})
    assert r.status_code == 200
    assert r.json()[0]["title"] == "Android Developer Intern"


def test_match_with_pdf_without_llm(client):
    pdf = make_pdf(RESUME)
    r = client.post("/match", files={"resume": ("cv.pdf", pdf, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["reranked"] is False
    assert len(body["matches"]) == 5
    assert all(m["reason"] is None for m in body["matches"])


def test_match_with_llm_returns_reasons(client):
    use_llm(json.dumps({"matches": [
        {"job_id": 29, "score": 92, "reason": "Spring Boot, Redis and Postgres match the loan backend"},
        {"job_id": 1, "score": 85, "reason": "Java and Redis idempotency overlap"},
    ]}))
    r = client.post("/match", data={"resume_text": RESUME, "wants": "backend internship"})
    body = r.json()
    assert body["reranked"] is True
    assert [m["id"] for m in body["matches"]] == [29, 1]
    assert body["matches"][0]["reason"].startswith("Spring Boot")


def test_match_falls_back_when_llm_returns_garbage(client):
    use_llm("not json at all")
    r = client.post("/match", data={"resume_text": RESUME})
    body = r.json()
    assert r.status_code == 200
    assert body["reranked"] is False
    assert len(body["matches"]) == 5


def test_match_exclude(client):
    r = client.post("/match", data={
        "resume_text": "sales business development cold calling leads " * 3,
        "exclude": "unpaid, sales",
        "top_k": 20,
    })
    for m in r.json()["matches"]:
        assert "sales" not in m["title"].lower()
        assert "unpaid" not in m["title"].lower()


def test_match_rejects_empty_input(client):
    assert client.post("/match", data={"resume_text": "too short"}).status_code == 422


def test_match_rejects_non_pdf(client):
    r = client.post("/match", files={"resume": ("cv.pdf", b"hello", "application/pdf")})
    assert r.status_code == 422
