import httpx

from scripts.ingest import fetch_greenhouse, html_to_text, ingest, read_csv
from tests.conftest import ROOT

SAMPLE = ROOT / "data" / "sample_jobs.csv"


def count(conn):
    return conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"]


def test_ingest_loads_all_sample_jobs(conn, embedder):
    inserted, skipped = ingest(conn, embedder, read_csv(SAMPLE))
    assert (inserted, skipped) == (30, 0)
    assert count(conn) == 30


def test_ingest_twice_does_not_duplicate(conn, embedder):
    jobs = read_csv(SAMPLE)
    ingest(conn, embedder, jobs)
    inserted, skipped = ingest(conn, embedder, jobs)
    assert (inserted, skipped) == (0, 30)
    assert count(conn) == 30


def test_duplicates_inside_one_batch_are_dropped(conn, embedder):
    job = read_csv(SAMPLE)[0]
    inserted, _ = ingest(conn, embedder, [job, dict(job)])
    assert inserted == 1


def test_html_to_text_handles_escaped_html():
    raw = "&lt;p&gt;Build &lt;strong&gt;APIs&lt;/strong&gt; in Python&lt;/p&gt;&lt;ul&gt;&lt;li&gt;SQL&lt;/li&gt;&lt;/ul&gt;"
    assert html_to_text(raw) == "Build APIs in Python SQL"


def test_fetch_greenhouse_parses_jobs_and_skips_missing_boards():
    def handler(request):
        if "/boards/acme/" in str(request.url):
            return httpx.Response(200, json={"jobs": [
                {"id": 11, "title": "Backend Intern", "location": {"name": "Gurugram"},
                 "absolute_url": "https://example.com/11",
                 "content": "&lt;p&gt;Python and Postgres&lt;/p&gt;"},
                {"id": 12, "title": "No description", "content": ""},
            ]})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    jobs = fetch_greenhouse("acme", client)
    assert len(jobs) == 1
    assert jobs[0]["external_id"] == "acme:11"
    assert jobs[0]["description"] == "Python and Postgres"
    assert fetch_greenhouse("doesnotexist", client) == []
