"""Load jobs into Postgres and embed them.

    python -m scripts.ingest csv data/sample_jobs.csv
    python -m scripts.ingest greenhouse stripe airbnb

Safe to run again: existing jobs are skipped (not re-embedded), and the
UNIQUE (source, external_id) constraint stops duplicates even if two runs
overlap.
"""
import argparse
import csv
import html
import sys
from html.parser import HTMLParser
from pathlib import Path

import httpx

from app import db
from app.embeddings import Embedder, get_embedder, job_to_text

GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
REQUIRED_COLUMNS = {"external_id", "title", "company", "description"}


# ---- sources: each returns a list of dicts ----

def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"{path}: missing columns {sorted(missing)}")
        return [
            {
                "source": "csv",
                "external_id": row["external_id"].strip(),
                "title": row["title"].strip(),
                "company": row["company"].strip(),
                "location": (row.get("location") or "").strip() or None,
                "url": (row.get("url") or "").strip() or None,
                "description": row["description"].strip(),
            }
            for row in reader
            if row["external_id"].strip() and row["description"].strip()
        ]


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(raw: str) -> str:
    # Greenhouse sends HTML that is itself HTML-escaped (&lt;p&gt;...)
    parser = _TextOnly()
    parser.feed(html.unescape(raw or ""))
    return " ".join(" ".join(parser.parts).split())


def fetch_greenhouse(board: str, client: httpx.Client) -> list[dict]:
    resp = client.get(GREENHOUSE_URL.format(board=board))
    if resp.status_code == 404:
        print(f"  ! board '{board}' not found, skipping")
        return []
    resp.raise_for_status()

    jobs = []
    for item in resp.json().get("jobs", []):
        description = html_to_text(item.get("content", ""))
        if not description:
            continue
        jobs.append({
            "source": "greenhouse",
            "external_id": f"{board}:{item['id']}",
            "title": item["title"].strip(),
            "company": board,
            "location": (item.get("location") or {}).get("name"),
            "url": item.get("absolute_url"),
            "description": description,
        })
    return jobs


# ---- write to the database ----

def ingest(conn, embedder: Embedder, jobs: list[dict], batch_size: int = 64) -> tuple[int, int]:
    """Returns (inserted, skipped)."""
    if not jobs:
        return 0, 0

    # skip jobs we already have, so re-runs don't pay for embeddings again
    keys = [(j["source"], j["external_id"]) for j in jobs]
    existing = {
        (r["source"], r["external_id"])
        for r in conn.execute(
            "SELECT source, external_id FROM jobs "
            "WHERE (source, external_id) IN (SELECT * FROM unnest(%s::text[], %s::text[]))",
            ([k[0] for k in keys], [k[1] for k in keys]),
        )
    }
    # also dedupe inside the input itself
    new_jobs = list({(j["source"], j["external_id"]): j for j in jobs
                     if (j["source"], j["external_id"]) not in existing}.values())

    inserted = 0
    for start in range(0, len(new_jobs), batch_size):
        batch = new_jobs[start:start + batch_size]
        vectors = embedder.embed_documents(
            [job_to_text(j["title"], j["company"], j["description"]) for j in batch]
        )
        with conn.cursor() as cur:
            for job, vec in zip(batch, vectors):
                cur.execute(
                    """
                    INSERT INTO jobs (source, external_id, title, company, location,
                                      url, description, embedding)
                    VALUES (%(source)s, %(external_id)s, %(title)s, %(company)s,
                            %(location)s, %(url)s, %(description)s, %(embedding)s)
                    ON CONFLICT (source, external_id) DO NOTHING
                    """,
                    {**job, "embedding": vec},
                )
                inserted += cur.rowcount
        conn.commit()
        print(f"  embedded and saved {min(start + batch_size, len(new_jobs))}/{len(new_jobs)}")

    return inserted, len(jobs) - inserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)
    p_csv = sub.add_parser("csv", help="load jobs from a CSV file")
    p_csv.add_argument("path", type=Path)
    p_gh = sub.add_parser("greenhouse", help="load jobs from public Greenhouse job boards")
    p_gh.add_argument("boards", nargs="+", help="board names, e.g. 'stripe' from boards.greenhouse.io/stripe")
    args = parser.parse_args(argv)

    if args.source == "csv":
        jobs = read_csv(args.path)
    else:
        jobs = []
        with httpx.Client(timeout=30) as client:
            for board in args.boards:
                print(f"fetching {board} ...")
                jobs += fetch_greenhouse(board, client)

    print(f"{len(jobs)} jobs read")
    db.init_db()
    with db.connect() as conn:
        inserted, skipped = ingest(conn, get_embedder(), jobs)
    print(f"done: {inserted} new, {skipped} already in the database")


if __name__ == "__main__":
    main()
