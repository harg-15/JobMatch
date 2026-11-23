# JobMatch

Upload a resume, get the 5 best matching jobs, each with a one line reason for why it fits.

Built with FastAPI, PostgreSQL + pgvector, local embeddings (fastembed) and an LLM for re-ranking.

## How it works

```
resume (PDF or text)
   │
   ├─ extract text (pypdf)
   ├─ embed it (bge-small-en-v1.5, runs locally)
   │
   ▼
Postgres + pgvector
   cosine distance search over job embeddings (HNSW index)
   + hard filters in SQL: excluded words, location
   → top 20 candidates
   │
   ▼
LLM re-rank
   scores the 20 jobs for this resume, returns JSON {job_id, score, reason}
   validated with pydantic → top 5
```

Two stages because each one is bad at something on its own:

- **Embeddings** are fast and cheap, but only know that two texts are "about similar things". A
  student's Spring Boot resume and a "Senior Backend Engineer, 6+ years" posting look very similar to them.
- **An LLM** can reason about seniority and what the person said they want, but sending it every job
  would be slow and expensive. So it only sees the 20 jobs vector search already shortlisted.

## Running it

### With Docker

```bash
cp .env.example .env        # add an LLM key, or leave it empty to skip re-ranking
docker compose up --build
docker compose exec api python -m scripts.ingest csv data/sample_jobs.csv
```

The first ingest downloads the embedding model, after that it's cached in a volume.

### Without Docker

Needs Python 3.11+ and Postgres 16 with the pgvector extension.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python -m scripts.ingest csv data/sample_jobs.csv
uvicorn app.main:app --reload
```

API docs are at http://localhost:8000/docs.

## Loading jobs

```bash
# from a CSV (columns: external_id, title, company, location, url, description)
python -m scripts.ingest csv data/sample_jobs.csv

# from public Greenhouse job boards - the name is the part after boards.greenhouse.io/
python -m scripts.ingest greenhouse stripe airbnb
```

`data/sample_jobs.csv` has 30 made-up postings for testing. They cover backend, ML,
frontend, data, sales etc. and include a few senior roles and unpaid internships on purpose.

Running ingest again only adds new jobs. Existing ones are skipped before embedding, and
`UNIQUE (source, external_id)` with `ON CONFLICT DO NOTHING` catches anything that slips through.

## API

```bash
# match a resume
curl -X POST localhost:8000/match \
  -F resume=@my_resume.pdf \
  -F wants="backend internship, Python or Java" \
  -F exclude="unpaid,sales"

# free text search
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "backend intern postgres redis", "limit": 5}'
```

Example `/match` response:

```json
{
  "reranked": true,
  "matches": [
    {"id": 29, "title": "Backend Developer Intern - Microservices", "company": "Logisync",
     "location": "Gurugram", "url": null, "score": 92, "reason": "..."}
  ]
}
```

`/match` options: `mode` = `rerank` (default), `vector` or `keyword`, `location`, `top_k`. You can
send `resume_text` instead of a PDF.

If the LLM is not configured, times out, or returns something that doesn't parse, `/match` still
works: it returns the vector search results with `"reranked": false`.

## Evaluation

```bash
python -m scripts.evaluate
```

Compares keyword search (Postgres full text), vector search, and vector + LLM re-rank on
precision@5, recall@5 and latency.

The labels live in `eval/labels.csv`: one row per (resume, job) pair that is a good match. Resumes
are text files in `eval/resumes/`. The included ones are small examples for the sample jobs; to get
meaningful numbers, load real jobs and label more pairs.

Note: precision@5 can't reach 1.0 for a resume that only has 2 good jobs in the data, which is why
recall is printed too.

## Design decisions

- **Filters in SQL, not in the prompt.** If someone says "no unpaid roles", that has to be
  guaranteed. An instruction in a prompt is a suggestion; a `WHERE` clause isn't.
- **LLM output is validated, not trusted.** The response goes through a pydantic model (score must
  be 0-100, reason not empty), job ids that weren't in the candidate list are dropped, and bad
  output gets one retry before falling back.
- **Local embeddings.** bge-small runs on CPU, needs no API key, and ingesting jobs costs nothing.
  It gives 384-dim vectors, which is why the column is `vector(384)`.
- **Query vs document embeddings.** bge expects an instruction prefix on the query side, so resumes
  use `query_embed` and jobs use `passage_embed`.
- **Keyword baseline.** A resume has hundreds of words and `plainto_tsquery` ANDs them, which
  matches nothing. The keyword search turns the resume into stemmed lexemes and ORs them instead.

## Known limitations

- The embedding model only reads the first ~512 tokens, so long job descriptions get cut off.
- HNSW with a `WHERE` filter can return fewer rows than `LIMIT` when the filter removes a lot,
  because filtering happens after the index scan. Fine at this size; at scale, raise
  `hnsw.ef_search` or use pgvector's iterative scans.
- One DB connection per request. A real deployment would use a connection pool.
- The LLM sees each job description cut to 800 characters to keep prompts small.

## Tests

```bash
pytest
```

Tests run against a real Postgres (`TEST_DATABASE_URL`, defaults to a local `jobmatch_test`
database) and use a small bag-of-words embedder so they don't need to download the model. The LLM
is replaced with a fake function, including cases where it returns invalid JSON or made-up job ids.
