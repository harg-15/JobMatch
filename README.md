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

To use a local model instead of an API, see the Ollama block in `.env.example`. Small local
models have short context windows, so re-ranking 10 candidates instead of 20 works better.

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
are text files in `eval/resumes/`. `eval/labels_sample.csv` holds the labels for the 30 made-up jobs
in `data/sample_jobs.csv`.

Note: precision@5 can't reach 1.0 for a resume that only has 2 good jobs in the data, which is why
recall is printed too.

### Results on real jobs

Data: 760 live postings from the Greenhouse boards of airbnb, figma, robinhood and coinbase
(loaded 2026-11-23). 4 student resumes, 20 labeled (resume, job) pairs: 7 backend, 6 frontend,
5 data analyst, 2 ML. Re-rank used `llama3.2:3b` (8k context) through Ollama on CPU, top 20
candidates in, top 5 out. Full output: `eval/results.txt`.

| mode    | precision@5 | recall@5 | median latency |
|---------|-------------|----------|----------------|
| keyword | 0.05        | 0.04     | 0.02 s         |
| vector  | 0.10        | 0.10     | 0.03 s         |
| rerank  | 0.10        | 0.10     | 511 s (CPU)    |

The LLM answered for all 4 resumes (no fallbacks to vector search).

What this shows:

- Vector search beats the keyword baseline, but both are low in absolute terms.
- **Re-ranking did not help here, because retrieval is the bottleneck.** For 3 of the 4 resumes none
  of the labeled jobs are in the vector top 20 at all (recall@20 was 0 for backend, frontend and ML;
  3 of 5 for data analyst), so the re-ranker never sees them. The pool is mostly senior and
  non-engineering roles, and embeddings rank "Senior Software Engineer" above "Software Engineer
  Intern" for a student resume. The re-ranker can only reorder what retrieval hands it.
- Latency for rerank is dominated by running a 3B model on CPU (about 3.5 tokens/s). A hosted
  model would be far faster, but would be a different number.

Caveats: only 20 labels from a single labeler (labeled by reading the postings, before looking at
what the system ranked), 4 resumes, and one run at temperature 0. Treat these as a baseline, not
as a precise measurement. A seniority-aware retrieval step (so intern and new-grad roles reach
the re-ranker) is the obvious next experiment.

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
