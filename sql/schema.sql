CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,              -- 'csv', 'greenhouse', ...
    external_id  TEXT NOT NULL,              -- id of the job in the source system
    title        TEXT NOT NULL,
    company      TEXT NOT NULL,
    location     TEXT,
    url          TEXT,
    description  TEXT NOT NULL,
    embedding    vector(384),                -- bge-small-en-v1.5 output size
    search_tsv   tsvector GENERATED ALWAYS AS (
                     to_tsvector('english', title || ' ' || description)
                 ) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- re-running an ingest must never create duplicate rows
    UNIQUE (source, external_id)
);

-- approximate nearest neighbour index for cosine distance (<=>)
CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw
    ON jobs USING hnsw (embedding vector_cosine_ops);

-- full text index, used by the keyword baseline
CREATE INDEX IF NOT EXISTS jobs_search_tsv_gin
    ON jobs USING gin (search_tsv);
