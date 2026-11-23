import hashlib
import os
import re
from pathlib import Path

import numpy as np
import psycopg
import pytest

from app import db

TEST_DB = os.getenv(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/jobmatch_test"
)
ROOT = Path(__file__).resolve().parent.parent


class FakeEmbedder:
    """Bag-of-words hashed into 384 dims. Texts that share words end up close,
    which is enough to test the search logic without downloading a model."""

    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9+#]+", text.lower()):
            if len(word) < 3:
                continue
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="session")
def database_url():
    try:
        psycopg.connect(TEST_DB, connect_timeout=3).close()
    except psycopg.OperationalError:
        pytest.skip(f"test database not reachable at {TEST_DB}")
    db.init_db(TEST_DB)
    return TEST_DB


@pytest.fixture
def conn(database_url):
    with db.connect(database_url) as c:
        c.execute("TRUNCATE jobs RESTART IDENTITY")
        c.commit()
        yield c


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def loaded(conn, embedder):
    """Database with the 30 sample jobs in it."""
    from scripts.ingest import ingest, read_csv

    ingest(conn, embedder, read_csv(ROOT / "data" / "sample_jobs.csv"))
    return conn
