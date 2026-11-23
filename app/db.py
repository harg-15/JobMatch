from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import settings

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def init_db(database_url: str | None = None) -> None:
    """Create the extension, table and indexes if they don't exist yet."""
    with psycopg.connect(database_url or settings.database_url, autocommit=True) as conn:
        conn.execute(SCHEMA_FILE.read_text())


def connect(database_url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(database_url or settings.database_url, row_factory=dict_row)
    register_vector(conn)  # lets us pass numpy arrays straight into queries
    return conn
