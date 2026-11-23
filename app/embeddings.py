"""Turns text into vectors.

bge models are trained with an instruction prefix on the *query* side, so
resumes go through embed_query() and job postings through embed_documents().
fastembed handles the prefix for us (query_embed vs passage_embed).
"""
from typing import Protocol

import numpy as np

from app.config import settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model_name: str = settings.embed_model):
        # imported here so tests don't need to download the model
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.dim = settings.embed_dim

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return list(self.model.passage_embed(texts, batch_size=32))

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self.model.query_embed(text)))


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    # loading the model takes a few seconds, so do it once per process
    global _embedder
    if _embedder is None:
        _embedder = FastEmbedder()
    return _embedder


def job_to_text(title: str, company: str, description: str) -> str:
    # the model only looks at the first ~512 tokens, so put the most
    # important fields first
    return f"{title} at {company}. {description}"
