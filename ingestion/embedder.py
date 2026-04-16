"""
ingestion/embedder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangChain OpenAIEmbeddings wrapper.

Provides:
  get_embeddings()   — singleton OpenAIEmbeddings instance (reused
                       across ingestion and retrieval)
  embed_documents()  — embed a batch of texts (for ingestion)
  embed_query()      — embed a single query string (for retrieval)

Both ingestion and retrieval import from here, guaranteeing the
exact same model + dimensions are used end-to-end.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
from functools import lru_cache
from typing import List

from langchain_openai import OpenAIEmbeddings

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
from utils.logger import get_logger

log = get_logger("embedder")


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Return a cached LangChain OpenAIEmbeddings instance.
    Cached so the same object is reused across the entire application
    (avoids creating duplicate clients during retrieval).
    """
    log.info(f"Initialising OpenAIEmbeddings — model: {EMBEDDING_MODEL}, "
             f"dimensions: {EMBEDDING_DIMENSIONS}")
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a list of document texts (used during ingestion)."""
    return get_embeddings().embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string (used during retrieval)."""
    return get_embeddings().embed_query(query)
