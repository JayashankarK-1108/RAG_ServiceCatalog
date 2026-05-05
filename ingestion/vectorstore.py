"""
ingestion/vectorstore.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangChain PineconeVectorStore integration.

Provides:
  get_vectorstore()          — returns LangChain PineconeVectorStore
                               (shared singleton, used by both ingestion
                               and the retrieval chain)
  ingest_documents()         — upserts LangChain Documents in batches
                               with checkpoint-based resume support
  delete_vectorstore_index() — wipe the Pinecone index for re-ingestion

LangChain's PineconeVectorStore.from_documents() handles:
  • Calling get_embeddings() internally
  • Serialising metadata alongside vectors
  • Batching upserts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import time
from typing import List, Optional, Dict
from functools import lru_cache

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import ForbiddenException
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.embedder import get_embeddings
from ingestion.checkpoint import load_checkpoint, save_checkpoint
from config.settings import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD,
    PINECONE_REGION, PINECONE_METRIC, PINECONE_NAMESPACE,
    EMBEDDING_DIMENSIONS, UPSERT_BATCH_SIZE, INGEST_DELAY_MS,
)
from utils.logger import get_logger

log = get_logger("vectorstore", "logs/ingest.log")


def _ensure_index() -> Pinecone:
    """Create the Pinecone index if it doesn't already exist, then wait until ready."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        log.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' …")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        except ForbiddenException as e:
            raise RuntimeError(
                f"Pinecone rejected index creation for '{PINECONE_INDEX_NAME}'. "
                f"You may have hit the maximum number of serverless indexes on your plan. "
                f"Delete an unused index at https://app.pinecone.io or set "
                f"PINECONE_INDEX_NAME to an existing index.\n"
                f"Original error: {e}"
            ) from e
        # Wait until the index is ready before returning
        log.info("Waiting for index to become ready …")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
        log.info("Index ready")
    else:
        log.info(f"Using existing index '{PINECONE_INDEX_NAME}'")
    return pc


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    """
    Return a cached LangChain PineconeVectorStore instance.
    Used by both the ingestion pipeline and the retrieval chain.
    """
    _ensure_index()
    log.info("Initialising LangChain PineconeVectorStore …")
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        namespace=PINECONE_NAMESPACE,
        pinecone_api_key=PINECONE_API_KEY,
    )


def ingest_documents(
    documents: List[Document],
    resume: bool = True,
) -> int:
    """
    Upsert LangChain Documents into Pinecone in batches.

    Each Document's page_content is embedded via LangChain OpenAIEmbeddings.
    Metadata is stored as-is alongside the vector.

    Args:
        documents: List of LangChain Document objects (post-splitter)
        resume:    Skip documents whose IDs are already in the checkpoint

    Returns:
        Number of documents newly upserted this run
    """
    # Build stable IDs from WU Id + chunk index
    def _doc_id(doc: Document) -> str:
        wu = doc.metadata.get("wu_id", "unknown")
        ci = doc.metadata.get("chunk_index", 0)
        return f"wu-{wu}-{ci}".replace(" ", "_").replace("/", "-")

    # Resume: skip already-done IDs
    checkpoint = load_checkpoint() if resume else {"upserted_ids": []}
    done_ids = set(checkpoint.get("upserted_ids", []))

    pending = [d for d in documents if _doc_id(d) not in done_ids]
    skipped = len(documents) - len(pending)
    if skipped:
        log.info(f"Resuming — skipping {skipped} already-upserted documents")

    log.info(f"Upserting {len(pending)} documents "
             f"(batch size={UPSERT_BATCH_SIZE}) …")

    vectorstore = get_vectorstore()
    newly_upserted = 0

    for i in tqdm(range(0, len(pending), UPSERT_BATCH_SIZE),
                  desc="Upserting to Pinecone", unit="batch"):
        batch = pending[i: i + UPSERT_BATCH_SIZE]
        ids   = [_doc_id(d) for d in batch]

        # LangChain PineconeVectorStore.add_documents() handles embedding + upsert
        vectorstore.add_documents(documents=batch, ids=ids)
        newly_upserted += len(batch)

        done_ids.update(ids)
        save_checkpoint(done_ids, total=len(documents))

        if i + UPSERT_BATCH_SIZE < len(pending):
            time.sleep(INGEST_DELAY_MS / 1000)

    log.info(f"✅ Upserted {newly_upserted} new documents | "
             f"Total: {newly_upserted + skipped}")
    return newly_upserted


def get_index_stats() -> Dict:
    pc = _ensure_index()
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    log.info(f"Index stats → total vectors: {stats.total_vector_count}")
    return stats


def delete_vectorstore_index() -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc.delete_index(PINECONE_INDEX_NAME)
    get_vectorstore.cache_clear()   # clear the lru_cache
    log.warning(f"Deleted Pinecone index '{PINECONE_INDEX_NAME}'")
