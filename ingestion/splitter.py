"""
ingestion/splitter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangChain Text Splitter stage.

Strategy:
  • Most catalog rows fit within CHUNK_SIZE characters, so they
    pass through as a single chunk (no split needed).
  • Long service descriptions (rare) are split by
    RecursiveCharacterTextSplitter while preserving all metadata.
  • The WU Id is prepended to EVERY chunk so retrieval always
    returns the identifier, even for partial chunks.

Flow:
  List[Document]  →  [RecursiveCharacterTextSplitter]  →  List[Document]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

log = get_logger("splitter")


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents using LangChain's RecursiveCharacterTextSplitter.

    Separators are chosen to keep structured labels intact:
      1. Split on newlines first  (preserves "Field: Value" lines)
      2. Fall back to sentences
      3. Fall back to words
      4. Fall back to characters

    All original metadata is copied to child chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,   # adds 'start_index' to metadata for traceability
    )

    split_docs: List[Document] = []
    original_count = len(documents)
    split_count = 0

    for doc in documents:
        wu_id = doc.metadata.get("wu_id", "unknown")

        if len(doc.page_content) <= CHUNK_SIZE:
            # Short enough — keep as-is, just tag it
            doc.metadata["chunk_index"] = 0
            doc.metadata["total_chunks"] = 1
            split_docs.append(doc)
        else:
            # Long description — split and tag each chunk
            chunks = splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
                # Ensure WU Id is always at the start of the chunk text
                if not chunk.page_content.startswith(f"Work Unit ID: {wu_id}"):
                    chunk.page_content = (
                        f"Work Unit ID: {wu_id}\n" + chunk.page_content
                    )
            split_docs.extend(chunks)
            split_count += 1
            log.info(f"  Split WU '{wu_id}' into {len(chunks)} chunks "
                     f"(original length: {len(doc.page_content)} chars)")

    log.info(
        f"Splitter complete — {original_count} docs → {len(split_docs)} chunks "
        f"({split_count} docs were split)"
    )
    return split_docs
