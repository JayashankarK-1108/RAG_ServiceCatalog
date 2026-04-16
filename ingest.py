"""
ingest.py — 🚀 LangChain Ingestion Pipeline CLI

Stages:
  1. ExcelServiceCatalogLoader  — reads Excel → LangChain Documents
  2. RecursiveCharacterTextSplitter — splits long descriptions
  3. OpenAIEmbeddings           — embeds via LangChain
  4. PineconeVectorStore        — upserts with checkpoint resume

Usage:
    python ingest.py                        # standard (resume enabled)
    python ingest.py --reset                # wipe index + re-ingest all
    python ingest.py --file path/to/file.xlsx
    python ingest.py --no-resume            # re-embed everything
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("logs", exist_ok=True)

from ingestion.loader      import ExcelServiceCatalogLoader
from ingestion.splitter    import split_documents
from ingestion.vectorstore import ingest_documents, delete_vectorstore_index, get_index_stats
from ingestion.checkpoint  import clear_checkpoint
from config.settings       import EXCEL_FILE_PATH
from utils.logger          import get_logger

log = get_logger("ingest", "logs/ingest.log")


def run(file_path=None, reset=False, resume=True):
    t0 = time.time()
    log.info("═" * 62)
    log.info("  SERVICE CATALOG RAG — LangChain Ingestion Pipeline")
    log.info("═" * 62)

    if reset:
        log.warning("--reset: clearing checkpoint + deleting Pinecone index …")
        clear_checkpoint()
        try:
            delete_vectorstore_index()
        except Exception as e:
            log.warning(f"Could not delete index: {e}")

    # ── Stage 1: LangChain Document Loader ───────────────────────────────────
    log.info("\n[1/3] Loading documents via ExcelServiceCatalogLoader …")
    loader = ExcelServiceCatalogLoader(file_path=file_path or EXCEL_FILE_PATH)
    documents = loader.load()
    log.info(f"  → {len(documents)} Documents loaded")

    if not documents:
        log.error("No documents loaded — check Excel path and column config.")
        sys.exit(1)

    # ── Stage 2: LangChain Text Splitter ─────────────────────────────────────
    log.info("\n[2/3] Splitting via RecursiveCharacterTextSplitter …")
    chunks = split_documents(documents)
    log.info(f"  → {len(chunks)} chunks after splitting")

    # ── Stage 3: LangChain Embeddings + PineconeVectorStore ──────────────────
    log.info("\n[3/3] Embedding + upserting via LangChain PineconeVectorStore …")
    upserted = ingest_documents(chunks, resume=resume)

    elapsed = time.time() - t0
    log.info("\n" + "═" * 62)
    log.info("  ✅ INGESTION COMPLETE")
    log.info(f"     Documents loaded   : {len(documents)}")
    log.info(f"     Chunks created     : {len(chunks)}")
    log.info(f"     Vectors upserted   : {upserted}")
    log.info(f"     Time elapsed       : {elapsed:.1f}s")
    log.info("═" * 62)

    stats = get_index_stats()
    log.info(f"Pinecone index total vectors: {stats.total_vector_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain Service Catalog Ingestion")
    parser.add_argument("--file",      "-f", type=str, default=None)
    parser.add_argument("--reset",     action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run(file_path=args.file, reset=args.reset, resume=not args.no_resume)
