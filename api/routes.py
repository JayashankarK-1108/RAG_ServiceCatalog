"""
api/routes.py
FastAPI route handlers — all backed by LangChain components.

GET  /health           — liveness check + index stats
POST /query            — full LangChain RAG: retrieval + LLM answer
POST /ingest           — trigger LangChain ingestion pipeline (background)
GET  /ingest/status    — poll ingestion progress
GET  /catalog/filters  — distinct metadata values for UI dropdowns
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from langchain_core.documents import Document

from api.models import (
    QueryRequest, QueryResponse, SourceDocument,
    IngestRequest, IngestResponse, HealthResponse,
)
from retrieval.chain import query as rag_query
from ingestion.checkpoint import load_checkpoint, clear_checkpoint
from config.settings import (
    PINECONE_INDEX_NAME, LLM_MODEL, EMBEDDING_MODEL,
    EXCEL_FILE_PATH, COLUMN_MAP, EXCEL_SHEET_NAME,
)
from utils.logger import get_logger

log = get_logger("routes")
router = APIRouter()

# In-memory ingestion state (single-process)
_ingest_state: dict = {"running": False, "last_result": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_to_source(doc: Document) -> SourceDocument:
    m = doc.metadata
    return SourceDocument(
        wu_id=               m.get("wu_id", ""),
        technology=          m.get("technology", ""),
        tech_tower=          m.get("tech_tower", ""),
        activities_category= m.get("activities_category", ""),
        business_scope=      m.get("business_scope", ""),
        hosting_environment= m.get("hosting_environment", ""),
        project_services=    m.get("project_services", ""),
        sla_notes=           m.get("sla_notes", ""),
        chunk_index=         int(m.get("chunk_index", 0)),
        total_chunks=        int(m.get("total_chunks", 1)),
    )


def _run_ingestion_sync(reset, resume):
    """Full LangChain ingestion pipeline — runs in background thread."""
    global _ingest_state
    _ingest_state["running"] = True
    try:
        from ingestion.loader     import ExcelServiceCatalogLoader
        from ingestion.splitter   import split_documents
        from ingestion.vectorstore import ingest_documents, delete_vectorstore_index

        if reset:
            clear_checkpoint()
            try:
                delete_vectorstore_index()
            except Exception:
                pass

        # 1. Load via LangChain Document Loader
        loader = ExcelServiceCatalogLoader(file_path=EXCEL_FILE_PATH)
        documents = loader.load()

        # 2. Split via LangChain Text Splitter
        chunks = split_documents(documents)

        # 3. Embed + upsert via LangChain PineconeVectorStore
        upserted = ingest_documents(chunks, resume=resume)

        _ingest_state["last_result"] = {
            "status":           "success",
            "documents_loaded": len(documents),
            "chunks_created":   len(chunks),
            "vectors_upserted": upserted,
            "message":          f"Ingested {upserted} vectors from "
                                f"{len(documents)} documents ({len(chunks)} chunks).",
        }
    except Exception as e:
        log.error(f"Ingestion failed: {e}")
        _ingest_state["last_result"] = {
            "status": "error", "documents_loaded": 0,
            "chunks_created": 0, "vectors_upserted": 0, "message": str(e),
        }
    finally:
        _ingest_state["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Liveness + Pinecone index stats."""
    try:
        from ingestion.vectorstore import get_index_stats
        stats = get_index_stats()
        return HealthResponse(
            status="ok",
            index_name=PINECONE_INDEX_NAME,
            vector_count=stats.total_vector_count,
            llm_model=LLM_MODEL,
            embed_model=EMBEDDING_MODEL,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pinecone unavailable: {e}")


@router.get("/debug/retrieval", tags=["System"])
def debug_retrieval(q: str = "job decommission"):
    """Diagnose retrieval pipeline step-by-step."""
    from pinecone import Pinecone
    from ingestion.embedder import get_embeddings
    from config.settings import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE

    result = {}

    # Step 1: Direct Pinecone connection
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        idx = pc.Index(PINECONE_INDEX_NAME)
        stats = idx.describe_index_stats()
        result["pinecone_total_vectors"] = stats.total_vector_count
        result["pinecone_namespaces"] = {k: v.vector_count for k, v in stats.namespaces.items()}
    except Exception as e:
        result["pinecone_error"] = str(e)
        return result

    # Step 2: Embed the query
    try:
        embeddings = get_embeddings()
        vec = embeddings.embed_query(q)
        result["embedding_dim"] = len(vec)
        result["embedding_ok"] = True
    except Exception as e:
        result["embedding_error"] = str(e)
        return result

    # Step 3: Direct Pinecone similarity query
    try:
        ns = PINECONE_NAMESPACE if PINECONE_NAMESPACE else None
        raw = idx.query(vector=vec, top_k=5, namespace=ns, include_metadata=True)
        result["direct_pinecone_matches"] = len(raw.matches)
        result["direct_pinecone_scores"] = [
            {"id": m.id, "score": round(m.score, 4)} for m in raw.matches
        ]
    except Exception as e:
        result["direct_query_error"] = str(e)

    # Step 4: LangChain vectorstore similarity_search
    try:
        from ingestion.vectorstore import get_vectorstore
        vs = get_vectorstore()
        docs = vs.similarity_search(q, k=5)
        result["langchain_docs_returned"] = len(docs)
        result["langchain_wu_ids"] = [d.metadata.get("wu_id") for d in docs]
    except Exception as e:
        result["langchain_error"] = str(e)

    return result


@router.post("/query", response_model=QueryResponse, tags=["Search"])
def query_catalog(req: QueryRequest):
    """
    Full LangChain RAG pipeline:
      1. Retrieve top-K catalog chunks from Pinecone (semantic search)
      2. Build a prompt with retrieved context
      3. Generate a natural language answer via ChatOpenAI (gpt-4o-mini)
      4. Return answer + source WU Ids + full chunk metadata
    """
    t0 = time.time()
    try:
        result = rag_query(
            question=req.query,
            filter_technology=req.filter_technology,
            filter_tech_tower=req.filter_tech_tower,
            filter_category=req.filter_category,
            top_k=req.top_k,
        )
    except Exception as e:
        log.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {e}")

    sources = [_doc_to_source(d) for d in result["source_documents"]]
    elapsed = (time.time() - t0) * 1000
    log.info(f"Query complete in {elapsed:.0f}ms | WU Ids: {result['wu_ids']}")

    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        wu_ids=result["wu_ids"],
        source_documents=sources,
        total_sources=len(sources),
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def trigger_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger the LangChain ingestion pipeline in the background.
    Poll GET /ingest/status for progress.
    """
    if _ingest_state["running"]:
        raise HTTPException(status_code=409,
            detail="Ingestion already running. Poll GET /ingest/status.")
    background_tasks.add_task(
        _run_ingestion_sync,
        reset=req.reset,
        resume=req.resume,
    )
    return IngestResponse(
        status="started", documents_loaded=0, chunks_created=0,
        vectors_upserted=0,
        message="Ingestion started. Poll GET /ingest/status for updates.",
    )


@router.get("/ingest/status", tags=["Ingestion"])
def ingest_status():
    """Poll ingestion progress via checkpoint + in-memory state."""
    cp = load_checkpoint()
    return {
        "running":         _ingest_state["running"],
        "upserted_so_far": len(cp.get("upserted_ids", [])),
        "total":           cp.get("total", 0),
        "last_updated":    cp.get("updated_at"),
        "last_result":     _ingest_state.get("last_result"),
    }


@router.get("/catalog/filters", tags=["Catalog"])
def get_filter_values():
    """Distinct metadata values — useful for building UI filter dropdowns."""
    try:
        import pandas as pd
        sheet = int(EXCEL_SHEET_NAME) if EXCEL_SHEET_NAME.isdigit() else EXCEL_SHEET_NAME
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet, dtype=str)

        def distinct(col):
            return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

        return {
            "technologies":          distinct(COLUMN_MAP["technology"]),
            "tech_towers":           distinct(COLUMN_MAP["tech_tower"]),
            "activities_categories": distinct(COLUMN_MAP["activities_category"]),
            "hosting_environments":  distinct(COLUMN_MAP["hosting_environment"]),
            "business_scopes":       distinct(COLUMN_MAP["business_scope"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
