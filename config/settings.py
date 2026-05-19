"""
config/settings.py
All configuration for the LangChain Service Catalog RAG.
Values can be overridden via environment variables or a .env file.

Excel column layout (8 fixed columns):
  A  WU Id                    B  Business Scope
  C  Hosting Environment      D  Tech Tower
  E  Technology               F  Activities Category
  G  Project related Services H  Column1 (SLA / effort notes)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI (embeddings only) ───────────────────────────────────────────────────
_openai_key = os.getenv("OPENAI_API_KEY")
if not _openai_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
OPENAI_API_KEY: str       = _openai_key
EMBEDDING_MODEL: str      = "text-embedding-3-small"   # LangChain OpenAIEmbeddings
EMBEDDING_DIMENSIONS: int = 1536

# ── Anthropic (LLM) ────────────────────────────────────────────────────────────
_anthropic_key = os.getenv("ANTHROPIC_API_KEY")
if not _anthropic_key:
    raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
ANTHROPIC_API_KEY: str    = _anthropic_key
LLM_MODEL: str            = "claude-sonnet-4-6"        # LangChain ChatAnthropic
LLM_TEMPERATURE: float    = 0.0                        # deterministic answers
LLM_MAX_TOKENS: int       = 1024

# ── Pinecone ───────────────────────────────────────────────────────────────────
_pinecone_key = os.getenv("PINECONE_API_KEY")
if not _pinecone_key:
    raise EnvironmentError("PINECONE_API_KEY environment variable is not set.")
PINECONE_API_KEY: str     = _pinecone_key
PINECONE_INDEX_NAME: str  = os.getenv("PINECONE_INDEX_NAME", "rag-servicecatalog")
PINECONE_CLOUD: str       = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str      = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC: str      = "cosine"
PINECONE_NAMESPACE: str   = os.getenv("PINECONE_NAMESPACE", "")

# ── Excel ──────────────────────────────────────────────────────────────────────
EXCEL_FILE_PATH: str  = os.getenv("EXCEL_FILE_PATH", "data/service_catalog.xlsx")
EXCEL_SHEET_NAME: str = os.getenv("EXCEL_SHEET_NAME", "0")   # "0" = first sheet

# Exact column header names in the Excel file
COLUMN_MAP: dict = {
    "wu_id":               "WU Id",
    "business_scope":      "Business Scope",
    "hosting_environment": "Hosting Environment",
    "tech_tower":          "Tech Tower",
    "technology":          "Technology",
    "activities_category": "Activities Category",
    "project_services":    "Project related Services",
    "sla_notes":           "Column1",
}

# ── LangChain Text Splitter ────────────────────────────────────────────────────
# Each Excel row becomes one Document; splitter is applied if description is long
CHUNK_SIZE: int     = 800    # characters per chunk
CHUNK_OVERLAP: int  = 100    # overlap between chunks

# ── LangChain Retriever ────────────────────────────────────────────────────────
TOP_K_RESULTS: int         = 5
MIN_SCORE_THRESHOLD: float = 0.30
SEARCH_TYPE: str           = "similarity"   # "similarity" | "mmr"
FETCH_K_MMR: int           = 20             # candidate pool for MMR re-ranking

# ── LangChain RAG Chain ────────────────────────────────────────────────────────
# System prompt injected into every RAG call
RAG_SYSTEM_PROMPT: str = """You are a helpful IT Service Catalog assistant.
Your job is to answer questions about technical activities by looking up the
Service Catalog and identifying the correct Work Unit (WU Id) and service details.

When answering:
1. Always state the WU Id(s) clearly at the start of your answer.
2. Explain what the service covers based on the catalog description.
3. Mention the SLA or effort level if available.
4. Be concise and actionable — the user needs to raise a ticket.
5. If multiple services match, list all relevant WU Ids.
6. If no relevant service is found, say so clearly and suggest the user
   contact the Service Desk."""

# ── Ingestion ─────────────────────────────────────────────────────────────────
UPSERT_BATCH_SIZE: int = 100
INGEST_DELAY_MS: int   = 50
CHECKPOINT_FILE: str   = "data/.ingest_checkpoint.json"
SKIP_EMPTY_ROWS: bool  = True

# ── FastAPI ────────────────────────────────────────────────────────────────────
API_HOST: str    = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int    = int(os.getenv("API_PORT", "8000"))
API_TITLE: str   = "Service Catalog RAG API (LangChain)"
API_VERSION: str = "2.0.0"
