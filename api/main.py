"""
api/main.py — FastAPI application entry point.

Run locally:
    python start_api.py
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config.settings import API_TITLE, API_VERSION, LLM_MODEL, EMBEDDING_MODEL
from utils.logger import get_logger

log = get_logger("main")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info(f"🚀 {API_TITLE} v{API_VERSION}")
    log.info(f"   LLM      : {LLM_MODEL}")
    log.info(f"   Embeddings: {EMBEDDING_MODEL}")
    log.info("   Docs     : http://localhost:8000/docs")
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
    description=f"""
## 🗂️ Service Catalog RAG API — LangChain Edition

Query the IT Service Catalog using **natural language**.
The system uses a full LangChain RAG pipeline:
- **LangChain Document Loader** — reads Excel row-by-row
- **LangChain Text Splitter** — RecursiveCharacterTextSplitter
- **LangChain OpenAIEmbeddings** — `{EMBEDDING_MODEL}`
- **LangChain PineconeVectorStore** — semantic retrieval
- **LangChain RetrievalQA (LCEL)** — ChatOpenAI `{LLM_MODEL}` generates the answer

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Full RAG — returns LLM answer + WU Ids + sources |
| `/health` | GET | Liveness + Pinecone stats |
| `/ingest` | POST | Trigger ingestion pipeline |
| `/ingest/status` | GET | Poll ingestion progress |
| `/catalog/filters` | GET | Filter dropdown values |
""",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the chatbot UI
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(_static_dir, "index.html"))
