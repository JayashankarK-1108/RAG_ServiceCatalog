# 🗂️ Service Catalog Agentic RAG — LangChain Edition

> Full LangChain RAG pipeline for querying the IT Service Catalog.
> Users ask natural language questions; the system retrieves matching catalog rows
> from Pinecone and uses an LLM to generate a clear, actionable answer with **WU Ids**.

---

## 🧱 LangChain Components Used

| Stage | LangChain Component | Purpose |
|-------|--------------------|---------| 
| Load | `BaseLoader` → `ExcelServiceCatalogLoader` | Reads Excel rows → `Document` objects |
| Split | `RecursiveCharacterTextSplitter` | Splits long descriptions into chunks |
| Embed | `OpenAIEmbeddings` | Vectorises chunks (`text-embedding-3-small`) |
| Store | `PineconeVectorStore` | Stores + retrieves vectors |
| Retrieve | `VectorStoreRetriever` | Semantic search with metadata filters |
| Generate | `ChatOpenAI` + `ChatPromptTemplate` | LLM answer generation (gpt-4o-mini) |
| Chain | **LCEL** `RunnablePassthrough` + `RunnableLambda` | Composable RAG pipeline |
| Parse | `StrOutputParser` | Extracts text from LLM response |

---

## 📐 Architecture

```
                        ┌──────────────────────────────────────┐
                        │         INGESTION PIPELINE           │
                        │                                      │
  Excel File            │  ExcelServiceCatalogLoader           │
  (500+ rows)  ────────▶│    ↓ List[Document]                  │
                        │  RecursiveCharacterTextSplitter       │
                        │    ↓ List[Document] (chunks)         │
                        │  OpenAIEmbeddings                    │
                        │    ↓ vectors                         │
                        │  PineconeVectorStore.add_documents() │
                        └──────────────────────────────────────┘
                                        │
                               Pinecone Index
                                        │
                        ┌──────────────────────────────────────┐
                        │         RETRIEVAL CHAIN (LCEL)       │
                        │                                      │
  User Query   ────────▶│  OpenAIEmbeddings.embed_query()      │
                        │    ↓ query vector                    │
                        │  PineconeVectorStore.as_retriever()  │
                        │    ↓ top-K Documents                 │
                        │  ChatPromptTemplate (context inject) │
                        │    ↓ formatted prompt                │
                        │  ChatOpenAI (gpt-4o-mini)            │
                        │    ↓ AIMessage                       │
                        │  StrOutputParser                     │
                        │    ↓ answer string                   │
                        └──────────────────────────────────────┘
                                        │
                             FastAPI POST /query
                             ↓
                    { answer, wu_ids, source_documents }
```

---

## 📁 Project Structure

```
service_catalog_rag_lc/
│
├── .github/workflows/
│   ├── ingest.yml          ← 🔁 LangChain ingestion (auto/manual/scheduled)
│   └── deploy.yml          ← 🚀 CI/CD lint + smoke test + Render deploy
│
├── api/
│   ├── main.py             ← FastAPI app + CORS + Swagger
│   ├── models.py           ← Pydantic v2 request/response schemas
│   └── routes.py           ← /health /query /ingest /catalog/filters
│
├── config/
│   └── settings.py         ← All config — reads env vars / .env
│
├── data/
│   └── service_catalog.xlsx
│
├── ingestion/
│   ├── loader.py           ← LangChain BaseLoader — Excel → Documents
│   ├── splitter.py         ← LangChain RecursiveCharacterTextSplitter
│   ├── embedder.py         ← LangChain OpenAIEmbeddings (singleton)
│   ├── checkpoint.py       ← Resume-on-failure (JSON checkpoint)
│   └── vectorstore.py      ← LangChain PineconeVectorStore + batch upsert
│
├── retrieval/
│   └── chain.py            ← LCEL RAG chain: retriever → prompt → LLM → parser
│
├── scripts/
│   ├── validate_excel.py   ← Pre-flight Excel validation
│   └── test_query.py       ← End-to-end API smoke tests
│
├── utils/
│   └── logger.py
│
├── ingest.py               ← 🚀 CLI ingestion runner
├── start_api.py            ← 🌐 Local API launcher
├── render.yaml             ← Render Blueprint (auto-detected)
├── Procfile                ← Render start command
├── runtime.txt             ← Python 3.11
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚙️ One-Time Setup

### 1. Clone & configure
```bash
git clone https://github.com/YOUR_ORG/service-catalog-rag.git
cd service-catalog-rag
cp .env.example .env        # fill in OPENAI_API_KEY + PINECONE_API_KEY
```

### 2. Add GitHub Secrets
`Settings → Secrets and variables → Actions`:

| Secret | Where to get it |
|--------|----------------|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `PINECONE_API_KEY` | Pinecone dashboard → API Keys |
| `PINECONE_INDEX_NAME` | e.g. `service-catalog` |
| `PINECONE_CLOUD` | e.g. `aws` |
| `PINECONE_REGION` | e.g. `us-east-1` |
| `RENDER_DEPLOY_HOOK` | Render → Service → Settings → Deploy Hook |

### 3. Deploy to Render
1. Go to https://dashboard.render.com/new/blueprint
2. Connect your GitHub repo — Render auto-detects `render.yaml`
3. Fill in the 3 `sync: false` secrets in the Render dashboard
4. Click **Apply** → your API is live

---

## 🔁 GitHub Actions Workflows

### `ingest.yml` — LangChain Ingestion

| Trigger | When |
|---------|------|
| **Auto** | Push to `main` changing `data/service_catalog.xlsx` |
| **Scheduled** | Every Sunday 00:00 UTC |
| **Manual** | Actions → "🗂️ Service Catalog Ingestion" → Run workflow |

**Manual inputs:**

| Input | Default | Description |
|-------|---------|-------------|
| `reset` | `false` | Wipe Pinecone index and re-ingest everything |
| `excel_file` | _(blank)_ | Custom Excel path |
| `redeploy_render` | `true` | Trigger Render redeploy after ingestion |

**Update the catalog:**
```bash
cp new_catalog.xlsx data/service_catalog.xlsx
git add data/service_catalog.xlsx
git commit -m "chore: update service catalog"
git push origin main
# → ingest.yml triggers automatically
```

---

## 💻 Local Development

```bash
pip install -r requirements.txt

# Validate Excel
python scripts/validate_excel.py

# Ingest (first run or after catalog update)
python ingest.py

# Start API
python start_api.py
# → http://localhost:8000/docs

# Run smoke tests
python scripts/test_query.py
```

---

## 🌐 API Reference

### `POST /query` — Full LangChain RAG

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I configure an existing Control-M job?",
    "top_k": 5
  }'
```

**With metadata pre-filters:**
```json
{
  "query": "migrate a job to another server",
  "top_k": 3,
  "filter_technology": "Control-M",
  "filter_category": "Migration"
}
```

**Response:**
```json
{
  "query": "How do I configure an existing Control-M job?",
  "answer": "To configure an existing Control-M job, you should raise a **Configuration** request under **WU Id IMS_WU_7036**.\n\nThis service covers job configuration and modification for existing applications — including modifying or updating an existing job or making configuration changes.\n\nThe effort level is **L3 Implementation activity**, measured per job. Contact the Scheduling team to proceed.",
  "wu_ids": ["IMS_WU_7036"],
  "source_documents": [
    {
      "wu_id": "IMS_WU_7036",
      "technology": "Control-M",
      "tech_tower": "Scheduling",
      "activities_category": "Configuration",
      "business_scope": "Global",
      "hosting_environment": "On_Prem",
      "project_services": "Job configuration/modification for existing application...",
      "sla_notes": "L3 Implementation activity - Measured per job",
      "chunk_index": 0,
      "total_chunks": 1
    }
  ],
  "total_sources": 1
}
```

### Other Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + vector count + model info |
| POST | `/ingest` | Trigger background ingestion |
| GET | `/ingest/status` | Poll progress (`upserted_so_far / total`) |
| GET | `/catalog/filters` | Distinct field values for UI dropdowns |

---

## ⚙️ Key Config (`config/settings.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | ChatOpenAI model for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAIEmbeddings model |
| `SEARCH_TYPE` | `similarity` | `similarity` or `mmr` (diversity re-ranking) |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `MIN_SCORE_THRESHOLD` | `0.30` | Minimum cosine similarity |
| `CHUNK_SIZE` | `800` | Max chars per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `LLM_TEMPERATURE` | `0.0` | Deterministic LLM output |
| `UPSERT_BATCH_SIZE` | `100` | Vectors per Pinecone upsert batch |

---

## 🔄 MMR vs Similarity Search

Change `SEARCH_TYPE` in `.env` or `config/settings.py`:

```
SEARCH_TYPE=mmr          # Maximal Marginal Relevance — reduces redundant results
SEARCH_TYPE=similarity   # Standard cosine similarity (default)
```

MMR is useful when multiple rows have similar descriptions — it ensures
diverse results are returned rather than near-duplicates.
