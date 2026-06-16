"""api/models.py — Pydantic v2 request/response models."""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
        description="Natural language question about a technical activity",
        examples=["How do I configure an existing Control-M job?"])
    top_k: int = Field(default=5, ge=1, le=20,
        description="Number of catalog chunks to retrieve")
    filter_technology: Optional[str] = Field(default=None,
        description="Pre-filter by Technology field (e.g. 'Control-M')")
    filter_tech_tower: Optional[str] = Field(default=None,
        description="Pre-filter by Tech Tower (e.g. 'Scheduling')")
    filter_category: Optional[str] = Field(default=None,
        description="Pre-filter by Activities Category (e.g. 'Migration')")


class IngestRequest(BaseModel):
    reset: bool = Field(default=False,
        description="Delete & recreate Pinecone index first")
    resume: bool = Field(default=True,
        description="Skip already-upserted IDs via checkpoint")


# ── Responses ─────────────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    wu_id:               str
    technology:          str
    tech_tower:          str
    activities_category: str
    business_scope:      str
    hosting_environment: str
    project_services:    str
    sla_notes:           str
    price_per_hour:      str = ""
    effort_hours:        str = ""
    total_price:         str = ""
    delivery_lead_days:  str = ""
    chunk_index:         int = 0
    total_chunks:        int = 1


class QueryResponse(BaseModel):
    query:            str
    answer:           str = Field(description="LLM-generated natural language answer")
    wu_ids:           List[str] = Field(description="All matched WU Ids")
    source_documents: List[SourceDocument] = Field(description="Retrieved catalog chunks")
    total_sources:    int


class IngestResponse(BaseModel):
    status:           str
    documents_loaded: int
    chunks_created:   int
    vectors_upserted: int
    message:          str


class HealthResponse(BaseModel):
    status:       str
    index_name:   str
    vector_count: int
    llm_model:    str
    embed_model:  str
