"""
retrieval/chain.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangChain RAG Chain for Service Catalog queries.

Architecture:
  User Query
      │
      ▼
  [LangChain Retriever]          ← PineconeVectorStore.as_retriever()
  Semantic search (top-K)            with optional metadata filters
      │
      ▼
  [Retrieved Documents]          ← LangChain Documents with metadata
      │
      ▼
  [RAG Prompt]                   ← SystemMessage + context + question
      │
      ▼
  [ChatAnthropic LLM]            ← claude-sonnet-4-6
      │
      ▼
  [StrOutputParser]
      │
      ▼
  Natural language answer + structured metadata

LangChain components used:
  • PineconeVectorStore.as_retriever()     — vector retriever
  • ChatPromptTemplate                     — structured prompt
  • ChatAnthropic                          — LLM
  • LCEL pipe (prompt | llm | parser)     — answer chain composition
  • StrOutputParser                        — parse LLM output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_anthropic import ChatAnthropic

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.vectorstore import get_vectorstore
from config.settings import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    TOP_K_RESULTS, SEARCH_TYPE, FETCH_K_MMR, MIN_SCORE_THRESHOLD,
    RAG_SYSTEM_PROMPT,
)
from utils.logger import get_logger

log = get_logger("chain")


# ── LLM singleton ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Cached LangChain ChatAnthropic instance."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
    log.info(f"Initialising ChatAnthropic — model: {LLM_MODEL}, "
             f"temperature: {LLM_TEMPERATURE}")
    return ChatAnthropic(
        api_key=ANTHROPIC_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# ── Prompt template ───────────────────────────────────────────────────────────

def _build_prompt() -> ChatPromptTemplate:
    """
    Build the RAG prompt template.
    The {context} placeholder is filled with retrieved catalog chunks.
    The {question} placeholder is filled with the user query.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            RAG_SYSTEM_PROMPT + "\n\n"
            "Service Catalog Context:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "{context}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━"
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])


# ── Context formatter ─────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """
    Format retrieved Documents into a single context string for the prompt.
    Each document is labelled with its WU Id for unambiguous LLM reference.
    """
    if not docs:
        return "No relevant services found in the catalog."

    sections = []
    for i, doc in enumerate(docs, 1):
        wu = doc.metadata.get("wu_id", "N/A")
        sections.append(f"[Service {i} — WU Id: {wu}]\n{doc.page_content}")

    return "\n\n".join(sections)


# ── Retriever builder ─────────────────────────────────────────────────────────

def _build_retriever(
    filter_technology: Optional[str] = None,
    filter_tech_tower: Optional[str] = None,
    filter_category: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
):
    """
    Build a LangChain PineconeVectorStore retriever with optional
    metadata pre-filters and configurable search strategy.

    Supports both 'similarity' and 'mmr' (Maximal Marginal Relevance)
    search types, both native to LangChain's vectorstore interface.
    """
    vectorstore = get_vectorstore()

    # Build Pinecone metadata filter dict
    pinecone_filter: Dict[str, Any] = {}
    if filter_technology:
        pinecone_filter["technology"] = {"$eq": filter_technology}
    if filter_tech_tower:
        pinecone_filter["tech_tower"] = {"$eq": filter_tech_tower}
    if filter_category:
        pinecone_filter["activities_category"] = {"$eq": filter_category}

    search_kwargs: Dict[str, Any] = {"k": top_k}
    if pinecone_filter:
        search_kwargs["filter"] = pinecone_filter
        log.info(f"Retriever metadata filter: {pinecone_filter}")

    if SEARCH_TYPE == "mmr":
        # MMR reduces redundancy among returned results
        search_kwargs["fetch_k"] = FETCH_K_MMR
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )
        log.info(f"Using MMR retriever (k={top_k}, fetch_k={FETCH_K_MMR})")
    else:
        # similarity_score_threshold filters out low-confidence matches
        search_kwargs["score_threshold"] = MIN_SCORE_THRESHOLD
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )
        log.info(f"Using similarity retriever (k={top_k}, "
                 f"score_threshold={MIN_SCORE_THRESHOLD})")

    return retriever


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def build_rag_chain(
    filter_technology: Optional[str] = None,
    filter_tech_tower: Optional[str] = None,
    filter_category: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
):
    """
    Assemble the LangChain answer-only chain (prompt → LLM → parser).
    The retriever is kept separate so source docs can be reused without
    a second Pinecone round-trip.
    """
    retriever = _build_retriever(
        filter_technology=filter_technology,
        filter_tech_tower=filter_tech_tower,
        filter_category=filter_category,
        top_k=top_k,
    )
    prompt = _build_prompt()
    llm    = get_llm()

    # Answer chain only — context is injected externally to avoid double retrieval
    answer_chain = prompt | llm | StrOutputParser()

    return answer_chain, retriever


def query(
    question: str,
    filter_technology: Optional[str] = None,
    filter_tech_tower: Optional[str] = None,
    filter_category: Optional[str] = None,
    top_k: int = TOP_K_RESULTS,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for a user query.

    Returns a dict with:
      - answer:            LLM-generated natural language response
      - source_documents:  List of matched catalog Documents with metadata
      - wu_ids:            Extracted list of matched WU Ids
    """
    if not question.strip():
        return {"answer": "Please provide a query.", "source_documents": [], "wu_ids": []}

    log.info(f"Query: \"{question[:80]}{'…' if len(question)>80 else ''}\"")

    answer_chain, retriever = build_rag_chain(
        filter_technology=filter_technology,
        filter_tech_tower=filter_tech_tower,
        filter_category=filter_category,
        top_k=top_k,
    )

    # Step 1: retrieve source documents once — reused for both context and response
    source_docs: List[Document] = retriever.invoke(question)

    # Fallback: if threshold filtered everything, retry with plain similarity
    if not source_docs and SEARCH_TYPE != "mmr":
        log.warning("Score threshold returned 0 docs — falling back to plain similarity")
        vectorstore = get_vectorstore()
        fb_kwargs: Dict[str, Any] = {"k": top_k}
        source_docs = vectorstore.similarity_search(question, **fb_kwargs)

    # Step 2: build context string from retrieved docs
    context = _format_docs(source_docs)

    # Step 3: generate answer using pre-built context (no second retrieval call)
    answer: str = answer_chain.invoke({"context": context, "question": question})

    # Step 4: extract WU Ids from source docs for structured response
    wu_ids = list(dict.fromkeys(
        doc.metadata.get("wu_id", "")
        for doc in source_docs
        if doc.metadata.get("wu_id")
    ))

    log.info(f"Answer generated | Sources: {wu_ids}")

    return {
        "answer":           answer,
        "source_documents": source_docs,
        "wu_ids":           wu_ids,
    }
