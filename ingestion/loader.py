"""
ingestion/loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangChain Document Loader for the Service Catalog Excel file.

Uses a CUSTOM loader (extending BaseLoader) because the built-in
UnstructuredExcelLoader doesn't preserve row-level metadata.
Each Excel row becomes ONE LangChain Document with:
  - page_content : rich semantic text (what gets embedded)
  - metadata     : all 8 column values + source info

Flow:
  Excel rows  →  [ExcelServiceCatalogLoader]  →  List[Document]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
from typing import Iterator, List

import pandas as pd
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import EXCEL_FILE_PATH, COLUMN_MAP, EXCEL_SHEET_NAME, EXCEL_HEADER_ROW, SKIP_EMPTY_ROWS
from utils.logger import get_logger

log = get_logger("loader", "logs/ingest.log")


def _clean(val) -> str:
    """Normalise a cell value to a clean string."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s in ("nan", "None") else s


def _build_page_content(row: dict) -> str:
    """
    Build the text that will be embedded for this service row.
    Structured labels + a natural-language summary sentence improve
    semantic matching across diverse query phrasings.
    """
    wu      = row.get("wu_id", "")
    cat     = row.get("activities_category", "")
    tech    = row.get("technology", "")
    tower   = row.get("tech_tower", "")
    env     = row.get("hosting_environment", "")
    scope   = row.get("business_scope", "")
    desc    = row.get("project_services", "")
    sla     = row.get("sla_notes", "")

    lines = [
        f"Work Unit ID: {wu}",
        f"Activity Category: {cat}",
        f"Technology: {tech}",
    ]
    if tower:  lines.append(f"Tech Tower: {tower}")
    if env:    lines.append(f"Hosting Environment: {env}")
    if scope:  lines.append(f"Business Scope: {scope}")
    if desc:   lines.append(f"Service Description: {desc}")
    if sla:    lines.append(f"SLA / Effort Level: {sla}")

    # Natural-language summary for fuzzy query coverage
    summary = []
    if cat:   summary.append(f"This is a {cat} activity")
    if tech:  summary.append(f"for {tech}")
    if tower: summary.append(f"in the {tower} tower")
    if env:   summary.append(f"hosted {env}")
    if sla:   summary.append(f"with SLA: {sla}")
    if summary:
        lines.append(". ".join(summary) + ".")

    return "\n".join(lines)


class ExcelServiceCatalogLoader(BaseLoader):
    """
    LangChain BaseLoader implementation for the Service Catalog Excel.

    Args:
        file_path:  Path to the .xlsx file
        sheet:      Sheet index (int) or name (str)
    """

    def __init__(
        self,
        file_path: str = EXCEL_FILE_PATH,
        sheet: str = EXCEL_SHEET_NAME,
    ):
        self.file_path = file_path
        self.sheet = int(sheet) if str(sheet).isdigit() else sheet

    def lazy_load(self) -> Iterator[Document]:
        """
        Yield one Document per valid Excel row.
        Implements the LangChain BaseLoader interface.
        """
        log.info(f"Loading Excel: {self.file_path} (sheet={self.sheet})")

        try:
            df = pd.read_excel(self.file_path, sheet_name=self.sheet,
                               header=EXCEL_HEADER_ROW, dtype=str)
        except FileNotFoundError:
            log.error(f"File not found: {self.file_path}")
            raise

        log.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

        missing = [v for v in COLUMN_MAP.values() if v not in df.columns]
        if missing:
            log.warning(f"Expected columns not found: {missing}")

        skipped = 0
        for idx, row in df.iterrows():
            fields = {
                key: _clean(row.get(col, ""))
                for key, col in COLUMN_MAP.items()
            }
            excel_row = int(idx) + 2

            if SKIP_EMPTY_ROWS and not fields["wu_id"]:
                log.warning(f"  Row {excel_row}: skipped — empty WU Id")
                skipped += 1
                continue

            page_content = _build_page_content(fields)

            # Metadata stored in Pinecone alongside the vector
            metadata = {
                **fields,
                "source":    self.file_path,
                "row_index": excel_row,
            }

            yield Document(page_content=page_content, metadata=metadata)

        log.info(f"Loader complete — yielded documents, skipped {skipped} rows")

    def load(self) -> List[Document]:
        """Load all documents into memory (for small-medium catalogs)."""
        docs = list(self.lazy_load())
        log.info(f"Loaded {len(docs)} Documents from Excel")
        return docs
