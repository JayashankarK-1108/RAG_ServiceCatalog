"""
ingestion/checkpoint.py
Save/load/clear ingestion progress for resume-on-failure support.
Persists the set of vector IDs already upserted to Pinecone.
"""

from __future__ import annotations
import json, os
from datetime import datetime
from typing import Set, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import CHECKPOINT_FILE
from utils.logger import get_logger

log = get_logger("checkpoint")


def load_checkpoint() -> Dict[str, Any]:
    if not os.path.exists(CHECKPOINT_FILE):
        return {"upserted_ids": [], "total": 0, "updated_at": None}
    with open(CHECKPOINT_FILE) as f:
        data = json.load(f)
    log.info(f"Checkpoint: {len(data.get('upserted_ids', []))} IDs already upserted")
    return data


def save_checkpoint(upserted_ids: Set[str], total: int) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "upserted_ids": list(upserted_ids),
            "total":        total,
            "updated_at":   datetime.utcnow().isoformat(),
        }, f, indent=2)
    log.info(f"Checkpoint saved — {len(upserted_ids)}/{total} upserted")


def clear_checkpoint() -> None:
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Checkpoint cleared")
