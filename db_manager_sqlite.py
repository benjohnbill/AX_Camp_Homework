import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np

DB_PATH = os.getenv("SQLITE_PATH", "data/narrative.db")
KNOWN_EMBEDDING_DIMS = {1536}


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def save_embedding(log_id: str, emb: np.ndarray) -> None:
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    with get_conn() as conn:
        conn.execute("UPDATE logs SET embedding = ? WHERE id = ?", (arr.tobytes(), log_id))
        conn.commit()


def _blob_to_float32(blob: bytes) -> Optional[np.ndarray]:
    if blob is None:
        return None

    raw = bytes(blob)

    candidates: Dict[np.dtype, np.ndarray] = {}

    for dtype in (np.float32, np.float64):
        if len(raw) % np.dtype(dtype).itemsize != 0:
            continue

        try:
            arr = np.frombuffer(raw, dtype=dtype)
            if arr.size > 0:
                candidates[np.dtype(dtype)] = arr
        except Exception:
            continue

    if not candidates:
        return None

    # Prefer vectors matching known embedding dimensions.
    for dtype in (np.float32, np.float64):
        arr = candidates.get(np.dtype(dtype))
        if arr is not None and arr.size in KNOWN_EMBEDDING_DIMS:
            return arr.astype(np.float32)

    # Default to float32 when shape hints are unavailable.
    arr32 = candidates.get(np.dtype(np.float32))
    if arr32 is not None:
        return arr32.astype(np.float32)

    arr64 = candidates.get(np.dtype(np.float64))
    if arr64 is not None:
        return arr64.astype(np.float32)

    return None


def get_all_embeddings() -> Tuple[List[str], Optional[np.ndarray]]:
    ids: List[str] = []
    vectors: List[np.ndarray] = []

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM logs WHERE embedding IS NOT NULL ORDER BY created_at DESC"
        ).fetchall()

    for row in rows:
        arr = _blob_to_float32(row["embedding"])
        if arr is None:
            continue
        ids.append(row["id"])
        vectors.append(arr)

    if not vectors:
        return ids, None

    # Keep only vectors with same shape as first one
    target_dim = vectors[0].shape[0]
    filtered_ids: List[str] = []
    filtered_vectors: List[np.ndarray] = []
    for idx, vec in zip(ids, vectors):
        if vec.shape[0] == target_dim:
            filtered_ids.append(idx)
            filtered_vectors.append(vec)

    if not filtered_vectors:
        return [], None

    return filtered_ids, np.vstack(filtered_vectors).astype(np.float32)


def load_logs_by_ids(ids: List[str]) -> List[Dict]:
    if not ids:
        return []

    placeholders = ",".join(["?"] * len(ids))
    sql = f"""
        SELECT id, content, meta_type, keywords, linked_constitutions
        FROM logs
        WHERE id IN ({placeholders})
    """

    with get_conn() as conn:
        rows = conn.execute(sql, ids).fetchall()

    result: List[Dict] = []
    for row in rows:
        keywords = []
        linked_constitutions = []

        if row["keywords"]:
            try:
                keywords = json.loads(row["keywords"])
            except Exception:
                keywords = []

        if row["linked_constitutions"]:
            try:
                linked_constitutions = json.loads(row["linked_constitutions"])
            except Exception:
                linked_constitutions = []

        result.append(
            {
                "id": row["id"],
                "content": row["content"],
                "meta_type": row["meta_type"],
                "keywords": keywords,
                "linked_constitutions": linked_constitutions,
            }
        )
    return result


def ensure_fts() -> None:
    """Best-effort FTS5 setup. Skip gracefully when SQLite lacks FTS5."""
    try:
        with get_conn() as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts
                USING fts5(id UNINDEXED, content, tokenize='unicode61');
                """
            )
            conn.execute("DELETE FROM logs_fts")
            conn.execute("INSERT INTO logs_fts(id, content) SELECT id, content FROM logs")
            conn.commit()
    except Exception as exc:
        print(f"[sqlite] FTS5 setup skipped: {exc}")
