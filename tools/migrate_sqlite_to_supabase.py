#!/usr/bin/env python3
import json
import os
import sqlite3
from typing import Iterable, List, Optional

import numpy as np
from tqdm import tqdm

import db_manager_postgres as pg

SQLITE_PATH = os.getenv("SQLITE_PATH", "data/narrative.db")
BATCH_SIZE = int(os.getenv("MIGRATION_BATCH", "200"))


def _safe_json_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _normalize_embedding(raw) -> Optional[np.ndarray]:
    if raw is None:
        return None

    # Stored as bytes (legacy sqlite blob)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        blob = bytes(raw)

        # Try float32 first
        arr32 = np.frombuffer(blob, dtype=np.float32)
        if arr32.size == 1536:
            return arr32.astype(np.float32)

        # Legacy float64 path
        arr64 = np.frombuffer(blob, dtype=np.float64)
        if arr64.size == 1536:
            return arr64.astype(np.float32)

        return None

    # Stored as JSON array text
    if isinstance(raw, str):
        try:
            vals = json.loads(raw)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            if arr.size == 1536:
                return arr
        except Exception:
            return None

    return None


def _get_logs_columns(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(logs)")
    return [row[1] for row in cur.fetchall()]


def iter_rows(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM logs")
    total = cur.fetchone()[0]

    columns = set(_get_logs_columns(conn))

    select_fields = [
        "id",
        "content",
        "meta_type",
        "parent_id",
        "action_plan",
        "created_at",
        "embedding",
        "emotion",
        "dimension",
        "keywords",
        "linked_constitutions",
        "duration",
        "debt_repaid",
        "kanban_status",
    ]

    for optional_col in ("quality_score", "reward_points"):
        if optional_col in columns:
            select_fields.append(optional_col)
        else:
            select_fields.append(f"0 AS {optional_col}")

    query = f"SELECT {', '.join(select_fields)} FROM logs ORDER BY created_at ASC"
    cur.execute(query)

    for row in tqdm(cur, total=total, desc="Migrating logs"):
        yield row


def main() -> None:
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(f"SQLite DB not found: {SQLITE_PATH}")

    total = 0
    migrated = 0
    skipped = 0

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    batch = []
    for row in iter_rows(conn):
        total += 1
        emb = _normalize_embedding(row["embedding"])

        payload = {
            "log_id": row["id"],
            "content": row["content"] or "",
            "meta_type": row["meta_type"] or "Fragment",
            "parent_id": row["parent_id"],
            "action_plan": row["action_plan"],
            "created_at": row["created_at"],
            "embedding": emb.tolist() if emb is not None else None,
            "emotion": row["emotion"],
            "dimension": row["dimension"],
            "keywords": _safe_json_list(row["keywords"]),
            "linked_constitutions": _safe_json_list(row["linked_constitutions"]),
            "duration": row["duration"],
            "debt_repaid": row["debt_repaid"] if row["debt_repaid"] is not None else 0,
            "kanban_status": row["kanban_status"],
            "quality_score": row["quality_score"] if row["quality_score"] is not None else 0,
            "reward_points": row["reward_points"] if row["reward_points"] is not None else 0,
        }

        batch.append(payload)
        if len(batch) >= BATCH_SIZE:
            for item in batch:
                try:
                    pg.upsert_log(**item)
                    migrated += 1
                except Exception as exc:
                    skipped += 1
                    print(f"[skip] id={item['log_id']} error={exc}")
            batch = []

    if batch:
        for item in batch:
            try:
                pg.upsert_log(**item)
                migrated += 1
            except Exception as exc:
                skipped += 1
                print(f"[skip] id={item['log_id']} error={exc}")

    conn.close()

    print("\n=== Migration Summary ===")
    print(f"Total rows:    {total}")
    print(f"Migrated rows: {migrated}")
    print(f"Skipped rows:  {skipped}")


if __name__ == "__main__":
    main()
