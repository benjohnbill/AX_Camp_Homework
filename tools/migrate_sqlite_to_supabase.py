#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from tqdm import tqdm

# Ensure project root is importable when running from tools/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_manager_postgres as pg

SQLITE_PATH = os.getenv("SQLITE_PATH", "data/narrative.db")
BATCH_SIZE = int(os.getenv("MIGRATION_BATCH", "200"))


@dataclass
class SectionResult:
    name: str
    total: int = 0
    migrated: int = 0
    skipped: int = 0
    errors: int = 0


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

    if isinstance(raw, (bytes, bytearray, memoryview)):
        blob = bytes(raw)
        arr32 = np.frombuffer(blob, dtype=np.float32)
        if arr32.size == 1536:
            return arr32.astype(np.float32)

        arr64 = np.frombuffer(blob, dtype=np.float64)
        if arr64.size == 1536:
            return arr64.astype(np.float32)
        return None

    if isinstance(raw, str):
        try:
            vals = json.loads(raw)
            arr = np.asarray(vals, dtype=np.float32).reshape(-1)
            if arr.size == 1536:
                return arr
        except Exception:
            return None

    return None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cur.fetchall()}


def _iter_logs_rows(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM logs")
    total = cur.fetchone()[0]

    columns = _table_columns(conn, "logs")
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
    for optional_col in ("quality_score", "reward_points", "tags"):
        if optional_col in columns:
            select_fields.append(optional_col)
        else:
            default_expr = "'[]' AS tags" if optional_col == "tags" else f"0 AS {optional_col}"
            select_fields.append(default_expr)

    query = f"SELECT {', '.join(select_fields)} FROM logs ORDER BY created_at ASC"
    cur.execute(query)

    for row in tqdm(cur, total=total, desc="Migrating logs"):
        yield row


def migrate_logs(sqlite_conn: sqlite3.Connection) -> SectionResult:
    result = SectionResult(name="logs")
    if not _table_exists(sqlite_conn, "logs"):
        result.skipped = 1
        return result

    batch = []
    for row in _iter_logs_rows(sqlite_conn):
        result.total += 1
        emb = _normalize_embedding(row["embedding"])

        payload = {
            "log_id": row["id"],
            "content": row["content"] or "",
            "meta_type": row["meta_type"] or "Log",
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
            "tags": _safe_json_list(row["tags"]),
        }
        batch.append(payload)

        if len(batch) >= BATCH_SIZE:
            _flush_logs_batch(batch, result)
            batch = []

    if batch:
        _flush_logs_batch(batch, result)
    return result


def _flush_logs_batch(batch: List[dict], result: SectionResult) -> None:
    for item in batch:
        try:
            pg.upsert_log(**item)
            result.migrated += 1
        except Exception as exc:
            result.errors += 1
            print(f"[logs][skip] id={item['log_id']} error={exc}")


def migrate_user_stats(sqlite_conn: sqlite3.Connection) -> SectionResult:
    result = SectionResult(name="user_stats")
    if not _table_exists(sqlite_conn, "user_stats"):
        result.skipped = 1
        return result

    sqlite_conn.row_factory = sqlite3.Row
    cur = sqlite_conn.cursor()
    cur.execute("SELECT * FROM user_stats WHERE id = 1")
    row = cur.fetchone()
    result.total = 1
    if not row:
        result.skipped = 1
        return result

    try:
        with pg.get_conn() as conn:
            with conn.cursor() as pcur:
                pcur.execute(
                    """
                    INSERT INTO user_stats (
                        id, last_login, current_streak, longest_streak, debt_count, recovery_points, last_log_at
                    )
                    VALUES (
                        1, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        last_login = EXCLUDED.last_login,
                        current_streak = EXCLUDED.current_streak,
                        longest_streak = EXCLUDED.longest_streak,
                        debt_count = EXCLUDED.debt_count,
                        recovery_points = EXCLUDED.recovery_points,
                        last_log_at = EXCLUDED.last_log_at
                    """,
                    (
                        row["last_login"],
                        row["current_streak"] or 0,
                        row["longest_streak"] or 0,
                        row["debt_count"] or 0,
                        row["recovery_points"] if "recovery_points" in row.keys() and row["recovery_points"] is not None else 0,
                        row["last_log_at"] if "last_log_at" in row.keys() else None,
                    ),
                )
            conn.commit()
        result.migrated = 1
    except Exception as exc:
        result.errors = 1
        print(f"[user_stats][skip] error={exc}")

    return result


def migrate_chat_history(sqlite_conn: sqlite3.Connection) -> SectionResult:
    result = SectionResult(name="chat_history")
    if not _table_exists(sqlite_conn, "chat_history"):
        result.skipped = 1
        return result

    sqlite_conn.row_factory = sqlite3.Row
    cur = sqlite_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chat_history")
    total = cur.fetchone()[0]
    result.total = total
    if total == 0:
        return result

    cur.execute("SELECT role, content, metadata, timestamp FROM chat_history ORDER BY id ASC")
    rows = cur.fetchall()
    with pg.get_conn() as conn:
        with conn.cursor() as pcur:
            for row in tqdm(rows, total=total, desc="Migrating chat_history"):
                try:
                    metadata = row["metadata"]
                    if metadata is None:
                        metadata = "{}"
                    pcur.execute(
                        """
                        INSERT INTO chat_history ("timestamp", role, content, metadata)
                        VALUES (%s, %s, %s, %s::jsonb)
                        """,
                        (row["timestamp"], row["role"], row["content"], metadata),
                    )
                    result.migrated += 1
                except Exception as exc:
                    result.errors += 1
                    print(f"[chat_history][skip] error={exc}")
        conn.commit()
    return result


def migrate_connections(sqlite_conn: sqlite3.Connection) -> SectionResult:
    result = SectionResult(name="connections")
    if not _table_exists(sqlite_conn, "connections"):
        result.skipped = 1
        return result

    sqlite_conn.row_factory = sqlite3.Row
    cur = sqlite_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM connections")
    total = cur.fetchone()[0]
    result.total = total
    if total == 0:
        return result

    cur.execute("SELECT source_id, target_id, type, created_at FROM connections ORDER BY id ASC")
    rows = cur.fetchall()

    with pg.get_conn() as conn:
        with conn.cursor() as pcur:
            for row in tqdm(rows, total=total, desc="Migrating connections"):
                try:
                    pcur.execute(
                        """
                        INSERT INTO connections (source_id, target_id, type, created_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source_id, target_id) DO UPDATE SET
                            type = EXCLUDED.type,
                            created_at = EXCLUDED.created_at
                        """,
                        (row["source_id"], row["target_id"], row["type"] or "manual", row["created_at"]),
                    )
                    result.migrated += 1
                except Exception as exc:
                    result.errors += 1
                    print(f"[connections][skip] {row['source_id']}->{row['target_id']} error={exc}")
        conn.commit()
    return result


def _sqlite_count(conn: sqlite3.Connection, table_name: str) -> int:
    if not _table_exists(conn, table_name):
        return 0
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    return int(cur.fetchone()[0])


def _postgres_count(table_name: str) -> int:
    with pg.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            return int(cur.fetchone()[0])


def print_post_migration_report(sqlite_conn: sqlite3.Connection) -> None:
    tables = ["logs", "user_stats", "chat_history", "connections"]
    print("\n=== Post Migration Count Report ===")
    for table in tables:
        s_count = _sqlite_count(sqlite_conn, table)
        try:
            p_count = _postgres_count(table)
            status = "OK" if p_count >= s_count else "CHECK"
            print(f"{table:12} sqlite={s_count:6d}  postgres={p_count:6d}  [{status}]")
        except Exception as exc:
            print(f"{table:12} sqlite={s_count:6d}  postgres=ERROR ({exc})")


def main() -> None:
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(f"SQLite DB not found: {SQLITE_PATH}")

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    sections = []
    try:
        sections.append(migrate_logs(sqlite_conn))
        sections.append(migrate_user_stats(sqlite_conn))
        sections.append(migrate_chat_history(sqlite_conn))
        sections.append(migrate_connections(sqlite_conn))
        print_post_migration_report(sqlite_conn)
    finally:
        sqlite_conn.close()

    print("\n=== Migration Summary ===")
    for s in sections:
        print(
            f"{s.name:12} total={s.total:6d} migrated={s.migrated:6d} "
            f"skipped={s.skipped:4d} errors={s.errors:4d}"
        )


if __name__ == "__main__":
    main()

