#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    import psycopg2
except Exception:
    psycopg2 = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools._env_utils import (
    connection_failure_hints,
    datastore_misconfigured_hint,
    load_runtime_env_from_secrets,
    short_error,
)


TABLES = ("logs", "user_stats", "chat_history", "connections")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SQLite->Supabase integrity gates.")
    parser.add_argument("--expect-postgres", action="store_true", help="Fail if DATASTORE is not postgres.")
    parser.add_argument("--max-dup-chat", type=int, default=0, help="Allowed duplicate source_chat_id groups.")
    parser.add_argument("--report-json", default="", help="Optional report output path.")
    parser.add_argument("--sqlite-path", default=os.getenv("SQLITE_PATH", "data/narrative.db"))
    return parser.parse_args()


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def load_sqlite_counts(sqlite_path: str) -> Optional[Dict[str, int]]:
    if not os.path.exists(sqlite_path):
        return None
    conn = sqlite3.connect(sqlite_path)
    try:
        counts = {}
        for table in TABLES:
            if _sqlite_table_exists(conn, table):
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = int(cur.fetchone()[0])
            else:
                counts[table] = 0
        return counts
    finally:
        conn.close()


def load_postgres_stats(database_url: str) -> Dict:
    stats: Dict = {}
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            counts = {}
            for table in TABLES:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = int(cur.fetchone()[0])
            stats["postgres_counts"] = counts

            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT source_chat_id
                    FROM chat_history
                    WHERE source_chat_id IS NOT NULL
                    GROUP BY source_chat_id
                    HAVING COUNT(*) > 1
                ) d
                """
            )
            stats["duplicate_chat_source_groups"] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM connections c
                LEFT JOIN logs s ON s.id = c.source_id
                LEFT JOIN logs t ON t.id = c.target_id
                WHERE s.id IS NULL OR t.id IS NULL
                """
            )
            stats["orphan_connections"] = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM connections WHERE source_id = target_id")
            stats["self_loop_connections"] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema='public'
                  AND table_name='chat_history'
                  AND column_name='source_chat_id'
                """
            )
            stats["has_source_chat_id_column"] = int(cur.fetchone()[0]) > 0

            cur.execute(
                """
                SELECT COUNT(*)
                FROM pg_indexes
                WHERE schemaname='public'
                  AND tablename='chat_history'
                  AND indexname='uq_chat_history_source_chat_id'
                """
            )
            stats["has_source_chat_id_unique_index"] = int(cur.fetchone()[0]) > 0
    finally:
        conn.close()

    return stats


def main() -> int:
    args = parse_args()
    env_info = load_runtime_env_from_secrets(PROJECT_ROOT)
    if env_info.get("loaded_keys"):
        print(f"[INFO] loaded from secrets: {', '.join(env_info['loaded_keys'])}")

    datastore = os.getenv("DATASTORE", "").strip().lower()
    database_url = os.getenv("DATABASE_URL")

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "datastore": datastore or "(unset)",
        "expect_postgres": args.expect_postgres,
        "max_dup_chat": args.max_dup_chat,
        "checks": {},
    }

    if args.expect_postgres and datastore != "postgres":
        report["checks"]["datastore"] = {"ok": False, "detail": f"DATASTORE={datastore or '(unset)'}"}
        print(f"[FAIL] DATASTORE is '{datastore or '(unset)'}' (expected: 'postgres').")
        hint = datastore_misconfigured_hint(datastore)
        if hint:
            print(f"[HINT] {hint}")
        if args.report_json:
            report_dir = os.path.dirname(args.report_json)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return 1
    report["checks"]["datastore"] = {"ok": True, "detail": datastore or "(unset)"}

    if psycopg2 is None:
        print("[FAIL] psycopg2 is not available.")
        return 1

    if not database_url:
        print("[FAIL] DATABASE_URL is not set.")
        return 1

    try:
        pg_stats = load_postgres_stats(database_url)
    except Exception as exc:
        print(f"[FAIL] Could not query postgres stats: {short_error(exc)}")
        for hint in connection_failure_hints(database_url, exc):
            print(f"[HINT] {hint}")
        return 1

    sqlite_counts = load_sqlite_counts(args.sqlite_path)
    pg_counts = pg_stats["postgres_counts"]

    parity = {}
    if sqlite_counts is not None:
        for table in TABLES:
            parity[table] = pg_counts.get(table, 0) >= sqlite_counts.get(table, 0)

    checks = {
        "source_chat_id_column": pg_stats["has_source_chat_id_column"],
        "source_chat_id_unique_index": pg_stats["has_source_chat_id_unique_index"],
        "duplicate_chat_source_groups": pg_stats["duplicate_chat_source_groups"] <= args.max_dup_chat,
        "orphan_connections": pg_stats["orphan_connections"] == 0,
        "self_loop_connections": pg_stats["self_loop_connections"] == 0,
    }
    if sqlite_counts is not None:
        checks["sqlite_postgres_count_parity_ge"] = all(parity.values())

    report["postgres_counts"] = pg_counts
    report["sqlite_counts"] = sqlite_counts
    report["parity_ge"] = parity
    report["metrics"] = {
        "duplicate_chat_source_groups": pg_stats["duplicate_chat_source_groups"],
        "orphan_connections": pg_stats["orphan_connections"],
        "self_loop_connections": pg_stats["self_loop_connections"],
    }
    report["checks"].update({k: {"ok": bool(v)} for k, v in checks.items()})

    failed_checks = [name for name, ok in checks.items() if not ok]
    if failed_checks:
        print("[FAIL] Integrity checks failed:")
        for name in failed_checks:
            print(f"  - {name}")
        exit_code = 1
    else:
        print("[PASS] Integrity checks passed.")
        exit_code = 0

    print(f"[INFO] postgres_counts={pg_counts}")
    if sqlite_counts is not None:
        print(f"[INFO] sqlite_counts={sqlite_counts}")
        print(f"[INFO] parity_ge={parity}")
    else:
        print(f"[INFO] sqlite database not found at '{args.sqlite_path}', parity check skipped.")

    if args.report_json:
        report_dir = os.path.dirname(args.report_json)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[INFO] report_json={args.report_json}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
