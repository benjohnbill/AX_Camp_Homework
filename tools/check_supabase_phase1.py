#!/usr/bin/env python3
import os
import sys

try:
    import psycopg2
except Exception:
    psycopg2 = None


REQUIRED_TABLES = {"logs", "user_stats", "chat_history", "connections"}
REQUIRED_INDEXES = {
    "idx_logs_embedding_ivfflat",
    "idx_logs_fts_en",
    "idx_logs_keywords_gin",
    "idx_logs_created_at",
    "idx_logs_meta_type",
    "idx_logs_parent_id",
    "idx_logs_kanban_status",
    "idx_logs_tags_gin",
    "idx_chat_history_timestamp",
    "idx_chat_history_role",
    "idx_connections_source",
    "idx_connections_target",
    "idx_connections_created_at",
    "idx_logs_content_trgm",
    "uq_chat_history_source_chat_id",
}
REQUIRED_COLUMNS = {
    ("chat_history", "source_chat_id"),
}


def main() -> int:
    database_url = os.getenv("DATABASE_URL")
    datastore = os.getenv("DATASTORE", "").strip().lower()

    if datastore != "postgres":
        print(f"[WARN] DATASTORE is '{datastore or '(unset)'}' (expected: 'postgres')")
    else:
        print("[OK] DATASTORE=postgres")

    if not database_url:
        print("[FAIL] DATABASE_URL is not set.")
        return 1

    if psycopg2 is None:
        print("[FAIL] psycopg2 is not installed in this environment.")
        print("       Install deps first: pip install -r requirements.txt")
        return 1

    try:
        conn = psycopg2.connect(database_url)
    except Exception as exc:
        print(f"[FAIL] Could not connect to DATABASE_URL: {exc}")
        return 1

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            tables = {row[0] for row in cur.fetchall()}

            cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname='public'")
            indexes = {row[0] for row in cur.fetchall()}

            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema='public'
                """
            )
            columns = {(row[0], row[1]) for row in cur.fetchall()}

            cur.execute("SELECT id FROM user_stats WHERE id = 1")
            singleton = cur.fetchone()
    finally:
        conn.close()

    missing_tables = sorted(REQUIRED_TABLES - tables)
    missing_indexes = sorted(REQUIRED_INDEXES - indexes)
    missing_columns = sorted(REQUIRED_COLUMNS - columns)

    if missing_tables:
        print("[FAIL] Missing tables:")
        for name in missing_tables:
            print(f"  - {name}")
    else:
        print("[OK] Required tables exist.")

    if missing_indexes:
        print("[FAIL] Missing indexes:")
        for name in missing_indexes:
            print(f"  - {name}")
    else:
        print("[OK] Required indexes exist.")

    if singleton:
        print("[OK] user_stats singleton row(id=1) exists.")
    else:
        print("[FAIL] user_stats singleton row(id=1) is missing.")

    if missing_columns:
        print("[FAIL] Missing required columns:")
        for table_name, column_name in missing_columns:
            print(f"  - {table_name}.{column_name}")
    else:
        print("[OK] Required columns exist.")

    if missing_tables or missing_indexes or missing_columns or not singleton:
        return 1
    print("[PASS] Phase 1 Supabase bootstrap checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
