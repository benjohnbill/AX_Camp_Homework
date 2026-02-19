#!/usr/bin/env python3
import argparse
import os
import sys
from urllib.parse import urlparse

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
    mask_dsn,
    short_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight auth check for Supabase Postgres.")
    parser.add_argument(
        "--from-secrets",
        dest="from_secrets",
        action="store_true",
        default=True,
        help="Load missing runtime env keys from .streamlit/secrets.toml (default: enabled).",
    )
    parser.add_argument(
        "--no-from-secrets",
        dest="from_secrets",
        action="store_false",
        help="Disable secrets fallback and only use process env.",
    )
    parser.add_argument("--connect-timeout", type=int, default=10, help="DB connection timeout in seconds.")
    parser.add_argument(
        "--skip-connect",
        action="store_true",
        help="Validate DSN shape only, skip live DB authentication test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.from_secrets:
        env_info = load_runtime_env_from_secrets(PROJECT_ROOT)
        if env_info.get("loaded_keys"):
            print(f"[INFO] loaded from secrets: {', '.join(env_info['loaded_keys'])}")

    datastore = os.getenv("DATASTORE", "").strip().lower()
    database_url = os.getenv("DATABASE_URL", "").strip()

    if datastore != "postgres":
        print(f"[FAIL] DATASTORE is '{datastore or '(unset)'}' (expected: 'postgres').")
        hint = datastore_misconfigured_hint(datastore)
        if hint:
            print(f"[HINT] {hint}")
        return 1
    print("[OK] DATASTORE=postgres")

    if not database_url:
        print("[FAIL] DATABASE_URL is not set.")
        return 1
    lower_url = database_url.lower()
    if "[your-password]" in lower_url or "your-password" in lower_url:
        print("[FAIL] DATABASE_URL still contains placeholder password text.")
        return 1

    parsed = urlparse(database_url)
    masked = mask_dsn(database_url)

    if parsed.scheme not in {"postgresql", "postgres"}:
        print(f"[FAIL] Invalid DB scheme: '{parsed.scheme or '(unset)'}'.")
        return 1
    if not parsed.hostname:
        print("[FAIL] DATABASE_URL host is missing.")
        return 1
    if not parsed.path or parsed.path == "/":
        print("[FAIL] DATABASE_URL database name is missing in path.")
        return 1
    if not parsed.username:
        print("[FAIL] DATABASE_URL username is missing.")
        return 1

    print(f"[INFO] DATABASE_URL(masked)={masked}")
    print(f"[INFO] host={parsed.hostname} port={parsed.port or '(default)'} db={parsed.path.lstrip('/')}")

    is_supabase_pooler = parsed.hostname.endswith("pooler.supabase.com")
    if is_supabase_pooler and not parsed.username.startswith("postgres."):
        print(
            "[FAIL] Session pooler URI usually requires username format 'postgres.<project_ref>'. "
            f"Current user='{parsed.username}'."
        )
        return 1

    if args.skip_connect:
        print("[PASS] Preflight URL checks passed (connect skipped).")
        return 0

    if psycopg2 is None:
        print("[FAIL] psycopg2 is not installed in this environment.")
        return 1

    try:
        conn = psycopg2.connect(database_url, connect_timeout=max(1, int(args.connect_timeout)))
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            row = cur.fetchone()
        conn.close()
        print(f"[OK] auth check succeeded (db={row[0]}, user={row[1]}).")
    except Exception as exc:
        print(f"[FAIL] auth check failed: {short_error(exc)}")
        for hint in connection_failure_hints(database_url, exc):
            print(f"[HINT] {hint}")
        return 1

    print("[PASS] Preflight auth check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
