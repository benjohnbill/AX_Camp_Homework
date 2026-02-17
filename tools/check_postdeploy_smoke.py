#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from db_backend import get_repository


REQUIRED_METHODS = [
    "get_fragment_count",
    "get_logs_paginated",
    "get_chat_history",
    "get_connection_counts",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-deploy smoke checks.")
    parser.add_argument(
        "--strict-postgres",
        dest="strict_postgres",
        action="store_true",
        default=True,
        help="Fail when DATASTORE is not postgres (default: enabled).",
    )
    parser.add_argument(
        "--no-strict-postgres",
        dest="strict_postgres",
        action="store_false",
        help="Allow non-postgres datastore for local exploratory checks.",
    )
    return parser.parse_args()


def main(strict_postgres: bool = True) -> int:
    datastore = os.getenv("DATASTORE", "").strip().lower()
    if datastore != "postgres":
        if strict_postgres:
            print(f"[FAIL] DATASTORE is '{datastore or '(unset)'}' (expected: 'postgres')")
            return 1
        print(f"[WARN] DATASTORE is '{datastore or '(unset)'}' (expected: 'postgres')")
    else:
        print("[OK] DATASTORE=postgres")

    repo = get_repository()
    missing = [name for name in REQUIRED_METHODS if not hasattr(repo, name)]
    if missing:
        print("[FAIL] Repository is missing required methods:")
        for name in missing:
            print(f"  - {name}")
        return 1

    try:
        fragment_count = repo.get_fragment_count()
        recent_logs = repo.get_logs_paginated(limit=5, offset=0, meta_type="Log")
        recent_chat = repo.get_chat_history(limit=5)
        conn_counts = repo.get_connection_counts()
    except Exception as exc:
        print(f"[FAIL] Smoke query failed: {exc}")
        return 1

    print(f"[OK] fragment_count={fragment_count}")
    print(f"[OK] recent_logs={len(recent_logs)}")
    print(f"[OK] recent_chat={len(recent_chat)}")
    print(f"[OK] connection_nodes={len(conn_counts)}")
    print(f"[INFO] checked_at_utc={datetime.now(timezone.utc).isoformat()}")
    print("[PASS] Post-deploy smoke checks passed.")
    return 0


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(strict_postgres=args.strict_postgres))
