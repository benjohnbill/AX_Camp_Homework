#!/usr/bin/env python3
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


def main() -> int:
    datastore = os.getenv("DATASTORE", "").strip().lower()
    if datastore != "postgres":
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
    sys.exit(main())
