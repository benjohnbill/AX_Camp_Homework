#!/usr/bin/env python3
"""Issue a local signed debug token from runtime auth secrets.

Prints only the token so PowerShell can capture it safely into a variable.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._env_utils import load_runtime_env_from_secrets

import universe_auth as ua


def _load_auth_env_from_local_secrets() -> None:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return
    try:
        data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    except Exception:
        return

    keys = (
        "UNIVERSE_JWT_SECRET",
        "UNIVERSE_AUTH_ISSUER",
        "UNIVERSE_AUTH_AUDIENCE",
    )
    for key in keys:
        if os.getenv(key):
            continue
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            os.environ[key] = text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Issue locally signed debug token for runtime E2E.")
    parser.add_argument("--user-id", required=True, help="Token user_id claim.")
    parser.add_argument("--aud", default="", help="Token audience claim (optional).")
    parser.add_argument("--ttl-minutes", type=int, default=10, help="Token TTL in minutes (1..120).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    load_runtime_env_from_secrets(str(ROOT))
    _load_auth_env_from_local_secrets()

    secret = str(os.getenv("UNIVERSE_JWT_SECRET") or "").strip()
    issuer = str(os.getenv("UNIVERSE_AUTH_ISSUER") or ua.DEFAULT_ISSUER).strip() or ua.DEFAULT_ISSUER
    default_audience = str(os.getenv("UNIVERSE_AUTH_AUDIENCE") or ua.DEFAULT_AUDIENCE).strip() or ua.DEFAULT_AUDIENCE
    audience = str(args.aud or default_audience).strip()

    if not secret:
        print("UNIVERSE_JWT_SECRET is missing.", file=sys.stderr)
        return 2

    try:
        token = ua.issue_debug_token(
            user_id=str(args.user_id).strip(),
            secret=secret,
            issuer=issuer,
            audience=audience,
            ttl_minutes=int(args.ttl_minutes),
        )
    except ua.AuthError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 3

    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

