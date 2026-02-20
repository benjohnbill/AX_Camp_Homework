#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import re
import secrets
from typing import Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 12:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def _fingerprint(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _upsert_toml_key(text: str, key: str, value: str) -> Tuple[str, bool]:
    pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=\s*.*$")
    line = f'{key} = "{value}"'
    if pattern.search(text):
        return pattern.sub(line, text), True
    if text and not text.endswith("\n"):
        text += "\n"
    return text + line + "\n", False


def rotate_local_secrets(write: bool = True) -> dict:
    jwt_secret = secrets.token_urlsafe(48)
    session_secret = secrets.token_urlsafe(48)

    current = ""
    if SECRETS_FILE.exists():
        current = SECRETS_FILE.read_text(encoding="utf-8")

    updated, _ = _upsert_toml_key(current, "UNIVERSE_JWT_SECRET", jwt_secret)
    updated, _ = _upsert_toml_key(updated, "UNIVERSE_SESSION_SECRET", session_secret)
    updated, _ = _upsert_toml_key(updated, "UNIVERSE_AUTH_ISSUER", "ax-camp-staging")
    updated, _ = _upsert_toml_key(updated, "UNIVERSE_AUTH_AUDIENCE", "android-universe")

    if write:
        SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECRETS_FILE.write_text(updated, encoding="utf-8")

    return {
        "jwt_masked": _mask(jwt_secret),
        "jwt_fp": _fingerprint(jwt_secret),
        "session_masked": _mask(session_secret),
        "session_fp": _fingerprint(session_secret),
        "secrets_file": str(SECRETS_FILE),
        "written": write,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate local staging universe JWT/session secrets.")
    parser.add_argument("--dry-run", action="store_true", help="Generate values without writing secrets.toml")
    args = parser.parse_args()

    result = rotate_local_secrets(write=not args.dry_run)
    print("[OK] staging debug JWT secret rotated.")
    print(f"[INFO] secrets_file={result['secrets_file']}")
    print(f"[INFO] jwt_secret(masked)={result['jwt_masked']} fp={result['jwt_fp']}")
    print(f"[INFO] session_secret(masked)={result['session_masked']} fp={result['session_fp']}")
    if args.dry_run:
        print("[INFO] dry-run mode: no file changes written.")
    print("[NOTE] Update Streamlit Cloud secrets with new values to invalidate previously shared tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
