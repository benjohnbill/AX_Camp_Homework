#!/usr/bin/env python3
import os
import pathlib
import re
from typing import Dict, List
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib  # type: ignore


RUNTIME_KEYS = ("DATASTORE", "DATABASE_URL", "OPENAI_API_KEY")


def mask_dsn(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"(postgres(?:ql)?://[^:/?#\s]+:)[^@/\s]+@", r"\1***@", value)


def short_error(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0] if str(exc) else type(exc).__name__


def datastore_misconfigured_hint(value: str) -> str:
    raw = (value or "").strip()
    if raw.startswith("postgresql://") or raw.startswith("postgres://"):
        return "DATASTORE must be literal 'postgres'. Put the URI into DATABASE_URL."
    return ""


def connection_failure_hints(database_url: str, exc: Exception) -> List[str]:
    message = short_error(exc).lower()
    parsed = urlparse(database_url or "")
    is_pooler = bool(parsed.hostname and parsed.hostname.endswith("pooler.supabase.com"))
    hints: List[str] = []

    auth_failure = "password authentication failed" in message or (
        "circuit breaker open" in message and "authentication" in message
    )
    if auth_failure:
        if is_pooler:
            if (parsed.username or "").startswith("postgres."):
                hints.append(
                    "Supabase Session Pooler username format looks valid. "
                    "DATABASE_URL password is likely stale or mismatched."
                )
            else:
                hints.append("For Supabase Session Pooler, username should be 'postgres.<project_ref>'.")
        hints.append(
            "Reset DB password in Supabase and update DATABASE_URL in local "
            ".streamlit/secrets.toml and Streamlit Cloud secrets."
        )
        hints.append("Restart the app/process after secrets update.")
        if "circuit breaker open" in message:
            hints.append("Wait 30-60 seconds before retry to let the pooler auth circuit reset.")
        return hints

    if "could not translate host name" in message or "name or service not known" in message:
        hints.append("Check DATABASE_URL host spelling and DNS/network access.")
    elif "connection refused" in message or "timeout expired" in message:
        hints.append("Check DB host/port reachability and transient network issues.")
    elif "no pg_hba.conf entry" in message:
        hints.append("Check allowed connection sources and SSL settings in Supabase.")
    return hints


def load_runtime_env_from_secrets(project_root: str) -> Dict[str, object]:
    """
    Load runtime keys from .streamlit/secrets.toml if missing in process env.
    Existing process env values always win.
    """
    root = pathlib.Path(project_root)
    secrets_path = root / ".streamlit" / "secrets.toml"
    result: Dict[str, object] = {
        "secrets_path": str(secrets_path),
        "loaded_keys": [],  # type: List[str]
        "secrets_found": False,
    }
    if not secrets_path.exists():
        return result

    try:
        data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    except Exception:
        return result

    result["secrets_found"] = True
    loaded_keys: List[str] = []
    for key in RUNTIME_KEYS:
        if os.getenv(key):
            continue
        value = data.get(key)
        if value is None:
            continue
        os.environ[key] = str(value)
        loaded_keys.append(key)
    result["loaded_keys"] = loaded_keys
    return result
