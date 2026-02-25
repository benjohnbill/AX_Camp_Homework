#!/usr/bin/env python3
"""Collect sanitized fixed-HTTPS auth/gateway evidence for cycle execution.

This script does not print or persist secret/token values.
It writes a compact JSON report that can be attached as backend evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import os
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._env_utils import load_runtime_env_from_secrets

import universe_auth as ua



class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_body(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text or "{}")
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body_bytes: Optional[bytes] = None,
    timeout_sec: int = 45,
) -> Tuple[int, Dict[str, str], str, Optional[str]]:
    opener = build_opener(NoRedirect)
    req = Request(url=url, method=method, headers=headers or {}, data=body_bytes)
    try:
        with opener.open(req, timeout=timeout_sec) as resp:
            status = int(resp.getcode())
            hdrs = {k: v for k, v in resp.headers.items()}
            text = resp.read(4000).decode("utf-8", "ignore")
            return status, hdrs, text, None
    except HTTPError as exc:
        status = int(exc.code)
        hdrs = {k: v for k, v in exc.headers.items()}
        text = exc.read(4000).decode("utf-8", "ignore")
        return status, hdrs, text, None
    except URLError as exc:
        return -1, {}, "", f"URLError: {exc}"
    except Exception as exc:  # pragma: no cover - defensive path
        return -1, {}, "", f"{type(exc).__name__}: {exc}"


def _request_with_retry(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body_bytes: Optional[bytes] = None,
    timeout_sec: int = 45,
    retries: int = 2,
) -> Tuple[int, Dict[str, str], str, Optional[str], int]:
    last: Tuple[int, Dict[str, str], str, Optional[str]] = (-1, {}, "", "not_executed")
    for idx in range(1, max(1, retries) + 1):
        last = _request(
            url=url,
            method=method,
            headers=headers,
            body_bytes=body_bytes,
            timeout_sec=timeout_sec,
        )
        status = last[0]
        if status != -1:
            return (*last, idx)
    return (*last, max(1, retries))


def _cookie_flags(set_cookie: str) -> Dict[str, bool]:
    attrs = [a.strip().lower() for a in set_cookie.split(";")[1:] if a.strip()]
    return {
        "has_httponly": any(a == "httponly" for a in attrs),
        "has_secure": any(a == "secure" for a in attrs),
        "has_samesite_none": any(a.startswith("samesite=none") for a in attrs),
    }


def _get_header(headers: Dict[str, str], key: str) -> str:
    key_l = key.lower()
    for hk, hv in headers.items():
        if hk.lower() == key_l:
            return hv
    return ""


def _probe_core_endpoints(base_debug: str, base_gateway: str) -> Dict[str, Any]:
    debug_url = f"{base_debug.rstrip('/')}/debug/token"
    healthz_url = f"{base_gateway.rstrip('/')}/healthz"
    session_url = f"{base_gateway.rstrip('/')}/gateway/session"
    universe_url = f"{base_gateway.rstrip('/')}/gateway/universe_3d"

    debug_status, _, debug_body, debug_err, debug_attempt = _request_with_retry(
        url=debug_url,
        method="POST",
        headers={"Content-Type": "application/json"},
        body_bytes=b"{}",
        retries=2,
    )
    debug_admin_key = str(os.getenv("DEBUG_TOKEN_ADMIN_KEY") or "").strip()
    debug_admin_headers = {
        "Content-Type": "application/json",
        "X-Debug-Admin-Key": debug_admin_key,
    }
    debug_admin_status, _, debug_admin_body, debug_admin_err, debug_admin_attempt = _request_with_retry(
        url=debug_url,
        method="POST",
        headers=debug_admin_headers,
        body_bytes=json.dumps({"user_id": "ct-probe-debug", "ttl_minutes": 5}).encode("utf-8"),
        retries=2,
    )
    debug_admin_json = _read_json_body(debug_admin_body)
    health_status, _, _, health_err, health_attempt = _request_with_retry(url=healthz_url, retries=3)
    session_status, _, session_body, session_err, session_attempt = _request_with_retry(url=session_url, retries=2)
    universe_status, _, universe_body, universe_err, universe_attempt = _request_with_retry(
        url=universe_url,
        retries=2,
    )

    return {
        "debug_token_no_admin": {
            "status": debug_status,
            "code": _read_json_body(debug_body).get("code", ""),
            "error": debug_err,
            "attempts": debug_attempt,
        },
        "debug_token_with_admin": {
            "status": debug_admin_status,
            "code": debug_admin_json.get("code", ""),
            "token_issued": bool(str(debug_admin_json.get("token", "")).strip()),
            "error": debug_admin_err,
            "attempts": debug_admin_attempt,
        },
        "gateway_healthz": {
            "status": health_status,
            "error": health_err,
            "attempts": health_attempt,
        },
        "gateway_session_no_auth": {
            "status": session_status,
            "code": _read_json_body(session_body).get("code", ""),
            "error": session_err,
            "attempts": session_attempt,
        },
        "gateway_universe_no_auth": {
            "status": universe_status,
            "code": _read_json_body(universe_body).get("code", ""),
            "error": universe_err,
            "attempts": universe_attempt,
        },
    }


def _probe_bearer_cookie(base_gateway: str, jwt_secret: str, issuer: str, audience: str) -> Dict[str, Any]:
    universe_url = f"{base_gateway.rstrip('/')}/gateway/universe_3d"

    token = ua.issue_debug_token(
        user_id="ct-probe-user",
        secret=jwt_secret,
        issuer=issuer,
        audience=audience,
        ttl_minutes=5,
    )
    first_status, first_hdrs, first_body, first_err = _request(
        url=universe_url,
        headers={"Authorization": f"Bearer {token}"},
    )
    first_json = _read_json_body(first_body)
    set_cookie = _get_header(first_hdrs, "Set-Cookie")
    cookie_pair = set_cookie.split(";", 1)[0] if set_cookie else ""
    follow_status, follow_hdrs, follow_body, follow_err = _request(
        url=universe_url,
        headers={"Cookie": cookie_pair} if cookie_pair else {},
    )
    follow_json = _read_json_body(follow_body)

    forbidden_token = ua.issue_debug_token(
        user_id="ct-probe-user-forbidden",
        secret=jwt_secret,
        issuer=issuer,
        audience="forbidden-audience",
        ttl_minutes=5,
    )
    forbidden_status, _, forbidden_body, forbidden_err = _request(
        url=universe_url,
        headers={"Authorization": f"Bearer {forbidden_token}"},
    )
    forbidden_json = _read_json_body(forbidden_body)

    return {
        "bearer_first": {
            "status": first_status,
            "x_auth_source": _get_header(first_hdrs, "X-Auth-Source"),
            "location_present": bool(_get_header(first_hdrs, "Location")),
            "set_cookie_present": bool(set_cookie),
            "cookie_flags": _cookie_flags(set_cookie),
            "code": first_json.get("code", ""),
            "error": first_err,
        },
        "cookie_follow_up": {
            "status": follow_status,
            "x_auth_source": _get_header(follow_hdrs, "X-Auth-Source"),
            "location_present": bool(_get_header(follow_hdrs, "Location")),
            "code": follow_json.get("code", ""),
            "error": follow_err,
        },
        "forbidden_audience": {
            "status": forbidden_status,
            "code": forbidden_json.get("code", ""),
            "error": forbidden_err,
        },
    }


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
        "UNIVERSE_SESSION_SECRET",
        "UNIVERSE_AUTH_ISSUER",
        "UNIVERSE_AUTH_AUDIENCE",
        "DEBUG_TOKEN_ADMIN_KEY",
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
    parser = argparse.ArgumentParser(description="Probe staging auth contract on fixed HTTPS endpoints.")
    parser.add_argument(
        "--debug-base-url",
        default="https://ax-camp-debug-token-staging.onrender.com",
        help="Base URL for debug token service.",
    )
    parser.add_argument(
        "--gateway-base-url",
        default="https://ax-camp-universe-gateway-staging.onrender.com",
        help="Base URL for gateway service.",
    )
    parser.add_argument(
        "--json-report",
        default="data/staging_auth_probe_latest.json",
        help="Output path for JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    load_runtime_env_from_secrets(str(ROOT))
    _load_auth_env_from_local_secrets()
    jwt_secret = str(os.getenv("UNIVERSE_JWT_SECRET") or "").strip()
    issuer = str(os.getenv("UNIVERSE_AUTH_ISSUER") or ua.DEFAULT_ISSUER).strip()
    audience = str(os.getenv("UNIVERSE_AUTH_AUDIENCE") or ua.DEFAULT_AUDIENCE).strip()

    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "debug_base_url": args.debug_base_url,
        "gateway_base_url": args.gateway_base_url,
        "env_presence": {
            "UNIVERSE_JWT_SECRET": bool(jwt_secret),
            "UNIVERSE_AUTH_ISSUER": bool(issuer),
            "UNIVERSE_AUTH_AUDIENCE": bool(audience),
            "DEBUG_TOKEN_ADMIN_KEY": bool(str(os.getenv("DEBUG_TOKEN_ADMIN_KEY") or "").strip()),
        },
        "core_endpoints": _probe_core_endpoints(args.debug_base_url, args.gateway_base_url),
    }

    if jwt_secret:
        report["bearer_cookie_contract"] = _probe_bearer_cookie(
            base_gateway=args.gateway_base_url,
            jwt_secret=jwt_secret,
            issuer=issuer,
            audience=audience,
        )
    else:
        report["bearer_cookie_contract"] = {
            "status": "skipped",
            "reason": "UNIVERSE_JWT_SECRET missing in runtime env/secrets fallback.",
        }

    out = Path(args.json_report)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] report saved: {out}")

    core = report["core_endpoints"]
    health_ok = core["gateway_healthz"]["status"] == 200
    universe_status = core["gateway_universe_no_auth"]["status"]
    if health_ok and universe_status in (401, 403, 307):
        print("[PASS] fixed HTTPS endpoints reachable with expected auth-coded behavior.")
        return 0
    print("[WARN] endpoint probe found unstable/unexpected responses. See JSON report.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
