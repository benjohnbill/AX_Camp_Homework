#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gateway_fastapi as gw
import universe_auth as ua


def _env() -> Dict[str, str]:
    return {
        "UNIVERSE_JWT_SECRET": "jwt-secret",
        "UNIVERSE_SESSION_SECRET": "session-secret",
        "UNIVERSE_AUTH_ISSUER": "ax-camp-staging",
        "UNIVERSE_AUTH_AUDIENCE": "android-universe",
        "UNIVERSE_SESSION_COOKIE": "ax_universe_session",
        "UNIVERSE_SESSION_TTL_SECONDS": "900",
        "UNIVERSE_UPSTREAM_EMBED_URL": "https://staging.example.com/?embed=universe_3d",
    }


def _issue_token(audience: str = "android-universe", user_id: str = "u1") -> str:
    return ua.issue_debug_token(
        user_id=user_id,
        secret="jwt-secret",
        issuer="ax-camp-staging",
        audience=audience,
        ttl_minutes=30,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gateway strict-auth E2E contract check.")
    parser.add_argument(
        "--json-report",
        default="data/gateway_e2e_latest.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_local_gateway_contract() -> Dict[str, Any]:
    app = gw.create_app(_env())
    client = TestClient(app, base_url="https://testserver")
    results: List[Dict[str, Any]] = []

    # 1) First bearer request
    token = _issue_token(user_id="e2e-user")
    first = client.get(
        "/gateway/universe_3d",
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=False,
    )
    set_cookie = first.headers.get("set-cookie", "")
    _assert(first.status_code == 307, "first bearer request must return 307")
    _assert(first.headers.get("x-auth-source") == "bearer", "first request source must be bearer")
    _assert("HttpOnly" in set_cookie, "session cookie must be HttpOnly")
    _assert("Secure" in set_cookie, "session cookie must be Secure")
    _assert("SameSite=none" in set_cookie, "session cookie must include SameSite=None")
    results.append(
        {
            "scenario": "first_bearer",
            "status_code": first.status_code,
            "x_auth_source": first.headers.get("x-auth-source"),
            "set_cookie": set_cookie,
            "pass": True,
        }
    )

    # 2) Cookie-only follow-up
    follow_up = client.get("/gateway/universe_3d", follow_redirects=False)
    _assert(follow_up.status_code == 307, "cookie follow-up must return 307")
    _assert(follow_up.headers.get("x-auth-source") == "cookie", "follow-up source must be cookie")
    results.append(
        {
            "scenario": "cookie_follow_up",
            "status_code": follow_up.status_code,
            "x_auth_source": follow_up.headers.get("x-auth-source"),
            "pass": True,
        }
    )

    # 3) Missing auth
    noauth_client = TestClient(app, base_url="https://testserver")
    missing = noauth_client.get("/gateway/universe_3d", follow_redirects=False)
    _assert(missing.status_code == 401, "missing auth must return 401")
    _assert(missing.json().get("code") == "missing_token", "missing auth code mismatch")
    results.append(
        {
            "scenario": "missing_auth",
            "status_code": missing.status_code,
            "code": missing.json().get("code"),
            "pass": True,
        }
    )

    # 4) Forbidden audience
    forbidden_token = _issue_token(audience="other-aud", user_id="forbidden-user")
    forbidden = noauth_client.get(
        "/gateway/universe_3d",
        headers={"Authorization": f"Bearer {forbidden_token}"},
        follow_redirects=False,
    )
    _assert(forbidden.status_code == 403, "forbidden audience must return 403")
    _assert(forbidden.json().get("code") == "forbidden_audience", "forbidden code mismatch")
    results.append(
        {
            "scenario": "forbidden_audience",
            "status_code": forbidden.status_code,
            "code": forbidden.json().get("code"),
            "pass": True,
        }
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "local-testclient",
        "status": "pass",
        "results": results,
    }


def main() -> int:
    args = parse_args()
    try:
        report = run_local_gateway_contract()
    except Exception as exc:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "local-testclient",
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 1
    else:
        code = 0

    report_path = Path(args.json_report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] report saved: {report_path}")
    print(f"[{report['status'].upper()}] gateway contract check")
    return code


if __name__ == "__main__":
    sys.exit(main())
