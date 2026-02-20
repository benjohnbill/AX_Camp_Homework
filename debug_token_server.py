import hmac
import json
import logging
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

import universe_auth as ua

LOGGER = logging.getLogger("debug_token_server")

STAGING_ENVS = {"staging", "development", "dev"}


def _utc_iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _error_payload(status: int, code: str, message: str) -> dict[str, Any]:
    return {"status": int(status), "code": code, "message": message}


def issue_debug_token_response(
    body: Mapping[str, Any],
    headers: Mapping[str, Any],
    environ: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    env = str(environ.get("APP_ENV") or environ.get("ENV") or "staging").strip().lower()
    if env not in STAGING_ENVS:
        return 403, _error_payload(403, "forbidden_env", "Debug token issuance is disabled outside staging/development.")

    admin_key_expected = str(environ.get("DEBUG_TOKEN_ADMIN_KEY") or "").strip()
    admin_key_given = str(headers.get("X-Debug-Admin-Key") or headers.get("x-debug-admin-key") or "").strip()
    if not admin_key_expected:
        return 403, _error_payload(403, "admin_key_not_configured", "Admin key is not configured.")
    if not admin_key_given or not hmac.compare_digest(admin_key_given, admin_key_expected):
        return 401, _error_payload(401, "unauthorized_admin", "Missing or invalid admin key.")

    user_id = str(body.get("user_id") or "").strip()
    if not user_id:
        return 403, _error_payload(403, "forbidden_user", "user_id is required.")

    issuer = str(environ.get("UNIVERSE_AUTH_ISSUER") or ua.DEFAULT_ISSUER).strip()
    audience_default = str(environ.get("UNIVERSE_AUTH_AUDIENCE") or ua.DEFAULT_AUDIENCE).strip()
    allowed_auds_raw = str(environ.get("DEBUG_TOKEN_ALLOWED_AUDIENCES") or audience_default).strip()
    allowed_auds = [a.strip() for a in allowed_auds_raw.split(",") if a.strip()]
    aud = str(body.get("aud") or audience_default).strip()
    if aud not in allowed_auds:
        return 403, _error_payload(403, "forbidden_audience", "aud is not allowed.")

    ttl_default = _safe_int(environ.get("DEBUG_TOKEN_TTL_DEFAULT_MINUTES"), ua.DEFAULT_DEBUG_TOKEN_TTL_MINUTES)
    ttl_minutes = _safe_int(body.get("ttl_minutes"), ttl_default)
    if ttl_minutes <= 0 or ttl_minutes > ua.MAX_DEBUG_TOKEN_TTL_MINUTES:
        return 403, _error_payload(
            403,
            "forbidden_ttl",
            f"ttl_minutes must be between 1 and {ua.MAX_DEBUG_TOKEN_TTL_MINUTES}.",
        )

    jwt_secret = str(environ.get("UNIVERSE_JWT_SECRET") or "").strip()
    if not jwt_secret:
        return 403, _error_payload(403, "auth_not_configured", "UNIVERSE_JWT_SECRET is not configured.")

    try:
        token = ua.issue_debug_token(
            user_id=user_id,
            secret=jwt_secret,
            issuer=issuer,
            audience=aud,
            ttl_minutes=ttl_minutes,
        )
        claims = ua.validate_jwt_claims(
            token=token,
            secret=jwt_secret,
            issuer=issuer,
            audience=aud,
        )
    except ua.AuthError as exc:
        return exc.status, _error_payload(exc.status, exc.code, exc.message)

    LOGGER.info(
        "debug_token_issued user_id=%s aud=%s exp=%s jti=%s token_masked=%s token_fp=%s",
        user_id,
        aud,
        claims.get("exp"),
        claims.get("jti"),
        ua.mask_token(token),
        ua.token_fingerprint(token),
    )
    return 200, {
        "status": 200,
        "code": "issued",
        "token": token,
        "token_type": "Bearer",
        "iss": claims.get("iss"),
        "aud": aud,
        "user_id": claims.get("user_id"),
        "expires_at_utc": _utc_iso(claims.get("exp")),
        "expires_in_seconds": int(claims.get("exp")) - int(claims.get("iat")),
        "jti": claims.get("jti"),
    }


class DebugTokenHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "UniverseDebugToken/1.0"

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/debug/token":
            self._send_json(404, _error_payload(404, "not_found", "Endpoint not found."))
            return

        content_length = _safe_int(self.headers.get("Content-Length"), 0)
        body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(body_raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be object")
        except Exception:
            self._send_json(400, _error_payload(400, "invalid_json", "Request body must be valid JSON object."))
            return

        status, payload = issue_debug_token_response(body=body, headers=self.headers, environ=os.environ)
        self._send_json(status, payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            payload = {
                "status": 200,
                "code": "ok",
                "service": "debug_token_server",
                "app_env": str(os.environ.get("APP_ENV") or os.environ.get("ENV") or "staging"),
            }
            self._send_json(200, payload)
            return
        self._send_json(404, _error_payload(404, "not_found", "Endpoint not found."))

    def log_message(self, format: str, *args: Any) -> None:
        # Keep server logs concise and avoid request body/token leaks.
        LOGGER.info("%s - - %s", self.address_string(), format % args)


def run_server(host: str = "0.0.0.0", port: int = 8787) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    server = ThreadingHTTPServer((host, int(port)), DebugTokenHTTPRequestHandler)
    LOGGER.info("debug_token_server_started host=%s port=%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    run_server(
        host=os.getenv("DEBUG_TOKEN_HOST", "0.0.0.0"),
        port=_safe_int(os.getenv("DEBUG_TOKEN_PORT"), 8787),
    )
