import base64
import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


DEFAULT_ISSUER = "ax-camp-staging"
DEFAULT_AUDIENCE = "android-universe"
DEFAULT_COOKIE_NAME = "ax_universe_session"
DEFAULT_SESSION_TTL_SECONDS = 60 * 60 * 8
DEFAULT_DEBUG_TOKEN_TTL_MINUTES = 60
MAX_DEBUG_TOKEN_TTL_MINUTES = 120


class AuthError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass
class AuthResult:
    ok: bool
    status: int
    payload: dict[str, Any]
    user_id: Optional[str] = None
    session_cookie_name: Optional[str] = None
    session_cookie_value: Optional[str] = None
    session_cookie_max_age: Optional[int] = None


def token_fingerprint(token: str) -> str:
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 14:
        return "***"
    return f"{token[:8]}...{token[-6:]}"


def _b64url_decode(segment: str) -> bytes:
    if not isinstance(segment, str) or not segment:
        raise AuthError(401, "invalid_token", "Token segment is empty.")
    padding = "=" * ((4 - len(segment) % 4) % 4)
    try:
        return base64.urlsafe_b64decode(segment + padding)
    except Exception as exc:
        raise AuthError(401, "invalid_token", "Malformed token encoding.") from exc


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode_json_segment(segment: str) -> dict[str, Any]:
    raw = _b64url_decode(segment)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AuthError(401, "invalid_token", "Token segment is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise AuthError(401, "invalid_token", "Token segment must be a JSON object.")
    return parsed


def _normalize_aud(aud_claim: Any) -> list[str]:
    if isinstance(aud_claim, str):
        return [aud_claim]
    if isinstance(aud_claim, list):
        return [str(v) for v in aud_claim]
    return []


def _verify_hs256_signature(token: str, secret: str) -> tuple[dict[str, Any], dict[str, Any]]:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError(401, "invalid_token", "Token format must be header.payload.signature.")

    header_segment, payload_segment, signature_segment = parts
    header = _decode_json_segment(header_segment)
    payload = _decode_json_segment(payload_segment)

    alg = str(header.get("alg") or "")
    if alg != "HS256":
        raise AuthError(401, "invalid_token", "Unsupported token algorithm.")

    signed_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signed_input, hashlib.sha256).digest()
    expected_sig_segment = _b64url_encode(expected_sig)
    if not hmac.compare_digest(expected_sig_segment, signature_segment):
        raise AuthError(401, "invalid_token", "Token signature mismatch.")

    return header, payload


def validate_jwt_claims(
    token: str,
    secret: str,
    issuer: str,
    audience: str,
    now_epoch: Optional[int] = None,
) -> dict[str, Any]:
    if not token:
        raise AuthError(401, "missing_token", "Bearer token is required.")
    if not secret:
        raise AuthError(401, "auth_not_configured", "Server auth secret is not configured.")

    _, payload = _verify_hs256_signature(token, secret)
    now_epoch = now_epoch or int(datetime.now(timezone.utc).timestamp())

    exp_raw = payload.get("exp")
    try:
        exp = int(exp_raw)
    except Exception as exc:
        raise AuthError(401, "invalid_token", "Token exp claim is missing or invalid.") from exc
    if exp <= now_epoch:
        raise AuthError(401, "token_expired", "Token has expired.")

    iss = str(payload.get("iss") or "")
    if iss != issuer:
        raise AuthError(403, "forbidden_issuer", "Token issuer is not allowed.")

    aud_list = _normalize_aud(payload.get("aud"))
    if audience not in aud_list:
        raise AuthError(403, "forbidden_audience", "Token audience is not allowed.")

    user_id = str(payload.get("user_id") or "")
    if not user_id:
        raise AuthError(403, "forbidden_user", "Token user_id claim is required.")

    return payload


def issue_session_token(
    user_id: str,
    secret: str,
    issuer: str,
    audience: str,
    ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
) -> str:
    if not secret:
        raise AuthError(401, "auth_not_configured", "Server auth secret is not configured.")
    if not user_id:
        raise AuthError(403, "forbidden_user", "Cannot issue session token without user_id.")

    iat = int(datetime.now(timezone.utc).timestamp())
    exp = iat + int(ttl_seconds)

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": issuer,
        "aud": audience,
        "sub": user_id,
        "user_id": user_id,
        "iat": iat,
        "exp": exp,
        "typ": "universe_session",
    }
    header_segment = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signed_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    signature_segment = _b64url_encode(hmac.new(secret.encode("utf-8"), signed_input, hashlib.sha256).digest())
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def issue_debug_token(
    user_id: str,
    secret: str,
    issuer: str,
    audience: str,
    ttl_minutes: int = DEFAULT_DEBUG_TOKEN_TTL_MINUTES,
) -> str:
    if not secret:
        raise AuthError(401, "auth_not_configured", "Server auth secret is not configured.")
    if not user_id:
        raise AuthError(403, "forbidden_user", "Cannot issue debug token without user_id.")

    ttl_minutes = int(ttl_minutes)
    if ttl_minutes <= 0 or ttl_minutes > MAX_DEBUG_TOKEN_TTL_MINUTES:
        raise AuthError(
            403,
            "forbidden_ttl",
            f"ttl_minutes must be between 1 and {MAX_DEBUG_TOKEN_TTL_MINUTES}.",
        )

    iat = int(datetime.now(timezone.utc).timestamp())
    exp = iat + ttl_minutes * 60
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": issuer,
        "aud": audience,
        "sub": user_id,
        "user_id": user_id,
        "iat": iat,
        "exp": exp,
        "jti": str(uuid.uuid4()),
        "typ": "debug_access",
    }
    header_segment = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signed_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    signature_segment = _b64url_encode(hmac.new(secret.encode("utf-8"), signed_input, hashlib.sha256).digest())
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def _extract_bearer_token(headers: Mapping[str, Any]) -> Optional[str]:
    candidates = ("authorization", "Authorization", "AUTHORIZATION")
    raw_header = None
    for key in candidates:
        raw_header = headers.get(key)
        if raw_header:
            break
    if not raw_header:
        return None

    raw = str(raw_header).strip()
    if not raw.lower().startswith("bearer "):
        raise AuthError(401, "invalid_authorization", "Authorization header must use Bearer scheme.")
    token = raw[7:].strip()
    if not token:
        raise AuthError(401, "missing_token", "Bearer token is empty.")
    return token


def authenticate_request(
    headers: Mapping[str, Any],
    cookies: Mapping[str, Any],
    jwt_secret: str,
    session_secret: str,
    issuer: str = DEFAULT_ISSUER,
    audience: str = DEFAULT_AUDIENCE,
    cookie_name: str = DEFAULT_COOKIE_NAME,
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
) -> AuthResult:
    try:
        cookie_token = str(cookies.get(cookie_name) or "").strip()
        if cookie_token:
            payload = validate_jwt_claims(
                token=cookie_token,
                secret=session_secret,
                issuer=issuer,
                audience=audience,
            )
            return AuthResult(
                ok=True,
                status=200,
                payload={"status": 200, "code": "ok", "source": "cookie"},
                user_id=str(payload.get("user_id") or ""),
            )

        bearer_token = _extract_bearer_token(headers)
        if not bearer_token:
            raise AuthError(401, "missing_token", "Provide Authorization: Bearer <JWT> on first request.")

        payload = validate_jwt_claims(
            token=bearer_token,
            secret=jwt_secret,
            issuer=issuer,
            audience=audience,
        )
        user_id = str(payload.get("user_id") or "")
        session_token = issue_session_token(
            user_id=user_id,
            secret=session_secret,
            issuer=issuer,
            audience=audience,
            ttl_seconds=session_ttl_seconds,
        )
        return AuthResult(
            ok=True,
            status=200,
            payload={"status": 200, "code": "ok", "source": "bearer"},
            user_id=user_id,
            session_cookie_name=cookie_name,
            session_cookie_value=session_token,
            session_cookie_max_age=session_ttl_seconds,
        )
    except AuthError as exc:
        return AuthResult(
            ok=False,
            status=exc.status,
            payload={
                "status": exc.status,
                "code": exc.code,
                "message": exc.message,
            },
        )
