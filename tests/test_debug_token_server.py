import debug_token_server as dts
import universe_auth as ua


def _environ(**overrides):
    base = {
        "APP_ENV": "staging",
        "DEBUG_TOKEN_ADMIN_KEY": "admin-secret",
        "UNIVERSE_JWT_SECRET": "jwt-secret",
        "UNIVERSE_AUTH_ISSUER": "iss-test",
        "UNIVERSE_AUTH_AUDIENCE": "android-universe",
        "DEBUG_TOKEN_ALLOWED_AUDIENCES": "android-universe",
        "DEBUG_TOKEN_TTL_DEFAULT_MINUTES": "60",
    }
    base.update(overrides)
    return base


def test_issue_debug_token_response_success():
    status, payload = dts.issue_debug_token_response(
        body={"user_id": "debug_user_001", "aud": "android-universe", "ttl_minutes": 45},
        headers={"X-Debug-Admin-Key": "admin-secret"},
        environ=_environ(),
    )
    assert status == 200
    assert payload["code"] == "issued"
    assert payload["user_id"] == "debug_user_001"
    assert payload["aud"] == "android-universe"
    claims = ua.validate_jwt_claims(
        token=payload["token"],
        secret="jwt-secret",
        issuer="iss-test",
        audience="android-universe",
    )
    assert claims["user_id"] == "debug_user_001"


def test_issue_debug_token_response_rejects_non_staging_env():
    status, payload = dts.issue_debug_token_response(
        body={"user_id": "u1"},
        headers={"X-Debug-Admin-Key": "admin-secret"},
        environ=_environ(APP_ENV="production"),
    )
    assert status == 403
    assert payload["code"] == "forbidden_env"


def test_issue_debug_token_response_rejects_missing_admin_key():
    status, payload = dts.issue_debug_token_response(
        body={"user_id": "u1"},
        headers={},
        environ=_environ(),
    )
    assert status == 401
    assert payload["code"] == "unauthorized_admin"


def test_issue_debug_token_response_rejects_forbidden_audience():
    status, payload = dts.issue_debug_token_response(
        body={"user_id": "u1", "aud": "unknown-aud"},
        headers={"X-Debug-Admin-Key": "admin-secret"},
        environ=_environ(),
    )
    assert status == 403
    assert payload["code"] == "forbidden_audience"


def test_issue_debug_token_response_rejects_large_ttl():
    status, payload = dts.issue_debug_token_response(
        body={"user_id": "u1", "ttl_minutes": 999},
        headers={"X-Debug-Admin-Key": "admin-secret"},
        environ=_environ(),
    )
    assert status == 403
    assert payload["code"] == "forbidden_ttl"
