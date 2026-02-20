import universe_auth as ua


def test_bearer_auth_success_issues_session_cookie():
    jwt_secret = "jwt-secret"
    session_secret = "session-secret"
    bearer = ua.issue_session_token(
        user_id="u1",
        secret=jwt_secret,
        issuer="iss-x",
        audience="aud-x",
        ttl_seconds=3600,
    )
    headers = {"Authorization": f"Bearer {bearer}"}
    cookies = {}

    result = ua.authenticate_request(
        headers=headers,
        cookies=cookies,
        jwt_secret=jwt_secret,
        session_secret=session_secret,
        issuer="iss-x",
        audience="aud-x",
        cookie_name="ax_cookie",
        session_ttl_seconds=300,
    )

    assert result.ok is True
    assert result.user_id == "u1"
    assert result.payload["source"] == "bearer"
    assert result.session_cookie_name == "ax_cookie"
    assert isinstance(result.session_cookie_value, str) and result.session_cookie_value


def test_cookie_auth_success_without_bearer():
    session_secret = "session-secret"
    cookie_token = ua.issue_session_token(
        user_id="u2",
        secret=session_secret,
        issuer="iss-y",
        audience="aud-y",
        ttl_seconds=3600,
    )

    result = ua.authenticate_request(
        headers={},
        cookies={"ax_cookie": cookie_token},
        jwt_secret="jwt-secret",
        session_secret=session_secret,
        issuer="iss-y",
        audience="aud-y",
        cookie_name="ax_cookie",
        session_ttl_seconds=300,
    )

    assert result.ok is True
    assert result.user_id == "u2"
    assert result.payload["source"] == "cookie"


def test_missing_token_returns_401():
    result = ua.authenticate_request(
        headers={},
        cookies={},
        jwt_secret="jwt-secret",
        session_secret="session-secret",
        issuer="iss-z",
        audience="aud-z",
    )
    assert result.ok is False
    assert result.status == 401
    assert result.payload["code"] == "missing_token"


def test_wrong_audience_returns_403():
    jwt_secret = "jwt-secret"
    bad_aud = ua.issue_session_token(
        user_id="u3",
        secret=jwt_secret,
        issuer="iss-a",
        audience="other-aud",
        ttl_seconds=3600,
    )
    result = ua.authenticate_request(
        headers={"Authorization": f"Bearer {bad_aud}"},
        cookies={},
        jwt_secret=jwt_secret,
        session_secret="session-secret",
        issuer="iss-a",
        audience="aud-a",
    )
    assert result.ok is False
    assert result.status == 403
    assert result.payload["code"] == "forbidden_audience"


def test_expired_token_returns_401():
    jwt_secret = "jwt-secret"
    expired = ua.issue_session_token(
        user_id="u4",
        secret=jwt_secret,
        issuer="iss-b",
        audience="aud-b",
        ttl_seconds=-1,
    )
    result = ua.authenticate_request(
        headers={"Authorization": f"Bearer {expired}"},
        cookies={},
        jwt_secret=jwt_secret,
        session_secret="session-secret",
        issuer="iss-b",
        audience="aud-b",
    )
    assert result.ok is False
    assert result.status == 401
    assert result.payload["code"] == "token_expired"


def test_issue_debug_token_contains_required_claims():
    token = ua.issue_debug_token(
        user_id="u5",
        secret="debug-secret",
        issuer="iss-debug",
        audience="aud-debug",
        ttl_minutes=30,
    )
    claims = ua.validate_jwt_claims(
        token=token,
        secret="debug-secret",
        issuer="iss-debug",
        audience="aud-debug",
    )
    assert claims["user_id"] == "u5"
    assert "jti" in claims and claims["jti"]


def test_mask_token_and_fingerprint_do_not_expose_full_token():
    token = "abcdefghijklmnopqrstuvwxyz.0123456789.abcdefghij"
    masked = ua.mask_token(token)
    fp = ua.token_fingerprint(token)
    assert masked != token
    assert token not in masked
    assert len(fp) == 12
