from fastapi.testclient import TestClient

import gateway_fastapi as gw
import universe_auth as ua


def _env(**overrides):
    base = {
        "UNIVERSE_JWT_SECRET": "jwt-secret",
        "UNIVERSE_SESSION_SECRET": "session-secret",
        "UNIVERSE_AUTH_ISSUER": "ax-camp-staging",
        "UNIVERSE_AUTH_AUDIENCE": "android-universe",
        "UNIVERSE_SESSION_COOKIE": "ax_universe_session",
        "UNIVERSE_SESSION_TTL_SECONDS": "900",
        "UNIVERSE_UPSTREAM_EMBED_URL": "https://staging.example.com/?embed=universe_3d",
    }
    base.update(overrides)
    return base


def _issue_bearer(secret="jwt-secret", issuer="ax-camp-staging", audience="android-universe", user_id="u1"):
    return ua.issue_debug_token(
        user_id=user_id,
        secret=secret,
        issuer=issuer,
        audience=audience,
        ttl_minutes=30,
    )


def _client(app):
    # Use https base URL so Secure cookies are persisted across requests.
    return TestClient(app, base_url="https://testserver")


def test_first_request_bearer_sets_httponly_cookie_and_redirects():
    app = gw.create_app(_env())
    client = _client(app)
    token = _issue_bearer()

    res = client.get(
        "/gateway/universe_3d",
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=False,
    )

    assert res.status_code == 307
    assert res.headers["location"] == "https://staging.example.com/?embed=universe_3d"
    assert res.headers["x-auth-source"] == "bearer"
    set_cookie = res.headers.get("set-cookie", "")
    assert "ax_universe_session=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=none" in set_cookie


def test_subsequent_request_cookie_only_is_accepted():
    app = gw.create_app(_env())
    client = _client(app)
    token = _issue_bearer(user_id="cookie_user")

    first = client.get(
        "/gateway/universe_3d",
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=False,
    )
    assert first.status_code == 307

    second = client.get("/gateway/universe_3d", follow_redirects=False)
    assert second.status_code == 307
    assert second.headers["x-auth-source"] == "cookie"


def test_missing_auth_returns_real_401_json():
    app = gw.create_app(_env())
    client = _client(app)

    res = client.get("/gateway/universe_3d", follow_redirects=False)

    assert res.status_code == 401
    payload = res.json()
    assert payload["code"] == "missing_token"
    assert payload["route"] == "gateway_universe_3d"


def test_forbidden_audience_returns_real_403_json():
    app = gw.create_app(_env())
    client = _client(app)
    bad_token = _issue_bearer(audience="other-aud")

    res = client.get(
        "/gateway/universe_3d",
        headers={"Authorization": f"Bearer {bad_token}"},
        follow_redirects=False,
    )

    assert res.status_code == 403
    payload = res.json()
    assert payload["code"] == "forbidden_audience"
    assert payload["route"] == "gateway_universe_3d"


def test_gateway_session_endpoint_accepts_cookie_after_bootstrap():
    app = gw.create_app(_env())
    client = _client(app)
    token = _issue_bearer(user_id="session_user")

    first = client.get(
        "/gateway/session",
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=False,
    )
    assert first.status_code == 200
    assert first.json()["source"] == "bearer"

    second = client.get("/gateway/session", follow_redirects=False)
    assert second.status_code == 200
    assert second.json()["source"] == "cookie"
    assert second.json()["user_id"] == "session_user"


def test_ocr_ingest_canonical_accepts_image_field(monkeypatch):
    app = gw.create_app(_env())
    client = _client(app)
    monkeypatch.setattr(gw.logic, "refine_image_to_narrative_with_ai", lambda content: f"len={len(content)}")

    res = client.post(
        "/v1/ocr/ingest",
        files={"image": ("sample.png", b"abc123", "image/png")},
    )

    assert res.status_code == 200
    assert res.json()["refined_text"] == "len=6"


def test_ocr_ingest_canonical_accepts_file_field(monkeypatch):
    app = gw.create_app(_env())
    client = _client(app)
    monkeypatch.setattr(gw.logic, "refine_image_to_narrative_with_ai", lambda content: f"len={len(content)}")

    res = client.post(
        "/v1/ocr/ingest",
        files={"file": ("sample.jpg", b"abcd", "image/jpeg")},
    )

    assert res.status_code == 200
    assert res.json()["refined_text"] == "len=4"


def test_ocr_ingest_alias_uses_same_handler(monkeypatch):
    app = gw.create_app(_env())
    client = _client(app)
    monkeypatch.setattr(gw.logic, "refine_image_to_narrative_with_ai", lambda content: "alias-ok")

    canonical = client.post(
        "/v1/ocr/ingest",
        files={"image": ("a.png", b"1", "image/png")},
    )
    alias = client.post(
        "/v1/narrative/vision",
        files={"file": ("b.png", b"2", "image/png")},
    )

    assert canonical.status_code == 200
    assert alias.status_code == 200
    assert canonical.json()["refined_text"] == "alias-ok"
    assert alias.json()["refined_text"] == "alias-ok"


def test_ocr_ingest_rejects_missing_multipart_file():
    app = gw.create_app(_env())
    client = _client(app)

    res = client.post("/v1/ocr/ingest")

    assert res.status_code == 400
    assert "image" in res.json()["error"]
    assert "file" in res.json()["error"]
