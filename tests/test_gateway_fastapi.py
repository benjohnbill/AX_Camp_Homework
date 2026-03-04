import time

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


def test_contract_refine_payload_text_required():
    app = gw.create_app(_env())
    client = _client(app)

    res = client.post("/v1/narrative/refine", json={"text": ""})

    assert res.status_code == 400
    assert "Empty text" in res.json()["error"]


def test_contract_refine_payload_success(monkeypatch):
    app = gw.create_app(_env())
    client = _client(app)
    monkeypatch.setattr(gw.logic, "refine_narrative_with_ai", lambda text: f"refined::{text}")

    res = client.post("/v1/narrative/refine", json={"text": "raw memo"})

    assert res.status_code == 200
    assert res.json()["refined_text"] == "refined::raw memo"


def test_contract_save_payload_text_required():
    app = gw.create_app(_env())
    client = _client(app)

    res = client.post("/v1/narrative", json={"text": ""})

    assert res.status_code == 400
    assert "Empty text" in res.json()["error"]


def test_contract_save_payload_success(monkeypatch):
    app = gw.create_app(_env())
    client = _client(app)
    monkeypatch.setattr(gw.logic, "save_log", lambda text, **kwargs: {"id": "log_1"})

    res = client.post("/v1/narrative", json={"text": "hello"})

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "ok"
    assert payload["log_id"] == "log_1"


def test_phase1_execution_core_loop_and_today():
    app = gw.create_app(_env())
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "plan"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    assert start.json()["flow_stage"] == "frog"

    focus_start = client.post(f"/v1/execution/session/{session_id}/focus/start")
    assert focus_start.status_code == 200
    assert focus_start.json()["flow_stage"] == "focus_running"

    focus_end = client.post(f"/v1/execution/session/{session_id}/focus/end")
    assert focus_end.status_code == 200
    assert focus_end.json()["flow_stage"] == "reflect_pending"

    reflect = client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={
            "reflection_good": "집중 시작이 빨랐다.",
            "reflection_hard": "중간 알림이 방해됐다.",
            "reflection_next_action": "내일은 알림 차단 후 시작한다.",
        },
    )
    assert reflect.status_code == 200
    assert reflect.json()["flow_stage"] == "done"

    today = client.get("/v1/execution/session/today")
    assert today.status_code == 200
    payload = today.json()
    assert payload["status"] == "ok"
    assert payload["session"]["id"] == session_id
    assert payload["session"]["flow_stage"] == "done"


def test_phase1_reflect_requires_three_required_fields():
    app = gw.create_app(_env())
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "focus_now"})
    session_id = start.json()["session_id"]

    reflect = client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={"reflection_good": "good", "reflection_hard": "hard"},
    )
    assert reflect.status_code == 400
    assert "required" in reflect.json()["error"]


def test_phase1_journal_promote_and_core_promote():
    app = gw.create_app(_env())
    client = _client(app)

    entry = client.post(
        "/v1/journal/entry",
        json={"entry_text": "오늘 계획 없이 바로 집중했다.", "next_action": "내일은 25분 먼저 시작."},
    )
    assert entry.status_code == 200
    entry_id = entry.json()["entry_id"]

    promote = client.post(f"/v1/journal/{entry_id}/promote")
    assert promote.status_code == 200
    session_id = promote.json()["session_id"]

    core = client.post(
        "/v1/core/promote",
        json={
            "source_type": "execution_session",
            "source_id": session_id,
            "title": "아침 첫 집중은 알림 차단",
            "body": "집중 전 방해 요소 제거가 재현 가능한 규칙이다.",
            "promoted_by": "user_test",
        },
    )
    assert core.status_code == 200
    assert core.json()["status"] == "ok"
    assert core.json()["core_entry_id"].startswith("core_")


def test_phase1_ocr_ingest_non_blocking_on_ai_failure(monkeypatch):
    app = gw.create_app(_env())
    client = _client(app)
    monkeypatch.setattr(
        gw.logic,
        "refine_image_to_narrative_with_ai",
        lambda content: (_ for _ in ()).throw(RuntimeError("ai down")),
    )

    res = client.post(
        "/v1/ocr/ingest",
        files={"image": ("sample.png", b"abc123", "image/png")},
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "accepted"
    assert payload["image_event_id"].startswith("img_")
    assert payload["ocr_status"] in {"queued", "running", "failed"}


def test_phase2_ai_job_lifecycle_and_polling():
    app = gw.create_app(_env(REDIRECTING_AI_DELAY_MS="50"))
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "plan"})
    session_id = start.json()["session_id"]

    commit = client.post(f"/v1/execution/session/{session_id}/commit")
    assert commit.status_code == 200
    assert len(commit.json()["queued_jobs"]) == 1

    end_focus = client.post(f"/v1/execution/session/{session_id}/focus/end")
    assert end_focus.status_code == 200
    assert len(end_focus.json()["queued_jobs"]) == 2

    insights = client.get(f"/v1/execution/session/{session_id}/insights")
    assert insights.status_code == 200
    payload = insights.json()
    assert payload["status"] == "ok"
    assert payload["job_status"]["auto_tag_extraction"] in {"queued", "running", "succeeded"}
    assert payload["job_status"]["similar_session_linking"] in {"queued", "running", "succeeded"}
    assert payload["job_status"]["next_action_recommendation"] in {"queued", "running", "succeeded"}

    job_id = payload["job_ids"]["next_action_recommendation"]
    terminal_state = None
    for _ in range(20):
        polled = client.get(f"/v1/jobs/{job_id}")
        assert polled.status_code == 200
        terminal_state = polled.json()["job"]["state"]
        if terminal_state in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    assert terminal_state == "succeeded"


def test_phase2_insight_fallback_when_ai_jobs_fail():
    app = gw.create_app(
        _env(REDIRECTING_AI_FAIL_JOB_TYPES="similar_session_linking,next_action_recommendation")
    )
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "focus_now"})
    session_id = start.json()["session_id"]
    client.post(f"/v1/execution/session/{session_id}/focus/end")

    client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={
            "reflection_good": "시작은 좋았다.",
            "reflection_hard": "알림 방해가 있었다.",
            "reflection_next_action": "내일은 알림을 끄고 시작한다.",
        },
    )

    # Give background workers a short window to transition to failed state.
    time.sleep(0.05)
    insights = client.get(f"/v1/execution/session/{session_id}/insights")
    assert insights.status_code == 200
    payload = insights.json()
    assert payload["status"] == "ok"
    assert payload["insight_source"] == "rule"
    assert payload["insights"]["next_action"] == "내일은 알림을 끄고 시작한다."
    assert payload["job_status"]["next_action_recommendation"] == "failed"
    assert payload["job_status"]["similar_session_linking"] == "failed"


def test_phase2_ai_job_idempotency_is_deterministic():
    app = gw.create_app(_env(REDIRECTING_AI_DELAY_MS="250"))
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "plan"})
    session_id = start.json()["session_id"]

    first = client.post(f"/v1/execution/session/{session_id}/commit")
    second = client.post(f"/v1/execution/session/{session_id}/commit")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["queued_jobs"][0] == second.json()["queued_jobs"][0]

    first_focus = client.post(f"/v1/execution/session/{session_id}/focus/end")
    second_focus = client.post(f"/v1/execution/session/{session_id}/focus/end")
    assert first_focus.status_code == 200
    assert second_focus.status_code == 200
    assert first_focus.json()["queued_jobs"] == second_focus.json()["queued_jobs"]


def test_phase2_core_loop_non_blocking_under_ai_delay_and_failure():
    app = gw.create_app(
        _env(
            REDIRECTING_AI_DELAY_MS="1000",
            REDIRECTING_AI_FAIL_JOB_TYPES="auto_tag_extraction,similar_session_linking,next_action_recommendation",
        )
    )
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "focus_now"})
    session_id = start.json()["session_id"]

    t0 = time.perf_counter()
    focus_end = client.post(f"/v1/execution/session/{session_id}/focus/end")
    elapsed_focus_end = time.perf_counter() - t0
    assert focus_end.status_code == 200
    assert elapsed_focus_end < 0.8

    t1 = time.perf_counter()
    reflect = client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={
            "reflection_good": "집중은 시작했다.",
            "reflection_hard": "중간에 알림이 많았다.",
            "reflection_next_action": "다음엔 알림 차단 후 시작한다.",
        },
    )
    elapsed_reflect = time.perf_counter() - t1
    assert reflect.status_code == 200
    assert reflect.json()["flow_stage"] == "done"
    assert elapsed_reflect < 0.8

    t2 = time.perf_counter()
    week = client.get("/v1/execution/insight/week")
    elapsed_week = time.perf_counter() - t2
    assert week.status_code == 200
    week_payload = week.json()
    assert week_payload["status"] == "ok"
    assert week_payload["insight_source"] == "rule"
    assert elapsed_week < 0.8


def test_phase2_week_insight_uses_ai_when_available():
    app = gw.create_app(_env(REDIRECTING_AI_DELAY_MS="10"))
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "plan"})
    session_id = start.json()["session_id"]
    client.post(f"/v1/execution/session/{session_id}/commit")
    client.post(f"/v1/execution/session/{session_id}/focus/end")

    # Wait shortly for background jobs to complete.
    time.sleep(0.05)
    week = client.get("/v1/execution/insight/week")
    assert week.status_code == 200
    payload = week.json()
    assert payload["status"] == "ok"
    assert payload["insight_source"] in {"rule", "ai"}
    assert payload["metrics"]["sessions_started"] >= 1


def test_phase25_plan_first_flow_frog_timebox_commit_focus_reflect():
    app = gw.create_app(_env())
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "plan"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    assert start.json()["flow_stage"] == "frog"

    frog = client.post(
        f"/v1/execution/session/{session_id}/frog",
        json={"frog_title": "핵심 제안서 초안", "frog_why": "마감 리스크 완화"},
    )
    assert frog.status_code == 200
    assert frog.json()["flow_stage"] == "timebox_edit"

    draft = client.post(
        f"/v1/execution/session/{session_id}/timebox/draft",
        json={
            "blocks": [
                {
                    "id": "blk_1",
                    "title": "초안 작성",
                    "goal": "서론/본론 구조 완성",
                    "starts_at": "2026-03-05T09:00:00Z",
                    "ends_at": "2026-03-05T10:00:00Z",
                }
            ],
            "manual_tags": ["proposal", "deadline"],
        },
    )
    assert draft.status_code == 200
    assert draft.json()["blocks_count"] == 1
    assert draft.json()["flow_stage"] == "timebox_edit"

    commit = client.post(f"/v1/execution/session/{session_id}/commit")
    assert commit.status_code == 200
    assert commit.json()["flow_stage"] == "focus_running"
    assert commit.json()["plan_status"] == "committed"
    assert len(commit.json()["queued_jobs"]) == 1

    focus_end = client.post(f"/v1/execution/session/{session_id}/focus/end")
    assert focus_end.status_code == 200
    assert focus_end.json()["flow_stage"] == "reflect_pending"
    assert len(focus_end.json()["queued_jobs"]) == 2

    reflect = client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={
            "reflection_good": "집중 블록을 예정대로 마쳤다.",
            "reflection_hard": "중간 알림으로 흐름이 끊겼다.",
            "reflection_next_action": "다음엔 시작 전 알림 차단.",
        },
    )
    assert reflect.status_code == 200
    assert reflect.json()["flow_stage"] == "done"


def test_phase25_focus_first_retro_flow():
    app = gw.create_app(_env())
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "focus_now"})
    session_id = start.json()["session_id"]
    assert start.json()["flow_stage"] == "focus_running"

    focus_end = client.post(f"/v1/execution/session/{session_id}/focus/end")
    assert focus_end.status_code == 200
    assert focus_end.json()["flow_stage"] == "retro_timebox"

    retro = client.post(
        f"/v1/execution/session/{session_id}/timebox/retro",
        json={
            "blocks": [
                {
                    "title": "집중 회고 블록",
                    "goal": "중단 원인 기록",
                    "starts_at": "2026-03-05T10:00:00Z",
                    "ends_at": "2026-03-05T10:20:00Z",
                }
            ]
        },
    )
    assert retro.status_code == 200
    assert retro.json()["flow_stage"] == "reflect_pending"
    assert retro.json()["blocks_count"] == 1

    reflect = client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={
            "reflection_good": "빠르게 몰입했다.",
            "reflection_hard": "초반에 컨텍스트 스위칭이 있었다.",
            "reflection_next_action": "첫 5분 계획 메모 후 시작한다.",
        },
    )
    assert reflect.status_code == 200
    assert reflect.json()["flow_stage"] == "done"


def test_phase25_ocr_session_link_and_reflection_curation():
    app = gw.create_app(_env())
    client = _client(app)

    start = client.post("/v1/execution/session/start", json={"entry_mode": "plan"})
    session_id = start.json()["session_id"]

    upload = client.post(
        f"/v1/execution/session/{session_id}/evidence/upload",
        files={"image": ("ev.png", b"evidence", "image/png")},
    )
    assert upload.status_code == 200
    upload_payload = upload.json()
    assert upload_payload["status"] == "accepted"
    assert upload_payload["session_id"] == session_id
    image_event_id = upload_payload["image_event_id"]

    link = client.post(
        f"/v1/execution/session/{session_id}/evidence/link",
        json={"links": [{"image_event_id": image_event_id, "decision": "linked", "user_meaning": "핵심 도표"}]},
    )
    assert link.status_code == 200
    assert link.json()["summary"]["linked"] == 1

    reflect = client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={
            "reflection_good": "증거를 연결해 회고가 쉬웠다.",
            "reflection_hard": "해석은 일부 지연됐다.",
            "reflection_next_action": "증거 캡처 즉시 1줄 의미를 남긴다.",
            "evidence_links": [
                {"image_event_id": image_event_id, "decision": "linked", "user_meaning": "핵심 도표 완성 시점"}
            ],
        },
    )
    assert reflect.status_code == 200
    summary = reflect.json()["evidence_link_summary"]
    assert summary["linked"] == 1
    assert summary["missing"] == 0


def test_phase25_state_transition_bundle_start_commit_focus_reflect_journal_promote_core():
    app = gw.create_app(_env(REDIRECTING_AI_DELAY_MS="500", REDIRECTING_AI_FAIL_JOB_TYPES="auto_tag_extraction"))
    client = _client(app)

    t0 = time.perf_counter()
    start = client.post("/v1/execution/session/start", json={"entry_mode": "plan"})
    session_id = start.json()["session_id"]
    client.post(f"/v1/execution/session/{session_id}/frog", json={"frog_title": "오늘 핵심", "frog_why": "우선순위 고정"})
    client.post(
        f"/v1/execution/session/{session_id}/timebox/draft",
        json={"blocks": [{"title": "핵심 블록", "starts_at": "2026-03-05T09:00:00Z", "ends_at": "2026-03-05T09:20:00Z"}]},
    )
    commit = client.post(f"/v1/execution/session/{session_id}/commit")
    focus_end = client.post(f"/v1/execution/session/{session_id}/focus/end")
    reflect = client.post(
        f"/v1/execution/session/{session_id}/reflect",
        json={
            "reflection_good": "핵심 단계를 모두 실행했다.",
            "reflection_hard": "중간 알림 방해가 있었다.",
            "reflection_next_action": "알림 차단 규칙을 계속 적용한다.",
        },
    )
    elapsed = time.perf_counter() - t0
    assert start.status_code == 200
    assert commit.status_code == 200
    assert focus_end.status_code == 200
    assert reflect.status_code == 200
    assert reflect.json()["flow_stage"] == "done"
    assert elapsed < 2.5

    journal = client.post(
        "/v1/journal/entry",
        json={"entry_text": "세션 회고 기록", "next_action": "내일 같은 시간 재시도", "manual_tags": ["retry"]},
    )
    assert journal.status_code == 200
    entry_id = journal.json()["entry_id"]
    promote = client.post(f"/v1/journal/{entry_id}/promote")
    assert promote.status_code == 200
    promoted_session_id = promote.json()["session_id"]

    core = client.post(
        "/v1/core/promote",
        json={
            "source_type": "execution_session",
            "source_id": promoted_session_id,
            "title": "아침 첫 집중 규칙",
            "body": "짧은 계획 이후 바로 집중 시작이 유효했다.",
            "promoted_by": "tester",
        },
    )
    assert core.status_code == 200
    assert core.json()["core_entry_id"].startswith("core_")
