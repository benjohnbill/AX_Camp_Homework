import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import date, datetime, timedelta, timezone
from threading import RLock
from typing import Any, Mapping

import uvicorn
from fastapi import FastAPI, Request, Response, Body, File, UploadFile, Form
from fastapi.responses import JSONResponse, RedirectResponse

import universe_auth as ua
import narrative_logic as logic

DEFAULT_UPSTREAM_EMBED_URL = "https://benjohnbill-ax-camp-homework.streamlit.app/?embed=universe_3d"


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _runtime_secret(environ: Mapping[str, Any], key: str, default: str = "") -> str:
    value = environ.get(key)
    return str(value).strip() if value else default


def _cache_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store", "Pragma": "no-cache"}


def _auth_payload(status: int, code: str, message: str) -> dict[str, Any]:
    return {"status": int(status), "code": code, "message": message, "route": "gateway_universe_3d"}


def create_app(environ: Mapping[str, Any] | None = None) -> FastAPI:
    env = dict(environ or os.environ)
    app = FastAPI(title="Universe Strict Auth Gateway", version="1.0.0")

    jwt_secret = _runtime_secret(env, "UNIVERSE_JWT_SECRET", "")
    session_secret = _runtime_secret(env, "UNIVERSE_SESSION_SECRET", "") or jwt_secret
    issuer = _runtime_secret(env, "UNIVERSE_AUTH_ISSUER", ua.DEFAULT_ISSUER)
    audience = _runtime_secret(env, "UNIVERSE_AUTH_AUDIENCE", ua.DEFAULT_AUDIENCE)
    cookie_name = _runtime_secret(env, "UNIVERSE_SESSION_COOKIE", ua.DEFAULT_COOKIE_NAME) or ua.DEFAULT_COOKIE_NAME
    session_ttl = _safe_int(
        _runtime_secret(env, "UNIVERSE_SESSION_TTL_SECONDS", str(ua.DEFAULT_SESSION_TTL_SECONDS)),
        ua.DEFAULT_SESSION_TTL_SECONDS,
    )
    upstream_embed_url = _runtime_secret(env, "UNIVERSE_UPSTREAM_EMBED_URL", DEFAULT_UPSTREAM_EMBED_URL)
    ocr_inline_timeout_ms = _safe_int(_runtime_secret(env, "REDIRECTING_OCR_INLINE_TIMEOUT_MS", "120"), 120)
    bg_workers = max(1, _safe_int(_runtime_secret(env, "REDIRECTING_BG_WORKERS", "4"), 4))
    ai_delay_ms = max(0, _safe_int(_runtime_secret(env, "REDIRECTING_AI_DELAY_MS", "0"), 0))
    ai_fail_job_types = {
        item.strip()
        for item in _runtime_secret(env, "REDIRECTING_AI_FAIL_JOB_TYPES", "").split(",")
        if item.strip()
    }

    runtime_lock = RLock()
    runtime_executor = ThreadPoolExecutor(max_workers=bg_workers)
    execution_sessions: dict[str, dict[str, Any]] = {}
    execution_session_ids: list[str] = []
    image_events: dict[str, dict[str, Any]] = {}
    journal_entries: dict[str, dict[str, Any]] = {}
    core_entries: dict[str, dict[str, Any]] = {}
    ai_jobs: dict[str, dict[str, Any]] = {}
    ai_jobs_by_idempotency: dict[str, str] = {}
    ai_job_links_by_session: dict[str, dict[str, str]] = {}

    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _utc_now_iso() -> str:
        return _utc_now().isoformat()

    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:16]}"

    def _safe_text(value: Any) -> str:
        return str(value or "").strip()

    def _parse_iso8601(raw: Any) -> datetime | None:
        text = _safe_text(raw)
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    def _resolve_session_date(raw: Any) -> str:
        text = _safe_text(raw)
        if not text:
            return date.today().isoformat()
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except Exception:
            return date.today().isoformat()

    def _resolve_anchor_date(raw: Any) -> date:
        text = _safe_text(raw)
        if not text:
            return date.today()
        try:
            return date.fromisoformat(text[:10])
        except Exception:
            return date.today()

    def _session_summary(session: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": session["id"],
            "session_date": session["session_date"],
            "flow_stage": session["flow_stage"],
            "plan_status": session["plan_status"],
            "entry_mode": session["entry_mode"],
        }

    def _latest_today_session_id() -> str | None:
        today_key = date.today().isoformat()
        with runtime_lock:
            for session_id in reversed(execution_session_ids):
                session = execution_sessions.get(session_id)
                if session and session.get("session_date") == today_key:
                    return str(session_id)
        return None

    def _resolve_evidence_session_link(explicit_session_id: str | None) -> tuple[str | None, str]:
        explicit = _safe_text(explicit_session_id)
        if explicit:
            with runtime_lock:
                if explicit in execution_sessions:
                    return explicit, "explicit"
            return None, "invalid_explicit"

        today_latest = _latest_today_session_id()
        if today_latest:
            return today_latest, "today_latest"
        return None, "unlinked"

    def _build_reflection_projection(session: dict[str, Any]) -> str:
        return (
            f"[ExecutionSession] good={session.get('reflection_good', '')} | "
            f"hard={session.get('reflection_hard', '')} | "
            f"next_action={session.get('reflection_next_action', '')}"
        )

    def _persist_reflection_projection(session: dict[str, Any]) -> None:
        # Reflection persistence must not block user-facing completion.
        def _save_projection() -> None:
            try:
                logic.save_log(_build_reflection_projection(session))
            except Exception:
                pass

        try:
            runtime_executor.submit(_save_projection)
        except Exception:
            pass

    def _rule_based_auto_tags(session: dict[str, Any]) -> list[str]:
        merged = " ".join(
            [
                _safe_text(session.get("reflection_good")),
                _safe_text(session.get("reflection_hard")),
                _safe_text(session.get("reflection_next_action")),
                _safe_text(session.get("frog_title")),
            ]
        ).lower()
        tags: list[str] = []
        if any(token in merged for token in ["focus", "집중"]):
            tags.append("focus")
        if any(token in merged for token in ["deadline", "마감"]):
            tags.append("deadline")
        if any(token in merged for token in ["interrupt", "알림", "방해"]):
            tags.append("interruptions")
        if not tags:
            tags = ["consistency"]
        return tags

    def _rule_based_next_action(session: dict[str, Any]) -> str:
        value = _safe_text(session.get("reflection_next_action"))
        if value:
            return value
        return "다음 세션 시작 전에 첫 25분 집중 블록을 캘린더에 고정한다."

    def _simulate_ai_job(job: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "")
        if ai_delay_ms > 0:
            time.sleep(ai_delay_ms / 1000.0)
        if job_type in ai_fail_job_types:
            raise RuntimeError(f"simulated_failure:{job_type}")

        if job_type == "auto_tag_extraction":
            return {"auto_tags": _rule_based_auto_tags(session)}
        if job_type == "similar_session_linking":
            return {"similar_pattern": "최근 완료 세션과 유사하게 시작 집중은 빠르고 중간 방해가 반복됨"}
        if job_type == "next_action_recommendation":
            return {"next_action": _rule_based_next_action(session)}
        return {}

    def _run_ai_job(job_id: str) -> None:
        with runtime_lock:
            job = ai_jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["attempt"] = int(job.get("attempt") or 0) + 1
            job["updated_at"] = _utc_now_iso()
            snapshot = dict(job)
            session = dict(execution_sessions.get(str(job.get("entity_id") or ""), {}))
        try:
            result = _simulate_ai_job(snapshot, session)
            with runtime_lock:
                job = ai_jobs.get(job_id)
                if not job:
                    return
                job["status"] = "succeeded"
                job["result"] = result
                job["last_error"] = None
                job["updated_at"] = _utc_now_iso()
        except Exception as exc:
            with runtime_lock:
                job = ai_jobs.get(job_id)
                if not job:
                    return
                job["status"] = "failed"
                job["last_error"] = str(exc)
                job["updated_at"] = _utc_now_iso()

    def _enqueue_ai_job(
        job_type: str,
        entity_type: str,
        entity_id: str,
        payload_json: dict[str, Any] | None = None,
        allow_retry_from_terminal: bool = False,
    ) -> dict[str, Any]:
        idempotency_key = f"{job_type}:{entity_type}:{entity_id}"
        with runtime_lock:
            existing_id = ai_jobs_by_idempotency.get(idempotency_key)
            if existing_id and existing_id in ai_jobs:
                existing_job = ai_jobs[existing_id]
                if existing_job.get("status") in {"queued", "running"}:
                    return dict(existing_job)
                if not allow_retry_from_terminal:
                    return dict(existing_job)

            job_id = _new_id("job")
            now_iso = _utc_now_iso()
            job = {
                "id": job_id,
                "job_type": job_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "payload_json": payload_json or {},
                "status": "queued",
                "attempt": 0,
                "max_attempts": 3,
                "idempotency_key": idempotency_key,
                "run_after": now_iso,
                "last_error": None,
                "result": {},
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            ai_jobs[job_id] = job
            ai_jobs_by_idempotency[idempotency_key] = job_id
            links = ai_job_links_by_session.setdefault(entity_id, {})
            links[job_type] = job_id
        runtime_executor.submit(_run_ai_job, job_id)
        return dict(job)

    def _ocr_background_refine(event_id: str, content: bytes, mime_type: str) -> None:
        with runtime_lock:
            event = image_events.get(event_id)
            if not event:
                return
            event["ocr_status"] = "running"
            event["updated_at"] = _utc_now_iso()
        try:
            try:
                refined = logic.refine_image_to_narrative_with_ai(content, mime_type=mime_type)
            except TypeError:
                # Backward-compatible for tests/patches that replace the function with a single-arg callable.
                refined = logic.refine_image_to_narrative_with_ai(content)
            with runtime_lock:
                event = image_events.get(event_id)
                if not event:
                    return
                event["ocr_status"] = "succeeded"
                event["ocr_text"] = _safe_text(refined)
                event["updated_at"] = _utc_now_iso()
        except Exception as exc:
            with runtime_lock:
                event = image_events.get(event_id)
                if not event:
                    return
                event["ocr_status"] = "failed"
                event["last_error"] = str(exc)
                event["updated_at"] = _utc_now_iso()

    def _authenticate(request: Request) -> ua.AuthResult:
        return ua.authenticate_request(
            headers=dict(request.headers),
            cookies=request.cookies,
            jwt_secret=jwt_secret,
            session_secret=session_secret,
            issuer=issuer,
            audience=audience,
            cookie_name=cookie_name,
            session_ttl_seconds=session_ttl,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "issuer": issuer,
            "audience": audience,
            "cookie_name": cookie_name,
            "upstream_embed_url": upstream_embed_url,
        }

    @app.get("/gateway/session")
    async def gateway_session(request: Request) -> JSONResponse:
        auth = _authenticate(request)
        if not auth.ok:
            payload = dict(auth.payload or _auth_payload(auth.status, "unauthorized", "Authentication failed."))
            payload.setdefault("route", "gateway_session")
            return JSONResponse(payload, status_code=auth.status, headers=_cache_headers())

        payload = {
            "status": 200,
            "code": "ok",
            "source": str((auth.payload or {}).get("source") or "unknown"),
            "user_id": auth.user_id or "",
            "route": "gateway_session",
        }
        response = JSONResponse(payload, status_code=200, headers=_cache_headers())
        if auth.session_cookie_name and auth.session_cookie_value:
            response.set_cookie(
                key=auth.session_cookie_name,
                value=auth.session_cookie_value,
                max_age=int(auth.session_cookie_max_age or session_ttl),
                httponly=True,
                secure=True,
                samesite="none",
                path="/",
            )
        return response

    @app.get("/gateway/universe_3d")
    async def gateway_universe_3d(request: Request) -> Response:
        auth = _authenticate(request)
        if not auth.ok:
            payload = dict(auth.payload or _auth_payload(auth.status, "unauthorized", "Authentication failed."))
            payload.setdefault("route", "gateway_universe_3d")
            return JSONResponse(payload, status_code=auth.status, headers=_cache_headers())

        response = RedirectResponse(upstream_embed_url, status_code=307, headers=_cache_headers())
        response.headers["X-Auth-Source"] = str((auth.payload or {}).get("source") or "unknown")
        response.headers["X-Auth-User"] = auth.user_id or ""
        if auth.session_cookie_name and auth.session_cookie_value:
            response.set_cookie(
                key=auth.session_cookie_name,
                value=auth.session_cookie_value,
                max_age=int(auth.session_cookie_max_age or session_ttl),
                httponly=True,
                secure=True,
                samesite="none",
                path="/",
            )
        return response

    @app.post("/v1/execution/session/start")
    async def execution_session_start(body: dict = Body(default={})) -> JSONResponse:
        session_date = _resolve_session_date(body.get("session_date"))
        entry_mode = _safe_text(body.get("entry_mode")) or "plan"
        if entry_mode not in {"plan", "focus_now"}:
            return JSONResponse({"error": "Invalid entry_mode"}, status_code=400)

        session_id = _new_id("sess")
        session = {
            "id": session_id,
            "session_date": session_date,
            "entry_mode": entry_mode,
            "flow_stage": "frog" if entry_mode == "plan" else "focus_running",
            "plan_status": "draft",
            "frog_title": "",
            "frog_why": "",
            "manual_tags": [],
            "timebox_blocks": [],
            "retro_blocks": [],
            "retro_saved": False,
            "evidence_links": [],
            "focus_started_at": None,
            "focus_ended_at": None,
            "focus_total_minutes": 0,
            "reflection_good": "",
            "reflection_hard": "",
            "reflection_next_action": "",
            "reflection_free_text": "",
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        with runtime_lock:
            execution_sessions[session_id] = session
            execution_session_ids.append(session_id)
        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "flow_stage": session["flow_stage"],
                "entry_mode": entry_mode,
            }
        )

    @app.post("/v1/execution/session/{session_id}/frog")
    async def execution_save_frog(session_id: str, body: dict = Body(default={})) -> JSONResponse:
        frog_title = _safe_text(body.get("frog_title"))
        frog_why = _safe_text(body.get("frog_why"))
        if not frog_title:
            return JSONResponse({"error": "frog_title is required"}, status_code=400)

        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            session["frog_title"] = frog_title
            session["frog_why"] = frog_why
            if session.get("entry_mode") == "plan":
                session["flow_stage"] = "timebox_edit"
            session["updated_at"] = _utc_now_iso()
            stage = str(session.get("flow_stage") or "timebox_edit")
        return JSONResponse({"status": "ok", "session_id": session_id, "flow_stage": stage})

    @app.post("/v1/execution/session/{session_id}/timebox/draft")
    async def execution_timebox_draft(session_id: str, body: dict = Body(default={})) -> JSONResponse:
        blocks_raw = body.get("blocks")
        blocks = blocks_raw if isinstance(blocks_raw, list) else []
        normalized_blocks: list[dict[str, Any]] = []
        for idx, raw in enumerate(blocks):
            item = raw if isinstance(raw, dict) else {}
            starts_at = _safe_text(item.get("starts_at"))
            ends_at = _safe_text(item.get("ends_at"))
            starts_dt = _parse_iso8601(starts_at)
            ends_dt = _parse_iso8601(ends_at)
            if starts_dt is not None and ends_dt is not None and starts_dt >= ends_dt:
                return JSONResponse({"error": f"invalid_timebox_range_at_index_{idx}"}, status_code=400)
            normalized_blocks.append(
                {
                    "id": _safe_text(item.get("id")) or _new_id("blk"),
                    "title": _safe_text(item.get("title")),
                    "goal": _safe_text(item.get("goal")),
                    "why": _safe_text(item.get("why")),
                    "inbox_note": _safe_text(item.get("inbox_note")),
                    "starts_at": starts_at,
                    "ends_at": ends_at,
                    "order_index": int(item.get("order_index") or idx),
                }
            )

        manual_tags_raw = body.get("manual_tags")
        manual_tags = [str(x).strip() for x in manual_tags_raw] if isinstance(manual_tags_raw, list) else []

        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            session["timebox_blocks"] = normalized_blocks
            session["manual_tags"] = [x for x in manual_tags if x]
            if session.get("entry_mode") == "plan":
                session["flow_stage"] = "timebox_edit"
            session["updated_at"] = _utc_now_iso()
            stage = str(session.get("flow_stage") or "timebox_edit")
        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "flow_stage": stage,
                "blocks_count": len(normalized_blocks),
            }
        )

    @app.post("/v1/execution/session/{session_id}/timebox/retro")
    async def execution_timebox_retro(session_id: str, body: dict = Body(default={})) -> JSONResponse:
        skip = bool(body.get("skip"))
        blocks_raw = body.get("blocks")
        blocks = blocks_raw if isinstance(blocks_raw, list) else []
        normalized_blocks: list[dict[str, Any]] = []
        if not skip:
            for idx, raw in enumerate(blocks):
                item = raw if isinstance(raw, dict) else {}
                starts_at = _safe_text(item.get("starts_at"))
                ends_at = _safe_text(item.get("ends_at"))
                starts_dt = _parse_iso8601(starts_at)
                ends_dt = _parse_iso8601(ends_at)
                if starts_dt is not None and ends_dt is not None and starts_dt >= ends_dt:
                    return JSONResponse({"error": f"invalid_retro_timebox_range_at_index_{idx}"}, status_code=400)
                normalized_blocks.append(
                    {
                        "id": _safe_text(item.get("id")) or _new_id("rblk"),
                        "title": _safe_text(item.get("title")),
                        "goal": _safe_text(item.get("goal")),
                        "why": _safe_text(item.get("why")),
                        "inbox_note": _safe_text(item.get("inbox_note")),
                        "starts_at": starts_at,
                        "ends_at": ends_at,
                        "order_index": int(item.get("order_index") or idx),
                    }
                )

        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            session["retro_blocks"] = normalized_blocks
            session["retro_saved"] = True
            session["flow_stage"] = "reflect_pending"
            session["updated_at"] = _utc_now_iso()
        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "flow_stage": "reflect_pending",
                "skip": skip,
                "blocks_count": len(normalized_blocks),
            }
        )

    @app.post("/v1/execution/session/{session_id}/focus/start")
    async def execution_focus_start(session_id: str) -> JSONResponse:
        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            session["focus_started_at"] = _utc_now_iso()
            session["flow_stage"] = "focus_running"
            session["updated_at"] = _utc_now_iso()
        return JSONResponse({"status": "ok", "session_id": session_id, "flow_stage": "focus_running"})

    @app.post("/v1/execution/session/{session_id}/focus/end")
    async def execution_focus_end(session_id: str) -> JSONResponse:
        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            ended_at = _utc_now()
            started_at = _parse_iso8601(session.get("focus_started_at"))
            minutes = 0
            if started_at is not None:
                minutes = max(0, int((ended_at - started_at).total_seconds() // 60))
            session["focus_ended_at"] = ended_at.isoformat()
            session["focus_total_minutes"] = minutes
            next_stage = "retro_timebox" if session.get("entry_mode") == "focus_now" else "reflect_pending"
            session["flow_stage"] = next_stage
            session["updated_at"] = _utc_now_iso()
        similar_job = _enqueue_ai_job(
            job_type="similar_session_linking",
            entity_type="execution_session",
            entity_id=session_id,
            payload_json={"session_id": session_id},
        )
        next_action_job = _enqueue_ai_job(
            job_type="next_action_recommendation",
            entity_type="execution_session",
            entity_id=session_id,
            payload_json={"session_id": session_id},
        )
        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "flow_stage": next_stage,
                "focus_total_minutes": minutes,
                "queued_jobs": [similar_job["id"], next_action_job["id"]],
            }
        )

    @app.post("/v1/execution/session/{session_id}/commit")
    async def execution_commit(session_id: str) -> JSONResponse:
        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            session["plan_status"] = "committed"
            session["flow_stage"] = "focus_running"
            session["updated_at"] = _utc_now_iso()
        job = _enqueue_ai_job(
            job_type="auto_tag_extraction",
            entity_type="execution_session",
            entity_id=session_id,
            payload_json={"session_id": session_id},
        )
        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "flow_stage": "focus_running",
                "plan_status": "committed",
                "queued_jobs": [job["id"]],
            }
        )

    @app.post("/v1/execution/session/{session_id}/reflect")
    async def execution_reflect(session_id: str, body: dict = Body(default={})) -> JSONResponse:
        reflection_good = _safe_text(body.get("reflection_good"))
        reflection_hard = _safe_text(body.get("reflection_hard"))
        reflection_next_action = _safe_text(body.get("reflection_next_action"))
        reflection_free_text = _safe_text(body.get("reflection_free_text"))
        evidence_links_raw = body.get("evidence_links")
        evidence_links = evidence_links_raw if isinstance(evidence_links_raw, list) else []
        if not reflection_good or not reflection_hard or not reflection_next_action:
            return JSONResponse(
                {"error": "reflection_good/reflection_hard/reflection_next_action are required"},
                status_code=400,
            )

        linked_count = 0
        skipped_count = 0
        missing_count = 0
        normalized_links: list[dict[str, Any]] = []
        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            for raw in evidence_links:
                item = raw if isinstance(raw, dict) else {}
                image_event_id = _safe_text(item.get("image_event_id"))
                decision = _safe_text(item.get("decision")).lower() or "linked"
                user_meaning = _safe_text(item.get("user_meaning"))
                if not image_event_id:
                    continue
                event = image_events.get(image_event_id)
                if not event:
                    missing_count += 1
                    normalized_links.append(
                        {
                            "image_event_id": image_event_id,
                            "decision": decision,
                            "status": "missing",
                            "user_meaning": user_meaning,
                        }
                    )
                    continue
                event["session_id"] = session_id
                if decision == "skipped":
                    event["link_status"] = "skipped"
                    skipped_count += 1
                else:
                    event["link_status"] = "linked"
                    linked_count += 1
                if user_meaning:
                    event["user_meaning"] = user_meaning
                event["updated_at"] = _utc_now_iso()
                normalized_links.append(
                    {
                        "image_event_id": image_event_id,
                        "decision": "skipped" if decision == "skipped" else "linked",
                        "status": "ok",
                        "user_meaning": user_meaning,
                    }
                )

            session["reflection_good"] = reflection_good
            session["reflection_hard"] = reflection_hard
            session["reflection_next_action"] = reflection_next_action
            session["reflection_free_text"] = reflection_free_text
            session["evidence_links"] = normalized_links
            session["flow_stage"] = "done"
            session["updated_at"] = _utc_now_iso()
            session_copy = dict(session)

        _persist_reflection_projection(session_copy)
        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "flow_stage": "done",
                "evidence_link_summary": {
                    "linked": linked_count,
                    "skipped": skipped_count,
                    "missing": missing_count,
                },
            }
        )

    @app.post("/v1/execution/session/{session_id}/evidence/link")
    async def execution_evidence_link(session_id: str, body: dict = Body(default={})) -> JSONResponse:
        links_raw = body.get("links")
        links = links_raw if isinstance(links_raw, list) else []
        if not links:
            return JSONResponse({"error": "links is required"}, status_code=400)

        linked_count = 0
        skipped_count = 0
        missing_count = 0
        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            for raw in links:
                item = raw if isinstance(raw, dict) else {}
                image_event_id = _safe_text(item.get("image_event_id"))
                decision = _safe_text(item.get("decision")).lower() or "linked"
                user_meaning = _safe_text(item.get("user_meaning"))
                if not image_event_id:
                    continue
                event = image_events.get(image_event_id)
                if not event:
                    missing_count += 1
                    continue
                event["session_id"] = session_id
                event["link_status"] = "skipped" if decision == "skipped" else "linked"
                if user_meaning:
                    event["user_meaning"] = user_meaning
                event["updated_at"] = _utc_now_iso()
                if decision == "skipped":
                    skipped_count += 1
                else:
                    linked_count += 1
            session["updated_at"] = _utc_now_iso()
        return JSONResponse(
            {
                "status": "ok",
                "session_id": session_id,
                "summary": {"linked": linked_count, "skipped": skipped_count, "missing": missing_count},
            }
        )

    @app.get("/v1/execution/session/today")
    async def execution_today() -> JSONResponse:
        today_key = date.today().isoformat()
        with runtime_lock:
            for session_id in reversed(execution_session_ids):
                session = execution_sessions.get(session_id)
                if session and session.get("session_date") == today_key:
                    return JSONResponse({"status": "ok", "session": _session_summary(session)})
        return JSONResponse({"status": "not_found"})

    @app.get("/v1/execution/session/{session_id}/insights")
    async def execution_session_insights(session_id: str) -> JSONResponse:
        with runtime_lock:
            session = execution_sessions.get(session_id)
            if not session:
                return JSONResponse({"error": "session_not_found"}, status_code=404)
            session_copy = dict(session)
            job_links = dict(ai_job_links_by_session.get(session_id) or {})
            jobs_snapshot = {
                job_type: dict(ai_jobs[job_id])
                for job_type, job_id in job_links.items()
                if job_id in ai_jobs
            }

        job_status = {
            "auto_tag_extraction": jobs_snapshot.get("auto_tag_extraction", {}).get("status", "not_queued"),
            "similar_session_linking": jobs_snapshot.get("similar_session_linking", {}).get("status", "not_queued"),
            "next_action_recommendation": jobs_snapshot.get("next_action_recommendation", {}).get("status", "not_queued"),
        }

        fallback_tags = _rule_based_auto_tags(session_copy)
        fallback_pattern = "최근 세션 패턴을 규칙 기반으로 집계 중입니다."
        fallback_next_action = _rule_based_next_action(session_copy)

        auto_tags = list(fallback_tags)
        similar_pattern = fallback_pattern
        next_action = fallback_next_action
        insight_source = "rule"

        auto_job = jobs_snapshot.get("auto_tag_extraction")
        if auto_job and auto_job.get("status") == "succeeded":
            auto_tags = list(auto_job.get("result", {}).get("auto_tags") or fallback_tags)
            insight_source = "ai"

        similar_job = jobs_snapshot.get("similar_session_linking")
        if similar_job and similar_job.get("status") == "succeeded":
            similar_pattern = _safe_text(similar_job.get("result", {}).get("similar_pattern")) or fallback_pattern
            insight_source = "ai"

        next_action_job = jobs_snapshot.get("next_action_recommendation")
        if next_action_job and next_action_job.get("status") == "succeeded":
            next_action = _safe_text(next_action_job.get("result", {}).get("next_action")) or fallback_next_action
            insight_source = "ai"

        return JSONResponse(
            {
                "status": "ok",
                "job_status": job_status,
                "job_ids": {k: v for k, v in job_links.items() if k in job_status},
                "insight_source": insight_source,
                "auto_tags": auto_tags,
                "insights": {
                    "similar_pattern": similar_pattern,
                    "next_action": next_action,
                },
            }
        )

    @app.get("/v1/execution/insight/week")
    async def execution_week_insight(anchor_date: str | None = None) -> JSONResponse:
        anchor = _resolve_anchor_date(anchor_date)
        week_start = anchor - timedelta(days=6)
        with runtime_lock:
            sessions = [dict(execution_sessions[sid]) for sid in execution_session_ids if sid in execution_sessions]
            links_snapshot = {sid: dict(ai_job_links_by_session.get(sid) or {}) for sid in execution_session_ids}
            jobs_snapshot = {job_id: dict(job) for job_id, job in ai_jobs.items()}

        sessions_in_range = [
            s
            for s in sessions
            if week_start <= _resolve_anchor_date(s.get("session_date")) <= anchor
        ]

        sessions_started = len(sessions_in_range)
        focus_completed = len([s for s in sessions_in_range if _safe_text(s.get("focus_ended_at"))])
        reflection_written = len([s for s in sessions_in_range if _safe_text(s.get("reflection_next_action"))])
        blockers: list[str] = []
        if any(
            "interrupt" in _safe_text(s.get("reflection_hard")).lower()
            or "알림" in _safe_text(s.get("reflection_hard"))
            for s in sessions_in_range
        ):
            blockers.append("interruptions")
        if not blockers:
            blockers = ["context_switch"]

        recommended_next_action = "오전 첫 블록에서 알림을 차단하고 25분 집중을 시작한다."
        insight_source = "rule"
        for session in reversed(sessions_in_range):
            sid = str(session.get("id") or "")
            next_action_job_id = _safe_text(links_snapshot.get(sid, {}).get("next_action_recommendation"))
            if not next_action_job_id:
                continue
            job = jobs_snapshot.get(next_action_job_id)
            if not job:
                continue
            if job.get("status") == "succeeded":
                next_action = _safe_text(job.get("result", {}).get("next_action"))
                if next_action:
                    recommended_next_action = next_action
                    insight_source = "ai"
                    break

        return JSONResponse(
            {
                "status": "ok",
                "week_range": {"start": week_start.isoformat(), "end": anchor.isoformat()},
                "insight_source": insight_source,
                "metrics": {
                    "sessions_started": sessions_started,
                    "focus_completed": focus_completed,
                    "reflection_written": reflection_written,
                },
                "top_blockers": blockers,
                "recommended_next_action": recommended_next_action,
            }
        )

    @app.get("/v1/jobs/{job_id}")
    async def get_job_status(job_id: str) -> JSONResponse:
        with runtime_lock:
            job = ai_jobs.get(job_id)
            if not job:
                return JSONResponse({"error": "job_not_found"}, status_code=404)
            payload = {
                "id": job["id"],
                "job_type": job["job_type"],
                "entity_type": job["entity_type"],
                "entity_id": job["entity_id"],
                "state": job["status"],
                "attempt": job["attempt"],
                "max_attempts": job["max_attempts"],
                "updated_at": job["updated_at"],
                "last_error": job["last_error"],
            }
        return JSONResponse({"status": "ok", "job": payload})

    @app.post("/v1/journal/entry")
    async def journal_entry(body: dict = Body(default={})) -> JSONResponse:
        entry_text = _safe_text(body.get("entry_text"))
        next_action = _safe_text(body.get("next_action"))
        if not entry_text or not next_action:
            return JSONResponse({"error": "entry_text and next_action are required"}, status_code=400)

        entry_id = _new_id("journal")
        with runtime_lock:
            journal_entries[entry_id] = {
                "id": entry_id,
                "entry_text": entry_text,
                "next_action": next_action,
                "manual_tags": body.get("manual_tags") if isinstance(body.get("manual_tags"), list) else [],
                "auto_tags": [],
                "promoted_session_id": None,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }

        # Legacy logs projection should not block Journal save.
        try:
            logic.save_log(entry_text)
        except Exception:
            pass
        return JSONResponse({"status": "ok", "entry_id": entry_id})

    @app.post("/v1/journal/{entry_id}/promote")
    async def journal_promote(entry_id: str) -> JSONResponse:
        with runtime_lock:
            entry = journal_entries.get(entry_id)
            if not entry:
                return JSONResponse({"error": "journal_not_found"}, status_code=404)
            session_id = _new_id("sess")
            session = {
                "id": session_id,
                "session_date": date.today().isoformat(),
                "entry_mode": "plan",
                "flow_stage": "reflect_pending",
                "plan_status": "draft",
                "frog_title": "",
                "frog_why": "",
                "manual_tags": [],
                "timebox_blocks": [],
                "retro_blocks": [],
                "retro_saved": False,
                "evidence_links": [],
                "focus_started_at": None,
                "focus_ended_at": None,
                "focus_total_minutes": 0,
                "reflection_good": entry.get("entry_text", ""),
                "reflection_hard": "",
                "reflection_next_action": entry.get("next_action", ""),
                "reflection_free_text": "",
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
            entry["promoted_session_id"] = session_id
            entry["updated_at"] = _utc_now_iso()
            execution_sessions[session_id] = session
            execution_session_ids.append(session_id)
        return JSONResponse({"status": "ok", "entry_id": entry_id, "session_id": session_id})

    @app.post("/v1/core/promote")
    async def core_promote(body: dict = Body(default={})) -> JSONResponse:
        source_type = _safe_text(body.get("source_type"))
        source_id = _safe_text(body.get("source_id"))
        title = _safe_text(body.get("title"))
        entry_body = _safe_text(body.get("body"))
        promoted_by = _safe_text(body.get("promoted_by"))
        if not source_type or not source_id or not title or not entry_body or not promoted_by:
            return JSONResponse(
                {"error": "source_type/source_id/title/body/promoted_by are required"},
                status_code=400,
            )
        if source_type not in {"execution_session", "journal"}:
            return JSONResponse({"error": "invalid_source_type"}, status_code=400)

        with runtime_lock:
            source_exists = (
                bool(execution_sessions.get(source_id))
                if source_type == "execution_session"
                else bool(journal_entries.get(source_id))
            )
            if not source_exists:
                return JSONResponse({"error": "source_not_found"}, status_code=404)
            core_id = _new_id("core")
            core_entries[core_id] = {
                "id": core_id,
                "source_type": source_type,
                "source_id": source_id,
                "title": title,
                "body": entry_body,
                "promoted_by": promoted_by,
                "promoted_at": _utc_now_iso(),
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
        return JSONResponse({"status": "ok", "core_entry_id": core_id})

    @app.post("/v1/narrative/refine")
    async def refine_narrative(request: Request, body: dict = Body(...)) -> JSONResponse:
        # Simple open endpoint for MVP.
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "Empty text"}, status_code=400)

        refined = logic.refine_narrative_with_ai(text)
        return JSONResponse({"refined_text": refined})

    async def _handle_ocr_ingest(
        image: UploadFile | None,
        file: UploadFile | None,
        session_id: str | None,
    ) -> JSONResponse:
        # Accept both multipart field names for Android/web compatibility.
        upload = image or file
        if upload is None:
            return JSONResponse(
                {"error": "Missing image file. Use multipart field `image` or `file`."},
                status_code=400,
            )

        try:
            content = await upload.read()
        except Exception as exc:
            return JSONResponse({"error": f"failed_to_read_upload: {exc}"}, status_code=400)

        event_id = _new_id("img")
        mime_type = _safe_text(upload.content_type) or "image/jpeg"
        now_iso = _utc_now_iso()
        resolved_session_id, link_rule = _resolve_evidence_session_link(session_id)
        with runtime_lock:
            image_events[event_id] = {
                "id": event_id,
                "session_id": resolved_session_id,
                "storage_uri": f"memory://upload/{event_id}",
                "capture_source": "api",
                "ocr_status": "queued",
                "ocr_text": "",
                "ai_summary": "",
                "link_status": "linked" if resolved_session_id else "inbox",
                "link_rule": link_rule,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            if resolved_session_id and resolved_session_id in execution_sessions:
                session = execution_sessions[resolved_session_id]
                evidence_links = session.get("evidence_links")
                if not isinstance(evidence_links, list):
                    evidence_links = []
                evidence_links.append(
                    {
                        "image_event_id": event_id,
                        "decision": "linked",
                        "status": "ok",
                        "user_meaning": "",
                    }
                )
                session["evidence_links"] = evidence_links
                session["updated_at"] = _utc_now_iso()

        future = runtime_executor.submit(_ocr_background_refine, event_id, content, mime_type)
        refined_text = ""
        inline_timeout = max(0, ocr_inline_timeout_ms) / 1000.0
        if inline_timeout > 0:
            try:
                future.result(timeout=inline_timeout)
                with runtime_lock:
                    refined_text = _safe_text(image_events.get(event_id, {}).get("ocr_text"))
            except TimeoutError:
                pass
            except Exception:
                pass

        with runtime_lock:
            event = dict(image_events.get(event_id) or {})
        return JSONResponse(
            {
                "status": "accepted",
                "image_event_id": event_id,
                "ocr_status": event.get("ocr_status", "queued"),
                "session_id": event.get("session_id"),
                "link_rule": event.get("link_rule"),
                "refined_text": refined_text,
            }
        )

    @app.post("/v1/ocr/ingest")
    async def ocr_ingest(
        image: UploadFile | None = File(default=None),
        file: UploadFile | None = File(default=None),
        session_id: str | None = Form(default=None),
    ) -> JSONResponse:
        return await _handle_ocr_ingest(image=image, file=file, session_id=session_id)

    @app.post("/v1/execution/session/{session_id}/evidence/upload")
    async def execution_evidence_upload(
        session_id: str,
        image: UploadFile | None = File(default=None),
        file: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        return await _handle_ocr_ingest(image=image, file=file, session_id=session_id)

    @app.post("/v1/narrative/vision")
    async def vision_narrative_alias(
        image: UploadFile | None = File(default=None),
        file: UploadFile | None = File(default=None),
        session_id: str | None = Form(default=None),
    ) -> JSONResponse:
        return await _handle_ocr_ingest(image=image, file=file, session_id=session_id)

    @app.post("/v1/narrative")
    async def save_narrative(request: Request, body: dict = Body(...)) -> JSONResponse:
        # Simple open save for MVP (In production, should be auth-gated)
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "Empty text"}, status_code=400)
        
        log = logic.save_log(text)
        return JSONResponse({"status": "ok", "log_id": log.get("id")})

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "gateway_fastapi:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=_safe_int(os.getenv("GATEWAY_PORT"), 8790),
        log_level=os.getenv("GATEWAY_LOG_LEVEL", "info"),
    )
