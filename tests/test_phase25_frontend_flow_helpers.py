from datetime import datetime, timedelta, timezone

from app import (
    _FLOW_STAGE_FOCUS_RUNNING,
    _FLOW_STAGE_REFLECTION,
    _FLOW_STAGE_RETRO_TIMEBOX,
    _collect_session_evidence_candidates,
    _next_stage_after_focus,
    _resolve_chronos_stage,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def test_resolve_chronos_stage_maps_legacy_values():
    assert _resolve_chronos_stage("") == "start_choice"
    assert _resolve_chronos_stage("idle") == "start_choice"
    assert _resolve_chronos_stage("setup") == "start_choice"
    assert _resolve_chronos_stage("timer") == _FLOW_STAGE_FOCUS_RUNNING


def test_next_stage_after_focus_by_entry_mode():
    assert _next_stage_after_focus("focus_now") == _FLOW_STAGE_RETRO_TIMEBOX
    assert _next_stage_after_focus("plan") == _FLOW_STAGE_REFLECTION


def test_collect_session_evidence_candidates_filters_by_session_start():
    now = datetime(2026, 3, 5, 11, 0, tzinfo=timezone.utc)
    logs = [
        {
            "id": "old",
            "meta_type": "Log",
            "content": "old before session",
            "created_at": _iso(now - timedelta(hours=3)),
        },
        {
            "id": "new-1",
            "meta_type": "Log",
            "content": "new during session",
            "created_at": _iso(now - timedelta(minutes=20)),
        },
        {
            "id": "new-2",
            "meta_type": "supporting_evidence",
            "content": "ocr evidence",
            "created_at": _iso(now - timedelta(minutes=10)),
        },
    ]

    selected = _collect_session_evidence_candidates(
        logs=logs,
        session_start_raw=_iso(now - timedelta(minutes=30)),
        limit=10,
    )

    ids = [row["id"] for row in selected]
    assert ids == ["new-2", "new-1"]
