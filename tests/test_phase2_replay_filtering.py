from datetime import datetime, timedelta, timezone

from app import _build_weekly_replay_payload, _parse_log_timestamp


def _as_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def test_parse_log_timestamp_normalizes_to_utc():
    parsed = _parse_log_timestamp("2026-03-04T10:00:00Z")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_build_weekly_replay_payload_strict_7_day_and_tiers():
    now_utc = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
    logs = [
        {
            "id": "t1",
            "meta_type": "session_completed",
            "created_at": _as_z(now_utc - timedelta(days=1)),
            "content": "completed session",
        },
        {
            "id": "t2",
            "meta_type": "session_interrupted",
            "created_at": _as_z(now_utc - timedelta(days=2)),
            "content": "interrupted session",
        },
        {
            "id": "t3",
            "meta_type": "supporting_evidence",
            "created_at": _as_z(now_utc - timedelta(days=3)),
            "content": "curated screenshot",
        },
        {
            "id": "old",
            "meta_type": "session_completed",
            "created_at": _as_z(now_utc - timedelta(days=9)),
            "content": "too old",
        },
        {
            "id": "unknown-type",
            "meta_type": "journal_entry",
            "created_at": _as_z(now_utc - timedelta(days=1)),
            "content": "not replay tier",
        },
        {
            "id": "bad-ts",
            "meta_type": "supporting_evidence",
            "created_at": "not-a-timestamp",
            "content": "bad timestamp",
        },
    ]

    filtered, counts = _build_weekly_replay_payload(
        logs=logs,
        now_utc=now_utc,
        lookback_days=7,
        limit=10,
    )

    ids = [row["id"] for row in filtered]
    assert ids == ["t1", "t2", "t3"]
    assert counts["tier1_completed"] == 1
    assert counts["tier2_interrupted"] == 1
    assert counts["tier3_supporting_evidence"] == 1


def test_build_weekly_replay_payload_respects_limit():
    now_utc = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
    logs = [
        {
            "id": "a",
            "meta_type": "session_completed",
            "created_at": _as_z(now_utc - timedelta(hours=1)),
            "content": "a",
        },
        {
            "id": "b",
            "meta_type": "session_interrupted",
            "created_at": _as_z(now_utc - timedelta(hours=2)),
            "content": "b",
        },
        {
            "id": "c",
            "meta_type": "supporting_evidence",
            "created_at": _as_z(now_utc - timedelta(hours=3)),
            "content": "c",
        },
    ]

    filtered, counts = _build_weekly_replay_payload(logs=logs, now_utc=now_utc, limit=2)

    assert [row["id"] for row in filtered] == ["a", "b"]
    assert counts["tier1_completed"] == 1
    assert counts["tier2_interrupted"] == 1
    assert counts["tier3_supporting_evidence"] == 0
