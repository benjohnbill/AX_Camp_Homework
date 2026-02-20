import json
from datetime import datetime, timezone

import universe_3d


def test_prepare_3d_payload_projects_and_converts_datetime():
    logs = [
        {
            "id": "l1",
            "content": "hello universe",
            "meta_type": "Log",
            "created_at": datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc),
            "embedding": [0.1, 0.2],
        },
        {"id": "l2", "content": "", "meta_type": "Log"},
        "invalid",
    ]
    cores = [
        {
            "id": "c1",
            "content": "Core A",
            "meta_type": "Core",
            "created_at": datetime(2026, 2, 19, 8, 30, tzinfo=timezone.utc),
            "extra": object(),
        },
        {"id": "c2", "content": "   "},
    ]

    safe_logs, safe_cores = universe_3d._prepare_3d_payload(logs, cores)

    assert len(safe_logs) == 1
    assert set(safe_logs[0].keys()) == {"id", "content", "meta_type", "created_at"}
    assert safe_logs[0]["created_at"].startswith("2026-02-20T12:00:00")
    assert "embedding" not in safe_logs[0]

    assert len(safe_cores) == 1
    assert set(safe_cores[0].keys()) == {"id", "content", "meta_type", "created_at"}
    assert safe_cores[0]["created_at"].startswith("2026-02-19T08:30:00")


def test_json_default_supports_datetime_bytes_and_set():
    payload = {
        "created_at": datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc),
        "raw": b"abc",
        "tags": {"x", "y"},
    }

    encoded = json.dumps(payload, default=universe_3d._json_default)

    assert '"created_at"' in encoded
    assert '"raw": "abc"' in encoded
    assert '"tags"' in encoded


def test_render_3d_universe_handles_datetime_payload(monkeypatch):
    captured = {"called": False}

    def fake_html(html_string, height, scrolling):
        captured["called"] = True
        assert "const logsData =" in html_string
        assert "const coresData =" in html_string
        assert isinstance(height, int)
        assert isinstance(scrolling, bool)

    monkeypatch.setattr(universe_3d.components, "html", fake_html)

    logs = [{"id": "l1", "content": "text", "meta_type": "Log", "created_at": datetime.now(timezone.utc)}]
    cores = [{"id": "c1", "content": "core", "meta_type": "Core", "created_at": datetime.now(timezone.utc)}]

    universe_3d.render_3d_universe(logs, cores)

    assert captured["called"] is True
