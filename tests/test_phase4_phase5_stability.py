import narrative_logic as logic


def test_build_context_from_logs_handles_none_values():
    logs = [
        {"created_at": None, "content": None},
        {"created_at": "2026-02-17T08:00:00Z", "content": "hello"},
    ]

    context = logic._build_context_from_logs(logs)

    assert "[unknown] " in context
    assert "[2026-02-17] hello" in context


def test_check_yesterday_promise_handles_none_created_at(monkeypatch):
    monkeypatch.setattr(
        logic,
        "load_logs",
        lambda: [
            {"meta_type": "Apology", "created_at": None, "id": "x1", "content": "c", "action_plan": "a"},
            {"meta_type": "Log", "created_at": "2026-02-16T09:00:00Z", "id": "x2", "content": "n"},
        ],
    )

    result = logic.check_yesterday_promise()

    assert result is None or isinstance(result, dict)
