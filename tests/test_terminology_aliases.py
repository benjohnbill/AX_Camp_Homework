import narrative_logic as logic
import db_manager as db


def test_entropy_mode_alias_matches_red_mode(monkeypatch):
    monkeypatch.setattr(db, "get_debt_count", lambda: 2)
    assert logic.is_red_mode() is True
    assert logic.is_entropy_mode() is True


def test_process_gap_alias_calls_process_apology(monkeypatch):
    sentinel = {"ok": True}

    def fake_process_apology(content, constitution_id, action_plan):
        assert content == "gap"
        assert constitution_id == "core-1"
        assert action_plan == "plan"
        return sentinel

    monkeypatch.setattr(logic, "process_apology", fake_process_apology)
    assert logic.process_gap("gap", "core-1", "plan") is sentinel
