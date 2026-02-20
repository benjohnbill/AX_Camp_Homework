"""
tests/test_interface_parity.py
방안 C: db_manager_postgres.py와 db_manager.py(SQLite fallback)의
공개 함수 인터페이스가 일치하는지 자동으로 검증한다.

실행: python -m pytest tests/test_interface_parity.py -v
"""

# SQLite fallback 시 반드시 존재해야 하는 핵심 함수 목록
REQUIRED_FUNCTIONS = [
    # CRUD
    "create_log",
    "get_all_logs",
    "get_log_by_id",
    "get_logs_for_analytics",
    "get_logs_last_7_days",
    "get_fragment_count",
    "get_random_echo",
    # Constitution / Core
    "get_cores",
    "get_constitutions",
    "create_gap",
    # Streak / Debt
    "check_streak_and_apply_penalty",
    "update_streak",
    "get_current_streak",
    "get_debt_count",
    "increment_debt",
    "decrement_debt",
    "add_recovery_points",
    # Chat
    "save_chat_message",
    "get_chat_history",
    "clear_chat_history",
    # Connections
    "add_connection",
    "get_connections",
    "get_connection_counts",
    # Kanban
    "get_kanban_cards",
    "update_kanban_status",
    "create_kanban_card",
    # Genesis
    "inject_genesis_data",
    "init_database",
    # Embeddings (via db_router)
    "get_all_embeddings",
    "save_embedding",
    "load_logs_by_ids",
    "ensure_fts",
]


def test_postgres_has_required_functions():
    """db_manager_postgres.py에 필수 함수가 모두 존재하는지 검증."""
    import db_manager_postgres as pg
    missing = [fn for fn in REQUIRED_FUNCTIONS if not hasattr(pg, fn)]
    assert not missing, (
        f"db_manager_postgres.py에 누락된 함수:\n" + "\n".join(f"  - {fn}" for fn in missing)
    )


def test_sqlite_has_required_functions():
    """db_manager.py(SQLite fallback)에 필수 함수가 모두 존재하는지 검증.
    fallback 전환 시 동작 불일치를 조기에 발견하기 위해 동일한 목록을 검사한다.
    """
    import db_manager as sq
    missing = [fn for fn in REQUIRED_FUNCTIONS if not hasattr(sq, fn)]
    assert not missing, (
        f"db_manager.py(SQLite)에 누락된 함수:\n" + "\n".join(f"  - {fn}" for fn in missing)
    )


def test_no_extra_critical_functions_only_in_postgres():
    """Postgres에만 있는 함수 중 SQLite fallback에서도 필요할 수 있는 함수를 경고로 나열한다.
    (hard failure가 아닌 정보성 검사)
    """
    import db_manager_postgres as pg
    import db_manager as sq

    pg_funcs = {fn for fn in dir(pg) if not fn.startswith("_")}
    sq_funcs = {fn for fn in dir(sq) if not fn.startswith("_")}

    postgres_only = pg_funcs - sq_funcs
    # 알려진 Postgres 전용 함수는 무시
    known_postgres_only = {
        "upsert_log", "get_conn", "get_connection", "keyword_search",
        "ensure_fts_index", "get_all_embeddings_meta", "get_all_embeddings_as_float32",
        "get_cores_with_stats", "get_logs_paginated", "get_logs_by_type",
        "update_log", "get_last_log_time", "get_recovery_points", "get_longest_streak",
        "reset_debt", "update_last_log_time",
        # [B-2] Chronos 타이머 함수
        "set_chronos_timer", "get_chronos_timer", "clear_chronos_timer",
    }
    unexpected = postgres_only - known_postgres_only
    if unexpected:
        print(f"\n[INFO] Postgres에만 있고 SQLite에 없는 함수 (검토 권장): {sorted(unexpected)}")
    # 정보성 검사이므로 항상 통과
    assert True
