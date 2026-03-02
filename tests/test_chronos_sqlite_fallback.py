import importlib
from datetime import datetime, timedelta


def _load_sqlite_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASTORE", "sqlite")
    import db_manager as dbm

    dbm = importlib.reload(dbm)
    monkeypatch.setattr(dbm, "DB_PATH", str(tmp_path / "narrative.db"), raising=True)
    dbm.init_database()
    return dbm


def test_chronos_timer_roundtrip_sqlite(monkeypatch, tmp_path):
    dbm = _load_sqlite_manager(monkeypatch, tmp_path)
    end_time = datetime.now() + timedelta(minutes=25)

    dbm.set_chronos_timer(end_time)
    restored = dbm.get_chronos_timer()

    assert restored is not None
    assert abs((restored - end_time).total_seconds()) < 1

    dbm.clear_chronos_timer()
    assert dbm.get_chronos_timer() is None

