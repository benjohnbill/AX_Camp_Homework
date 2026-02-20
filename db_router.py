"""
db_router.py
단일 진입점: db_backend.get_repository()에 위임한다.
DATASTORE 판단 로직은 db_backend에서만 수행된다.
"""
from db_backend import get_repository as _get_repo


def ensure_fts():
    return _get_repo().ensure_fts()


def get_all_embeddings():
    return _get_repo().get_all_embeddings()


def load_logs_by_ids(ids):
    return _get_repo().load_logs_by_ids(ids)


def save_embedding(log_id, emb):
    return _get_repo().save_embedding(log_id, emb)


def upsert_log(*args, **kwargs):
    return _get_repo().upsert_log(*args, **kwargs)
