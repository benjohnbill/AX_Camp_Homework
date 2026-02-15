import os

DATASTORE = os.getenv("DATASTORE", "sqlite").strip().lower()

if DATASTORE == "postgres":
    from db_manager_postgres import (  # noqa: F401
        ensure_fts,
        get_all_embeddings,
        load_logs_by_ids,
        save_embedding,
        upsert_log,
    )
else:
    from db_manager_sqlite import (  # noqa: F401
        ensure_fts,
        get_all_embeddings,
        load_logs_by_ids,
        save_embedding,
    )

    def upsert_log(*args, **kwargs):
        raise NotImplementedError("upsert_log is only available when DATASTORE=postgres")
