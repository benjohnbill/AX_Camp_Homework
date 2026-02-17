import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class Repository(Protocol):
    def __getattr__(self, name: str): ...


class ModuleRepository:
    """Thin adapter that exposes backend module functions through one object."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name: str):
        return getattr(self._module, name)


_repo_cache = None


def get_repository() -> Repository:
    global _repo_cache
    if _repo_cache is not None:
        return _repo_cache

    datastore = os.getenv("DATASTORE", "sqlite").strip().lower()
    if datastore == "postgres":
        import db_manager_postgres as backend
    else:
        import db_manager as backend

    _repo_cache = ModuleRepository(backend)
    return _repo_cache

