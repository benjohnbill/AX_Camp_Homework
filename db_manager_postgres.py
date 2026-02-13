import json
import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.getenv("DATABASE_URL")


@contextmanager
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


def _to_float32_vector(embedding: Optional[Iterable[float]]) -> Optional[List[float]]:
    if embedding is None:
        return None
    arr = np.asarray(list(embedding), dtype=np.float32).reshape(-1)
    if arr.size != 1536:
        print(f"[postgres] Skip invalid embedding dimension={arr.size}, expected=1536")
        return None
    return arr.tolist()


def upsert_log(
    log_id: str,
    content: str,
    meta_type: str = "Fragment",
    parent_id: Optional[str] = None,
    action_plan: Optional[str] = None,
    created_at: Optional[str] = None,
    embedding: Optional[Iterable[float]] = None,
    emotion: Optional[str] = None,
    dimension: Optional[str] = None,
    keywords: Optional[Iterable[str]] = None,
    linked_constitutions: Optional[Iterable[str]] = None,
    duration: Optional[int] = None,
    debt_repaid: int = 0,
    kanban_status: Optional[str] = None,
    quality_score: int = 0,
    reward_points: int = 0,
) -> None:
    vec = _to_float32_vector(embedding)
    keywords_json = json.dumps(list(keywords or []), ensure_ascii=False)
    linked_json = json.dumps(list(linked_constitutions or []), ensure_ascii=False)

    sql = """
    INSERT INTO logs (
        id, content, meta_type, parent_id, action_plan, created_at,
        embedding, emotion, dimension, keywords, linked_constitutions,
        duration, debt_repaid, kanban_status, quality_score, reward_points
    )
    VALUES (
        %(id)s, %(content)s, %(meta_type)s, %(parent_id)s, %(action_plan)s,
        COALESCE(%(created_at)s::timestamptz, NOW()), %(embedding)s, %(emotion)s,
        %(dimension)s, %(keywords)s::jsonb, %(linked_constitutions)s::jsonb,
        %(duration)s, %(debt_repaid)s, %(kanban_status)s, %(quality_score)s, %(reward_points)s
    )
    ON CONFLICT (id) DO UPDATE SET
        content = EXCLUDED.content,
        meta_type = EXCLUDED.meta_type,
        parent_id = EXCLUDED.parent_id,
        action_plan = EXCLUDED.action_plan,
        created_at = EXCLUDED.created_at,
        embedding = EXCLUDED.embedding,
        emotion = EXCLUDED.emotion,
        dimension = EXCLUDED.dimension,
        keywords = EXCLUDED.keywords,
        linked_constitutions = EXCLUDED.linked_constitutions,
        duration = EXCLUDED.duration,
        debt_repaid = EXCLUDED.debt_repaid,
        kanban_status = EXCLUDED.kanban_status,
        quality_score = EXCLUDED.quality_score,
        reward_points = EXCLUDED.reward_points;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "id": log_id,
                    "content": content,
                    "meta_type": meta_type,
                    "parent_id": parent_id,
                    "action_plan": action_plan,
                    "created_at": created_at,
                    "embedding": vec,
                    "emotion": emotion,
                    "dimension": dimension,
                    "keywords": keywords_json,
                    "linked_constitutions": linked_json,
                    "duration": duration,
                    "debt_repaid": debt_repaid,
                    "kanban_status": kanban_status,
                    "quality_score": quality_score,
                    "reward_points": reward_points,
                },
            )
        conn.commit()


def save_embedding(log_id: str, emb: np.ndarray) -> None:
    vec = _to_float32_vector(np.asarray(emb, dtype=np.float32).reshape(-1))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE logs SET embedding = %s WHERE id = %s", (vec, log_id))
        conn.commit()


def get_all_embeddings() -> Tuple[List[str], Optional[np.ndarray]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, embedding FROM logs WHERE embedding IS NOT NULL ORDER BY created_at DESC")
            rows = cur.fetchall()

    ids: List[str] = []
    vectors: List[np.ndarray] = []

    for log_id, vec in rows:
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != 1536:
                print(f"[postgres] skip id={log_id} invalid dim={arr.size}")
                continue
            ids.append(log_id)
            vectors.append(arr)
        except Exception:
            continue

    if not vectors:
        return [], None
    return ids, np.vstack(vectors).astype(np.float32)


def get_all_embeddings_meta() -> Tuple[List[str], Optional[np.ndarray]]:
    return get_all_embeddings()


def load_logs_by_ids(ids: List[str]) -> List[Dict]:
    if not ids:
        return []

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, meta_type, keywords, linked_constitutions
                FROM logs
                WHERE id = ANY(%s)
                """,
                (ids,),
            )
            rows = cur.fetchall()

    result: List[Dict] = []
    for log_id, content, meta_type, keywords, linked in rows:
        result.append(
            {
                "id": log_id,
                "content": content,
                "meta_type": meta_type,
                "keywords": keywords or [],
                "linked_constitutions": linked or [],
            }
        )
    return result


def ensure_fts() -> None:
    """Ensure FTS index exists; skip gracefully if service permission is restricted."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_logs_fts_en
                    ON logs USING GIN (to_tsvector('english', COALESCE(content, '')))
                    """
                )
            conn.commit()
    except Exception as exc:
        print(f"[postgres] ensure_fts skipped: {exc}")
