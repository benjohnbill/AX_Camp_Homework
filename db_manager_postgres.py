import json
import os
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL")

GENESIS_DATA = [
    {
        "meta_type": "Core",
        "content": "순간의 열정이나 강렬한 욕망을 통한 동기부여를 지속가능하게 만들고 싶다.",
    },
    {
        "meta_type": "Log",
        "content": "플래너에 적힌 과업을 무시하고 하루종일 코딩만 했다.",
    },
    {
        "meta_type": "Log",
        "content": "12시에 자겠다고 했지만 지금 12시 39분이다.",
    },
]


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


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def _to_float32_vector(embedding: Optional[Iterable[float]]) -> Optional[List[float]]:
    if embedding is None:
        return None
    arr = np.asarray(list(embedding), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size != 1536:
        print(f"[postgres] Skip invalid embedding dimension={arr.size}, expected=1536")
        return None
    return arr.tolist()


def _json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _row_to_dict(row: Dict) -> Dict:
    d = dict(row)
    d["keywords"] = _json_list(d.get("keywords"))
    d["linked_constitutions"] = _json_list(d.get("linked_constitutions"))
    d["tags"] = _json_list(d.get("tags"))
    emb = d.get("embedding")
    if emb is None:
        d["embedding"] = []
    elif isinstance(emb, list):
        d["embedding"] = emb
    else:
        try:
            d["embedding"] = np.asarray(emb, dtype=np.float32).reshape(-1).tolist()
        except Exception:
            d["embedding"] = []
    d["timestamp"] = d.get("created_at")
    d["text"] = d.get("content", "")
    return d


def init_database():
    # Schema is managed via SQL files. Keep this as a safe no-op plus seed.
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO user_stats (id) VALUES (1) ON CONFLICT (id) DO NOTHING")
                cur.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS source_chat_id BIGINT")
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_history_source_chat_id "
                    "ON chat_history (source_chat_id)"
                )
            conn.commit()
    except Exception:
        pass


def upsert_log(
    log_id: str,
    content: str,
    meta_type: str = "Log",
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
    tags: Optional[Iterable[str]] = None,
) -> None:
    vec = _to_float32_vector(embedding)
    keywords_json = json.dumps(list(keywords or []), ensure_ascii=False)
    linked_json = json.dumps(list(linked_constitutions or []), ensure_ascii=False)
    tags_json = json.dumps(list(tags or []), ensure_ascii=False)

    sql = """
    INSERT INTO logs (
        id, content, meta_type, parent_id, action_plan, created_at,
        embedding, emotion, dimension, keywords, linked_constitutions,
        duration, debt_repaid, kanban_status, quality_score, reward_points, tags
    )
    VALUES (
        %(id)s, %(content)s, %(meta_type)s, %(parent_id)s, %(action_plan)s,
        COALESCE(%(created_at)s::timestamptz, NOW()), %(embedding)s, %(emotion)s,
        %(dimension)s, %(keywords)s::jsonb, %(linked_constitutions)s::jsonb,
        %(duration)s, %(debt_repaid)s, %(kanban_status)s, %(quality_score)s, %(reward_points)s, %(tags)s::jsonb
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
        reward_points = EXCLUDED.reward_points,
        tags = EXCLUDED.tags;
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
                    "tags": tags_json,
                },
            )
        conn.commit()


def create_log(
    content: str,
    meta_type: str = "Log",
    parent_id: Optional[str] = None,
    action_plan: Optional[str] = None,
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
    tags: Optional[Iterable[str]] = None,
) -> Dict:
    log_id = str(uuid.uuid4())
    upsert_log(
        log_id=log_id,
        content=content,
        meta_type=meta_type,
        parent_id=parent_id,
        action_plan=action_plan,
        embedding=embedding,
        emotion=emotion,
        dimension=dimension,
        keywords=keywords,
        linked_constitutions=linked_constitutions,
        duration=duration,
        debt_repaid=debt_repaid,
        kanban_status=kanban_status,
        quality_score=quality_score,
        reward_points=reward_points,
        tags=tags,
    )
    return get_log_by_id(log_id)


def create_gap(
    content: str,
    core_id: str,
    action_plan: str = "",
    embedding: Optional[Iterable[float]] = None,
    quality_score: int = 0,
    reward_points: int = 0,
    tags: Optional[Iterable[str]] = None,
) -> Dict:
    return create_log(
        content=content,
        meta_type="Gap",
        parent_id=core_id,
        action_plan=action_plan,
        embedding=embedding,
        linked_constitutions=[core_id] if core_id else [],
        quality_score=quality_score,
        reward_points=reward_points,
        tags=tags,
    )


def get_log_by_id(log_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM logs WHERE id = %s", (log_id,))
            row = cur.fetchone()
    return _row_to_dict(row) if row else None


def get_all_logs() -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM logs ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def get_logs_paginated(limit: int = 10, offset: int = 0, meta_type: str = None) -> List[Dict]:
    sql = "SELECT * FROM logs"
    params: List = []
    if meta_type:
        sql += " WHERE meta_type = %s"
        params.append(meta_type)
    sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def get_logs_by_type(meta_type: str) -> List[Dict]:
    return get_logs_paginated(limit=5000, offset=0, meta_type=meta_type)


def get_logs_for_analytics() -> List[Dict]:
    return get_all_logs()


def get_logs_last_7_days() -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM logs WHERE created_at >= NOW() - INTERVAL '7 days' ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def get_fragment_count() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM logs WHERE meta_type IN ('Log', 'Fragment')")
            return cur.fetchone()[0]


def get_random_echo() -> Optional[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM logs
                WHERE meta_type IN ('Log', 'Fragment')
                ORDER BY RANDOM()
                LIMIT 1
                """
            )
            row = cur.fetchone()
    return _row_to_dict(row) if row else None


def get_cores() -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM logs WHERE meta_type IN ('Core', 'Constitution') ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def get_constitutions() -> List[Dict]:
    return get_cores()


def get_cores_with_stats() -> List[Dict]:
    cores = get_cores()
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT parent_id AS core_id,
                       COUNT(*) FILTER (WHERE meta_type IN ('Log','Fragment')) AS log_count,
                       COALESCE(SUM(duration), 0) AS total_duration,
                       MAX(created_at) FILTER (WHERE meta_type IN ('Log','Fragment')) AS last_activity
                FROM logs
                WHERE parent_id IS NOT NULL
                GROUP BY parent_id
                """
            )
            stats = {r["core_id"]: r for r in cur.fetchall()}

    out = []
    for core in cores:
        row = stats.get(core["id"], {})
        count = int(row.get("log_count", 0) or 0)
        duration = int(row.get("total_duration", 0) or 0)
        health = -1.0 if count == 0 else min(1.0, 0.3 + (duration * 0.005) + count * 0.05)
        c = dict(core)
        c["log_count"] = count
        c["duration"] = duration
        c["health_score"] = health
        out.append(c)
    return out


def update_log(log_id: str, **kwargs) -> bool:
    if not kwargs:
        return False
    valid = {
        "content",
        "meta_type",
        "parent_id",
        "action_plan",
        "emotion",
        "dimension",
        "keywords",
        "linked_constitutions",
        "duration",
        "debt_repaid",
        "kanban_status",
        "quality_score",
        "reward_points",
        "tags",
    }
    sets = []
    vals = []
    for k, v in kwargs.items():
        if k not in valid:
            continue
        if k in {"keywords", "linked_constitutions", "tags"}:
            v = json.dumps(v or [], ensure_ascii=False)
            sets.append(f"{k} = %s::jsonb")
        else:
            sets.append(f"{k} = %s")
        vals.append(v)
    if not sets:
        return False
    vals.append(log_id)
    sql = f"UPDATE logs SET {', '.join(sets)} WHERE id = %s"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, vals)
            changed = cur.rowcount > 0
        conn.commit()
    return changed


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


def get_all_embeddings_as_float32() -> Tuple[List[str], np.ndarray]:
    ids, arr = get_all_embeddings()
    if arr is None:
        return [], np.zeros((0, 1536), dtype=np.float32)
    return ids, arr


def save_embedding(log_id: str, emb: np.ndarray) -> None:
    vec = _to_float32_vector(np.asarray(emb, dtype=np.float32).reshape(-1))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE logs SET embedding = %s WHERE id = %s", (vec, log_id))
        conn.commit()


def load_logs_by_ids(ids: List[str]) -> List[Dict]:
    if not ids:
        return []
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM logs WHERE id = ANY(%s)", (ids,))
            rows = cur.fetchall()
    by_id = {r["id"]: _row_to_dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def ensure_fts() -> None:
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


def ensure_fts_index() -> None:
    ensure_fts()


def _contains_hangul(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def keyword_search(query_text: str, limit: int = 100) -> Tuple[List[str], List[float]]:
    query_text = (query_text or "").strip()
    if not query_text:
        return [], []

    is_short_query = len(query_text) < 3
    prefers_korean_fallback = is_short_query or _contains_hangul(query_text)
    if prefers_korean_fallback:
        weight_fts, weight_trgm, weight_ilike, trgm_threshold = 0.2, 0.6, 0.2, 0.05
    else:
        weight_fts, weight_trgm, weight_ilike, trgm_threshold = 0.6, 0.3, 0.1, 0.08

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT
                        id,
                        (
                            %s * COALESCE(ts_rank_cd(
                                to_tsvector('english', COALESCE(content, '')),
                                plainto_tsquery('english', %s)
                            ), 0.0)
                            + %s * COALESCE(similarity(COALESCE(content, ''), %s), 0.0)
                            + %s * CASE
                                WHEN COALESCE(content, '') ILIKE ('%%' || %s || '%%') THEN 1.0
                                ELSE 0.0
                              END
                        ) AS score
                    FROM logs
                    WHERE
                        to_tsvector('english', COALESCE(content, '')) @@ plainto_tsquery('english', %s)
                        OR COALESCE(content, '') ILIKE ('%%' || %s || '%%')
                        OR similarity(COALESCE(content, ''), %s) >= %s
                    ORDER BY score DESC, created_at DESC
                    LIMIT %s
                    """,
                    (
                        weight_fts,
                        query_text,
                        weight_trgm,
                        query_text,
                        weight_ilike,
                        query_text,
                        query_text,
                        query_text,
                        query_text,
                        trgm_threshold,
                        limit,
                    ),
                )
            except Exception as exc:
                print(f"[postgres] keyword_search hybrid fallback: {exc}")
                cur.execute(
                    """
                    SELECT id,
                           CASE
                               WHEN COALESCE(content, '') ILIKE ('%%' || %s || '%%') THEN 1.0
                               ELSE 0.0
                           END AS score
                    FROM logs
                    WHERE COALESCE(content, '') ILIKE ('%%' || %s || '%%')
                    ORDER BY score DESC, created_at DESC
                    LIMIT %s
                    """,
                    (query_text, query_text, limit),
                )
            rows = cur.fetchall()
    ids = [r[0] for r in rows]
    scores = [float(r[1] or 0.0) for r in rows]
    return ids, scores


def update_streak() -> dict:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("INSERT INTO user_stats (id) VALUES (1) ON CONFLICT (id) DO NOTHING")
            cur.execute("SELECT last_login, current_streak, longest_streak FROM user_stats WHERE id = 1")
            row = cur.fetchone() or {}

            today = date.today()
            yesterday = today - timedelta(days=1)
            last_login = row.get("last_login")
            current = int(row.get("current_streak") or 0)
            longest = int(row.get("longest_streak") or 0)
            status = "kept"

            if last_login == today:
                status = "same_day"
            elif last_login == yesterday:
                current += 1
                longest = max(longest, current)
                status = "increased"
            else:
                current = 1
                longest = max(longest, current)
                status = "broken"

            cur.execute(
                """
                UPDATE user_stats
                SET last_login=%s, current_streak=%s, longest_streak=%s
                WHERE id=1
                """,
                (today, current, longest),
            )
        conn.commit()

    return {"streak": current, "longest": longest, "status": status}


def check_streak_and_apply_penalty() -> dict:
    streak_info = update_streak()
    penalty_applied = streak_info.get("status") == "broken" and streak_info.get("streak", 0) == 1
    if penalty_applied:
        increment_debt(1)
    return {"streak_info": streak_info, "penalty_applied": penalty_applied}


def get_current_streak() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_streak FROM user_stats WHERE id=1")
            row = cur.fetchone()
    return int(row[0]) if row else 0


def get_debt_count() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT debt_count FROM user_stats WHERE id=1")
            row = cur.fetchone()
    return int(row[0]) if row else 0


def increment_debt(amount: int = 1) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_stats SET debt_count = GREATEST(0, COALESCE(debt_count,0) + %s) WHERE id=1",
                (amount,),
            )
        conn.commit()


def decrement_debt(amount: int = 1) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_stats SET debt_count = GREATEST(0, COALESCE(debt_count,0) - %s) WHERE id=1 RETURNING debt_count",
                (amount,),
            )
            row = cur.fetchone()
        conn.commit()
    return int(row[0]) if row else 0


def add_recovery_points(points: int) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_stats SET recovery_points = GREATEST(0, COALESCE(recovery_points,0) + %s) WHERE id=1 RETURNING recovery_points",
                (points,),
            )
            row = cur.fetchone()
        conn.commit()
    return int(row[0]) if row else 0


def get_last_log_time() -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(created_at) FROM logs")
            row = cur.fetchone()
    if not row or not row[0]:
        return None
    return row[0].strftime("%Y-%m-%d %H:%M:%S")


def save_chat_message(role: str, content: str, metadata: dict = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (role, content, metadata) VALUES (%s, %s, %s::jsonb)",
                (role, content, json.dumps(metadata or {}, ensure_ascii=False)),
            )
        conn.commit()


def get_chat_history(limit: int = 50) -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT id, "timestamp", role, content, metadata FROM chat_history ORDER BY "timestamp" DESC LIMIT %s',
                (limit,),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def clear_chat_history() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_history")
        conn.commit()


def add_connection(source_id: str, target_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO connections (source_id, target_id)
                VALUES (%s, %s)
                ON CONFLICT (source_id, target_id) DO NOTHING
                """,
                (source_id, target_id),
            )
            created = cur.rowcount > 0
        conn.commit()
    return created


def get_connections() -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM connections ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_connection_counts() -> Dict[str, int]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source_id, COUNT(*) FROM connections GROUP BY source_id")
            rows = cur.fetchall()
    return {r[0]: int(r[1]) for r in rows}


def get_kanban_cards() -> Dict[str, List[Dict]]:
    out = {"draft": [], "orbit": [], "landed": []}
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM logs WHERE kanban_status IS NOT NULL ORDER BY created_at DESC")
            rows = cur.fetchall()
    for row in rows:
        d = _row_to_dict(row)
        status = d.get("kanban_status") or "draft"
        if status in out:
            out[status].append(d)
    return out


def update_kanban_status(log_id: str, new_status: str) -> bool:
    if new_status not in ("draft", "orbit", "landed"):
        return False
    return update_log(log_id, kanban_status=new_status)


def create_kanban_card(content: str, constitution_id: str, status: str = "draft") -> Dict:
    if not constitution_id:
        raise ValueError("constitution_id is required")
    return create_log(
        content=content,
        meta_type="Log",
        parent_id=constitution_id,
        linked_constitutions=[constitution_id],
        kanban_status=status,
    )


def inject_genesis_data(embedding_func):
    if get_fragment_count() > 0:
        return
    print("Injecting Genesis Data...")
    for row in GENESIS_DATA:
        content = row["content"]
        embedding = None
        try:
            embedding = embedding_func(content)
        except Exception:
            embedding = None
        create_log(content=content, meta_type=row["meta_type"], embedding=embedding)
    increment_debt(1)
