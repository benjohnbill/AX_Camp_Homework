from typing import Dict, List

import numpy as np


def _combine_scores(vec_score: float, text_score: float, alpha: float) -> float:
    return (alpha * vec_score) + ((1.0 - alpha) * text_score)


def rank_hybrid_rows(rows: List[Dict], alpha: float = 0.7, top_k: int = 10) -> List[Dict]:
    ranked = []
    for row in rows:
        vec_score = float(row.get("vec_score", 0.0) or 0.0)
        text_score = float(row.get("text_score", 0.0) or 0.0)
        combined = _combine_scores(vec_score, text_score, alpha)
        r = dict(row)
        r["combined_score"] = combined
        ranked.append(r)
    ranked.sort(key=lambda r: r["combined_score"], reverse=True)
    return ranked[:top_k]


def hybrid_search_pg(conn, q_emb, query_text: str, top_k: int = 10, alpha: float = 0.7, cand_k: int = 200):
    """
    Two-step hybrid search:
      1) top cand_k by vector distance (embedding <-> qvec)
      2) rank candidates with ts_rank_cd + weighted combination

    Returns rows with id, content, vec_score, text_score, combined_score.
    """
    try:
        from pgvector.psycopg2 import register_vector

        register_vector(conn)
    except Exception as exc:
        raise RuntimeError(
            "pgvector adapter is not available. Install pgvector and verify extension setup."
        ) from exc

    qvec = np.asarray(q_emb, dtype=np.float32).reshape(-1)
    if qvec.size != 1536:
        raise ValueError(f"q_emb dimension must be 1536, got {qvec.size}")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, (embedding <-> %s::vector) AS distance
            FROM logs
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> %s::vector ASC
            LIMIT %s
            """,
            (qvec.tolist(), qvec.tolist(), cand_k),
        )
        candidates = cur.fetchall()

    if not candidates:
        return []

    id_list = [row[0] for row in candidates]
    dist_map = {row[0]: float(row[2]) for row in candidates}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                content,
                ts_rank_cd(
                    to_tsvector('english', COALESCE(content, '')),
                    plainto_tsquery('english', %s)
                ) AS text_score
            FROM logs
            WHERE id = ANY(%s)
            """,
            (query_text, id_list),
        )
        text_rows = cur.fetchall()

    rows = []
    for log_id, content, text_score in text_rows:
        distance = dist_map.get(log_id, 999.0)
        vec_score = 1.0 / (1.0 + distance)
        text_score_val = float(text_score or 0.0)
        combined = _combine_scores(vec_score, text_score_val, alpha)
        rows.append(
            {
                "id": log_id,
                "content": content,
                "vec_score": vec_score,
                "text_score": text_score_val,
                "combined_score": combined,
            }
        )

    rows.sort(key=lambda r: r["combined_score"], reverse=True)
    return rows[:top_k]
