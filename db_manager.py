"""
db_manager.py
Antigravity v5: SQLite Database Layer
3-Body Hierarchy: Constitution (Sun) / Apology (Planet) / Fragment (Dust)
+ Chronos Docking / Soul Finviz / Narrative Kanban
"""

import sqlite3
import uuid
import json
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, List
import os

# ============================================================
# Database Configuration
# ============================================================
DB_PATH = "data/narrative.db"

# Genesis Data (Seed for Testing Red Protocol)
GENESIS_DATA = [
    {
        "meta_type": "Constitution",
        "content": "순간의 열정이나 강렬한 욕망을 통한 동기부여를 지속가능하게 만들고 싶다."
    },
    {
        "meta_type": "Fragment",
        "content": "플래너에 적힌 과업을 무시하고 하루종일 코딩만 했다."
    },
    {
        "meta_type": "Fragment",
        "content": "12시에 자겠다고 했지만 지금 12시 39분이다."
    }
]


# ============================================================
# Database Initialization
# ============================================================
def get_connection() -> sqlite3.Connection:
    """Thread-safe connection"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """
    데이터베이스 테이블 생성 및 스키마 마이그레이션을 관리합니다.
    [변경 이유] 하나의 큰 함수를 테이블 생성, 초기 데이터 삽입, 마이그레이션으로 분리하여 관리 용이성 향상.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    _create_base_tables(cursor)
    _init_user_stats(cursor)
    _run_schema_migrations(cursor)
    _init_fts_table(cursor)
    
    conn.commit()
    conn.close()


def _create_base_tables(cursor):
    """기본적인 3-Body Hierarchy 테이블들을 생성합니다."""
    # Logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            meta_type TEXT DEFAULT 'Fragment',
            parent_id TEXT REFERENCES logs(id),
            action_plan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding BLOB, emotion TEXT, dimension TEXT, keywords TEXT
        )
    """)
    
    # User stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            id INTEGER PRIMARY KEY DEFAULT 1,
            last_login DATE,
            current_streak INTEGER DEFAULT 0,
            longest_streak INTEGER DEFAULT 0,
            debt_count INTEGER DEFAULT 0
        )
    """)
    
    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT
        )
    """)
    
    # Manual Connections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL, target_id TEXT NOT NULL,
            type TEXT DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_id) REFERENCES logs(id),
            FOREIGN KEY(target_id) REFERENCES logs(id),
            UNIQUE(source_id, target_id)
        )
    """)


def _init_user_stats(cursor):
    """사용자 통계 테이블의 초기 행을 생성합니다."""
    cursor.execute("SELECT COUNT(*) FROM user_stats")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO user_stats (id, last_login, current_streak, longest_streak, debt_count)
            VALUES (1, NULL, 0, 0, 0)
        """)


def _run_schema_migrations(cursor):
    """기존 스키마에 새로운 컬럼들을 추가하는 마이그레이션 로직입니다."""
    # user_stats 컬럼 추가
    user_stats_cols = [("recovery_points", "INTEGER DEFAULT 0"), ("last_log_at", "TIMESTAMP")]
    for col, ctype in user_stats_cols:
        try:
            cursor.execute(f"ALTER TABLE user_stats ADD COLUMN {col} {ctype}")
        except sqlite3.OperationalError: pass

    # logs 컬럼 추가 (Chronos / Kanban)
    logs_cols = [
        ("linked_constitutions", "TEXT"), ("duration", "INTEGER"),
        ("debt_repaid", "INTEGER DEFAULT 0"), ("kanban_status", "TEXT"),
        ("quality_score", "INTEGER DEFAULT 0"), ("reward_points", "INTEGER DEFAULT 0")
    ]
    for col, ctype in logs_cols:
        try:
            cursor.execute(f"ALTER TABLE logs ADD COLUMN {col} {ctype}")
        except sqlite3.OperationalError: pass
    
    # [NEW v5.4] Tagging System
    try:
        cursor.execute("ALTER TABLE logs ADD COLUMN tags TEXT")
    except sqlite3.OperationalError: pass

def get_all_used_tags() -> List[str]:
    """Get all distinct tags used in logs"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT tags FROM logs WHERE tags IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    
    tags = set()
    for row in rows:
        try:
            t = json.loads(row['tags'])
            if isinstance(t, list):
                tags.update(t)
        except: pass
    return sorted(list(tags))


def _init_fts_table(cursor):
    """Full-Text Search를 위한 가상 테이블을 초기화합니다."""
    try:
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(content, keywords, content='logs', content_rowid='id')")
        cursor.execute("INSERT OR IGNORE INTO logs_fts(rowid, content, keywords) SELECT rowid, content, keywords FROM logs")
    except Exception as e:
        print(f"Warning: FTS5 failed: {e}")


def inject_genesis_data(embedding_func):
    """DB가 비어있을 때 초기 데이터를 주입합니다."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM logs")
    if cursor.fetchone()[0] == 0:
        print("🌱 Injecting Genesis Data...")
        for data in GENESIS_DATA:
            _insert_genesis_record(cursor, data, embedding_func)
        
        cursor.execute("UPDATE user_stats SET debt_count = 1 WHERE id = 1")
        conn.commit()
    
    conn.close()


def _insert_genesis_record(cursor, data, embedding_func):
    """개별 제네시스 레코드를 삽입하는 내부 도우미 함수입니다."""
    log_id = str(uuid.uuid4())
    content = data["content"]
    
    try:
        embedding = embedding_func(content)
        embedding_blob = np.array(embedding).tobytes()
    except:
        embedding_blob = None
    
    cursor.execute("""
        INSERT INTO logs (id, content, meta_type, created_at, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (log_id, content, data["meta_type"], datetime.now().isoformat(), embedding_blob))


# ============================================================
# Streak System
# ============================================================
def update_streak() -> dict:
    """Check and update streak on login. Returns streak info."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT last_login, current_streak, longest_streak FROM user_stats WHERE id = 1")
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return {"streak": 0, "longest": 0, "status": "new"}
        
    last_login = row["last_login"]
    current_streak = row["current_streak"] or 0
    longest_streak = row["longest_streak"] or 0
    
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    
    status = "kept"
    
    if last_login == today:
        status = "same_day"
    elif last_login == yesterday:
        current_streak += 1
        if current_streak > longest_streak:
            longest_streak = current_streak
        status = "increased"
    else:
        current_streak = 1  # Reset if missed a day
        status = "broken"
        
    cursor.execute("""
        UPDATE user_stats 
        SET last_login = ?, current_streak = ?, longest_streak = ?
        WHERE id = 1
    """, (today, current_streak, longest_streak))
    
    conn.commit()
    conn.close()
    
    return {
        "streak": current_streak,
        "longest": longest_streak,
        "status": status
    }


def get_current_streak() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT current_streak FROM user_stats WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    return result['current_streak'] if result else 0


def update_last_log_time():
    """Update the last_log_at timestamp to NOW"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE user_stats SET last_log_at = CURRENT_TIMESTAMP WHERE id=1")
    conn.commit()
    conn.close()


def get_last_log_time() -> str:
    """Get the last time the user created a log"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT last_log_at FROM user_stats WHERE id=1")
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def get_longest_streak() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT longest_streak FROM user_stats WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    return result['longest_streak'] if result else 0


def get_recovery_points() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT recovery_points FROM user_stats WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    return result['recovery_points'] if result and result['recovery_points'] is not None else 0


def add_recovery_points(points: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE user_stats SET recovery_points = COALESCE(recovery_points, 0) + ? WHERE id = 1", (points,))
    conn.commit()
    cursor.execute("SELECT recovery_points FROM user_stats WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    return result['recovery_points'] if result else 0


def get_debt_count() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT debt_count FROM user_stats WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0


def increment_debt(amount: int = 1) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE user_stats SET debt_count = debt_count + ? WHERE id = 1", (amount,))
    conn.commit()
    conn.close()


def decrement_debt(amount: int = 1) -> int:
    """Decrement debt by amount, ensuring it doesn't go below 0. Returns new debt count."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get current debt
    cursor.execute("SELECT debt_count FROM user_stats WHERE id = 1")
    current = cursor.fetchone()[0]
    
    new_debt = max(0, current - amount)
    
    cursor.execute("UPDATE user_stats SET debt_count = ? WHERE id = 1", (new_debt,))
    conn.commit()
    conn.close()
    return new_debt


def reset_debt() -> None:
    """Reset debt to 0 (Catharsis) - Legacy/Special Use"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE user_stats SET debt_count = 0 WHERE id = 1")
    conn.commit()
    conn.close()


# ============================================================
# CRUD Operations
# ============================================================
def create_log(
    content: str,
    meta_type: str = "Fragment",
    embedding: list = None,
    emotion: str = None,
    dimension: str = None,
    keywords: list = None,
    parent_id: str = None,
    action_plan: str = None,
    # [NEW v5] Chronos / Kanban fields
    linked_constitutions: list = None,
    duration: int = None,
    debt_repaid: int = 0,
    kanban_status: str = None,
    quality_score: int = 0,
    reward_points: int = 0,
    # [NEW v5.4] Tagging System
    tags: list = None
) -> dict:
    """Create a new log entry"""
    log_id = str(uuid.uuid4())
    
    # [FIX] Use BLOB for embedding storage (consistent with schema)
    embedding_blob = np.array(embedding).tobytes() if embedding else None
    
    conn = get_connection()
    cursor = conn.cursor()
    
    keywords_json = json.dumps(keywords) if keywords else None
    linked_const_json = json.dumps(linked_constitutions) if linked_constitutions else None
    tags_json = json.dumps(tags) if tags else None

    cursor.execute("""
        INSERT INTO logs (id, content, meta_type, parent_id, action_plan, 
                         created_at, embedding, emotion, dimension, keywords,
                         linked_constitutions, duration, debt_repaid, kanban_status,
                         quality_score, reward_points, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        log_id, content, meta_type, parent_id, action_plan,
        datetime.now().isoformat(), embedding_blob, emotion, dimension, keywords_json,
        linked_const_json, duration, debt_repaid, kanban_status,
        quality_score, reward_points, tags_json
    ))
    
    conn.commit()
    conn.close()
    
    # [NEW v5.1] Update last_log_at on any creation
    update_last_log_time()
    
    return get_log_by_id(log_id)


def get_logs_for_analytics() -> List[dict]:
    """Get all logs with created_at and meta_type for analytics"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, meta_type, created_at, duration, linked_constitutions FROM logs")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def create_apology(
    content: str,
    constitution_id: str,
    action_plan: str,
    embedding: list = None,
    quality_score: int = 0,
    reward_points: int = 0,
    tags: list = None
) -> dict:
    """Create an Apology linked to a Constitution"""
    return create_log(
        content=content,
        meta_type="Apology",
        embedding=embedding,
        parent_id=constitution_id,
        action_plan=action_plan,
        quality_score=quality_score,
        reward_points=reward_points,
        tags=tags
    )


def get_logs_last_7_days() -> List[dict]:
    """Get logs created in the last 7 days for Weekly Summary"""
    conn = get_connection()
    cursor = conn.cursor()
    
    seven_days_ago = (date.today() - timedelta(days=7)).isoformat()
    
    cursor.execute("""
        SELECT * FROM logs 
        WHERE created_at >= ? 
        ORDER BY created_at ASC
    """, (seven_days_ago,))
    
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(row) for row in rows]


def get_connection_counts() -> dict:
    """
    Get connection counts for each log ID (Node Growth Physics).
    Returns: {log_id: count, ...}
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Count connections where log is source or target
    cursor.execute("""
        SELECT id, 
        (SELECT COUNT(*) FROM logs l2 WHERE l2.parent_id = logs.id) as child_count
        FROM logs
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    # Dictionary mapping ID to count
    return {row['id']: row['child_count'] for row in rows}


def get_log_by_id(log_id: str) -> Optional[dict]:
    """Get a single log by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_all_logs() -> list:
    """Get all logs ordered by creation time (desc)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(row) for row in rows]


def get_logs_paginated(limit: int = 10, offset: int = 0, meta_type: str = None) -> list:
    """Get logs with pagination, optionally filtering by meta_type"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if meta_type:
        cursor.execute("""
            SELECT * FROM logs 
            WHERE meta_type = ?
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """, (meta_type, limit, offset))
    else:
        cursor.execute("""
            SELECT * FROM logs 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(row) for row in rows]


def get_fragment_count() -> int:
    """Get total count of fragments"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM logs WHERE meta_type = 'Fragment'")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_log_counts() -> dict:
    """
    [Optimized] Get counts for all log types in one query.
    Returns: {'Constitution': N, 'Apology': N, 'Fragment': N}
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT meta_type, COUNT(*) FROM logs GROUP BY meta_type")
    rows = cursor.fetchall()
    conn.close()
    
    counts = {"Constitution": 0, "Apology": 0, "Fragment": 0}
    for meta_type, count in rows:
        if meta_type in counts:
            counts[meta_type] = count
    return counts


def get_random_echo() -> dict:
    """Get a random log for Echo System (Prioritize meaningful ones if possible)"""
    conn = get_connection()
    cursor = conn.cursor()
    # Simple random selection for now
    cursor.execute("SELECT * FROM logs ORDER BY RANDOM() LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_logs_by_type(meta_type: str) -> List[dict]:
    """Get logs by meta_type"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs WHERE meta_type = ? ORDER BY created_at DESC", (meta_type,))
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(row) for row in rows]


def get_constitutions() -> List[dict]:
    """Get all Constitution logs"""
    return get_logs_by_type("Constitution")


def get_apologies() -> List[dict]:
    """Get all Apology logs"""
    return get_logs_by_type("Apology")


def get_fragments() -> List[dict]:
    """Get all Fragment logs"""
    return get_logs_by_type("Fragment")


def get_yesterday_apology() -> Optional[dict]:
    """Get the most recent Apology from yesterday"""
    conn = get_connection()
    cursor = conn.cursor()
    
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    
    cursor.execute("""
        SELECT * FROM logs 
        WHERE meta_type = 'Apology' 
        AND date(created_at) >= ? AND date(created_at) < ?
        ORDER BY created_at DESC LIMIT 1
    """, (yesterday, today))
    
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def update_log(log_id: str, **kwargs) -> bool:
    """Update log fields"""
    conn = get_connection()
    cursor = conn.cursor()
    
    updates = []
    values = []
    
    for key, value in kwargs.items():
        if key == "embedding":
            updates.append("embedding = ?")
            values.append(np.array(value).tobytes() if value else None)
        elif key == "keywords":
            updates.append("keywords = ?")
            values.append(json.dumps(value or [], ensure_ascii=False))
        elif key == "linked_constitutions":
            updates.append("linked_constitutions = ?")
            values.append(json.dumps(value or [], ensure_ascii=False))
        else:
            updates.append(f"{key} = ?")
            values.append(value)
    
    if not updates:
        return False
    
    values.append(log_id)
    cursor.execute(f"UPDATE logs SET {', '.join(updates)} WHERE id = ?", values)
    
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def delete_log(log_id: str) -> bool:
    """Delete a log"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs WHERE id = ?", (log_id,))
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


# ============================================================
# Chat History
# ============================================================
def save_chat_message(role: str, content: str, metadata: dict = None) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (role, content, metadata)
        VALUES (?, ?, ?)
    """, (role, content, json.dumps(metadata) if metadata else None))
    conn.commit()
    conn.close()


def get_chat_history(limit: int = 50) -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in reversed(rows)]


def clear_chat_history() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()


# ============================================================
# Manual Connections (Constellations)
# ============================================================
def add_connection(source_id: str, target_id: str) -> bool:
    """Add a manual connection between two logs"""
    if source_id == target_id:
        return False
        
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Check bidirectional
        cursor.execute("""
            SELECT id FROM connections 
            WHERE (source_id = ? AND target_id = ?) 
            OR (source_id = ? AND target_id = ?)
        """, (source_id, target_id, target_id, source_id))
        
        if cursor.fetchone():
            return False  # Already exists
            
        cursor.execute("""
            INSERT INTO connections (source_id, target_id)
            VALUES (?, ?)
        """, (source_id, target_id))
        conn.commit()
        success = True
    except Exception as e:
        success = False
    finally:
        conn.close()
    return success


def get_connections() -> List[dict]:
    """Get all manual connections"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if table exists first (migration support)
    try:
        cursor.execute("SELECT * FROM connections")
        rows = cursor.fetchall()
    except:
        return []
    finally:
        conn.close()
        
    return [dict(row) for row in rows]


def delete_connection(source_id: str, target_id: str) -> bool:
    """Delete a manual connection"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM connections 
            WHERE (source_id = ? AND target_id = ?) 
            OR (source_id = ? AND target_id = ?)
        """, (source_id, target_id, target_id, source_id))
        conn.commit()
        success = True
    except:
        success = False
    finally:
        conn.close()
    return success


# ============================================================
# [NEW v5] Chronos & Soul Finviz Queries
# ============================================================
def get_constitutions_with_stats() -> List[dict]:
    """
    각 헌법별 통계(성취 시간, 기록 수, 건강도 등)를 집계하여 반환합니다.
    [변경 이유] 거대한 루프 내부 로직을 별도 함수로 분리하여 가독성 및 유지보수성 향상.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    constitutions = get_constitutions()
    stats = []
    
    for const in constitutions:
        const_stats = _fetch_single_constitution_stats(cursor, const)
        stats.append(const_stats)
    
    conn.close()
    return stats


def _fetch_single_constitution_stats(cursor, const: dict) -> dict:
    """단일 헌법에 대한 상세 통계를 쿼리하고 계산합니다."""
    const_id = const["id"]
    
    # Fragment 통계 집계
    cursor.execute("SELECT duration, linked_constitutions, created_at FROM logs WHERE meta_type = 'Fragment'")
    rows = cursor.fetchall()
    
    duration, count, last_act = _aggregate_fragment_data(rows, const_id, const.get("created_at", ""))
    
    # Apology 통계 집계
    cursor.execute("SELECT COUNT(*) as cnt FROM logs WHERE meta_type = 'Apology' AND parent_id = ?", (const_id,))
    apology_count = cursor.fetchone()["cnt"]
    
    health = _calculate_constitution_health(duration, count, last_act)
    
    return {
        "id": const_id,
        "content": const["content"],
        "total_duration": duration,
        "fragment_count": count,
        "apology_count": apology_count,
        "last_activity": last_act,
        "days_since_activity": _days_since(last_act),
        "health_score": round(health, 2),
        "size": max(1, duration + count * 10)
    }


def _aggregate_fragment_data(rows: list, const_id: str, initial_last_act: str) -> tuple:
    """로그 목록을 순회하며 특정 헌법에 연결된 데이터들을 합산합니다."""
    total_duration = 0
    fragment_count = 0
    last_activity = initial_last_act
    
    for row in rows:
        linked = row["linked_constitutions"]
        if not linked: continue
        
        try:
            linked_ids = json.loads(linked)
        except:
            linked_ids = []
            
        if const_id in linked_ids:
            fragment_count += 1
            total_duration += (row["duration"] or 0)
            if row["created_at"] > last_activity:
                last_activity = row["created_at"]
                
    return total_duration, fragment_count, last_activity


def _calculate_constitution_health(duration: int, count: int, last_act: str) -> float:
    """
    헌법의 건강도(-1.0 ~ 1.0)를 계산합니다.
    [변경 이유] 복잡한 점수 계산 수식을 격리하여 실험 및 수정이 용이하게 함.
    """
    days_since = _days_since(last_act)
    has_debt = get_debt_count() > 0
    
    if has_debt or days_since > 7:
        health = -1.0 + min(0.5, count * 0.1)
    elif days_since > 3:
        health = 0.0
    else:
        # 가중치 기반 점수 부여 (시간, 깊이, 빈도)
        base = 0.3 + (duration * 0.005)
        count_bonus = count * 0.05
        health = min(1.0, base + count_bonus)
        
    return max(-1.0, min(1.0, health))


def _days_since(iso_timestamp: str) -> int:
    """Calculate days since an ISO timestamp"""
    if not iso_timestamp:
        return 999
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return (datetime.now() - dt.replace(tzinfo=None)).days
    except:
        return 999


# ============================================================
# [NEW v5] Narrative Kanban Operations
# ============================================================
def get_kanban_cards() -> dict:
    """Get logs grouped by kanban_status. Returns {'draft': [...], 'orbit': [...], 'landed': [...]}"""
    conn = get_connection()
    cursor = conn.cursor()
    
    result = {"draft": [], "orbit": [], "landed": []}
    
    cursor.execute("""
        SELECT * FROM logs 
        WHERE kanban_status IS NOT NULL 
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    for row in rows:
        d = _row_to_dict(row)
        status = d.get("kanban_status", "draft")
        if status in result:
            result[status].append(d)
    
    return result


def update_kanban_status(log_id: str, new_status: str) -> bool:
    """Move a kanban card to a new status column"""
    if new_status not in ("draft", "orbit", "landed"):
        return False
    return update_log(log_id, kanban_status=new_status)


def create_kanban_card(content: str, constitution_id: str, status: str = "draft") -> dict:
    """Create a new Kanban card - MUST orbit a Constitution (no orphans)."""
    if not constitution_id:
        raise ValueError("A Kanban card cannot be born without a mother star (constitution_id required).")
    return create_log(
        content=content,
        meta_type="Fragment",
        parent_id=constitution_id,
        linked_constitutions=[constitution_id],
        kanban_status=status
    )


# ============================================================
# Helpers
# ============================================================
def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert SQLite Row to dict with proper type handling"""
    d = dict(row)
    
    # Parse keywords JSON
    if d.get("keywords"):
        try:
            d["keywords"] = json.loads(d["keywords"])
        except:
            d["keywords"] = []
    else:
        d["keywords"] = []
    
    # [NEW v5] Parse linked_constitutions JSON
    if d.get("linked_constitutions"):
        try:
            d["linked_constitutions"] = json.loads(d["linked_constitutions"])
        except:
            d["linked_constitutions"] = []
    else:
        d["linked_constitutions"] = []

    # [NEW v5.4] Parse tags JSON
    if d.get("tags"):
        try:
            d["tags"] = json.loads(d["tags"])
        except:
            d["tags"] = []
    else:
        d["tags"] = []
    
    # Convert embedding blob to list
    if d.get("embedding"):
        try:
            # Handle both blob and list formats (legacy vs new)
            if isinstance(d["embedding"], bytes):
                # Try float32 first
                try:
                    d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).tolist()
                except:
                    # Fallback to float64
                    d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float64).tolist()
            # If it's already a list, do nothing
        except:
            d["embedding"] = []
    else:
        d["embedding"] = []
    
    # Ensure timestamp field exists
    if "created_at" in d:
        d["timestamp"] = d["created_at"]
    
    # Add 'text' alias for compatibility
    d["text"] = d.get("content", "")
    
    return d


def reset_database() -> None:
    """Reset database for testing (deletes all data)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs")
    cursor.execute("DELETE FROM chat_history")
    cursor.execute("UPDATE user_stats SET last_login = NULL, current_streak = 0, longest_streak = 0, debt_count = 0")
    conn.commit()
    conn.close()


# Initialize on import
init_database()


# ============================================================
# [NEW v5.1] Hybrid Search Helpers
# ============================================================
def get_all_embeddings_as_float32():
    """Efficiently load all embeddings for ANN indexing"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM logs WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    
    ids = []
    embeds = []
    
    DIM = 1536
    
    for log_id, blob in rows:
        if not blob:
            continue
            
        try:
            # Try float32
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.size != DIM:
                # Fallback to float64 and convert
                arr = np.frombuffer(blob, dtype=np.float64).astype(np.float32)
            
            if arr.size == DIM:
                ids.append(log_id)
                embeds.append(arr)
        except:
            pass
            
    conn.close()
    
    if not embeds:
        return ids, np.zeros((0, DIM), dtype=np.float32)
        
    return ids, np.vstack(embeds)


def load_logs_by_ids(log_ids: List[str]) -> List[dict]:
    """Bulk retrieve logs by ID list (preserves order)"""
    if not log_ids:
        return []
        
    conn = get_connection()
    cursor = conn.cursor()
    
    placeholders = ",".join("?" for _ in log_ids)
    cursor.execute(f"SELECT * FROM logs WHERE id IN ({placeholders})", log_ids)
    
    rows = cursor.fetchall()
    conn.close()
    
    # Create lookup dict
    log_map = {row['id']: _row_to_dict(row) for row in rows}
    
    # Return in requested order
    return [log_map[lid] for lid in log_ids if lid in log_map]


def ensure_fts_index():
    """Sync FTS index with main table"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Rebuild FTS for simplicity (or incremental update)
        cursor.execute("INSERT OR REPLACE INTO logs_fts(rowid, content, keywords) SELECT rowid, content, keywords FROM logs")
        conn.commit()
    except:
        pass
    conn.close()

