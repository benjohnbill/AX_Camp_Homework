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
    """Create tables if not exist"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Logs table (3-Body Hierarchy)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            meta_type TEXT DEFAULT 'Fragment',
            parent_id TEXT REFERENCES logs(id),
            action_plan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding BLOB,
            emotion TEXT,
            dimension TEXT,
            keywords TEXT
        )
    """)
    
    # User stats table (Streak & Debt)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_stats (
            id INTEGER PRIMARY KEY DEFAULT 1,
            last_login DATE,
            current_streak INTEGER DEFAULT 0,
            longest_streak INTEGER DEFAULT 0,
            debt_count INTEGER DEFAULT 0
        )
    """)
    
    # Initialize user_stats if empty
    cursor.execute("SELECT COUNT(*) FROM user_stats")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO user_stats (id, last_login, current_streak, longest_streak, debt_count)
            VALUES (1, NULL, 0, 0, 0)
        """)
    
    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT
        )
    """)
    
    # Manual Connections table (Constellations)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            type TEXT DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_id) REFERENCES logs(id),
            FOREIGN KEY(target_id) REFERENCES logs(id),
            UNIQUE(source_id, target_id)
        )
    """)
    
    conn.commit()

    # user_stats migration columns
    user_stats_migrations = [
        ("recovery_points", "INTEGER DEFAULT 0"),
    ]
    for col_name, col_type in user_stats_migrations:
        try:
            cursor.execute(f"ALTER TABLE user_stats ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    
    # ============================================================
    # [NEW v5] Schema Migration — Add Chronos / Kanban columns
    # ============================================================
    migration_columns = [
        ("linked_constitutions", "TEXT"),      # JSON array of Constitution IDs
        ("duration", "INTEGER"),               # Minutes spent (Chronos)
        ("debt_repaid", "INTEGER DEFAULT 0"),  # 1 if this log repaid a debt
        ("kanban_status", "TEXT"),              # 'draft' | 'orbit' | 'landed' | NULL
        ("quality_score", "INTEGER DEFAULT 0"),
        ("reward_points", "INTEGER DEFAULT 0"),
    ]
    
    for col_name, col_type in migration_columns:
        try:
            cursor.execute(f"ALTER TABLE logs ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists — safe to ignore

    # ============================================================
    # [NEW v5.1] FTS5 Virtual Table for Hybrid Search
    # ============================================================
    try:
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(content, keywords, content='logs', content_rowid='id')")
        # Initial population if empty
        cursor.execute("INSERT OR IGNORE INTO logs_fts(rowid, content, keywords) SELECT rowid, content, keywords FROM logs")
        conn.commit()
    except Exception as e:
        print(f"Warning: FTS5 not available or failed: {e}")
    
    conn.close()


def inject_genesis_data(embedding_func):
    """Inject seed data if database is empty"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM logs")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("🌱 Injecting Genesis Data...")
        for data in GENESIS_DATA:
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
        
        # Set initial debt for testing Red Protocol
        cursor.execute("UPDATE user_stats SET debt_count = 1 WHERE id = 1")
        
        conn.commit()
        print(f"✅ Injected {len(GENESIS_DATA)} genesis records + debt_count=1 for testing")
    
    conn.close()


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
    reward_points: int = 0
) -> dict:
    """Create a new log entry"""
    conn = get_connection()
    cursor = conn.cursor()
    
    log_id = str(uuid.uuid4())
    embedding_blob = np.array(embedding).tobytes() if embedding else None
    keywords_json = json.dumps(keywords or [], ensure_ascii=False)
    linked_const_json = json.dumps(linked_constitutions or [], ensure_ascii=False)
    
    cursor.execute("""
        INSERT INTO logs (id, content, meta_type, parent_id, action_plan, 
                         created_at, embedding, emotion, dimension, keywords,
                         linked_constitutions, duration, debt_repaid, kanban_status,
                         quality_score, reward_points)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        log_id, content, meta_type, parent_id, action_plan,
        datetime.now().isoformat(), embedding_blob, emotion, dimension, keywords_json,
        linked_const_json, duration, debt_repaid, kanban_status,
        quality_score, reward_points
    ))
    
    conn.commit()
    conn.close()
    
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
    reward_points: int = 0
) -> dict:
    """Create an Apology linked to a Constitution"""
    return create_log(
        content=content,
        meta_type="Apology",
        embedding=embedding,
        parent_id=constitution_id,
        action_plan=action_plan,
        quality_score=quality_score,
        reward_points=reward_points
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


def get_fragments_paginated(limit: int = 10, offset: int = 0) -> list:
    """Get fragments with pagination"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM logs 
        WHERE meta_type = 'Fragment' 
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
    Get each Constitution with aggregated stats for Soul Finviz Treemap.
    Returns: list of dicts with id, content, total_duration, fragment_count, last_activity, health_score
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    constitutions = get_constitutions()
    
    stats = []
    for const in constitutions:
        const_id = const["id"]
        
        # Count fragments linked to this constitution via linked_constitutions JSON
        cursor.execute("SELECT duration, linked_constitutions, created_at FROM logs WHERE meta_type = 'Fragment'")
        rows = cursor.fetchall()
        
        total_duration = 0
        fragment_count = 0
        last_activity = const.get("created_at", "")
        
        for row in rows:
            linked = row["linked_constitutions"]
            if linked:
                try:
                    linked_ids = json.loads(linked)
                except:
                    linked_ids = []
                if const_id in linked_ids:
                    fragment_count += 1
                    total_duration += (row["duration"] or 0)
                    if row["created_at"] > last_activity:
                        last_activity = row["created_at"]
        
        # Also count Apologies linked via parent_id
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM logs 
            WHERE meta_type = 'Apology' AND parent_id = ?
        """, (const_id,))
        apology_count = cursor.fetchone()["cnt"]
        
        # Check debt status
        has_debt = get_debt_count() > 0
        
        # Calculate health score (-1.0 to 1.0)
        days_since = _days_since(last_activity)
        
        if has_debt or days_since > 7:
            health = -1.0 + min(0.5, fragment_count * 0.1)
        elif days_since > 3:
            health = 0.0
        else:
            health = min(1.0, 0.3 + fragment_count * 0.1 + total_duration * 0.005)
        
        health = max(-1.0, min(1.0, health))
        
        stats.append({
            "id": const_id,
            "content": const["content"],
            "total_duration": total_duration,
            "fragment_count": fragment_count,
            "apology_count": apology_count,
            "last_activity": last_activity,
            "days_since_activity": days_since,
            "health_score": round(health, 2),
            # Size for treemap = duration + fragment_count weighted
            "size": max(1, total_duration + fragment_count * 10)
        })
    
    conn.close()
    return stats


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
    """Create a new Kanban card — MUST orbit a Constitution (no orphans)."""
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

