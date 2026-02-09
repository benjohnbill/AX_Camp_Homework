"""
db_manager.py
Antigravity v4: SQLite Database Layer
3-Body Hierarchy: Constitution (Sun) / Apology (Planet) / Fragment (Dust)
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
    
    today = date.today()
    last_login = None
    if row['last_login']:
        try:
            last_login = datetime.strptime(row['last_login'], "%Y-%m-%d").date()
        except:
            last_login = None
    
    current_streak = row['current_streak'] or 0
    longest_streak = row['longest_streak'] or 0
    streak_broken = False
    
    if last_login is None:
        # First login ever
        current_streak = 1
    elif last_login == today:
        # Already logged in today
        pass
    elif last_login == today - timedelta(days=1):
        # Consecutive day
        current_streak += 1
    else:
        # Streak broken
        streak_broken = True
        current_streak = 1
    
    # Update longest streak
    if current_streak > longest_streak:
        longest_streak = current_streak
    
    # Save updates
    cursor.execute("""
        UPDATE user_stats 
        SET last_login = ?, current_streak = ?, longest_streak = ?
        WHERE id = 1
    """, (today.isoformat(), current_streak, longest_streak))
    
    conn.commit()
    conn.close()
    
    return {
        "current_streak": current_streak,
        "longest_streak": longest_streak,
        "streak_broken": streak_broken
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


# ============================================================
# Debt System (Red Protocol)
# ============================================================
def get_debt_count() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT debt_count FROM user_stats WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    return result['debt_count'] if result else 0


def increment_debt(amount: int = 1) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE user_stats SET debt_count = debt_count + ? WHERE id = 1", (amount,))
    conn.commit()
    cursor.execute("SELECT debt_count FROM user_stats WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    return result['debt_count']


def reset_debt() -> None:
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
    action_plan: str = None
) -> dict:
    """Create a new log entry"""
    conn = get_connection()
    cursor = conn.cursor()
    
    log_id = str(uuid.uuid4())
    embedding_blob = np.array(embedding).tobytes() if embedding else None
    keywords_json = json.dumps(keywords or [], ensure_ascii=False)
    
    cursor.execute("""
        INSERT INTO logs (id, content, meta_type, parent_id, action_plan, 
                         created_at, embedding, emotion, dimension, keywords)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        log_id, content, meta_type, parent_id, action_plan,
        datetime.now().isoformat(), embedding_blob, emotion, dimension, keywords_json
    ))
    
    conn.commit()
    conn.close()
    
    return get_log_by_id(log_id)


def create_apology(content: str, constitution_id: str, action_plan: str, embedding: list = None) -> dict:
    """Create an Apology linked to a Constitution"""
    return create_log(
        content=content,
        meta_type="Apology",
        embedding=embedding,
        parent_id=constitution_id,
        action_plan=action_plan
    )


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
    
    # Convert embedding blob to list
    if d.get("embedding"):
        try:
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float64).tolist()
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
