"""
database.py
Antigravity: SQLite Database Layer
"""

import sqlite3
import uuid
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
import os

# ============================================================
# Database Configuration
# ============================================================
DB_PATH = "data/narrative.db"

# Genesis Data (Seed)
GENESIS_DATA = [
    {
        "meta_type": "Constitution",
        "content": "순간의 열정이나 강렬한 욕망을 통한 동기부여를 지속가능하게 만들고 싶다.",
        "status": "active"
    },
    {
        "meta_type": "Fragment",
        "content": "사실, 플래너에 독일어 공부/일기 쓰기/태블릿에 4K Berlin View 깔기 등의 과업이 있었지만, 오늘은 하루종일 Antigravity와 코딩하느라, 하지 못했다. 아니, 아마도 안한 것일 것이다.",
        "status": "active"
    },
    {
        "meta_type": "Fragment",
        "content": "며칠 전에, 수면의 중요성에 관한 글을 읽으며 정리했다. 그리고, 반드시 12시에 자겠다고 생각했다. ... 지금, 코딩하다보니 12시 39분이다.",
        "status": "active"
    },
    {
        "meta_type": "Thirst",
        "content": "나는 왜 글쓰기를 하지? 나의 글쓰기는 자위행위와 같은가? 넘치는 실재의 흐름을 분출해버리는 일종의 증상적인 행위인가?",
        "status": "active"
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
    
    # Logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            meta_type TEXT DEFAULT 'Fragment',
            status TEXT DEFAULT 'active',
            parent_id TEXT REFERENCES logs(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT,
            dimension TEXT,
            keywords TEXT,
            vector_embedding BLOB,
            is_virtual INTEGER DEFAULT 0,
            virtual_type TEXT,
            amendment_reason TEXT,
            pending_proof_count INTEGER DEFAULT 0
        )
    """)
    
    # Tags table (Many-to-Many)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id TEXT PRIMARY KEY,
            log_id TEXT REFERENCES logs(id),
            name TEXT NOT NULL
        )
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
    
    # Debts table (Strategic Sacrifice - Constitution violations)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS debts (
            id TEXT PRIMARY KEY,
            constitution_id TEXT REFERENCES logs(id),
            fragment_id TEXT REFERENCES logs(id),
            debt_type TEXT NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            repaid_at TIMESTAMP,
            is_repaid INTEGER DEFAULT 0
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
            
            # Get embedding
            try:
                embedding = embedding_func(content)
                embedding_blob = np.array(embedding).tobytes()
            except:
                embedding_blob = None
            
            cursor.execute("""
                INSERT INTO logs (id, content, meta_type, status, created_at, vector_embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                log_id,
                content,
                data["meta_type"],
                data["status"],
                datetime.now().isoformat(),
                embedding_blob
            ))
        
        conn.commit()
        print(f"✅ Injected {len(GENESIS_DATA)} genesis records")
    
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
    tags: list = None,
    is_virtual: bool = False,
    virtual_type: str = None
) -> dict:
    """Create a new log entry"""
    conn = get_connection()
    cursor = conn.cursor()
    
    log_id = str(uuid.uuid4())
    embedding_blob = np.array(embedding).tobytes() if embedding else None
    keywords_json = json.dumps(keywords or [], ensure_ascii=False)
    
    cursor.execute("""
        INSERT INTO logs (id, content, meta_type, status, created_at, 
                          emotion, dimension, keywords, vector_embedding,
                          is_virtual, virtual_type)
        VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?)
    """, (
        log_id, content, meta_type, datetime.now().isoformat(),
        emotion, dimension, keywords_json, embedding_blob,
        1 if is_virtual else 0, virtual_type
    ))
    
    # Insert tags
    if tags:
        for tag in tags:
            tag_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO tags (id, log_id, name) VALUES (?, ?, ?)",
                (tag_id, log_id, tag.strip())
            )
    
    conn.commit()
    conn.close()
    
    return get_log_by_id(log_id)


def get_log_by_id(log_id: str) -> Optional[dict]:
    """Get a single log by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM logs WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return _row_to_dict(row)
    return None


def get_all_logs(include_virtual: bool = True, include_archived: bool = True) -> List[dict]:
    """Get all logs"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM logs WHERE 1=1"
    if not include_virtual:
        query += " AND is_virtual = 0"
    if not include_archived:
        query += " AND status != 'archived'"
    query += " ORDER BY created_at DESC"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_dict(row) for row in rows]


def update_log(log_id: str, **kwargs) -> bool:
    """Update log fields"""
    conn = get_connection()
    cursor = conn.cursor()
    
    updates = []
    values = []
    
    for key, value in kwargs.items():
        if key == "embedding":
            updates.append("vector_embedding = ?")
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
    cursor.execute(
        f"UPDATE logs SET {', '.join(updates)} WHERE id = ?",
        values
    )
    
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def delete_log(log_id: str) -> bool:
    """Delete a log"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Delete tags first
    cursor.execute("DELETE FROM tags WHERE log_id = ?", (log_id,))
    cursor.execute("DELETE FROM logs WHERE id = ?", (log_id,))
    
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def get_logs_by_type(meta_type: str) -> List[dict]:
    """Get logs by meta_type"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM logs WHERE meta_type = ? AND status = 'active' ORDER BY created_at DESC",
        (meta_type,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_dict(row) for row in rows]


def get_virtual_nodes() -> List[dict]:
    """Get all ghost/virtual nodes"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM logs WHERE is_virtual = 1")
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_dict(row) for row in rows]


def clear_virtual_nodes() -> int:
    """Delete all virtual nodes"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM logs WHERE is_virtual = 1")
    count = cursor.rowcount
    
    conn.commit()
    conn.close()
    return count


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
    
    cursor.execute(
        "SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
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
# Strategic Sacrifice (Debt System)
# ============================================================
def record_strategic_sacrifice(constitution_id: str, fragment_id: str, debt_type: str, reason: str = None) -> dict:
    """
    Record a Strategic Sacrifice - user consciously violates Constitution.
    Creates a 'Debt' that must be repaid later.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    debt_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO debts (id, constitution_id, fragment_id, debt_type, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (debt_id, constitution_id, fragment_id, debt_type, reason, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    
    return {
        "id": debt_id,
        "constitution_id": constitution_id,
        "fragment_id": fragment_id,
        "debt_type": debt_type,
        "reason": reason,
        "is_repaid": False
    }


def get_unpaid_debts() -> List[dict]:
    """Get all unpaid debts"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM debts WHERE is_repaid = 0 ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def repay_debt(debt_id: str) -> bool:
    """Mark a debt as repaid"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE debts SET is_repaid = 1, repaid_at = ? WHERE id = ?",
        (datetime.now().isoformat(), debt_id)
    )
    
    conn.commit()
    success = cursor.rowcount > 0
    conn.close()
    return success


def get_debt_count() -> int:
    """Get count of unpaid debts"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM debts WHERE is_repaid = 0")
    count = cursor.fetchone()[0]
    conn.close()
    return count


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
    if d.get("vector_embedding"):
        try:
            d["embedding"] = np.frombuffer(d["vector_embedding"], dtype=np.float64).tolist()
        except:
            d["embedding"] = []
        del d["vector_embedding"]
    else:
        d["embedding"] = []
    
    # Convert is_virtual to bool
    d["is_virtual"] = bool(d.get("is_virtual", 0))
    
    # Ensure timestamp field exists
    if "created_at" in d:
        d["timestamp"] = d["created_at"]
    
    return d


# Initialize on import
init_database()
