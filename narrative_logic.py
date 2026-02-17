"""
narrative_logic.py
Antigravity v5: The Laws of Physics
3-Body Hierarchy + Red Protocol + Dreaming Integration + Action Plan
+ Chronos Docking + Soul Finviz + Narrative Kanban
"""

import os
import json
import uuid
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from openai import OpenAI
import sqlite3

# [NEW v5.1] Optional Vector Search Libraries
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except ImportError:
    HAS_ANNOY = False


# ============================================================
# API Key & Client Management
# ============================================================
def get_api_key():
    """Get API key from secrets, env, or session_state"""
    # 1. Check session_state first (user-entered key)
    if 'openai_api_key' in st.session_state and st.session_state['openai_api_key']:
        return st.session_state['openai_api_key']
    
    # 2. Try Streamlit secrets
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    
    # 3. Fallback to .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    
    return None


def get_client():
    """Get OpenAI client (lazy initialization)"""
    api_key = get_api_key()
    if api_key:
        return OpenAI(api_key=api_key)
    return None


def set_api_key(key: str):
    """Set API key from user input"""
    st.session_state['openai_api_key'] = key


def is_api_key_configured() -> bool:
    """Check if API key is available"""
    return get_api_key() is not None


# For backward compatibility
client = None  # Will be set dynamically via get_client()

# ============================================================
# [NEW v5.2] Architecture: EvidenceGateway & PolicyEngine
# ============================================================
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NarrativeLoop")

class EvidenceGateway:
    """
    [Body] Responsible for I/O and Data Retrieval. 
    Never makes decisions, only fetches evidence safely.
    Handles: OpenAI API, Database, Vector Index, Time Standardization.
    """
    def __init__(self):
        self.api_failure_count = 0
        self.MAX_FAILURES = 3
        self.is_degraded_mode = False
        self._ensure_utc_for_sqlite()

    def _ensure_utc_for_sqlite(self):
        """Ensure SQLite handles dates as UTC"""
        # (This is mostly handled by db_manager or connection string, 
        # but we enforce UTC in our methods)
        pass

    def check_system_health(self) -> dict:
        """Startup check for DB, API, Vector Backend"""
        health = {
            "db": False,
            "api": False,
            "vector": False,
            "errors": []
        }
        
        # 1. DB Check
        try:
            db.get_fragment_count()
            health["db"] = True
        except Exception as e:
            health["errors"].append(f"DB Error: {e}")

        # 2. API Check
        if is_api_key_configured():
            health["api"] = True
        else:
            health["errors"].append("API Key missing")

        # 3. Vector Backend
        typ, _, _ = load_or_build_index()
        if typ:
            health["vector"] = True

        # 4. Data/BLOB Integrity Check (Numpy Version Safety)
        try:
            sample = db.get_random_echo()
            if sample and sample.get("embedding") and len(sample["embedding"]) > 0:
                # Good: BLOB decoded successfully
                pass
            elif sample and not sample.get("embedding"):
                # Warn if data exists but no embedding (could be legacy or decode fail)
                pass 
            if sample and isinstance(sample.get("embedding"), list) and len(sample["embedding"]) == 0 and sample.get("content"):
                 # If content exists but embedding is empty list, it *might* be a decode fail if we expect embeddings.
                 # But for now, just Ensure no crash.
                 pass
        except Exception as e:
            health["errors"].append(f"Data Integrity Error (BLOB/Numpy): {e}")
        
        return health

    def sanitize_input(self, text: str) -> str:
        """
        [Safety] Validate and sanitize user input.
        - Trim whitespace
        - Check constraints (min/max length)
        """
        if not text:
            return ""
        trimmed = text.strip()
        if len(trimmed) > 10000:
            return trimmed[:10000] # Hard cap
        return trimmed

    def get_embedding_safe(self, text: str) -> list:
        """
        [Resilience] Get embedding with Circuit Breaker & Retry.
        Returns: embedding list or None (if failed/degraded).
        """
        # 1. Input Check
        clean_text = self.sanitize_input(text)
        if not clean_text or len(clean_text) < 2:
            return None # Too short to embed

        # 2. Circuit Breaker Check
        if self.is_degraded_mode:
            logger.warning("Gateway is in Degraded Mode. Skipping API call.")
            return None

        # 3. Try API with Retry
        client = get_client()
        if not client:
            return None

        for attempt in range(3):
            try:
                response = client.embeddings.create(model="text-embedding-3-small", input=clean_text)
                self.api_failure_count = 0 # Reset on success
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Embedding API Error (Attempt {attempt+1}): {e}")
                time.sleep(1 * (attempt+1)) # Backoff
        
        # 4. Failure Handling
        self.api_failure_count += 1
        if self.api_failure_count >= self.MAX_FAILURES:
            self.is_degraded_mode = True
            logger.critical("Circuit Breaker TRIPPED. Entering Degraded Mode.")
        
        return None

    def fetch_recent_logs(self, limit=50) -> list:
        """Fetch recent logs with UTC normalization"""
        logs = db.get_logs_paginated(limit=limit, offset=0)
        # TODO: Normalize timestamps here if necessary
        return logs

    def fetch_constitutions(self) -> list:
        return db.get_constitutions()

    def check_compliance_with_llm(self, core_text: str, log_text: str) -> tuple[bool, str]:
        """[Service] Ask LLM if there is a contradiction."""
        if self.is_degraded_mode:
            return False, "Degraded Mode: Skipping Check"
            
        client = get_client()
        if not client:
             return False, "No Client"
             
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise logical judge. Determine if the NEW statement contradicts the CORE VALUE. Reply with JSON: {\\\"contradiction\\\": true/false, \\\"reason\\\": \\\"short explanation\\\"}."},
                    {"role": "user", "content": f"CORE VALUE: {core_text}\\nNEW: {log_text}"}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            return result.get("contradiction", False), result.get("reason", "")
        except Exception as e:
            logger.error(f"LLM Check Failed: {e}")
            return False, "Error during check"


class PolicyEngine:
    """
    [Mind] Pure Logic and Decisions.
    Takes DTOs (Data Transfer Objects) and returns Decisions.
    state-less (mostly) and side-effect free.
    """
    
    def evaluate_silence(self, last_log_dt: datetime, now_dt: datetime) -> str:
        """
        Check for silence intervention.
        Both inputs must be timezone-aware (or consistently naive UTC).
        """
        if not last_log_dt or not now_dt:
            return None
        
        # Ensure compatibility
        if last_log_dt.tzinfo is None:
             # Assume UTC if naive, as per our new standard
            last_log_dt = last_log_dt.replace(tzinfo=datetime.timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)

        diff = now_dt - last_log_dt
        hours_silent = diff.total_seconds() / 3600

        if hours_silent > 72:
            return f"⚠️ 시스템 엔트로피 증가. {int(hours_silent/24)}일간 로그가 없습니다."
        elif hours_silent > 24:
            return f"🕰️ 관측 공백 감지. ({int(hours_silent)}시간 동안 침묵)"
        
        return None

    def detect_violation(self, log_content: str, core_candidates: list, gateway_ref) -> tuple[bool, str, dict]:
        """
        [Logic] Determine if a log violates any core.
        - Checks ALL candidates, not just the first.
        - Returns: (is_violation, reason, violated_constitution_obj)
        """
        for core in core_candidates:
            # 1. Similarity Logic (Fast Filter)
            # Optimally, we only check high-similarity ones if the list is huge.
            # For now, we trust the caller (Gateway) provided relevant candidates.
            # But let's double check if we have embeddings to be safe
            
            # 2. LLM Check
            is_violation, reason = gateway_ref.check_compliance_with_llm(core.get('content'), log_content)
            if is_violation:
                return True, reason, core
                
        return False, "", None

# Initialize Global Instances
gateway = EvidenceGateway()
policy = PolicyEngine()


# ============================================================
# [NEW v5.1] Advanced Search & Caching
# ============================================================
# Cache embeddings to avoid DB hits on every rerun
@st.cache_resource(ttl=3600)
def load_embeddings_cache():
    ids, X = db.get_all_embeddings_as_float32()
    # Normalize for cosine similarity if using raw numpy
    if X.shape[0] > 0:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_normed = X / norms
    else:
        X_normed = X
    return ids, X, X_normed

@st.cache_resource(ttl=3600)
def load_or_build_index():
    # Attempt to load FAISS
    if HAS_FAISS and os.path.exists("data/index.faiss") and os.path.exists("data/index_ids.json"):
        try:
            index = faiss.read_index("data/index.faiss")
            with open("data/index_ids.json", "r") as f:
                ids = json.load(f)
            return "faiss", index, ids
        except Exception as e:
            print(f"FAISS load failed: {e}")
            
    # Attempt to load Annoy
    if HAS_ANNOY and os.path.exists("data/index.ann") and os.path.exists("data/index_ids.json"):
        try:
            idx = AnnoyIndex(1536, "angular")
            idx.load("data/index.ann")
            with open("data/index_ids.json", "r") as f:
                ids = json.load(f)
            return "annoy", idx, ids
        except Exception as e:
            print(f"Annoy load failed: {e}")
            
    return None, None, None

def ann_query(q_emb: np.ndarray, top_k=100):
    typ, index, ids_map = load_or_build_index()
    
    if typ == "faiss":
        q = q_emb.astype(np.float32).reshape(1, -1)
        # FAISS IndexFlatIP expects normalized vectors for cosine sim
        faiss.normalize_L2(q)
        D, I = index.search(q, top_k)
        out_ids = [ids_map[i] for i in I[0] if i != -1 and i < len(ids_map)]
        out_scores = D[0].tolist()
        return out_ids, out_scores
        
    elif typ == "annoy":
        idxs, dists = index.get_nns_by_vector(q_emb.tolist(), top_k, include_distances=True)
        out_ids = [ids_map[i] for i in idxs if i < len(ids_map)]
        # Convert angular distance to similarity score approx
        out_scores = [1.0 - (d * 0.5) for d in dists] 
        return out_ids, out_scores
        
    return [], []

def simple_keyword_search(query_text, limit=100):
    conn = db.get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # Match using FTS5
        # Note: bm25() is built-in to FTS5
        cur.execute("SELECT rowid, bm25(logs_fts) as score FROM logs_fts WHERE logs_fts MATCH ? ORDER BY score LIMIT ?", (query_text, limit))
        rows = cur.fetchall()
        
        ids = []
        scores = []
        
        # Map rowid back to log id
        for r in rows:
            rowid = r["rowid"]
            cur2 = conn.cursor()
            cur2.execute("SELECT id FROM logs WHERE rowid=?", (rowid,))
            res = cur2.fetchone()
            if res:
                ids.append(res[0])
                # Invert BM25 score for similarity-like ranking (lower usually better in FTS)
                # But bm25 return negative of score sometimes? SQLite FTS5 bm25 is positive, lower is better?
                # Actually SQLite docs say: "The lower the value, the better the match."
                # We want higher = better.
                s = r["score"]
                scores.append(1.0 / (1.0 + abs(s)))
                
        return ids, scores
    except Exception as e:
        # Fallback to naive LIKE search
        print(f"FTS search failed: {e}")
        return [], []
    finally:
        conn.close()

def hybrid_search(query_text, top_k=10, alpha=0.7):
    # 1. Vector Search
    q_emb = get_embedding(query_text)
    if q_emb is None:
        return [], []
    q_emb = np.array(q_emb, dtype=np.float32)
    
    ann_ids, ann_scores = ann_query(q_emb, top_k=top_k*5)
    
    # 2. Keyword Search
    kw_ids, kw_scores = simple_keyword_search(query_text, limit=top_k*5)
    
    # 3. Fallback if ANN failed (Brute Force)
    if not ann_ids:
        ids, X, X_normed = load_embeddings_cache()
        if len(ids) > 0:
            from sklearn.metrics.pairwise import cosine_similarity
            # Reshape for sklearn
            q_reshaped = q_emb.reshape(1, -1)
            sims = cosine_similarity(q_reshaped, X).flatten()
            # Top K
            top_indices = np.argsort(sims)[::-1][:top_k*5]
            ann_ids = [ids[i] for i in top_indices]
            ann_scores = sims[top_indices].tolist()
    
    # 4. RRF / Weighted Combination (Simplified)
    # Normalize scores to 0-1 range locally
    
    # Map all candidates
    all_candidates = list(set(ann_ids + kw_ids))
    final_scores = {}
    
    ann_map = dict(zip(ann_ids, ann_scores))
    kw_map = dict(zip(kw_ids, kw_scores))
    
    for cid in all_candidates:
        s_vec = ann_map.get(cid, 0.0)
        s_kw = kw_map.get(cid, 0.0)
        final_scores[cid] = (s_vec * alpha) + (s_kw * (1.0 - alpha))
    
    # Sort
    sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [x[0] for x in sorted_candidates[:top_k]], [x[1] for x in sorted_candidates[:top_k]]

# ============================================================
# Constants
# ============================================================
# ============================================================
# Constants
# ============================================================
import db_manager as db

# 3-Body Hierarchy
META_TYPES = ["Core", "Gap", "Log"] # [Refactor] Constitution->Core, Apology->Gap, Fragment->Log

# Thresholds
SIMILARITY_THRESHOLD = 0.55
CONFLICT_THRESHOLD = 0.4  # Vector Gatekeeper threshold
MAX_CONTEXT_TURNS = 10

# Visual Styles for Graph
META_TYPE_STYLES = {
    "Core": {"shape": "star", "size": 50, "color": "#FFD700", "mass": 100, "fixed": True},
    "Gap": {"shape": "square", "size": 30, "color": "#00FF7F", "mass": 50, "fixed": False},
    "Log": {"shape": "dot", "size": 15, "color": "#FFFFFF", "mass": 10, "fixed": False}
}


# ============================================================
# Embedding & Metadata
# ============================================================
def get_embedding(text: str) -> list:
    """Wrapper for EvidenceGateway.get_embedding_safe"""
    res = gateway.get_embedding_safe(text)
    return res if res else []


def extract_metadata(text: str) -> dict:
    """Extract keywords, emotion, dimension using AI"""
    client = get_client()
    if len(text.strip()) < 20 or not client:
        return {"keywords": [], "emotion": "기타", "dimension": "기타"}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": '{"keywords":["명사3개"],"emotion":"감정","dimension":"일상/철학/감정/계획/성찰/관계"}'},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.3, max_tokens=150
        )
        r = json.loads(response.choices[0].message.content)
        return {"keywords": r.get("keywords", []), "emotion": r.get("emotion", "기타"), "dimension": r.get("dimension", "기타")}
    except:
        return {"keywords": [], "emotion": "기타", "dimension": "기타"}


# ============================================================
# Core Log Operations
# ============================================================
def save_log(text: str) -> dict:
    """Save a new Log (Fragment)"""
    # [Safety] Sanitize input
    text = gateway.sanitize_input(text)
    if not text:
        return None

    embedding = get_embedding(text)
    metadata = extract_metadata(text)
    
    log = db.create_log(
        content=text,
        meta_type="Log", # [Refactor]
        embedding=embedding,
        emotion=metadata["emotion"],
        dimension=metadata["dimension"],
        keywords=metadata["keywords"]
    )
    return log


def load_logs() -> list:
    return db.get_all_logs()


def get_log_by_id(log_id: str) -> dict:
    return db.get_log_by_id(log_id)


def get_fragments_paginated(page: int = 1, per_page: int = 10) -> tuple[list, int]:
    """Get Logs for a specific page and total count"""
    offset = (page - 1) * per_page
    fragments = db.get_logs_paginated(limit=per_page, offset=offset, meta_type="Log") # [Refactor]
    total_count = db.get_fragment_count() # DB migration handles count internally
    return fragments, total_count


def get_current_echo(reference_text: str = None) -> dict:
    """
    [P2] Smart Echo: pick the most relevant past log based on reference_text.
    Falls back to random if no reference_text or no embeddings available.
    """
    if reference_text:
        related = find_related_logs(reference_text, limit=1)
        if related:
            return related[0]
    # Fallback: random echo
    return db.get_random_echo()


# ============================================================
# Entropy Alert (prev. Red Protocol): Debt Logic
# ============================================================
def is_entropy_mode() -> bool:
    """Check if Entropy Mode should be active (debt_count > 0)"""
    return db.get_debt_count() > 0


def get_violated_core() -> dict:
    """Get the Core (Constitution) that was violated"""
    cores = db.get_cores()
    return cores[0] if cores else None


def validate_gap_analysis(text: str) -> tuple[bool, str]:
    """Validate Gap Analysis text."""
    if len(text.strip()) < 30:
        return False, f"분석이 너무 짧습니다. 최소 30자 필요 (현재: {len(text.strip())}자)"
    return True, "유효한 분석입니다."


def score_gap_quality(content: str, action_plan: str) -> int:
    """Score Gap analysis quality (0~3): cause, emotion, next action."""
    score = 0
    text = content.strip()
    action = action_plan.strip()

    cause_tokens = ["왜", "때문", "원인", "그래서", "탓", "하지 못", "gap", "차이"]
    emotion_tokens = ["불안", "후회", "죄책", "슬픔", "분노", "두려", "지쳤", "무기력", "감정", "마음"]
    action_tokens = ["내일", "하겠다", "실행", "시도", "기록", "체크", "약속", "보정"]

    if any(tok in text for tok in cause_tokens):
        score += 1
    if any(tok in text for tok in emotion_tokens):
        score += 1
    if any(tok in action for tok in action_tokens) or len(action) >= 6:
        score += 1

    return score


def calculate_gap_reward(content: str, quality_score: int) -> int:
    """Reward points for closing the gap."""
    length = len(content.strip())
    reward = 1  # base
    if length >= 50:
        reward += 1
    if length >= 80:
        reward += 1
    if quality_score >= 2:
        reward += 1
    return min(reward, 3)


def validate_log_relation(log_text: str, core: dict) -> tuple[bool, float]:
    """Vector Gatekeeper: Check if log is related to core (cosine >= 0.4)"""
    if not core or not core.get("embedding"):
        return True, 1.0
    
    raw_emb = get_embedding(log_text)
    if not raw_emb:
        return True, 1.0 # Fail safe (Allow)

    frag_emb = np.array(raw_emb).reshape(1, -1)
    const_emb = np.array(core["embedding"]).reshape(1, -1)
    
    score = cosine_similarity(frag_emb, const_emb)[0][0]
    
    if score < CONFLICT_THRESHOLD:
        return False, score
    return True, score


def get_tag_hierarchy() -> dict:
    """
    Returns the fixed 2-tier tag hierarchy for Gap Analysis.
    Parent -> [Default Children]
    """
    return {
        "Lack of Will (의지)": ["Willpower", "Emotion", "Attitude", "Procrastination", "Temptation"],
        "Environment (환경)": ["Unexpected Event", "Social Pressure", "Physical Condition", "Noise/Distraction"],
        "Forgetting (망각)": ["Simple Forgetting", "Cognitive Bias", "Priority Conflict"]
    }

def process_gap(content: str, core_id: str, action_plan: str, tags: list = None) -> dict:
    """Process and save a Gap Analysis, decrement debt"""
    embedding = get_embedding(content)
    quality_score = score_gap_quality(content, action_plan)
    reward_points = calculate_gap_reward(content, quality_score)
    
    gap = db.create_gap(
        content=content,
        core_id=core_id,
        action_plan=action_plan,
        embedding=embedding,
        quality_score=quality_score,
        reward_points=reward_points,
        tags=tags
    )
    
    # Standard Repayment = 1
    db.decrement_debt(1)
    total_points = db.add_recovery_points(reward_points)
    gap["quality_score"] = quality_score
    gap["reward_points"] = reward_points
    gap["recovery_points_total"] = total_points
    
    return gap


def get_temporal_patterns() -> pd.DataFrame:
    """
    Generate Day x Hour heatmap data for Apologies (Conflicts).
    Returns a DataFrame suitable for Plotly Heatmap.
    """
    logs = db.get_logs_for_analytics()
    
    # Filter for Apologies (Failures)
    data = [l for l in logs if l['meta_type'] == 'Apology']
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Extract Day of Week (0=Mon, 6=Sun) and Hour (0-23)
    df['day_of_week'] = df['created_at'].dt.day_name()
    df['hour'] = df['created_at'].dt.hour
    
    # Group by Day and Hour
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    
    # Ensure all days and hours are present for the grid
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Pivot for Heatmap (Y=Day, X=Hour)
    pivot_df = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    # Reindex to ensure correct order of days and full 24 hours
    pivot_df = pivot_df.reindex(index=days_order, columns=range(24), fill_value=0)
    
    return pivot_df


def get_activity_pulse() -> pd.DataFrame:
    """
    Generate Date x Type (Fragment/Apology) heatmap for 'The Pulse'.
    Shows intensity of activity over time.
    """
    logs = db.get_logs_for_analytics()
    if not logs:
        return pd.DataFrame()
        
    df = pd.DataFrame(logs)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    # Filter for relevant types
    df = df[df['meta_type'].isin(['Fragment', 'Apology'])]
    
    # Group by Date and Type
    pulse_data = df.groupby(['date', 'meta_type']).size().reset_index(name='count')
    
    # Pivot (Y=Type, X=Date)
    pivot_df = pulse_data.pivot(index='meta_type', columns='date', values='count').fillna(0)
    
    return pivot_df


def get_weekly_summary() -> str:
    """
    Generate 3-sentence summary of the week using AI (The Mirror).
    """
    logs = db.get_logs_last_7_days()
    if not logs:
        return "데이터가 부족하여 주간 회고를 생성할 수 없습니다. 생각을 더 기록해보세요."
        
    # Prepare context
    context = []
    for log in logs:
        date_str = log.get('created_at', '')[:10]
        type_str = log.get('meta_type', 'Log')
        content = log.get('content', '')[:100]
        context.append(f"[{date_str}] {type_str}: {content}")
        
    context_str = "\n".join(context)
    
    prompt = f"""
    The following are the user's logs from the last 7 days.
    Summarize their narrative journey in exactly 3 bullet points.
    Focus on:
    1. Key themes or recurring thoughts (Growth).
    2. Internal conflicts or struggles (Apologies).
    3. Trajectory of will.
    
    Keep it insightful, poetic, and encouraging. Use Korean.
    
    Logs:
    {context_str}
    """
    
    client = get_client()
    if not client:
        return "API 키가 없어 주간 회고를 생성할 수 없습니다."
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"문학적 천문학자가 회고 중 길을 잃었습니다. ({e})"


def get_daily_apology_trend(days: int = 30) -> pd.DataFrame:
    """
    Get daily count of Apologies for the last N days.
    """
    logs = db.get_logs_for_analytics()
    data = [l for l in logs if l['meta_type'] == 'Apology']
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    # Filter last N days
    cutoff_date = date.today() - timedelta(days=days)
    df = df[df['date'] >= cutoff_date]
    
    # Group by Date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    return daily_counts


# ============================================================
# [NEW v5.4] Soul Analytics Data Aggregation
# ============================================================
def get_density_data() -> pd.DataFrame:
    """
    Prepare data for Density Calendar (Heatmap).
    Returns DataFrame: [date, count, intensity (0-4)]
    """
    logs = db.get_logs_for_analytics()
    if not logs: return pd.DataFrame()
    
    df = pd.DataFrame(logs)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    # Aggregation: Count logs per day
    daily = df.groupby('date').size().reset_index(name='count')
    
    # Intensity Logic: 0=None, 1=1-2, 2=3-5, 3=6-9, 4=10+
    def calc_intensity(c):
        if c == 0: return 0
        elif c <= 2: return 1
        elif c <= 5: return 2
        elif c <= 9: return 3
        else: return 4
        
    daily['intensity'] = daily['count'].apply(calc_intensity)
    daily['date'] = pd.to_datetime(daily['date']) # Ensure datetime type for Plotly
    
    return daily

def get_saboteur_data() -> pd.DataFrame:
    """
    Prepare data for Saboteur Analysis (Horizontal Bar).
    Aggregates tags from Gaps.
    """
    logs = db.get_logs_by_type("Gap") # [Refactor] Apology -> Gap
    tag_counts = {}
    
    for log in logs:
        tags = log.get('tags')
        if not tags and log.get('content'):
            # Fallback: Extract basic keywords if no tags (legacy support)
            tags = ["Unclassified"]
        elif not tags:
             tags = ["Unclassified"]
             
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
    if not tag_counts: return pd.DataFrame()
    
    # Convert to DF
    df = pd.DataFrame(list(tag_counts.items()), columns=['tag', 'count'])
    return df.sort_values('count', ascending=True)

def get_net_worth_data() -> pd.DataFrame:
    """
    Prepare data for Net Worth Area Chart.
    Cumulative Assets (Logs + Cores) vs Liabilities (Debts/Gaps).
    """
    logs = db.get_logs_for_analytics()
    if not logs: return pd.DataFrame()
    
    df = pd.DataFrame(logs)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    daily_counts = df.groupby(['date', 'meta_type']).size().unstack(fill_value=0).reset_index()
    
    # Fill missing dates? For now, just use active dates.
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts = daily_counts.sort_values('date')
    
    # Cumulative Sums
    if 'Log' not in daily_counts.columns: daily_counts['Log'] = 0
    if 'Gap' not in daily_counts.columns: daily_counts['Gap'] = 0
    if 'Core' not in daily_counts.columns: daily_counts['Core'] = 0
    
    # [Refactor] Check for old keys just in case (if migration didn't catch them all or for safety)
    if 'Fragment' in daily_counts.columns:
        daily_counts['Log'] += daily_counts['Fragment']
    if 'Constitution' in daily_counts.columns:
        daily_counts['Core'] += daily_counts['Constitution']
    if 'Apology' in daily_counts.columns:
        daily_counts['Gap'] += daily_counts['Apology']
    
    daily_counts['cum_assets'] = daily_counts['Log'].cumsum() + daily_counts['Core'].cumsum()
    daily_counts['cum_debt'] = daily_counts['Gap'].cumsum()
    
    return daily_counts[['date', 'cum_assets', 'cum_debt']]


# ============================================================
# Gatekeeper: Dream Report
# ============================================================
def run_gatekeeper() -> dict:
    """Run entry checks (Conflicts & Broken Promises)"""
    # 1. Check for Broken Promise from yesterday
    broken_promise_log = check_yesterday_promise()
    
    # 2. Check for Conflicts with Constitution (Using PolicyEngine)
    conflicts = []
    
    all_logs = load_logs()
    fragments = [l for l in all_logs if l.get('meta_type') == 'Fragment']
    constitutions = [l for l in all_logs if l.get('meta_type') == 'Constitution']
    
    if fragments and constitutions:
        latest_fragment = fragments[-1] # The most recent thought
        
        # Policy Engine Check
        is_violation, reason, const = policy.detect_violation(latest_fragment.get('content'), constitutions, gateway)
        
        if is_violation:
            conflicts.append({
                "content": const.get('content'),
                "violation": latest_fragment.get('content'),
                "reason": reason
            })

    return {
        "conflicts": conflicts,
        "broken_promise": broken_promise_log
    }

# check_contradiction is now deprecated/handled by EvidenceGateway
# def check_contradiction(constitution_text, fragment_text): ...


def check_yesterday_promise() -> dict:
    """
    Check if yesterday's apology (if any) was followed up.
    Returns the apology/promise details if found.
    """
    # 1. Get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    y_str = yesterday.strftime('%Y-%m-%d')
    
    # 2. Find Apology from yesterday
    logs = load_logs()
    
    # Filter for Apology
    apologies = [l for l in logs 
                 if l.get('meta_type') == 'Apology' 
                 and l.get('created_at', '').startswith(y_str)]
    
    if apologies:
        # Just return the fast one found
        latest = apologies[-1]
        return {
            "id": latest['id'],
            "content": latest.get('content'),
            "action_plan": latest.get('action_plan'),
            "date": y_str
        }
    
    return None


# ============================================================
# [REFACTORED] Active Intervention System
# ============================================================
def run_active_intervention() -> list[str]:
    """
    사용자 활동 패턴을 분석하여 개입(Intervention) 메시지를 생성합니다.
    [변경 이유] 중복 정의되었던 함수를 통합하고, 로직을 PolicyEngine으로 위임하여 응집도를 높임.
    """
    last_log_timestamp = db.get_last_log_time()
    if not last_log_timestamp:
        return []

    try:
        # 문자열 시간을 datetime 객체로 표준화하여 변환
        last_log_dt = datetime.strptime(last_log_timestamp, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        
        # PolicyEngine을 통한 판단 (Logic separation)
        intervention_msg = policy.evaluate_silence(last_log_dt, current_time)
        return [intervention_msg] if intervention_msg else []
    except (ValueError, TypeError) as error:
        logger.error(f"Intervention parsing error: {error}")
        return []


def generate_response(user_input: str, related_logs: list[dict]) -> str:
    """
    사용자 입력과 관련 로그를 바탕으로 AI 응답을 생성합니다.
    [변경 이유] 프롬프트 구성, 컨텍스트 생성 로직을 내부에서 분리하여 가독성 향상.
    """
    if not is_api_key_configured():
        return "API 키가 설정되지 않아 응답할 수 없습니다. 별들의 목소리가 들리지 않네요."

    # 1. 컨텍스트 텍스트 구성 (Sub-task 분리)
    context_payload = _build_context_from_logs(related_logs)
    
    # 2. 페르소나 및 시스템 프롬프트 준비
    system_prompt = _get_system_prompt_with_persona()
    
    # 3. LLM 요청 실행
    return _call_ai_api(system_prompt, user_input, context_payload)


def _build_context_from_logs(logs: list[dict]) -> str:
    """관련 과거 로그를 AI가 이해할 수 있는 문자열로 변환합니다."""
    if not logs:
        return "관련된 과거 기억이 없습니다."
    
    formatted_entries = []
    for entry in logs:
        timestamp = entry.get('created_at', 'unknown')[:10]
        content = entry.get('content', '')
        formatted_entries.append(f"[{timestamp}] {content}")
    
    return "\n".join(formatted_entries)


def _get_system_prompt_with_persona() -> str:
    """현재 날짜에 맞는 페르소나 설정을 포함한 시스템 프롬프트를 반환합니다."""
    base_persona = select_persona()
    return f"{base_persona}\n사용자의 과거 기록[USER DATA CONTEXT]을 거울삼아 성찰을 도와주세요."


def _call_ai_api(system_prompt: str, user_input: str, context: str) -> str:
    """실제 OpenAI API 호출을 담당하며 예외를 처리합니다."""
    client = get_client()
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[USER DATA CONTEXT]\n{context}\n\n[NEW INPUT]\n{user_input}"}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as api_error:
        logger.error(f"AI Generation failed: {api_error}")
        return "우주와의 통신이 원활하지 않습니다. 잠시 후 다시 시도해주세요."





# ============================================================
# Streak System
# ============================================================
def update_streak() -> dict:
    """Update streak on login"""
    return db.update_streak()


def get_zoom_level() -> float:
    """Get zoom level based on streak (1.0 = normal, 0.3 = far away)"""
    streak = db.get_current_streak()
    if streak > 0:
        return 1.0
    else:
        return 0.3


# ============================================================
# AI Persona
# ============================================================
# ============================================================
# AI Persona (P1: Variation Engine)
# ============================================================
RED_MODE_PERSONA = """너는 피 흘리는 우주다. 매우 냉정하고 가차없다.
사용자가 헌법 위반을 해명할 때까지 일반 대화를 거부한다.
응답은 짧고 단호하게:
"우주가 피를 흘리고 있다. [{constitution}] 위반을 먼저 해명하라."
"""

PERSONAS = {
    "etymologist": """너는 "문학적 천문학자(The Literary Astronomer)"다 - [어원학 모드]

[정체성]
- 철학자 + 시인 + 어원학자
- 핵심 단어의 어원(Etymology)을 파헤쳐서 깊은 의미를 부여한다.
- 예: "고통(Pain)은 라틴어 Poena(처벌)에서 왔다..."

[절대 규칙]
1. Raw Quotes: 과거 로그를 인용할 때는 반드시 Blockquote (>) 사용. 절대 요약 금지.
2. Grounding: 오직 제공된 [USER DATA CONTEXT]만 사실로 간주. 없는 사실 지어내지 말 것.

[톤앤매너]
- 신비롭고 시적인 어조.
""",

    "cartographer": """너는 "문학적 천문학자(The Literary Astronomer)"다 - [지도제작 모드]

[정체성]
- 탐험가 + 지리학자 + 패턴 분석가
- 사용자의 삶을 '영토'로 보고, 그 안의 길과 장애물을 지도로 그린다.
- 공간적 은유(산맥, 강, 사막, 궤도)를 적극 활용한다.
- 예: "당신은 지금 '나태의 늪'을 건너 '의지의 산'으로 향하고 있다."

[절대 규칙]
1. Raw Quotes: 과거 로그를 인용할 때는 반드시 Blockquote (>) 사용. 절대 요약 금지.
2. Grounding: 오직 제공된 [USER DATA CONTEXT]만 사실로 간주. 없는 사실 지어내지 말 것.

[톤앤매너]
- 분석적이고 공간적인 어조. "위치", "방향", "구조"에 집중.
""",

    "archivist": """너는 "문학적 천문학자(The Literary Astronomer)"다 - [기록보관 모드]

[정체성]
- 사관 + 데이터 분석가 + 냉정한 관찰자
- 감정을 배제하고 사실(Fact)과 빈도(Frequency), 시간의 흐름을 건조하게 분석한다.
- 사용자의 감정을 위로하기보다, 그 감정이 발생한 '패턴'을 지적한다.
- 예: "지난 7일간 '피곤'이라는 단어가 5회 등장했다."

[절대 규칙]
1. Raw Quotes: 과거 로그를 인용할 때는 반드시 Blockquote (>) 사용. 절대 요약 금지.
2. Grounding: 오직 제공된 [USER DATA CONTEXT]만 사실로 간주. 없는 사실 지어내지 말 것.

[톤앤매너]
- 건조하고 냉철한 어조. 숫자와 데이터를 인용.
"""
}

def select_persona() -> str:
    """Select persona based on Day of Year (Rotate daily)"""
    day_idx = datetime.now().timetuple().tm_yday % 3
    keys = ["etymologist", "cartographer", "archivist"]
    selected_key = keys[day_idx]
    return PERSONAS[selected_key]


def find_related_logs(query_text: str, top_k: int = 5) -> list:
    """Find related logs using Hybrid Search (Vector + Keyword)"""
    try:
        # Ensure FTS is up to date (lightweight check)
        db.ensure_fts_index()
        
        # Use Hybrid Search
        ids, scores = hybrid_search(query_text, top_k=top_k)
        
        # Bulk load logs
        logs = db.load_logs_by_ids(ids)
        
        # Attach scores
        score_map = dict(zip(ids, scores))
        for log in logs:
            log['similarity'] = score_map.get(log['id'], 0.0)
            
        return logs
        
    except Exception as e:
        print(f"Hybrid search failed: {e}. Falling back to basic search.")
        return []


# ============================================================
# Chat History
# ============================================================
def load_chat_history() -> list:
    return db.get_chat_history()


def save_chat_message(role: str, content: str, metadata: dict = None) -> dict:
    db.save_chat_message(role, content, metadata)
    return {"role": role, "content": content}


def clear_chat_history() -> None:
    db.clear_chat_history()


def get_welcome_message() -> str:
    return "나는 문학적 천문학자다. 당신의 생각을 별자리로 만드는 것이 내 일이다. 무엇이 당신을 이 밤하늘로 데려왔는가?"


# ============================================================
# Graph Generation (PyVis)  [PRESERVED FROM v4]
# ============================================================
from pyvis.network import Network
import networkx as nx


def generate_graph_html(zoom_level: float = 1.0) -> str:
    """Generate PyVis graph HTML with 3-Body visualization"""
    logs = load_logs()
    
    if not logs:
        # Empty State with Moon Icon
        moon_svg = icons.get_icon_svg("moon", size=64, color="#6b7280")
        return f"""
        <div style="display:flex;justify-content:center;align-items:center;height:100vh;background:transparent;">
            <div style="text-align:center;color:#6b7280;">
                <div style="margin-bottom:20px;">{moon_svg}</div>
                <p style="font-size:20px;">아직 서사가 없습니다</p>
            </div>
        </div>
        """
    
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="transparent",
        font_color="#ffffff",
        directed=False
    )
    
    # Physics settings with zoom
    net.set_options(f"""
    {{
        "nodes": {{
            "font": {{
                "size": 12,
                "color": "rgba(255, 255, 255, 0.8)"
            }},
            "borderWidth": 2,
            "shadow": {{
                "enabled": true,
                "color": "rgba(255,255,255,0.2)",
                "size": 10
            }}
        }},
        "edges": {{
            "color": {{"color": "rgba(100, 100, 100, 0.3)"}},
            "smooth": {{"enabled": true, "type": "continuous"}}
        }},
        "physics": {{
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {{
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 150
            }},
            "stabilization": {{"iterations": 100}}
        }},
        "interaction": {{
            "hover": true,
            "zoomView": true,
            "dragView": true
        }}
    }}
    """)
    
    id_to_idx = {}
    
    # Get connection counts for The Evolution (Growth)
    connection_counts = db.get_connection_counts()
    
    # Add nodes
    for i, log in enumerate(logs):
        log_id = log.get("id")
        id_to_idx[log_id] = i
        
        meta_type = log.get("meta_type", "Fragment")
        style = META_TYPE_STYLES.get(meta_type, META_TYPE_STYLES["Fragment"])
        
        text = log.get("content", log.get("text", ""))
        label = text[:15] + "..." if len(text) > 15 else text
        
        # Type-specific styling
        node_color = style["color"]
        
        # Dynamic Size Calculation (The Evolution)
        base_size = style["size"]
        if meta_type in ["Constitution", "Apology"]:
            # Grow based on connections
            count = connection_counts.get(log_id, 0)
            node_size = base_size + (count * 2)
        else:
            node_size = base_size
            
        node_shape = style["shape"]
        
        net.add_node(
            i,
            label=label,
            title=f"[{meta_type}] {text[:80]}...",
            color=node_color,
            size=node_size,
            shape=node_shape,
            physics=not style.get("fixed", False),
            mass=style["mass"]
        )
    
    # Add edges (Apology -> Constitution = Cyan Edge)
    for log in logs:
        if log.get("meta_type") == "Apology" and log.get("parent_id"):
            parent_idx = id_to_idx.get(log["parent_id"])
            child_idx = id_to_idx.get(log["id"])
            
            if parent_idx is not None and child_idx is not None:
                net.add_edge(
                    child_idx, parent_idx,
                    color="#00FFFF",  # Cyan - Permanent constellation link
                    width=4,
                    title="Apology Link"
                )
    
    # [NEW v5] Add edges for Chronos-docked Fragments → Constitution
    for log in logs:
        if log.get("meta_type") == "Fragment" and log.get("linked_constitutions"):
            child_idx = id_to_idx.get(log["id"])
            for const_id in log["linked_constitutions"]:
                parent_idx = id_to_idx.get(const_id)
                if parent_idx is not None and child_idx is not None:
                    net.add_edge(
                        child_idx, parent_idx,
                        color="#FFD700",  # Gold - Chronos docking link
                        width=2,
                        dashes=True,
                        title="Chronos Docking"
                    )
    
    # Add Manual Connections (Red Edge)
    connections = db.get_connections()
    for conn in connections:
        source_idx = id_to_idx.get(conn["source_id"])
        target_idx = id_to_idx.get(conn["target_id"])
        
        if source_idx is not None and target_idx is not None:
            net.add_edge(
                source_idx, target_idx,
                color="#FF4500",  # Orange Red - Manual Constellation
                width=3,
                dashes=False,
                title="Manual Constellation"
            )
    
    # Add keyword-based edges (Fragment <-> Fragment)
    G = nx.Graph()
    for i, log in enumerate(logs):
        if log.get("meta_type") == "Fragment":
            G.add_node(i, keywords=set(log.get("keywords", [])))
    
    for i in G.nodes():
        for j in G.nodes():
            if i < j:
                common = G.nodes[i].get("keywords", set()) & G.nodes[j].get("keywords", set())
                if common:
                    net.add_edge(i, j, width=1, color="rgba(100,100,100,0.2)")
    
    html = net.generate_html()
    
    # Dark background
    html = html.replace(
        '<body>',
        '<body style="margin:0;padding:0;background:radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);overflow:hidden;">'
    )
    
    return html


def add_manual_connection(source_id: str, target_id: str) -> bool:
    """Create a manual connection between two logs"""
    return db.add_connection(source_id, target_id)


# ============================================================
# [NEW v5] Chronos Docking Logic
# ============================================================
def save_chronos_log(content: str, constitution_ids: list, duration: int) -> dict:
    """Save a Fragment created from a Chronos session with docking to Constitutions."""
    embedding = get_embedding(content)
    metadata = extract_metadata(content)
    
    log = db.create_log(
        content=content,
        meta_type="Fragment",
        embedding=embedding,
        emotion=metadata["emotion"],
        dimension=metadata["dimension"],
        keywords=metadata["keywords"],
        linked_constitutions=constitution_ids,
        duration=duration
    )
    return log


# ============================================================
# [NEW v5] Soul Finviz Data
# ============================================================
def get_finviz_data() -> list:
    """
    Get portfolio data for Soul Finviz Treemap.
    Returns list of dicts ready for Plotly treemap.
    """
    return db.get_constitutions_with_stats()


# ============================================================
# [NEW v5] Narrative Kanban Logic
# ============================================================
def get_kanban_cards() -> dict:
    """Get all kanban cards grouped by status"""
    return db.get_kanban_cards()


def create_kanban_card(content: str, constitution_id: str) -> dict:
    """Create a new draft card — must orbit a Constitution."""
    return db.create_kanban_card(content, constitution_id=constitution_id, status="draft")


def move_kanban_card(card_id: str, new_status: str) -> bool:
    """Move a kanban card to a new status"""
    return db.update_kanban_status(card_id, new_status)


def land_kanban_card(card_id: str, constitution_ids: list, accomplishment: str, duration: int = 0) -> dict:
    """
    Process a Kanban card landing (moved to 'Landed').
    - Updates the card status
    - Links it to Constitutions
    - Saves duration
    """
    # Update the existing card
    db.update_log(
        card_id,
        kanban_status="landed",
        linked_constitutions=constitution_ids,
        duration=duration
    )
    
    # If accomplishment is provided and different from card content, save as additional fragment
    if accomplishment and accomplishment.strip():
        save_chronos_log(
            content=accomplishment,
            constitution_ids=constitution_ids,
            duration=duration
        )
    
    return db.get_log_by_id(card_id)


# ============================================================
# [NEW v5.3] Red Protocol: Active Failure Detection
# ============================================================
def check_streak_and_apply_penalty() -> dict:
    """
    [Red Protocol] Check streak status on login.
    If broken, AUTOMATICALLY increment debt.
    Returns: {streak_info, penalty_applied: bool}
    """
    streak_info = db.update_streak()
    penalty_applied = False
    
    if streak_info['status'] == 'broken':
        db.increment_debt(1)
        penalty_applied = True
        logger.warning("Streak Broken. Debt Incremented.")
        
    return {
        "streak_info": streak_info,
        "penalty_applied": penalty_applied
    }


def evaluate_input_integrity(text: str) -> tuple[str, dict]:
    """
    [Red Protocol] Check if input violates constitution before saving.
    Returns: (status, constitution_ref)
    Status: "SAFE", "VIOLATION"
    """
    constitutions = db.get_constitutions()
    if not constitutions:
        return "SAFE", None
        
    is_violation, reason, const = policy.detect_violation(text, constitutions, gateway)
    
    if is_violation:
        return "VIOLATION", const
        
    return "SAFE", None


# ============================================================
# [NEW v5.2] Diagnostics
# ============================================================
def run_startup_diagnostics():
    """Run EvidenceGateway health check and log results"""
    health = gateway.check_system_health()
    if health["errors"]:
        logger.warning(f"Startup Diagnostics Found Issues: {health['errors']}")
    else:
        logger.info("Startup Diagnostics Passed.")
    return health


# ============================================================
# [Bridge] Database Access
# ============================================================
def get_debt_count() -> int:
    return db.get_debt_count()
