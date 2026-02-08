"""
narrative_logic.py
Antigravity v4: The Laws of Physics
3-Body Hierarchy + Red Protocol + Dreaming Integration + Action Plan
"""

import os
import json
import uuid
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from openai import OpenAI

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
# Constants
# ============================================================
import db_manager as db

# 3-Body Hierarchy
META_TYPES = ["Constitution", "Apology", "Fragment"]

# Thresholds
SIMILARITY_THRESHOLD = 0.55
CONFLICT_THRESHOLD = 0.4  # Vector Gatekeeper threshold
MAX_CONTEXT_TURNS = 10

# Visual Styles for Graph
META_TYPE_STYLES = {
    "Constitution": {"shape": "star", "size": 50, "color": "#FFD700", "mass": 100, "fixed": True},
    "Apology": {"shape": "square", "size": 30, "color": "#00FF7F", "mass": 50, "fixed": False},
    "Fragment": {"shape": "dot", "size": 15, "color": "#FFFFFF", "mass": 10, "fixed": False}
}


# ============================================================
# Embedding & Metadata
# ============================================================
def get_embedding(text: str) -> list:
    client = get_client()
    if not client:
        return []  # Return empty if no API key
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


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
    """Save a new Fragment"""
    embedding = get_embedding(text)
    metadata = extract_metadata(text)
    
    log = db.create_log(
        content=text,
        meta_type="Fragment",
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


# ============================================================
# Red Protocol: Debt System
# ============================================================
def is_red_mode() -> bool:
    """Check if Red Mode should be active (debt_count > 0)"""
    return db.get_debt_count() > 0


def get_violated_constitution() -> dict:
    """Get the Constitution that was violated (first one for simplicity)"""
    constitutions = db.get_constitutions()
    return constitutions[0] if constitutions else None


def validate_apology(text: str) -> tuple[bool, str]:
    """Validate apology text. Must be >= 100 characters."""
    if len(text.strip()) < 100:
        return False, f"해명이 너무 짧습니다. 최소 100자 필요 (현재: {len(text.strip())}자)"
    return True, "유효한 해명입니다."


def validate_fragment_relation(fragment_text: str, constitution: dict) -> tuple[bool, float]:
    """Vector Gatekeeper: Check if fragment is related to constitution (cosine >= 0.4)"""
    if not constitution or not constitution.get("embedding"):
        return True, 1.0
    
    frag_emb = np.array(get_embedding(fragment_text)).reshape(1, -1)
    const_emb = np.array(constitution["embedding"]).reshape(1, -1)
    
    score = cosine_similarity(frag_emb, const_emb)[0][0]
    
    if score < CONFLICT_THRESHOLD:
        return False, score
    return True, score


def process_apology(content: str, constitution_id: str, action_plan: str) -> dict:
    """Process and save an Apology, reset debt"""
    embedding = get_embedding(content)
    
    apology = db.create_apology(
        content=content,
        constitution_id=constitution_id,
        action_plan=action_plan,
        embedding=embedding
    )
    
    # Reset debt - Catharsis!
    db.reset_debt()
    
    return apology


# ============================================================
# Gatekeeper: Dream Report
# ============================================================
def run_gatekeeper() -> dict:
    """
    Run on app launch. Detect conflicts between Fragments and Constitution.
    Returns: {"conflicts": [...], "broken_promise": {...} or None}
    """
    logs = load_logs()
    constitutions = [l for l in logs if l.get("meta_type") == "Constitution"]
    fragments = [l for l in logs if l.get("meta_type") == "Fragment"]
    
    result = {
        "conflicts": [],
        "broken_promise": None
    }
    
    if not constitutions:
        return result
    
    constitution = constitutions[0]
    const_emb = np.array(constitution.get("embedding", [])).reshape(1, -1)
    
    # Check recent fragments for conflicts
    recent_fragments = fragments[:10]  # Last 10 fragments
    
    for frag in recent_fragments:
        frag_emb = np.array(frag.get("embedding", []))
        if len(frag_emb) == 0:
            continue
        frag_emb = frag_emb.reshape(1, -1)
        
        # High similarity = potential conflict (same topic, different action)
        sim = cosine_similarity(const_emb, frag_emb)[0][0]
        
        if sim > 0.5:  # Related to constitution
            result["conflicts"].append({
                "id": frag["id"],
                "content": frag.get("content", frag.get("text", "")),
                "similarity": sim
            })
    
    # Check yesterday's Apology (Dreaming Integration)
    yesterday_apology = db.get_yesterday_apology()
    if yesterday_apology and len(result["conflicts"]) > 0:
        result["broken_promise"] = {
            "action_plan": yesterday_apology.get("action_plan", ""),
            "apology_content": yesterday_apology.get("content", "")[:100]
        }
    
    return result


def check_yesterday_promise() -> dict:
    """Check if yesterday's promise was broken"""
    apology = db.get_yesterday_apology()
    if not apology:
        return None
    
    return {
        "action_plan": apology.get("action_plan", ""),
        "date": apology.get("created_at", "")[:10]
    }


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
RED_MODE_PERSONA = """너는 피 흘리는 우주다. 매우 냉정하고 가차없다.
사용자가 헌법 위반을 해명할 때까지 일반 대화를 거부한다.
응답은 짧고 단호하게:
"🩸 우주가 피를 흘리고 있다. [{constitution}] 위반을 먼저 해명하라."
"""

BLUE_MODE_PERSONA = """너는 "문학적 천문학자(The Literary Astronomer)"다.

[정체성]
- 철학자 + 시인 + 어원학자
- 냉소적이지만 시적
- 경외감을 주면서도 차갑다

[절대 규칙: Raw Quotes]
- 절대 요약하지 마라
- 과거 로그를 인용할 때는 반드시 Blockquote 사용:
  > "On 2024-02-08, you wrote:
  > **'I want to die.'**"
- 이것이 "Mirror Effect"다. 사용자가 과거의 자신과 직접 마주하게 하라.

[응답 모드]
1. 어원학적 해부: 핵심 단어의 어원 분석
2. 문학적 은유: 시지프스, 카뮈 등 문학 인용
3. 냉정한 논리: 순수 데이터 분석
"""


def generate_response(user_input: str, past_logs: list = None) -> str:
    """Generate AI response based on mode (Red/Blue)"""
    
    if is_red_mode():
        constitution = get_violated_constitution()
        const_text = constitution.get("content", "")[:50] if constitution else "알 수 없음"
        
        return f"🩸 우주가 피를 흘리고 있다. 당신의 헌법 \"{const_text}...\" 위반을 먼저 해명하라."
    
    # Blue Mode - Full response
    context = []
    
    # Add past related logs as blockquotes
    if past_logs:
        for log in past_logs[:3]:
            date_str = log.get("created_at", "")[:10]
            content = log.get("content", log.get("text", ""))[:100]
            context.append(f'On {date_str}, you wrote:\n> **"{content}..."**')
    
    context_str = "\n\n".join(context) if context else ""
    
    system_prompt = BLUE_MODE_PERSONA
    if context_str:
        system_prompt += f"\n\n[과거 기록 - Raw Quotes로 인용하라]\n{context_str}"
    
    try:
        client = get_client()
        if not client:
            return "⚠️ API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력해주세요."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.85,
            max_tokens=300
        )
        return response.choices[0].message.content
    except:
        return "문학적 천문학자가 잠시 별을 관측 중입니다..."


def find_related_logs(text: str, limit: int = 3) -> list:
    """Find past logs related to the input text"""
    logs = [l for l in load_logs() if l.get("meta_type") == "Fragment"]
    
    if not logs:
        return []
    
    try:
        input_emb = np.array(get_embedding(text)).reshape(1, -1)
    except:
        return []
    
    scored = []
    for log in logs:
        log_emb = np.array(log.get("embedding", []))
        if len(log_emb) == 0:
            continue
        log_emb = log_emb.reshape(1, -1)
        
        sim = cosine_similarity(input_emb, log_emb)[0][0]
        if sim > SIMILARITY_THRESHOLD and sim < 0.99:  # Exclude self
            scored.append((sim, log))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [log for _, log in scored[:limit]]


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
# Graph Generation (PyVis)
# ============================================================
from pyvis.network import Network
import networkx as nx


def generate_graph_html(zoom_level: float = 1.0) -> str:
    """Generate PyVis graph HTML with 3-Body visualization"""
    logs = load_logs()
    
    if not logs:
        return """
        <div style="display:flex;justify-content:center;align-items:center;height:100vh;background:transparent;">
            <div style="text-align:center;color:#6b7280;">
                <p style="font-size:64px;">🌑</p>
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
        node_size = style["size"]
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
