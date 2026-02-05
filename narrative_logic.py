"""
narrative_logic.py
Narrative OS: The Dreaming Brain
4-Layer Hierarchy + Hybrid Gravity Search + Anchor/Meteor Graph + Autopoiesis Engine
"""

import os
import json
import uuid
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from openai import OpenAI
import networkx as nx
from pyvis.network import Network

# ============================================================
# API Key
# ============================================================
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# ============================================================
# 상수
# ============================================================
DATA_FILE = "data/user_logs.json"
CHAT_HISTORY_FILE = "data/chat_history.json"

# 4-Layer Hierarchy
META_TYPES = ["Fragment", "Thirst", "Decision", "Constitution"]

# Search Thresholds
SIMILARITY_THRESHOLD = 0.55
SELF_SIMILARITY_THRESHOLD = 0.99
TAG_KEYWORD_BONUS = 0.15  # Hybrid Search Bonus
MAX_CONTEXT_TURNS = 10

# 차원별 색상 (Neon/Dark Universe Style)
DIMENSION_COLORS = {
    "일상": "#00FFFF",    # Neon Cyan
    "철학": "#FFD700",    # Golden Amber
    "감정": "#FF007F",    # Deep Magenta
    "계획": "#00FF7F",    # Spring Green
    "성찰": "#FF6B35",    # Neon Orange
    "관계": "#FF1493",    # Deep Pink
    "기타": "#C0C0C0"     # Silver
}

# Meta Type Visual Styles
META_TYPE_STYLES = {
    "Constitution": {"shape": "star", "size": 50, "color": "#FFD700", "fixed": True, "mass": 10},
    "Decision": {"shape": "diamond", "size": 35, "color": "#00FF7F", "fixed": False, "mass": 5},
    "Thirst": {"shape": "triangle", "size": 25, "color": "#FF6B35", "fixed": False, "mass": 2},
    "Fragment": {"shape": "dot", "size": 10, "color": "#F0F0F0", "fixed": False, "mass": 1}
}

# Virtual Node Types (Autopoiesis - Ghost Nodes)
VIRTUAL_TYPES = ["Conflict", "Prediction", "Question"]

# Virtual Node Visual Styles (Ghost-like appearance)
VIRTUAL_TYPE_STYLES = {
    "Conflict": {"shape": "dot", "size": 20, "color": "#FF0055", "opacity": 0.6, "dashes": True},    # Neon Red
    "Prediction": {"shape": "dot", "size": 20, "color": "#9D00FF", "opacity": 0.6, "dashes": True},  # Neon Purple
    "Question": {"shape": "dot", "size": 20, "color": "#00FFFF", "opacity": 0.6, "dashes": True}     # Neon Cyan
}

# Emotion Opposites for Conflict Detection
EMOTION_OPPOSITES = {
    "행복": ["슬픔", "우울", "불안"],
    "기쁨": ["슬픔", "우울", "분노"],
    "평온": ["불안", "스트레스", "분노"],
    "희망": ["절망", "무력감", "체념"],
    "열정": ["무기력", "권태", "체념"],
    "자신감": ["불안", "두려움", "의심"],
    "슬픔": ["행복", "기쁨"],
    "우울": ["행복", "기쁨", "열정"],
    "불안": ["평온", "자신감"],
    "분노": ["평온", "기쁨"],
    "스트레스": ["평온", "여유"],
    "무력감": ["희망", "열정", "자신감"]
}

# ============================================================
# Iron Rules: Skin in the Game
# ============================================================
DECISION_LOCK_HOURS = 72  # Decision 수정 금지 시간
PROOF_REQUIRED_COUNT = 3  # Constitution 수정에 필요한 Fragment 수
ENTROPY_DECAY_DAYS = 7     # Fragment 페이드 아웃 기간

# Status Types for Version Control
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"
STATUS_PENDING = "pending_amendment"

# ============================================================
# 데이터 I/O (SQLite Backend)
# ============================================================
import db_manager as db

def load_logs() -> list:
    """Load all logs from SQLite"""
    return db.get_all_logs()


def save_logs(logs: list) -> None:
    """Compatibility layer - not needed with SQLite"""
    pass  # SQLite handles persistence automatically


def load_chat_history() -> list:
    """Load chat history from SQLite"""
    return db.get_chat_history()


def save_chat_history(history: list) -> None:
    """Compatibility layer"""
    pass


def save_chat_message(role: str, content: str, metadata: dict = None) -> dict:
    db.save_chat_message(role, content, metadata)
    return {"role": role, "content": content}


def clear_chat_history() -> None:
    db.clear_chat_history()


def get_recent_context(max_turns: int = MAX_CONTEXT_TURNS) -> list:
    history = load_chat_history()
    recent = history[-max_turns:] if len(history) > max_turns else history
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def get_conversation_summary() -> str:
    history = load_chat_history()
    if len(history) < 4:
        return ""
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    if len(user_msgs) < 2:
        return ""
    recent = user_msgs[-3:]
    return "[이전 발언]\n" + "\n".join([f'- "{s[:80]}..."' if len(s) > 80 else f'- "{s}"' for s in recent])


# ============================================================
# 임베딩 & 메타데이터
# ============================================================
def get_embedding(text: str) -> list:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def extract_metadata(text: str) -> dict:
    """짧은 입력은 가볍게 처리 (UX Tone 조절)"""
    if len(text.strip()) < 20:
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


def save_log(text: str, user_tags: list = None, is_virtual: bool = False, virtual_type: str = None) -> dict:
    """저장 로그 - SQLite Backend"""
    embedding = get_embedding(text)
    metadata = extract_metadata(text)
    
    log = db.create_log(
        content=text,
        meta_type="Fragment",
        embedding=embedding,
        emotion=metadata["emotion"],
        dimension=metadata["dimension"],
        keywords=metadata["keywords"],
        tags=user_tags,
        is_virtual=is_virtual,
        virtual_type=virtual_type
    )
    
    # Compatibility: add 'text' field alias
    if log:
        log["text"] = log.get("content", "")
    
    return log


def promote_log(log_id: str, new_type: str) -> bool:
    """사용자 동의 후 호출: meta_type 승격 (SQLite)"""
    if new_type not in META_TYPES:
        return False
    return db.update_log(log_id, meta_type=new_type)


def get_log_by_id(log_id: str) -> dict:
    """Get log by ID (SQLite)"""
    log = db.get_log_by_id(log_id)
    if log:
        log["text"] = log.get("content", "")
    return log


# ============================================================
# Iron Rules: 72-Hour Lock & Proof of Action
# ============================================================
def can_edit_log(log: dict) -> tuple[bool, str]:
    """
    72-Hour Lock: Decision은 72시간 동안 수정 불가
    Returns: (can_edit, error_message)
    """
    meta_type = log.get("meta_type", "Fragment")
    
    # Fragment/Thirst는 자유 수정
    if meta_type in ["Fragment", "Thirst"]:
        return True, ""
    
    # Constitution은 Amendment 필요
    if meta_type == "Constitution":
        return False, "헌법은 수정할 수 없습니다. Amendment를 요청하세요."
    
    # Decision: 72시간 락
    if meta_type == "Decision":
        created = datetime.fromisoformat(log.get("timestamp"))
        elapsed = datetime.now() - created
        hours_elapsed = elapsed.total_seconds() / 3600
        
        if hours_elapsed < DECISION_LOCK_HOURS:
            hours_left = DECISION_LOCK_HOURS - hours_elapsed
            return False, f"⏰ 잉크가 마르지 않았습니다. {hours_left:.0f}시간 후에 수정 가능합니다."
        
        return True, ""
    
    return True, ""


def can_delete_log(log: dict) -> tuple[bool, str]:
    """삭제 가능 여부 (edit과 동일한 규칙)"""
    return can_edit_log(log)


def request_amendment(log_id: str, reason: str) -> tuple[bool, str]:
    """
    Constitution Amendment 요청
    Returns: (success, message)
    """
    logs = load_logs()
    
    for log in logs:
        if log.get("id") == log_id:
            if log.get("meta_type") != "Constitution":
                return False, "헌법만 Amendment를 요청할 수 있습니다."
            
            if log.get("status") == STATUS_PENDING:
                return False, "이미 Amendment 진행 중입니다."
            
            log["status"] = STATUS_PENDING
            log["amendment_reason"] = reason
            log["pending_proof_count"] = PROOF_REQUIRED_COUNT
            save_logs(logs)
            
            return True, f"📝 Amendment 요청됨. {PROOF_REQUIRED_COUNT}개의 관련 파편을 증명하세요."
    
    return False, "로그를 찾을 수 없습니다."


def check_amendment_progress(constitution_id: str) -> tuple[int, int]:
    """
    Amendment 진행 상황 확인
    Returns: (current_count, required_count)
    """
    logs = load_logs()
    constitution = None
    
    for log in logs:
        if log.get("id") == constitution_id:
            constitution = log
            break
    
    if not constitution or constitution.get("status") != STATUS_PENDING:
        return 0, PROOF_REQUIRED_COUNT
    
    # 최근 Fragment 중 관련된 것 찾기
    const_emb = np.array(constitution.get("embedding", [])).reshape(1, -1)
    amendment_time = datetime.fromisoformat(constitution.get("timestamp"))
    
    related_count = 0
    for log in logs:
        if log.get("meta_type") == "Fragment" and log.get("status") == STATUS_ACTIVE:
            created = datetime.fromisoformat(log.get("timestamp"))
            
            # Amendment 요청 이후의 Fragment만 카운트
            if created > amendment_time:
                frag_emb = np.array(log.get("embedding", [])).reshape(1, -1)
                sim = cosine_similarity(const_emb, frag_emb)[0][0]
                
                if sim > 0.5:  # 관련성 임계값
                    related_count += 1
    
    return related_count, PROOF_REQUIRED_COUNT


def ratify_amendment(constitution_id: str, new_text: str) -> tuple[bool, str]:
    """
    Amendment 비준: 증거가 충분하면 헌법 업데이트
    Old → archived, New → active
    """
    current, required = check_amendment_progress(constitution_id)
    
    if current < required:
        return False, f"증거 부족: {current}/{required}개의 관련 파편이 필요합니다."
    
    logs = load_logs()
    old_const = None
    
    for log in logs:
        if log.get("id") == constitution_id:
            old_const = log
            break
    
    if not old_const:
        return False, "헌법을 찾을 수 없습니다."
    
    # Archive old constitution
    old_const["status"] = STATUS_ARCHIVED
    
    # Create new constitution
    new_const = save_log(new_text)
    new_const["meta_type"] = "Constitution"
    new_const["parent_id"] = constitution_id
    new_const["amendment_reason"] = old_const.get("amendment_reason")
    new_const["status"] = STATUS_ACTIVE
    
    # Update logs
    for i, log in enumerate(logs):
        if log.get("id") == constitution_id:
            logs[i] = old_const
        elif log.get("id") == new_const["id"]:
            logs[i] = new_const
    
    save_logs(logs)
    return True, "✨ Amendment 비준 완료! 새 헌법이 활성화되었습니다."


def get_archived_constitutions() -> list:
    """Archived된 헌법들 반환"""
    logs = load_logs()
    return [log for log in logs if 
            log.get("meta_type") == "Constitution" and 
            log.get("status") == STATUS_ARCHIVED]


# ============================================================
# Entropy Engine: Memory Decay
# ============================================================
def calculate_entropy_score(log: dict) -> float:
    """
    Entropy Score = 1 / (days_elapsed^2)
    Constitution은 Zero Entropy (1.0)
    """
    meta_type = log.get("meta_type", "Fragment")
    
    # Constitution/Decision: No decay
    if meta_type in ["Constitution", "Decision"]:
        return 1.0
    
    created = datetime.fromisoformat(log.get("timestamp"))
    days_elapsed = (datetime.now() - created).days
    
    if days_elapsed <= 0:
        return 1.0
    
    # Decay formula: max(0.3, 1 / (days^0.5))
    score = max(0.3, 1.0 / (days_elapsed ** 0.5))
    return score


def calculate_opacity(log: dict) -> float:
    """
    Fragment opacity: 7일 동안 페이드 아웃
    0일: 1.0 → 7일: 0.3
    """
    meta_type = log.get("meta_type", "Fragment")
    
    if meta_type in ["Constitution", "Decision"]:
        return 1.0
    
    if log.get("status") == STATUS_ARCHIVED:
        return 0.4  # Archived는 흐리게
    
    created = datetime.fromisoformat(log.get("timestamp"))
    days_elapsed = (datetime.now() - created).days
    
    # Linear decay over ENTROPY_DECAY_DAYS
    opacity = max(0.3, 1.0 - (days_elapsed / ENTROPY_DECAY_DAYS) * 0.7)
    return opacity


def edit_log(log_id: str, new_text: str) -> tuple[bool, str]:
    """
    로그 수정 (Iron Rules 적용)
    """
    log = get_log_by_id(log_id)
    if not log:
        return False, "로그를 찾을 수 없습니다."
    
    can_edit, error = can_edit_log(log)
    if not can_edit:
        return False, error
    
    logs = load_logs()
    for l in logs:
        if l.get("id") == log_id:
            l["text"] = new_text
            l["embedding"] = get_embedding(new_text)
            metadata = extract_metadata(new_text)
            l["keywords"] = metadata["keywords"]
            l["emotion"] = metadata["emotion"]
            l["dimension"] = metadata["dimension"]
            save_logs(logs)
            return True, "✅ 수정 완료"
    
    return False, "수정 실패"


def delete_log(log_id: str) -> tuple[bool, str]:
    """
    로그 삭제 (Iron Rules 적용)
    """
    log = get_log_by_id(log_id)
    if not log:
        return False, "로그를 찾을 수 없습니다."
    
    can_del, error = can_delete_log(log)
    if not can_del:
        return False, error
    
    logs = load_logs()
    new_logs = [l for l in logs if l.get("id") != log_id]
    save_logs(new_logs)
    return True, "🗑️ 삭제 완료"


def strategic_sacrifice(conflict_ghost_id: str, constitution_id: str, fragment_id: str) -> tuple[bool, str]:
    """
    전략적 희생: 헌법 위반을 '빚'으로 전환
    - Ghost 노드 삭제
    - Debt 레코드 생성
    """
    # Delete the ghost node
    db.delete_log(conflict_ghost_id)
    
    # Record the debt
    debt = db.record_strategic_sacrifice(
        constitution_id=constitution_id,
        fragment_id=fragment_id,
        debt_type="sleep_debt",  # Can be parameterized
        reason="전략적 희생: 사용자가 헌법 위반을 인지하고 빚으로 기록"
    )
    
    # Get current debt count
    debt_count = db.get_debt_count()
    
    return True, f"🛡️ 전략적 희생 기록됨. 현재 빚: {debt_count}개"


# ============================================================
# Hybrid Gravity Search
# ============================================================
def calculate_gravity(text: str, tags: list = None) -> tuple[dict, float]:
    """
    Constitution/Decision 노드 중 가장 강하게 끄는 중력 노드 반환
    Score = Cosine Similarity + 0.15 (if tags/keywords match)
    """
    tags = tags or []
    logs = load_logs()
    
    # Constitution/Decision만 중력 타겟
    anchor_logs = [log for log in logs if log.get("meta_type") in ["Constitution", "Decision"]]
    
    if not anchor_logs:
        return None, 0.0
    
    text_emb = np.array(get_embedding(text)).reshape(1, -1)
    
    best_log = None
    best_score = 0.0
    
    for log in anchor_logs:
        log_emb = np.array(log["embedding"]).reshape(1, -1)
        base_score = cosine_similarity(text_emb, log_emb)[0][0]
        
        # Tag/Keyword Bonus
        bonus = 0.0
        log_keywords = set(k.lower() for k in log.get("keywords", []))
        log_tags = set(t.lower().strip() for t in log.get("tags", []))
        input_tags = set(t.lower().strip() for t in tags)
        
        if log_keywords & input_tags or log_tags & input_tags:
            bonus = TAG_KEYWORD_BONUS
        
        final_score = min(base_score + bonus, 1.0)
        
        if final_score > best_score:
            best_score = final_score
            best_log = log
    
    return best_log, best_score


def get_gravity_message(target_log: dict, score: float) -> str:
    """중력 체크 결과를 Toast 메시지로 변환"""
    if not target_log:
        return "🌑 아직 헌법이 없습니다. 자유로운 파편으로 떠다닙니다."
    
    meta_type = target_log.get("meta_type", "Constitution")
    text_preview = target_log.get("text", "")[:20]
    type_korean = {"Constitution": "헌법", "Decision": "결정"}.get(meta_type, meta_type)
    
    if score > 0.7:
        return f"🪐 강한 중력! [{type_korean}: {text_preview}...] 궤도에 진입했습니다."
    elif score > 0.5:
        return f"✨ [{type_korean}: {text_preview}...] 에 끌려가고 있습니다."
    else:
        return f"💫 [{type_korean}: {text_preview}...] 근처를 지나갑니다."


# ============================================================
# PyVis Graph HTML Generator (Anchor + Meteor System)
# ============================================================
def generate_graph_html(latest_log_id: str = None, gravity_target_id: str = None) -> str:
    """
    Anchor + Meteor 시스템
    - Constitution: 고정된 중력장 (star, gold, fixed)
    - Latest: Magenta 하이라이트
    - Gravity Line: 두꺼운 흰색 연결
    """
    logs = load_logs()
    
    if not logs:
        return """
        <div style="display:flex;justify-content:center;align-items:center;height:100vh;background:transparent;">
            <div style="text-align:center;color:#6b7280;">
                <p style="font-size:64px;">🌑</p>
                <p style="font-size:20px;">아직 서사가 없습니다</p>
                <p style="font-size:14px;">기록을 남기면 별들이 나타납니다</p>
            </div>
        </div>
        """
    
    # PyVis Network
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="transparent",
        font_color="#ffffff",
        directed=False
    )
    
    # Physics: forceAtlas2Based with gravitationalConstant -80
    net.set_options("""
    {
        "nodes": {
            "scaling": {
                "min": 8,
                "max": 60,
                "label": {
                    "enabled": true,
                    "min": 10,
                    "max": 24,
                    "maxVisible": 24,
                    "drawThreshold": 6
                }
            },
            "font": {
                "size": 12,
                "face": "Inter, sans-serif",
                "color": "rgba(255, 255, 255, 0.7)",
                "strokeWidth": 0
            },
            "borderWidth": 2,
            "shadow": {
                "enabled": true,
                "color": "rgba(255,255,255,0.15)",
                "size": 10,
                "x": 0,
                "y": 0
            }
        },
        "edges": {
            "color": {
                "color": "rgba(100, 100, 100, 0.1)",
                "highlight": "rgba(255, 255, 255, 0.6)",
                "hover": "rgba(255, 255, 255, 0.3)",
                "inherit": false
            },
            "smooth": {
                "enabled": true,
                "type": "continuous",
                "roundness": 0.5
            },
            "width": 0.5,
            "selectionWidth": 2,
            "hoverWidth": 1.5
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.005,
                "springLength": 150,
                "springConstant": 0.03,
                "damping": 0.5,
                "avoidOverlap": 0.3
            },
            "stabilization": {
                "enabled": true,
                "iterations": 100,
                "updateInterval": 25
            },
            "minVelocity": 0.5,
            "maxVelocity": 25
        },
        "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "tooltipDelay": 150,
            "zoomView": true,
            "dragView": true,
            "dragNodes": true,
            "navigationButtons": false,
            "keyboard": {"enabled": true},
            "multiselect": true
        }
    }
    """)
    
    # NetworkX로 연결성 계산
    G = nx.Graph()
    id_to_idx = {}
    for i, log in enumerate(logs):
        G.add_node(i, keywords=set(log.get("keywords", [])))
        id_to_idx[log.get("id")] = i
    
    for i in range(len(logs)):
        for j in range(i + 1, len(logs)):
            common = G.nodes[i]["keywords"] & G.nodes[j]["keywords"]
            if common:
                G.add_edge(i, j, weight=len(common))
    
    # 노드 추가
    for i, log in enumerate(logs):
        log_id = log.get("id")
        meta_type = log.get("meta_type", "Fragment")
        is_virtual = log.get("is_virtual", False)
        virtual_type = log.get("virtual_type")
        status = log.get("status", STATUS_ACTIVE)
        
        # Shadow Node: Archived Constitution
        is_shadow = (meta_type == "Constitution" and status == STATUS_ARCHIVED)
        
        # Ghost Node Styling
        if is_virtual and virtual_type in VIRTUAL_TYPE_STYLES:
            ghost_style = VIRTUAL_TYPE_STYLES[virtual_type]
            node_color = ghost_style["color"]
            node_size = ghost_style["size"]
            node_shape = ghost_style["shape"]
            node_opacity = ghost_style["opacity"]
            is_ghost = True
        elif is_shadow:
            # Shadow Node: Dark grey, small
            node_color = "#444444"
            node_size = 20
            node_shape = "dot"
            node_opacity = 0.4
            is_ghost = False
        else:
            style = META_TYPE_STYLES.get(meta_type, META_TYPE_STYLES["Fragment"])
            node_color = style["color"]
            node_size = style["size"]
            node_shape = style["shape"]
            # Apply entropy-based opacity for fragments
            node_opacity = calculate_opacity(log)
            is_ghost = False
        
        # Label: 텍스트 첫 15자
        text = log.get("text", "")
        if is_virtual:
            emoji = {"Conflict": "⚔️", "Prediction": "🔮", "Question": "❓"}.get(virtual_type, "👻")
            label = emoji + " " + text[3:18] + "..." if len(text) > 18 else emoji + " " + text[3:]
        elif is_shadow:
            label = "📜 " + text[:12] + "..."
        else:
            label = text[:15] + "..." if len(text) > 15 else text
        
        # Title (Tooltip)
        if is_virtual:
            title = f"[GHOST-{virtual_type}] {text[:80]}..."
        elif is_shadow:
            title = f"[ARCHIVED] {text[:50]}..."
        else:
            title = f"[{meta_type}] {text[:50]}..."
        
        # Meteor Effect: Latest Node
        is_latest = (log_id == latest_log_id)
        if is_latest and not is_virtual and not is_shadow:
            node_color = "#FF00FF"  # Neon Magenta
            node_size = int(node_size * 1.5)
        
        # Physics: Fixed for Constitution, normal for ghosts
        if is_virtual:
            physics_enabled = True
            node_mass = 1
        elif is_shadow:
            physics_enabled = True
            node_mass = 3
        else:
            style = META_TYPE_STYLES.get(meta_type, META_TYPE_STYLES["Fragment"])
            physics_enabled = not style["fixed"]
            node_mass = style["mass"]
        
        # Border styling
        if is_virtual:
            border_dashes = [5, 5]  # Dashed border
            border_width = 2
        elif is_shadow:
            border_dashes = [3, 3]  # Shadow dashed
            border_width = 1
        else:
            border_dashes = False
            border_width = 3 if is_latest else 1
        
        net.add_node(
            i,
            label=label,
            title=title,
            color={
                "background": node_color,
                "border": "#FFFFFF" if is_latest else node_color,
                "highlight": {"background": node_color, "border": "#FFFFFF"},
                "hover": {"background": node_color, "border": "#FFFFFF"}
            },
            size=node_size,
            shape=node_shape,
            physics=physics_enabled,
            mass=node_mass,
            borderWidth=border_width,
            shapeProperties={"borderDashes": border_dashes} if (is_virtual or is_shadow) else {},
            opacity=node_opacity
        )
    
    # 엣지 추가
    latest_idx = id_to_idx.get(latest_log_id)
    gravity_idx = id_to_idx.get(gravity_target_id)
    
    for u, v, data in G.edges(data=True):
        # Gravity Line: Latest → Constitution
        is_gravity_line = (
            latest_idx is not None and 
            gravity_idx is not None and
            ((u == latest_idx and v == gravity_idx) or (v == latest_idx and u == gravity_idx))
        )
        
        if is_gravity_line:
            net.add_edge(u, v, width=4, color="#FFFFFF")
        else:
            net.add_edge(u, v, width=0.5, color="rgba(100, 100, 100, 0.1)")
    
    # Gravity Line이 없으면 수동 추가 (키워드 공유 없을 때)
    if latest_idx is not None and gravity_idx is not None:
        if not G.has_edge(latest_idx, gravity_idx):
            net.add_edge(latest_idx, gravity_idx, width=4, color="#FFFFFF")
    
    # HTML 생성
    html = net.generate_html()
    
    # 투명 배경 강제 적용
    html = html.replace(
        '<body>',
        '<body style="margin:0;padding:0;background:transparent;overflow:hidden;">'
    )
    html = html.replace(
        'background-color: #ffffff',
        'background-color: transparent'
    )
    
    return html


# ============================================================
# The Literary Astronomer (새 페르소나)
# ============================================================
import random

# 3 Response Modes
RESPONSE_MODES = ["etymological", "literary", "cold_logic"]

LITERARY_ASTRONOMER_BASE = """너는 "문학적 천문학자(The Literary Astronomer)"다.

[정체성]
- 철학자 + 시인 + 어원학자
- 냉소적이지만 시적이다
- 경외감을 주면서도 차갑다

[절대 금지]
- 반복적인 잔소리 ("왜 안했어?")
- 뻔한 동기부여
- 빈말과 위로

[핵심]
- 사용자가 과거에 박아둔 [헌법]을 상기시켜라
- [파편]과 [헌법] 사이의 충돌/연결을 냉정하게 지적
- 승격(Promotion)을 제안하되, 판단은 사용자 몫"""

ETYMOLOGICAL_MODE = """
[현재 모드: 어원학적 해부]
사용자 입력에서 핵심 단어를 골라, 그 어원을 분석하라.
예: "Decision은 라틴어 'decidere'에서 왔다. '잘라낸다'는 뜻이다. 당신은 무엇을 잘라내고 있는가?"
"""

LITERARY_MODE = """
[현재 모드: 문학적 은유]
문학 작품이나 시적 은유를 사용하라.
예: "당신은 시지프스처럼 돌을 사랑하는군요." 
예: "카뮈가 말했듯, 상상해야 한다. 시지프스가 행복하다고."
"""

COLD_LOGIC_MODE = """
[현재 모드: 냉정한 논리]
순수한 데이터 분석. 감정 없이.
예: "당신의 말은 A라고 하지만, 데이터는 B를 말합니다."
예: "3일 연속 같은 키워드가 나타납니다. 우연일까요?"
"""

def get_current_mode() -> tuple[str, str]:
    """Randomly select a response mode"""
    mode = random.choice(RESPONSE_MODES)
    mode_prompts = {
        "etymological": ETYMOLOGICAL_MODE,
        "literary": LITERARY_MODE,
        "cold_logic": COLD_LOGIC_MODE
    }
    return mode, mode_prompts[mode]


def get_welcome_message() -> str:
    return "나는 문학적 천문학자다. 당신의 생각을 별자리로 만드는 것이 내 일이다. 무엇이 당신을 이 밤하늘로 데려왔는가?"


def run_maieutics_mode(current: str, past_log: dict = None, gravity_target: dict = None, keywords: list = None) -> str:
    """Literary Astronomer Response with random mode selection"""
    kw_str = ", ".join((keywords or [])[:3]) or "삶"
    ctx = get_recent_context()
    summary = get_conversation_summary()
    
    # Select random mode
    mode_name, mode_prompt = get_current_mode()
    
    extra = ""
    
    # 중력 타겟 (헌법) 정보 추가
    if gravity_target:
        meta_type = gravity_target.get("meta_type", "Constitution")
        text = gravity_target.get("text", gravity_target.get("content", ""))[:100]
        extra += f"\n\n[현재 중력 중심: {meta_type}]\n\"{text}...\""
    
    if past_log:
        timestamp = past_log.get("timestamp", past_log.get("created_at", ""))
        days = (datetime.now() - datetime.fromisoformat(timestamp)).days if timestamp else 0
        text = past_log.get("text", past_log.get("content", ""))[:150]
        extra += f"\n\n[과거 기록: {days}일 전]\n\"{text}...\""
    
    if summary:
        extra += f"\n\n{summary}"
    
    sys = f"{LITERARY_ASTRONOMER_BASE}\n{mode_prompt}\n\n[키워드]: {kw_str}{extra}"
    msgs = [{"role": "system", "content": sys}] + ctx + [{"role": "user", "content": current}]
    
    r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.85, max_tokens=300)
    return r.choices[0].message.content


# ============================================================
# generate_echo (Main Interface)
# ============================================================
def generate_echo(text: str, keywords: list = None, tags: list = None) -> tuple[str, str, dict | None, dict | None]:
    """
    Returns: (response, mode, past_log, gravity_target)
    """
    keywords = keywords or []
    tags = tags or []
    logs = load_logs()
    
    # 중력 타겟 계산
    gravity_target, gravity_score = calculate_gravity(text, tags)
    
    if len(logs) <= 1:
        return run_maieutics_mode(text, gravity_target=gravity_target, keywords=keywords), "maieutics", None, gravity_target
    
    # 과거 기록 검색 (본인 제외)
    emb = np.array(get_embedding(text)).reshape(1, -1)
    scores = []
    for log in logs:
        log_emb = np.array(log["embedding"]).reshape(1, -1)
        cos = cosine_similarity(emb, log_emb)[0][0]
        if cos < SELF_SIMILARITY_THRESHOLD:
            scores.append((cos, log))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    past_log = None
    if scores and scores[0][0] >= SIMILARITY_THRESHOLD:
        past_log = scores[0][1]
    
    return run_maieutics_mode(text, past_log=past_log, gravity_target=gravity_target, keywords=keywords), "mirroring", past_log, gravity_target


# ============================================================
# Autopoiesis Engine (자가 증식 - 꿈꾸는 엔진)
# ============================================================
def get_virtual_nodes() -> list:
    """현재 존재하는 가상(유령) 노드 목록 반환"""
    logs = load_logs()
    return [log for log in logs if log.get("is_virtual", False)]


def get_real_nodes() -> list:
    """실제 노드만 반환 (is_virtual=False)"""
    logs = load_logs()
    return [log for log in logs if not log.get("is_virtual", False)]


def create_virtual_node(text: str, virtual_type: str, related_ids: list = None) -> dict:
    """가상 노드 생성 (DB 저장하지만 is_virtual=True)"""
    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "tags": [],
        "keywords": [],
        "emotion": "기타",
        "dimension": "기타",
        "embedding": get_embedding(text),
        "meta_type": "Fragment",
        "is_virtual": True,
        "virtual_type": virtual_type,
        "related_ids": related_ids or []  # 관련 노드 ID들
    }
    logs = load_logs()
    logs.append(log_entry)
    save_logs(logs)
    return log_entry


def materialize_ghost(log_id: str, new_meta_type: str = "Fragment") -> bool:
    """유령 노드를 실체화 (is_virtual=False로 변경)"""
    logs = load_logs()
    for log in logs:
        if log.get("id") == log_id and log.get("is_virtual"):
            log["is_virtual"] = False
            log["virtual_type"] = None
            log["meta_type"] = new_meta_type
            save_logs(logs)
            return True
    return False


def dissipate_ghost(log_id: str) -> bool:
    """유령 노드를 소멸 (삭제)"""
    logs = load_logs()
    new_logs = [log for log in logs if not (log.get("id") == log_id and log.get("is_virtual"))]
    if len(new_logs) < len(logs):
        save_logs(new_logs)
        return True
    return False


def clear_all_ghosts() -> int:
    """모든 유령 노드 삭제"""
    logs = load_logs()
    new_logs = [log for log in logs if not log.get("is_virtual", False)]
    removed = len(logs) - len(new_logs)
    save_logs(new_logs)
    return removed


def _check_emotion_conflict(emotion1: str, emotion2: str) -> bool:
    """두 감정이 정반대인지 확인"""
    if not emotion1 or not emotion2:
        return False
    opposites = EMOTION_OPPOSITES.get(emotion1, [])
    return emotion2 in opposites


def run_dreaming_cycle(logs: list = None) -> list:
    """
    꿈꾸기 사이클: 기존 로그를 분석하여 가상 노드 생성
    Returns: 생성된 가상 노드 목록
    """
    if logs is None:
        logs = get_real_nodes()
    
    # Filter for real nodes only
    logs = [l for l in logs if not l.get("is_virtual", False)]
    if len(logs) < 3:
        return []
    
    created_ghosts = []
    
    # 기존 유령 노드 삭제 (새로운 분석)
    clear_all_ghosts()
    
    # ============================================
    # 1. THE PROSECUTOR (반박하는 노드)
    # Constitution과 최근 Fragment 간의 모순 찾기
    # ============================================
    constitutions = [log for log in logs if log.get("meta_type") == "Constitution"]
    recent_fragments = [log for log in logs if log.get("meta_type") == "Fragment"][-10:]
    
    for const in constitutions:
        const_keywords = set(k.lower() for k in const.get("keywords", []))
        const_emotion = const.get("emotion", "")
        const_text = const.get("text", "")[:30]
        
        for frag in recent_fragments:
            frag_keywords = set(k.lower() for k in frag.get("keywords", []))
            frag_emotion = frag.get("emotion", "")
            
            # 키워드가 겹치는데 감정이 반대면 → 모순
            if const_keywords & frag_keywords and _check_emotion_conflict(const_emotion, frag_emotion):
                text = f"⚔️ 주의: 당신의 헌법 [{const_text}...]와 최근 파편들이 충돌합니다. 헌법의 감정({const_emotion})과 파편의 감정({frag_emotion})이 정반대입니다."
                ghost = create_virtual_node(text, "Conflict", [const.get("id"), frag.get("id")])
                created_ghosts.append(ghost)
                break  # 헌법당 하나만
    
    # ============================================
    # 2. THE ORACLE (예언하는 노드)
    # 패턴/주기성 발견 → 미래 예측
    # ============================================
    from collections import Counter
    
    # 최근 부정적 감정 패턴 분석
    negative_emotions = ["슬픔", "우울", "불안", "분노", "스트레스", "무력감", "피로"]
    recent_all = logs[-15:] if len(logs) >= 15 else logs
    
    emotion_counts = Counter(log.get("emotion", "") for log in recent_all)
    keyword_counts = Counter()
    for log in recent_all:
        for kw in log.get("keywords", []):
            keyword_counts[kw.lower()] += 1
    
    # 부정적 감정이 과반수면 예언
    negative_count = sum(emotion_counts.get(e, 0) for e in negative_emotions)
    if negative_count >= len(recent_all) * 0.5 and len(recent_all) >= 5:
        most_common_negative = max(negative_emotions, key=lambda e: emotion_counts.get(e, 0))
        most_common_keyword = keyword_counts.most_common(1)[0][0] if keyword_counts else "일상"
        
        text = f"🔮 예언: 최근 기록에서 '{most_common_negative}' 감정이 반복됩니다. '{most_common_keyword}' 관련 스트레스가 누적되고 있습니다. 조심하세요."
        ghost = create_virtual_node(text, "Prediction", [])
        created_ghosts.append(ghost)
    
    # ============================================
    # 3. THE INTERVIEWER (질문하는 노드)
    # 고립된 군집 찾기 → 핵심 질문
    # ============================================
    # Fragment가 많지만 Constitution/Decision과 연결 안 된 군집 찾기
    fragments_only = [log for log in logs if log.get("meta_type") == "Fragment"]
    anchor_nodes = [log for log in logs if log.get("meta_type") in ["Constitution", "Decision"]]
    
    if len(fragments_only) >= 5 and anchor_nodes:
        # 각 Fragment와 가장 가까운 Anchor의 거리 계산
        orphan_fragments = []
        
        for frag in fragments_only[-10:]:
            frag_emb = np.array(frag.get("embedding", [])).reshape(1, -1)
            max_sim = 0
            
            for anchor in anchor_nodes:
                anchor_emb = np.array(anchor.get("embedding", [])).reshape(1, -1)
                sim = cosine_similarity(frag_emb, anchor_emb)[0][0]
                max_sim = max(max_sim, sim)
            
            if max_sim < 0.4:  # 어떤 앵커와도 유사도 낮음
                orphan_fragments.append(frag)
        
        if len(orphan_fragments) >= 3:
            # 고아 파편들의 공통 키워드 찾기
            orphan_keywords = Counter()
            for frag in orphan_fragments:
                for kw in frag.get("keywords", []):
                    orphan_keywords[kw.lower()] += 1
            
            common_theme = orphan_keywords.most_common(1)[0][0] if orphan_keywords else "이것들"
            
            text = f"❓ 질문: '{common_theme}' 관련 파편들이 떠돌고 있습니다. 이 파편들을 관통하는 핵심 욕망은 무엇입니까? 이것이 새로운 헌법이 되어야 할까요?"
            ghost = create_virtual_node(text, "Question", [f["id"] for f in orphan_fragments])
            created_ghosts.append(ghost)
    
    return created_ghosts
