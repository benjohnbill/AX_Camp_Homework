"""
narrative_logic.py
Narrative OS: The Brain
PyVis 기반 Knowledge Graph + Narrative Maieutics
"""

import os
import json
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
SIMILARITY_THRESHOLD = 0.55
SELF_SIMILARITY_THRESHOLD = 0.99
KEYWORD_MATCH_BONUS = 0.12
TAG_MATCH_BONUS = 0.08
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


# ============================================================
# 데이터 I/O
# ============================================================
def load_logs() -> list:
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []


def save_logs(logs: list) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def load_chat_history() -> list:
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []


def save_chat_history(history: list) -> None:
    os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_chat_message(role: str, content: str, metadata: dict = None) -> dict:
    history = load_chat_history()
    message = {"timestamp": datetime.now().isoformat(), "role": role, "content": content}
    if metadata:
        message["metadata"] = metadata
    history.append(message)
    save_chat_history(history)
    return message


def clear_chat_history() -> None:
    save_chat_history([])


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


def save_log(text: str, user_tags: list = None) -> dict:
    embedding = get_embedding(text)
    metadata = extract_metadata(text)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text, "tags": user_tags or [],
        "keywords": metadata["keywords"], "emotion": metadata["emotion"],
        "dimension": metadata["dimension"], "embedding": embedding
    }
    logs = load_logs()
    logs.append(log_entry)
    save_logs(logs)
    return log_entry


# ============================================================
# PyVis Graph HTML Generator (Obsidian Living Universe)
# ============================================================
def generate_graph_html() -> str:
    """
    Obsidian 스타일의 '살아있는 우주' 그래프 생성
    - 무중력 물리엔진 (stabilization OFF)
    - 네온 노드 + 은은한 엣지
    - 투명 배경 (CSS Gradient용)
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
    
    # PyVis Network (투명 배경)
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="transparent",
        font_color="#ffffff",
        directed=False
    )
    
    # 극한 튜닝: Obsidian Style Living Universe
    # - 작은 노드 (Stars)
    # - 아주 얇은 엣지 (Gravity Lines)
    # - 광활한 물리 엔진 (Expanding Universe)
    net.set_options("""
    {
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 4,
                "max": 12,
                "label": {
                    "enabled": true,
                    "min": 10,
                    "max": 20,
                    "maxVisible": 20,
                    "drawThreshold": 8
                }
            },
            "font": {
                "size": 12,
                "face": "Inter, sans-serif",
                "color": "rgba(255, 255, 255, 0.6)",
                "strokeWidth": 0
            },
            "borderWidth": 0,
            "shadow": {
                "enabled": true,
                "color": "rgba(255,255,255,0.1)",
                "size": 5,
                "x": 0,
                "y": 0
            }
        },
        "edges": {
            "color": {
                "color": "rgba(100, 100, 100, 0.15)",
                "highlight": "rgba(255, 255, 255, 0.4)",
                "hover": "rgba(255, 255, 255, 0.2)",
                "inherit": false
            },
            "smooth": {
                "enabled": true,
                "type": "continuous",
                "roundness": 0.5
            },
            "width": 0.5,
            "selectionWidth": 1.5,
            "hoverWidth": 1.0
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -40,
                "centralGravity": 0.003,
                "springLength": 120,
                "springConstant": 0.04,
                "damping": 0.4,
                "avoidOverlap": 0.2
            },
            "stabilization": {
                "enabled": true,
                "iterations": 80,
                "updateInterval": 20
            },
            "minVelocity": 0.75,
            "maxVelocity": 30
        },
        "interaction": {
            "hover": true,
            "hoverConnectedEdges": true,
            "tooltipDelay": 200,
            "zoomView": true,
            "dragView": true,
            "dragNodes": true,
            "navigationButtons": false,
            "keyboard": {
                "enabled": true
            },
            "multiselect": true
        }
    }
    """)
    
    # NetworkX로 연결성 계산
    G = nx.Graph()
    for i, log in enumerate(logs):
        G.add_node(i, keywords=set(log.get("keywords", [])))
    
    for i in range(len(logs)):
        for j in range(i + 1, len(logs)):
            common = G.nodes[i]["keywords"] & G.nodes[j]["keywords"]
            if common:
                G.add_edge(i, j, weight=len(common))
    
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    # 노드 추가
    for i, log in enumerate(logs):
        dimension = log.get("dimension", "기타")
        node_color = DIMENSION_COLORS.get(dimension, DIMENSION_COLORS["기타"])
        
        # Scaling을 위한 Value 설정 (연결성에 비례)
        degree = degrees.get(i, 0)
        value = 2 + (degree / max_degree) * 8 if max_degree > 0 else 3
        
        net.add_node(
            i,
            label=label,
            # title 제거: 호버 시 툴팁 없음
            color={
                "background": node_color,
                "border": node_color,
                "highlight": {"background": node_color, "border": "#ffffff"},
                "hover": {"background": node_color, "border": "#ffffff"}
            },
            value=value, # size 대신 value 사용해야 scaling 작동
            shape="dot"
        )
    
    # 엣지 추가
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, width=1)
    
    # HTML 생성
    html = net.generate_html()
    
    # 투명 배경 강제 적용
    html = html.replace(
        '<body>',
        '<body style="margin:0;padding:0;background:transparent;overflow:hidden;">'
    )
    
    # canvas 배경도 투명하게
    html = html.replace(
        'background-color: #ffffff',
        'background-color: transparent'
    )
    
    return html


# ============================================================
# Hybrid Search
# ============================================================
def calculate_keyword_bonus(c: list, p: list) -> float:
    if not c or not p:
        return 0.0
    inter = set(k.lower() for k in c) & set(k.lower() for k in p)
    return min(len(inter) * KEYWORD_MATCH_BONUS, 0.20) if inter else 0.0


def calculate_tag_bonus(c: list, p: list) -> float:
    if not c or not p:
        return 0.0
    return TAG_MATCH_BONUS if set(t.lower().strip() for t in c) & set(t.lower().strip() for t in p) else 0.0


def calculate_days_diff(ts: str) -> int:
    return (datetime.now() - datetime.fromisoformat(ts)).days


def get_temporal_context(d: int) -> str:
    if d < 7:
        return "최근 기록. 감정 지속성 확인."
    elif d <= 30:
        return "한 달 이내. 패턴 반복 확인."
    elif d <= 365:
        return "오래된 기록. 근본적 변화 확인."
    return "1년+. 영원회귀 질문."


# ============================================================
# Maieutics
# ============================================================
MAIEUTICS_PROMPT = """너는 "Narrative Maieutician" (서사적 산파술사)이다.

[판단] 매 응답마다 선택:
1. 해석: 모순/패턴 지적
2. 질문: 필요시 1개만
3. 종결: 행동 유도

[연결] 이전 맥락과 연결. 모순 직접 지적.
[톤] 직설적. 3문장 이내. 빈말 금지."""


def get_welcome_message() -> str:
    return "나는 서사적 산파다. 당신의 생각을 꺼내는 것이 내 일이다. 무엇이 당신을 여기로 데려왔는가?"


def run_maieutics_mode(current: str, past_log: dict = None, days_diff: int = 0, keywords: list = None) -> str:
    kw_str = ", ".join((keywords or [])[:3]) or "삶"
    ctx = get_recent_context()
    summary = get_conversation_summary()
    
    extra = ""
    if past_log:
        extra = f"\n\n[과거 기록: {days_diff}일 전]\n\"{past_log['text'][:150]}...\"\n[맥락]: {get_temporal_context(days_diff)}"
    if summary:
        extra += f"\n\n{summary}"

    sys = f"{MAIEUTICS_PROMPT}\n\n[키워드]: {kw_str}{extra}"
    msgs = [{"role": "system", "content": sys}] + ctx + [{"role": "user", "content": current}]
    
    r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.75, max_tokens=200)
    return r.choices[0].message.content


# ============================================================
# generate_echo
# ============================================================
def generate_echo(text: str, keywords: list = None, tags: list = None) -> tuple[str, str, dict | None, list]:
    keywords = keywords or []
    tags = tags or []
    logs = load_logs()
    
    if len(logs) <= 1:
        return run_maieutics_mode(text, keywords=keywords), "maieutics", None, keywords
    
    emb = np.array(get_embedding(text)).reshape(1, -1)
    scores = []
    for log in logs:
        log_emb = np.array(log["embedding"]).reshape(1, -1)
        cos = cosine_similarity(emb, log_emb)[0][0]
        final = min(cos + calculate_keyword_bonus(keywords, log.get("keywords", [])) + calculate_tag_bonus(tags, log.get("tags", [])), 1.0)
        scores.append((final, cos, log))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    best_log, best_score = None, 0.0
    for f, c, log in scores:
        if c < SELF_SIMILARITY_THRESHOLD:
            best_score, best_log = f, log
            break
    
    if not best_log or best_score < SIMILARITY_THRESHOLD:
        return run_maieutics_mode(text, keywords=keywords), "maieutics", None, keywords
    
    days = calculate_days_diff(best_log['timestamp'])
    return run_maieutics_mode(text, past_log=best_log, days_diff=days, keywords=keywords), "mirroring", best_log, keywords
