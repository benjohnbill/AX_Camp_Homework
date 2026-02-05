"""
app.py
Narrative OS: Single Page, Dual View
PyVis Full-Screen Obsidian Universe
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import narrative_logic as logic
import streamlit as st
import os

def get_api_key():
    # 1. 우선순위: 시스템(secrets 또는 .env)에 키가 있는지 먼저 확인
    # 로컬에서는 secrets.toml을 읽어서 자동으로 작동함
    system_key = st.secrets.get("OPENAI_API_KEY")

    if system_key:
        return system_key
    
    # 2. 후순위: 시스템에 키가 없다면(배포 서버), 사용자에게 입력창 제시
    # 사이드바에 입력창을 만듭니다.
    user_key = st.sidebar.text_input(
        "OpenAI API Key를 입력하세요 (BYOK)", 
        type="password",
        help="본인의 API Key를 사용합니다. 저장되지 않습니다."
    )
    
    if user_key:
        return user_key
    
    # 3. 키가 아예 없으면 멈춤
    st.info("⚠️ 작동하려면 API Key가 필요합니다. 사이드바에 키를 입력해주세요.")
    st.stop()

# --- 메인 로직 시작 ---
api_key = get_api_key()

# 클라이언트 생성 (이제 api_key는 무조건 존재함)
from openai import OpenAI
client = OpenAI(api_key=api_key)

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Narrative OS",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# Session State
# ============================================================
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = "main"

if 'messages' not in st.session_state:
    saved = logic.load_chat_history()
    if saved:
        st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved]
    else:
        welcome = logic.get_welcome_message()
        st.session_state.messages = [{"role": "assistant", "content": welcome}]
        logic.save_chat_message("assistant", welcome)

if 'last_metadata' not in st.session_state:
    st.session_state['last_metadata'] = {}


# ============================================================
# GRAPH VIEW (Full Screen Obsidian Universe)
# ============================================================
if st.session_state['view_mode'] == "graph":
    
    # Full-screen CSS Hack with Radial Gradient Universe
    st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at center, #1a1a2e 0%, #0d0d15 50%, #000000 100%);
        }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        header {visibility: hidden !important; height: 0 !important;}
        footer {visibility: hidden !important; height: 0 !important;}
        .stSidebar {display: none !important;}
        #MainMenu {visibility: hidden !important;}
        
        /* Graph iframe 전체 화면 */
        iframe {
            background: transparent !important;
        }
        
        .floating-back-btn {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 9999;
            background: rgba(30, 30, 50, 0.9);
            border: 1px solid rgba(233, 69, 96, 0.5);
            color: #e94560;
            padding: 12px 24px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }
        
        .floating-back-btn:hover {
            background: rgba(233, 69, 96, 0.2);
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(233, 69, 96, 0.3);
        }
        
        .legend-bar {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            background: rgba(20, 20, 35, 0.9);
            border: 1px solid rgba(255,255,255,0.15);
            padding: 12px 30px;
            border-radius: 30px;
            backdrop-filter: blur(15px);
            display: flex;
            gap: 25px;
            box-shadow: 0 0 30px rgba(0,0,0,0.5);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ccc;
            font-size: 12px;
            font-weight: 500;
        }
        
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            box-shadow: 0 0 8px currentColor;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Floating Back Button (Streamlit 버튼 대신 폼 사용)
    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        if st.button("⬅️ 복귀", key="back_btn"):
            st.session_state['view_mode'] = "main"
            st.rerun()
    
    # PyVis Graph HTML 렌더링
    graph_html = logic.generate_graph_html()
    components.html(graph_html, height=900, scrolling=False)
    
    # Legend Bar
    legend_html = '<div class="legend-bar">'
    for dim, color in logic.DIMENSION_COLORS.items():
        legend_html += f'<div class="legend-item"><div class="legend-dot" style="background:{color};"></div>{dim}</div>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)


# ============================================================
# MAIN VIEW (기록 + 채팅)
# ============================================================
else:
    # Main View 스타일
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #0a0a12 0%, #12121f 50%, #0f1524 100%);
        }
        
        .main-header {
            text-align: center;
            padding: 15px 0 25px 0;
        }
        
        .main-header h1 {
            background: linear-gradient(90deg, #e94560, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: 4px;
        }
        
        .section-title {
            color: #9ca3af;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .keyword-tag {
            display: inline-block;
            background: rgba(233, 69, 96, 0.15);
            color: #e94560;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin: 2px;
        }
        
        .emotion-tag {
            display: inline-block;
            background: rgba(155, 89, 182, 0.15);
            color: #9b59b6;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin: 2px;
        }
        
        .dimension-tag {
            display: inline-block;
            background: rgba(52, 152, 219, 0.15);
            color: #3498db;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin: 2px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("### 🧠 Narrative OS")
        st.markdown("---")
        
        logs = logic.load_logs()
        
        # Universe 진입 버튼
        if st.button("🌌 서사 우주 진입", type="primary", use_container_width=True):
            st.session_state['view_mode'] = "graph"
            st.rerun()
        
        st.markdown("---")
        
        # 통계
        st.markdown("#### 📈 서사 통계")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("서사", f"{len(logs)}개")
        with c2:
            st.metric("대화", f"{len(st.session_state.messages)}개")
        
        if logs:
            emotions = [l.get("emotion", "기타") for l in logs]
            for e, c in pd.Series(emotions).value_counts().head(3).items():
                st.caption(f"• {e}: {c}개")
        
        st.markdown("---")
        
        if st.button("🗑️ 전체 초기화", type="secondary", use_container_width=True):
            logic.save_logs([])
            logic.clear_chat_history()
            st.session_state.messages = []
            st.session_state['last_metadata'] = {}
            st.rerun()
        
        if st.button("💬 대화만 초기화", use_container_width=True):
            logic.clear_chat_history()
            w = logic.get_welcome_message()
            st.session_state.messages = [{"role": "assistant", "content": w}]
            logic.save_chat_message("assistant", w)
            st.rerun()
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🧠 NARRATIVE OS</h1>
        <p style="color: #6b7280;">지능형 서사 연결망</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2 Column Layout
    left, right = st.columns([1, 1], gap="large")
    
    # Left: 기록
    with left:
        st.markdown('<p class="section-title">✍️ 오늘의 기록</p>', unsafe_allow_html=True)
        
        text = st.text_area("기록", height=200, placeholder="생각을 적어보세요...", label_visibility="collapsed")
        tags = st.text_input("태그", placeholder="#의지 #계획 (선택)")
        
        if st.button("🔗 기록 및 서사 연결", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("내용을 입력하세요.")
            else:
                with st.spinner("🧠"):
                    tag_list = [t.strip() for t in tags.replace("#", " #").split("#") if t.strip()]
                    saved = logic.save_log(text, tag_list)
                    
                    st.session_state.messages.append({"role": "user", "content": text})
                    logic.save_chat_message("user", text, metadata={
                        "keywords": saved.get("keywords", []),
                        "emotion": saved.get("emotion", ""),
                        "dimension": saved.get("dimension", "")
                    })
                    
                    echo, mode, _, kw = logic.generate_echo(
                        text, keywords=saved.get("keywords", []), tags=tag_list
                    )
                    
                    st.session_state.messages.append({"role": "assistant", "content": echo})
                    logic.save_chat_message("assistant", echo)
                    
                    st.session_state['last_metadata'] = {
                        "keywords": saved.get("keywords", []),
                        "emotion": saved.get("emotion", ""),
                        "dimension": saved.get("dimension", ""),
                        "mode": mode
                    }
                st.rerun()
    
    # Right: 채팅
    with right:
        st.markdown('<p class="section-title">🪞 서사의 메아리</p>', unsafe_allow_html=True)
        
        m = st.session_state.get('last_metadata', {})
        if m.get('keywords') or m.get('emotion') or m.get('dimension'):
            h = ""
            for k in m.get('keywords', [])[:4]:
                h += f'<span class="keyword-tag">{k}</span>'
            if m.get('emotion'):
                h += f'<span class="emotion-tag">😶 {m["emotion"]}</span>'
            if m.get('dimension'):
                h += f'<span class="dimension-tag">📂 {m["dimension"]}</span>'
            st.markdown(h, unsafe_allow_html=True)
            st.markdown("")
        
        chat = st.container(height=350)
        with chat:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        if prompt := st.chat_input("생각을 이어가세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            logic.save_chat_message("user", prompt)
            
            with st.spinner("🧠"):
                meta = logic.extract_metadata(prompt)
                echo, mode, _, _ = logic.generate_echo(prompt, keywords=meta.get("keywords", []))
                st.session_state.messages.append({"role": "assistant", "content": echo})
                logic.save_chat_message("assistant", echo)
                st.session_state['last_metadata'] = {
                    "keywords": meta.get("keywords", []),
                    "emotion": meta.get("emotion", ""),
                    "dimension": meta.get("dimension", ""),
                    "mode": mode
                }
            st.rerun()
