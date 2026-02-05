"""
app.py
Antigravity Narrative OS: 3-Mode Architecture
Stream (Chat) | Universe (Graph) | Desk (Editor)
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime
import narrative_logic as logic
import db_manager as db

# ============================================================
# Initialization
# ============================================================
# Inject Genesis Data on first run
db.inject_genesis_data(logic.get_embedding)

# Auto-trigger Dreaming Cycle to detect initial conflicts
if 'genesis_dreaming_done' not in st.session_state:
    logs = logic.load_logs()
    if logs:
        created_ghosts = logic.run_dreaming_cycle(logs)
        ghost_count = len(created_ghosts)
        if ghost_count > 0:
            st.toast(f"🌌 시스템이 {ghost_count}개의 충돌을 감지했습니다.", icon="⚔️")
    st.session_state['genesis_dreaming_done'] = True


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Antigravity",
    page_icon="🌌",
    layout="wide"
)


# ============================================================
# Session State
# ============================================================
if 'mode' not in st.session_state:
    st.session_state['mode'] = "stream"  # stream | universe | desk

if 'messages' not in st.session_state:
    saved = logic.load_chat_history()
    if saved:
        st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved]
    else:
        welcome = logic.get_welcome_message()
        st.session_state.messages = [{"role": "assistant", "content": welcome}]
        logic.save_chat_message("assistant", welcome)

if 'selected_cards' not in st.session_state:
    st.session_state['selected_cards'] = []

if 'last_log_id' not in st.session_state:
    st.session_state['last_log_id'] = None

if 'gravity_target_id' not in st.session_state:
    st.session_state['gravity_target_id'] = None

if 'pending_promotion' not in st.session_state:
    st.session_state['pending_promotion'] = None


# ============================================================
# Global CSS: The Breathing Universe
# ============================================================
st.markdown("""
<style>
    /* Deep Universe Background */
    .stApp {
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        position: relative;
    }
    
    /* Twinkling Stars Animation */
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    @keyframes move-stars {
        from { background-position: 0 0; }
        to { background-position: -10000px 5000px; }
    }
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(2px 2px at 160px 120px, rgba(255,255,255,0.9), transparent),
            radial-gradient(1px 1px at 230px 80px, #fff, transparent),
            radial-gradient(2px 2px at 300px 180px, rgba(255,255,255,0.7), transparent);
        background-repeat: repeat;
        background-size: 350px 350px;
        animation: move-stars 200s linear infinite, twinkle 3s ease-in-out infinite;
        z-index: 0;
        opacity: 0.6;
        pointer-events: none;
    }
    
    /* Ensure content is above stars */
    .main .block-container {
        position: relative;
        z-index: 1;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 15px 0;
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #e94560, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 3px;
        margin-bottom: 5px;
    }
    
    /* Card Styling for Desk */
    .card-item {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .card-item:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(233, 69, 96, 0.3);
    }
    
    .card-selected {
        border-color: #FFD700 !important;
        background: rgba(255,215,0,0.1) !important;
    }
    
    /* Type badges */
    .type-fragment { background: rgba(240,240,240,0.1); color: #F0F0F0; }
    .type-constitution { background: rgba(255,215,0,0.15); color: #FFD700; }
    .type-decision { background: rgba(0,255,127,0.15); color: #00FF7F; }
    .type-thirst { background: rgba(255,107,53,0.15); color: #FF6B35; }
    
    .type-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 10px;
        font-weight: 500;
    }
    
    /* Ghost styling */
    .ghost-conflict { color: #FF0055; }
    .ghost-prediction { color: #9D00FF; }
    .ghost-question { color: #00FFFF; }
    
    /* Mode indicator */
    .mode-indicator {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        z-index: 1000;
        background: rgba(255,255,255,0.1);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR: Mode Switcher & System Info
# ============================================================
with st.sidebar:
    st.markdown("### 🌌 ANTIGRAVITY")
    st.markdown("---")
    
    # 3-Mode Radio Selector
    mode = st.radio(
        "Mode",
        ["🌊 Stream", "🌌 Universe", "🖊️ Desk"],
        index=["stream", "universe", "desk"].index(st.session_state['mode']),
        label_visibility="collapsed"
    )
    
    # Update mode state
    mode_map = {"🌊 Stream": "stream", "🌌 Universe": "universe", "🖊️ Desk": "desk"}
    if mode_map[mode] != st.session_state['mode']:
        st.session_state['mode'] = mode_map[mode]
        st.rerun()
    
    st.markdown("---")
    
    # Statistics
    logs = logic.load_logs()
    constitution_count = len([l for l in logs if l.get("meta_type") == "Constitution" and not l.get("is_virtual")])
    decision_count = len([l for l in logs if l.get("meta_type") == "Decision" and not l.get("is_virtual")])
    fragment_count = len([l for l in logs if l.get("meta_type") == "Fragment" and not l.get("is_virtual")])
    thirst_count = len([l for l in logs if l.get("meta_type") == "Thirst" and not l.get("is_virtual")])
    
    c1, c2 = st.columns(2)
    c1.metric("⭐ 헌법", constitution_count)
    c2.metric("◆ 결정", decision_count)
    c1.metric("● 파편", fragment_count)
    c2.metric("▲ 갈증", thirst_count)
    
    # Debt Counter
    debt_count = db.get_debt_count()
    if debt_count > 0:
        st.warning(f"🛡️ 미상환 빚: {debt_count}개")
    
    st.markdown("---")
    
    # Ghost Nodes (Autopoiesis)
    st.markdown("#### 👻 유령 노드")
    ghost_nodes = logic.get_virtual_nodes()
    
    if st.button(f"🌙 시스템 꿈꾸기 ({len(ghost_nodes)})", use_container_width=True):
        with st.spinner("꿈을 꾸는 중..."):
            created = logic.run_dreaming_cycle()
            if created:
                st.toast(f"👻 {len(created)}개의 유령 노드가 생성되었습니다!", icon="🌙")
            else:
                st.toast("💤 꿈에서 아무것도 발견하지 못했습니다.", icon="😴")
        st.rerun()
    
    # Ghost List
    for ghost in ghost_nodes:
        v_type = ghost.get("virtual_type", "Unknown")
        emoji = {"Conflict": "⚔️", "Prediction": "🔮", "Question": "❓"}.get(v_type, "👻")
        color_class = {"Conflict": "ghost-conflict", "Prediction": "ghost-prediction", "Question": "ghost-question"}.get(v_type, "")
        
        with st.expander(f"{emoji} {v_type}", expanded=False):
            content = ghost.get("text", ghost.get("content", ""))[:80]
            st.markdown(f"<span class='{color_class}'>{content}...</span>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 수용", key=f"mat_{ghost['id']}", use_container_width=True):
                    logic.materialize_ghost(ghost['id'], "Thirst")
                    st.toast("✨ 유령이 실체화되었습니다!", icon="✅")
                    st.rerun()
            with col2:
                if st.button("❌ 기각", key=f"dis_{ghost['id']}", use_container_width=True):
                    logic.dissipate_ghost(ghost['id'])
                    st.toast("💨 유령이 소멸되었습니다.", icon="❌")
                    st.rerun()
            
            # Strategic Sacrifice for Conflicts
            if v_type == "Conflict":
                if st.button("🛡️ 전략적 희생", key=f"sac_{ghost['id']}", use_container_width=True,
                            help="헌법 위반을 인정하고 '빚'으로 기록"):
                    constitutions = db.get_logs_by_type("Constitution")
                    const_id = constitutions[0]["id"] if constitutions else None
                    success, msg = logic.strategic_sacrifice(ghost['id'], const_id, ghost['id'])
                    st.toast(msg, icon="🛡️")
                    st.rerun()
    
    st.markdown("---")
    
    # Reset buttons
    if st.button("🗑️ 전체 초기화", type="secondary", use_container_width=True):
        logic.save_logs([])
        logic.clear_chat_history()
        db.clear_virtual_nodes()
        st.session_state.messages = []
        st.session_state['selected_cards'] = []
        st.session_state['last_log_id'] = None
        st.session_state['gravity_target_id'] = None
        st.rerun()


# ============================================================
# MODE 1: THE STREAM (Chat-First)
# ============================================================
if st.session_state['mode'] == "stream":
    st.markdown("""
    <div class="main-header">
        <h1>🌊 THE STREAM</h1>
        <p style="color: #6b7280;">Fast capture. Hot state. Think aloud.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Promotion Handler
    if st.session_state.get('pending_promotion'):
        promo = st.session_state['pending_promotion']
        st.info(f"💡 승격 제안: 이 파편을 {promo['suggested_type']}으로 승격하시겠습니까?")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("⭐ 헌법으로", key="promo_const"):
                logic.promote_log(promo['log_id'], "Constitution")
                st.toast("⭐ 헌법으로 승격되었습니다!", icon="⭐")
                st.session_state['pending_promotion'] = None
                st.rerun()
        with col2:
            if st.button("◆ 결정으로", key="promo_dec"):
                logic.promote_log(promo['log_id'], "Decision")
                st.toast("◆ 결정으로 승격되었습니다!", icon="◆")
                st.session_state['pending_promotion'] = None
                st.rerun()
        with col3:
            if st.button("▲ 갈증으로", key="promo_thirst"):
                logic.promote_log(promo['log_id'], "Thirst")
                st.toast("▲ 갈증으로 승격되었습니다!", icon="▲")
                st.session_state['pending_promotion'] = None
                st.rerun()
        with col4:
            if st.button("❌ 무시", key="promo_skip"):
                st.session_state['pending_promotion'] = None
                st.rerun()
    
    # Chat Input
    if user_input := st.chat_input("무엇이 당신을 여기로 데려왔는가?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        logic.save_chat_message("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Save as Fragment first
        log_entry = logic.save_log(user_input)
        st.session_state['last_log_id'] = log_entry['id']
        
        # Calculate gravity
        gravity_target, gravity_score = logic.calculate_gravity(user_input, [])
        if gravity_target:
            st.session_state['gravity_target_id'] = gravity_target.get('id')
            const_name = (gravity_target.get("text", gravity_target.get("content", "")))[:30]
            st.toast(f"🪐 '{const_name}...'와 궤도 연결됨", icon="🪐")
        
        # Smart Response Logic
        is_simple = len(user_input.strip()) < 30 or user_input.strip().endswith(('.','.'))
        
        if is_simple:
            # Simple input -> Silent save + Toast only
            st.toast("💾 저장됨.", icon="💾")
            response = None
        else:
            # Deep input -> Generate response with Persona
            response, mode, past_log, grav = logic.generate_echo(user_input)
            
            # Display response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            logic.save_chat_message("assistant", response)
            
            # Check for promotion suggestion
            if "헌법" in response or "결정" in response or "승격" in response:
                st.session_state['pending_promotion'] = {
                    'log_id': log_entry['id'],
                    'suggested_type': 'Constitution' if "헌법" in response else 'Decision'
                }
        
        st.rerun()


# ============================================================
# MODE 2: THE UNIVERSE (Graph View)
# ============================================================
elif st.session_state['mode'] == "universe":
    st.markdown("""
    <div class="main-header">
        <h1>🌌 THE UNIVERSE</h1>
        <p style="color: #6b7280;">Contemplation. Timeless. See the whole.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate and display graph
    graph_html = logic.generate_graph_html(
        latest_log_id=st.session_state.get('last_log_id'),
        gravity_target_id=st.session_state.get('gravity_target_id')
    )
    
    components.html(graph_html, height=700, scrolling=False)
    
    # Legend
    legend_items = [
        ("⭐", "#FFD700", "헌법"),
        ("◆", "#00FF7F", "결정"),
        ("▲", "#FF6B35", "갈증"),
        ("●", "#F0F0F0", "파편"),
        ("●", "#FF00FF", "최신"),
        ("⚔️", "#FF0055", "충돌"),
        ("🔮", "#9D00FF", "예언"),
    ]
    
    legend_html = '<div style="display:flex;gap:20px;justify-content:center;margin-top:10px;">'
    for symbol, color, name in legend_items:
        legend_html += f'<span style="color:{color};">{symbol} {name}</span>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)


# ============================================================
# MODE 3: THE DESK (Editor)
# ============================================================
elif st.session_state['mode'] == "desk":
    st.markdown("""
    <div class="main-header">
        <h1>🖊️ THE DESK</h1>
        <p style="color: #6b7280;">Narrative creation. Cool state. Write essays.</p>
    </div>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1.5])
    
    # LEFT: Card Selection
    with left_col:
        st.markdown("### 📚 카드 선택")
        st.markdown("*에세이에 포함할 카드를 선택하세요*")
        
        logs = logic.load_logs()
        real_logs = [l for l in logs if not l.get("is_virtual")]
        
        for log in real_logs[:20]:  # Show last 20
            meta_type = log.get("meta_type", "Fragment")
            type_class = f"type-{meta_type.lower()}"
            content = log.get("text", log.get("content", ""))[:60]
            log_id = log.get("id")
            
            is_selected = log_id in st.session_state['selected_cards']
            selected_class = "card-selected" if is_selected else ""
            
            # Card display
            type_emoji = {"Constitution": "⭐", "Decision": "◆", "Thirst": "▲", "Fragment": "●"}.get(meta_type, "●")
            
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="card-item {selected_class}">
                    <span class="type-badge {type_class}">{type_emoji} {meta_type}</span>
                    <p style="margin:5px 0 0 0;color:#ccc;font-size:13px;">{content}...</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if is_selected:
                    if st.button("➖", key=f"desel_{log_id}"):
                        st.session_state['selected_cards'].remove(log_id)
                        st.rerun()
                else:
                    if st.button("➕", key=f"sel_{log_id}"):
                        st.session_state['selected_cards'].append(log_id)
                        st.rerun()
    
    # RIGHT: Essay Editor
    with right_col:
        st.markdown("### ✍️ 에세이 작성")
        
        # Show selected cards summary
        if st.session_state['selected_cards']:
            st.markdown(f"*{len(st.session_state['selected_cards'])}개 카드 선택됨*")
            
            # Compile selected content
            selected_content = []
            for log_id in st.session_state['selected_cards']:
                log = logic.get_log_by_id(log_id)
                if log:
                    text = log.get("text", log.get("content", ""))
                    selected_content.append(f"> {text[:100]}...")
            
            with st.expander("선택된 카드 미리보기", expanded=False):
                st.markdown("\n\n".join(selected_content))
        
        # Essay text area
        essay = st.text_area(
            "에세이를 작성하세요",
            height=400,
            placeholder="선택한 카드들을 바탕으로 에세이를 작성하세요...",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 저장", use_container_width=True):
                if essay.strip():
                    log_entry = logic.save_log(essay)
                    st.toast("💾 에세이가 저장되었습니다!", icon="✍️")
                    st.session_state['selected_cards'] = []
                    st.rerun()
        with col2:
            if st.button("🗑️ 초기화", use_container_width=True):
                st.session_state['selected_cards'] = []
                st.rerun()


# ============================================================
# Mode Indicator (Bottom Right)
# ============================================================
mode_names = {"stream": "🌊 STREAM", "universe": "🌌 UNIVERSE", "desk": "🖊️ DESK"}
st.markdown(f"""
<div class="mode-indicator">
    {mode_names[st.session_state['mode']]}
</div>
""", unsafe_allow_html=True)
