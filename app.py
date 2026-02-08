"""
app.py
Antigravity v4: 3-Mode Architecture
Stream (Chat) | Universe (Graph) | Desk (Editor)
Gatekeeper Modal + Red Protocol + Silent Zoom
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import narrative_logic as logic
import db_manager as db

# ============================================================
# Page Config (MUST be first)
# ============================================================
st.set_page_config(
    page_title="Antigravity",
    page_icon="🌌",
    layout="wide"
)

# ============================================================
# Initialization
# ============================================================
# Initialize database and inject Genesis Data
db.inject_genesis_data(logic.get_embedding)

# Update streak on login
if 'streak_updated' not in st.session_state:
    streak_info = logic.update_streak()
    st.session_state['streak_info'] = streak_info
    st.session_state['streak_updated'] = True

# ============================================================
# Session State
# ============================================================
if 'mode' not in st.session_state:
    st.session_state['mode'] = "stream"

if 'messages' not in st.session_state:
    saved = logic.load_chat_history()
    if saved:
        st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved]
    else:
        welcome = logic.get_welcome_message()
        st.session_state.messages = [{"role": "assistant", "content": welcome}]
        logic.save_chat_message("assistant", welcome)

if 'gatekeeper_dismissed' not in st.session_state:
    st.session_state['gatekeeper_dismissed'] = False

if 'first_input_of_session' not in st.session_state:
    st.session_state['first_input_of_session'] = True

if 'selected_cards' not in st.session_state:
    st.session_state['selected_cards'] = []

# ============================================================
# Check Red Mode
# ============================================================
red_mode = logic.is_red_mode()

# ============================================================
# CSS: The Atmosphere
# ============================================================
if red_mode:
    background_css = """
        .stApp {
            background: linear-gradient(to bottom, #330000 0%, #000000 100%) !important;
        }
        @keyframes jitter {
            0%, 100% { transform: translate(0, 0); }
            25% { transform: translate(-2px, 1px); }
            50% { transform: translate(1px, -2px); }
            75% { transform: translate(-1px, 2px); }
        }
        iframe {
            animation: jitter 0.15s infinite;
        }
    """
else:
    background_css = """
        .stApp {
            background-color: #0e1117;
            background-image: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
            background-attachment: fixed;
        }
    """

st.markdown(f"""
<style>
    {background_css}
    
    /* Twinkling Stars */
    @keyframes twinkle {{
        0%, 100% {{ opacity: 0.3; }}
        50% {{ opacity: 1; }}
    }}
    
    @keyframes move-stars {{
        from {{ background-position: 0 0; }}
        to {{ background-position: -10000px 5000px; }}
    }}
    
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(2px 2px at 160px 120px, rgba(255,255,255,0.9), transparent);
        background-repeat: repeat;
        background-size: 350px 350px;
        animation: move-stars 200s linear infinite, twinkle 3s ease-in-out infinite;
        z-index: 0;
        opacity: 0.6;
        pointer-events: none;
    }}
    
    /* Hide Header/Footer */
    header {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 8rem;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(14, 17, 23, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}
    
    /* Bottom input area transparent */
    section[data-testid="stBottom"] {{
        background: transparent !important;
    }}
    
    /* Text readability */
    h1, h2, h3, h4, h5, h6, p, li, span, div {{
        color: #ffffff !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        background-color: rgba(30, 35, 45, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }}
    
    .stButton > button:hover {{
        background-color: rgba(233, 69, 96, 0.3) !important;
        border-color: rgba(233, 69, 96, 0.6) !important;
    }}
    
    /* Chat input */
    .stChatInput input {{
        background-color: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-radius: 20px !important;
    }}
    
    /* Text areas */
    .stTextArea textarea {{
        background-color: rgba(20, 25, 35, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }}
    
    /* Streak counter */
    .streak-counter {{
        font-size: 24px;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        padding: 10px;
        background: rgba(255, 215, 0, 0.1);
        border-radius: 10px;
        margin-bottom: 15px;
    }}
    
    /* Red mode warning */
    .red-warning {{
        background: rgba(255, 0, 0, 0.2);
        border: 2px solid #ff0000;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }}
    
    /* Gatekeeper modal */
    .gatekeeper-modal {{
        background: rgba(0, 0, 0, 0.95);
        border: 2px solid #ff4444;
        border-radius: 15px;
        padding: 30px;
        margin: 50px auto;
        max-width: 600px;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GATEKEEPER MODAL (Entry Ritual)
# ============================================================
if not st.session_state.get('gatekeeper_dismissed'):
    report = logic.run_gatekeeper()
    
    if report['conflicts'] or report.get('broken_promise'):
        st.markdown("""
        <div class="gatekeeper-modal">
            <h2 style="color: #ff4444; text-align: center;">⚠️ DREAM REPORT</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.error(f"🌙 시스템이 당신이 자는 동안 {len(report['conflicts'])}개의 모순을 발견했습니다.")
        
        # Show conflicts
        for conflict in report['conflicts'][:3]:
            content = conflict.get('content', '')[:80]
            st.markdown(f"> {content}...")
        
        # Show broken promise if exists
        if report.get('broken_promise'):
            promise = report['broken_promise']
            st.warning(f"📝 **어제의 약속:** {promise.get('action_plan', '없음')}")
            st.error("왜 또 여기 있는가?")
        
        # Burn & Enter button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔥 Burn & Enter", use_container_width=True, type="primary"):
                st.session_state['gatekeeper_dismissed'] = True
                st.rerun()
        
        st.stop()  # Block the rest of the app
    else:
        st.session_state['gatekeeper_dismissed'] = True


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Streak Counter
    streak = db.get_current_streak()
    longest = db.get_longest_streak()
    
    st.markdown(f"""
    <div class="streak-counter">
        🔥 {streak}일 연속
    </div>
    """, unsafe_allow_html=True)
    
    if longest > streak:
        st.caption(f"최장 기록: {longest}일")
    
    # Mode Selection
    st.markdown("---")
    
    options = {"stream": "🌊 Stream", "universe": "🌌 Universe", "desk": "🖊️ Desk"}
    
    mode = st.radio(
        "Navigation",
        list(options.keys()),
        format_func=lambda x: options[x],
        index=list(options.keys()).index(st.session_state['mode']),
        label_visibility="collapsed"
    )
    st.session_state['mode'] = mode
    
    st.markdown("---")
    
    # Statistics
    logs = logic.load_logs()
    const_count = len([l for l in logs if l.get("meta_type") == "Constitution"])
    apology_count = len([l for l in logs if l.get("meta_type") == "Apology"])
    frag_count = len([l for l in logs if l.get("meta_type") == "Fragment"])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⭐", const_count, help="Constitution")
    c2.metric("🩹", apology_count, help="Apology")
    c3.metric("💫", frag_count, help="Fragment")
    
    # Debt Counter (Red Protocol)
    debt = db.get_debt_count()
    if debt > 0:
        st.error(f"🩸 빚: {debt}")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 대화 초기화", use_container_width=True):
        logic.clear_chat_history()
        st.session_state.messages = [{"role": "assistant", "content": logic.get_welcome_message()}]
        st.session_state['first_input_of_session'] = True
        st.rerun()


# ============================================================
# RED MODE: Apology Form
# ============================================================
if red_mode:
    st.markdown("""
    <div class="red-warning">
        <h2 style="color: #ff4444; text-align: center;">🩸 우주가 피를 흘리고 있습니다</h2>
    </div>
    """, unsafe_allow_html=True)
    
    constitution = logic.get_violated_constitution()
    if constitution:
        st.markdown(f"**위반된 헌법:**")
        st.info(f"⭐ {constitution.get('content', '')}")
    
    st.markdown("---")
    st.markdown("### 📝 해명서 작성")
    st.caption("최소 100자 이상의 해명 + 내일의 약속이 필요합니다")
    
    apology_text = st.text_area(
        "해명 (Explanation)",
        placeholder="왜 헌법을 위반했는가? 무슨 일이 있었는가?",
        height=150
    )
    
    action_plan = st.text_input(
        "내일의 약속 (Action Plan)",
        placeholder="내일은 무엇을 다르게 할 것인가?"
    )
    
    # Validation
    is_valid = len(apology_text.strip()) >= 100 and len(action_plan.strip()) > 0
    char_count = len(apology_text.strip())
    
    if char_count < 100:
        st.warning(f"해명 글자 수: {char_count}/100")
    
    if st.button("🩹 제출하고 속죄하기", disabled=not is_valid, use_container_width=True, type="primary"):
        # Process apology
        logic.process_apology(
            content=apology_text,
            constitution_id=constitution['id'] if constitution else None,
            action_plan=action_plan
        )
        
        # Catharsis!
        st.balloons()
        st.success("✨ 우주가 다시 푸르게 변했습니다. Constellation이 생성되었습니다.")
        st.session_state['first_input_of_session'] = True
        st.rerun()
    
    st.stop()  # Block normal chat in Red Mode


# ============================================================
# MODE 1: THE STREAM
# ============================================================
if st.session_state['mode'] == "stream":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🌊 THE STREAM</h1>
        <p style="color: #6b7280;">Atomic thoughts. Hot state. Think aloud.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat Input
    if user_input := st.chat_input("What is the single sentence that defines you right now?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        logic.save_chat_message("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Save as Fragment
        log_entry = logic.save_log(user_input)
        
        # First input = Silent Save + Meteor Effect
        if st.session_state.get('first_input_of_session'):
            st.toast("💫 저장됨. Meteor Effect.", icon="☄️")
            st.session_state['first_input_of_session'] = False
        else:
            # Subsequent inputs = AI Response with Raw Quotes
            related_logs = logic.find_related_logs(user_input)
            response = logic.generate_response(user_input, related_logs)
            
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            logic.save_chat_message("assistant", response)
        
        st.rerun()


# ============================================================
# MODE 2: THE UNIVERSE
# ============================================================
elif st.session_state['mode'] == "universe":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🌌 THE UNIVERSE</h1>
        <p style="color: #6b7280;">Contemplation. Timeless. See the whole.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get zoom level based on streak
    zoom = logic.get_zoom_level()
    
    if zoom < 1.0:
        st.warning("🌑 Streak이 끊겼습니다. 우주가 멀어집니다...")
    
    # Generate graph
    graph_html = logic.generate_graph_html(zoom_level=zoom)
    components.html(graph_html, height=650, scrolling=False)
    
    # Legend
    st.markdown("""
    <div style="display:flex;gap:20px;justify-content:center;margin-top:15px;">
        <span style="color:#FFD700;">⭐ Constitution</span>
        <span style="color:#00FF7F;">🩹 Apology</span>
        <span style="color:#FFFFFF;">💫 Fragment</span>
        <span style="color:#00FFFF;">━ Constellation Link</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MODE 3: THE DESK
# ============================================================
elif st.session_state['mode'] == "desk":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🖊️ THE DESK</h1>
        <p style="color: #6b7280;">Narrative creation. Cool state. Write essays.</p>
    </div>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1.5])
    
    with left_col:
        st.markdown("### 📚 Cards")
        
        logs = [l for l in logic.load_logs() if l.get("meta_type") == "Fragment"]
        
        for log in logs[:15]:
            content = log.get("content", log.get("text", ""))[:60]
            log_id = log.get("id")
            is_selected = log_id in st.session_state['selected_cards']
            
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"💫 {content}...")
            with col2:
                if is_selected:
                    if st.button("➖", key=f"d_{log_id}"):
                        st.session_state['selected_cards'].remove(log_id)
                        st.rerun()
                else:
                    if st.button("➕", key=f"s_{log_id}"):
                        st.session_state['selected_cards'].append(log_id)
                        st.rerun()
    
    with right_col:
        st.markdown("### ✍️ Essay")
        
        if st.session_state['selected_cards']:
            st.caption(f"{len(st.session_state['selected_cards'])} cards selected")
        
        essay = st.text_area("Write your narrative", height=400, placeholder="Connect your fragments into a story...")
        
        if st.button("💾 Save Essay", use_container_width=True):
            if essay.strip():
                logic.save_log(essay)
                st.toast("💾 Essay saved!", icon="✍️")
                st.session_state['selected_cards'] = []
                st.rerun()


# ============================================================
# Mode Indicator
# ============================================================
mode_names = {"stream": "🌊 STREAM", "universe": "🌌 UNIVERSE", "desk": "🖊️ DESK"}
st.markdown(f"""
<div style="position:fixed;bottom:20px;right:20px;padding:8px 16px;border-radius:20px;
    font-size:12px;background:rgba(0,0,0,0.6);color:#fff;border:1px solid rgba(255,255,255,0.1);">
    {mode_names[st.session_state['mode']]}
</div>
""", unsafe_allow_html=True)
