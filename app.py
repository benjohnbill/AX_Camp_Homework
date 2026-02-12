"""
app.py
Antigravity v5: 5-Mode Architecture
Stream (Chat) | Chronos (Timer) | Universe (Graph+Finviz) | Control (Kanban) | Desk (Editor)
Gatekeeper Modal + Red Protocol + Silent Zoom + Chronos Docking + Soul Finviz
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import time
import json

import narrative_logic as logic
import db_manager as db
import icons

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page Config (MUST be first)
# ============================================================
st.set_page_config(
    page_title="Antigravity",
    page_icon=icons.get_icon("galaxy"),
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
    welcome = logic.get_welcome_message()
    st.session_state.messages = [{"role": "assistant", "content": welcome}]

if 'current_echo' not in st.session_state:
    st.session_state['current_echo'] = logic.get_current_echo()

if 'gatekeeper_dismissed' not in st.session_state:
    st.session_state['gatekeeper_dismissed'] = False

if 'first_input_of_session' not in st.session_state:
    st.session_state['first_input_of_session'] = True

if 'selected_cards' not in st.session_state:
    st.session_state['selected_cards'] = []

# [NEW v5] Chronos Timer State
if 'chronos_running' not in st.session_state:
    st.session_state['chronos_running'] = False
if 'chronos_end_time' not in st.session_state:
    st.session_state['chronos_end_time'] = None
if 'chronos_duration' not in st.session_state:
    st.session_state['chronos_duration'] = 25
if 'chronos_finished' not in st.session_state:
    st.session_state['chronos_finished'] = False

# [NEW v5] Kanban Docking Modal State
if 'docking_modal_active' not in st.session_state:
    st.session_state['docking_modal_active'] = False
if 'docking_card_id' not in st.session_state:
    st.session_state['docking_card_id'] = None

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
    
    /* Icons alignment */
    svg {{
        vertical-align: text-bottom;
        margin-right: 6px;
    }}
    
    /* [NEW v5] Chronos Timer Display */
    .chronos-timer {{
        font-size: 72px;
        font-weight: 900;
        text-align: center;
        font-family: 'Courier New', monospace;
        letter-spacing: 8px;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        margin: 40px 0;
    }}
    
    /* [NEW v5] Kanban Column */
    .kanban-card {{
        background: rgba(20, 25, 35, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        transition: all 0.2s;
    }}
    .kanban-card:hover {{
        border-color: rgba(0, 255, 255, 0.5);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.1);
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GATEKEEPER MODAL (Entry Ritual)  [PRESERVED FROM v4]
# ============================================================
if not st.session_state.get('gatekeeper_dismissed'):
    report = logic.run_gatekeeper()
    
    if report['conflicts'] or report.get('broken_promise'):
        st.markdown(f"""
        <div class="gatekeeper-modal">
            <h2 style="color: #ff4444; text-align: center;">{icons.get_icon("alert-triangle", color="#ff4444")} DREAM REPORT</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.error(f"{icons.get_icon('moon')} 시스템이 당신이 자는 동안 {len(report['conflicts'])}개의 모순을 발견했습니다.")
        
        # Show conflicts
        for conflict in report['conflicts'][:3]:
            content = conflict.get('content', '')[:80]
            st.markdown(f"> {content}...")
        
        # Show broken promise if exists
        if report.get('broken_promise'):
            promise = report['broken_promise']
            st.warning(f"{icons.get_icon('pencil')} **어제의 약속:** {promise.get('action_plan', '없음')}")
            st.error("왜 또 여기 있는가?")
        
        # Burn & Enter button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Burn & Enter", use_container_width=True, type="primary"):
                st.session_state['gatekeeper_dismissed'] = True
                st.rerun()
        
        st.stop()
    else:
        # The Mirror of the Week (Weekly Summary)
        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 20px;
            margin: 20px auto;
            max_width: 600px;
        ">
            <h3 style="color: #00BFFF; margin-top:0;">{icons.get_icon("sparkles")} The Mirror of the Week</h3>
            <p style="font-style: italic; color: #ccc;">이번 주, 당신의 궤적은 어디로 향했습니까?</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("별들이 당신의 일주일을 회고하고 있습니다..."):
            weekly_summary = logic.get_weekly_summary()
        
        st.info(weekly_summary)
        
        if st.button("Enter the Void", use_container_width=True, type="primary"):
            st.session_state['gatekeeper_dismissed'] = True
            st.rerun()
            
        st.stop()


# ============================================================
# SIDEBAR  [UPDATED v5: 5 modes]
# ============================================================
with st.sidebar:
    # Streak Counter
    streak = db.get_current_streak()
    longest = db.get_longest_streak()
    
    st.markdown(f"""
    <div class="streak-counter">
        {icons.get_icon("flame", color="#FFD700")} {streak}일 연속
    </div>
    """, unsafe_allow_html=True)
    
    if longest > streak:
        st.caption(f"최장 기록: {longest}일")
    
    # Mode Selection — [UPDATED v5: 5 modes]
    st.markdown("---")
    
    options = {
        "stream": "⚡ Stream",
        "chronos": "⏱️ Chronos",
        "universe": "🌌 Universe",
        "control": "📋 Control",
        "desk": "📖 Desk"
    }
    
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
    c1.metric("Const", const_count)
    c2.metric("Apol", apology_count)
    c3.metric("Frag", frag_count)
    
    # Debt Counter (Red Protocol)
    debt = db.get_debt_count()
    if debt > 0:
        st.error(f"Debt: {debt}")
    
    st.markdown("---")
    
    # API Key Configuration
    if not logic.is_api_key_configured():
        st.warning("API 키 필요")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            help="OpenAI API 키를 입력하세요"
        )
        if api_key_input:
            logic.set_api_key(api_key_input)
            st.success("API 키 설정됨")
            st.rerun()
    else:
        st.success("API 연결됨")
    
    st.markdown("---")
    
    # Reset button
    if st.button("대화 초기화", use_container_width=True):
        logic.clear_chat_history()
        st.session_state.messages = [{"role": "assistant", "content": logic.get_welcome_message()}]
        st.session_state['first_input_of_session'] = True
        st.rerun()


# ============================================================
# RED MODE: Apology Form  [PRESERVED FROM v4]
# ============================================================
if red_mode:
    st.markdown(f"""
    <div class="red-warning">
        <h2 style="color: #ff4444; text-align: center;">{icons.get_icon("alert-triangle", size=32)} 우주가 피를 흘리고 있습니다</h2>
    </div>
    """, unsafe_allow_html=True)
    
    constitution = logic.get_violated_constitution()
    if constitution:
        st.markdown(f"**위반된 헌법:**")
        st.markdown(f"{icons.get_icon('star')} {constitution.get('content', '')}", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("---")
    
    st.markdown(f"### {icons.get_icon('pencil')} 해명서 작성", unsafe_allow_html=True)
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
    
    is_valid = len(apology_text.strip()) >= 100 and len(action_plan.strip()) > 0
    char_count = len(apology_text.strip())
    
    if char_count < 100:
        st.warning(f"해명 글자 수: {char_count}/100")
    
    if st.button("제출하고 속죄하기", disabled=not is_valid, use_container_width=True, type="primary"):
        logic.process_apology(
            content=apology_text,
            constitution_id=constitution['id'] if constitution else None,
            action_plan=action_plan
        )
        st.balloons()
        st.success("우주가 다시 푸르게 변했습니다. Constellation이 생성되었습니다.")
        st.session_state['first_input_of_session'] = True
        st.rerun()

    st.stop()  # Block normal chat in Red Mode


# ============================================================
# MODE 1: THE STREAM  [PRESERVED FROM v4]
# ============================================================
if st.session_state['mode'] == "stream":
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>{icons.get_icon("waves", size=40)} THE STREAM</h1>
        <p style="color: #6b7280;">Atomic thoughts. Hot state. Think aloud.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # [Echo System] Display random past memory
    echo = st.session_state.get('current_echo')
    if echo:
        echo_content = echo.get('content', echo.get('text', ''))
        echo_date = echo.get('created_at', '')[:10]
        
        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.05); 
            border-left: 3px solid rgba(255, 255, 255, 0.3);
            padding: 15px 20px; 
            margin: 0 auto 30px auto; 
            max-width: 700px;
            border-radius: 4px;
            font-style: italic;
            color: #ccc;
        ">
            <div style="font-size: 0.9em; margin-bottom: 8px; color: #888;">
                {icons.get_icon("sparkles", size=14)} Echo from {echo_date}
            </div>
            "{echo_content}"
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat Input
    if user_input := st.chat_input("What is the single sentence that defines you right now?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        logic.save_chat_message("user", user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        log_entry = logic.save_log(user_input)
        
        if st.session_state.get('first_input_of_session'):
            st.toast("저장됨. Meteor Effect.", icon="☄️")
            st.session_state['first_input_of_session'] = False
        else:
            related_logs = logic.find_related_logs(user_input)
            response = logic.generate_response(user_input, related_logs)
            
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            logic.save_chat_message("assistant", response)
        
        st.rerun()


# ============================================================
# MODE 2: CHRONOS  [NEW v5]
# ============================================================
elif st.session_state['mode'] == "chronos":
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>⏱️ CHRONOS</h1>
        <p style="color: #6b7280;">Time is the currency. Dock it to your soul.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if timer has expired
    if st.session_state['chronos_running'] and st.session_state['chronos_end_time']:
        remaining = st.session_state['chronos_end_time'] - datetime.now()
        if remaining.total_seconds() <= 0:
            st.session_state['chronos_running'] = False
            st.session_state['chronos_finished'] = True
    
    # ---- STATE: DOCKING MODAL (Timer Finished) ----
    if st.session_state['chronos_finished']:
        st.markdown("""
        <div style="
            background: rgba(0, 255, 255, 0.05);
            border: 2px solid #00FFFF;
            border-radius: 15px;
            padding: 30px;
            margin: 20px auto;
            max-width: 700px;
            text-align: center;
        ">
            <h2 style="color: #00FFFF;">⚓ DOCKING PROTOCOL</h2>
            <p style="color: #ccc;">시간이 완료되었습니다. 이 시간은 어떤 헌법에 귀속됩니까?</p>
        </div>
        """, unsafe_allow_html=True)
        
        constitutions = db.get_constitutions()
        const_options = {c['content'][:50]: c['id'] for c in constitutions}
        
        if not const_options:
            st.warning("헌법이 없습니다. Stream에서 먼저 헌법을 생성하세요.")
        else:
            selected_consts = st.multiselect(
                "이 시간이 섬긴 헌법을 선택하세요 (1개 이상 필수)",
                options=list(const_options.keys()),
                key="docking_constitutions"
            )
            
            accomplishment = st.text_area(
                "무엇을 성취했습니까? (최소 10자)",
                placeholder="이 시간 동안 무엇을 했고, 무엇을 느꼈는가...",
                key="docking_accomplishment",
                height=100
            )
            
            char_count = len(accomplishment.strip())
            if accomplishment.strip() and char_count < 10:
                st.error(f"서사가 너무 짧습니다. 더 깊이 파세요. ({char_count}/10자)")
            
            can_dock = len(selected_consts) > 0 and char_count >= 10
            
            if st.button("⚓ Dock & Seal", disabled=not can_dock, use_container_width=True, type="primary"):
                const_ids = [const_options[name] for name in selected_consts]
                duration = st.session_state.get('chronos_duration', 25)
                
                logic.save_chronos_log(
                    content=accomplishment,
                    constitution_ids=const_ids,
                    duration=duration
                )
                
                st.balloons()
                st.toast(f"🎯 {duration}분이 헌법에 도킹되었습니다!", icon="⚓")
                
                # Reset Chronos state
                st.session_state['chronos_finished'] = False
                st.session_state['chronos_end_time'] = None
                st.session_state['chronos_running'] = False
                st.rerun()
    
    # ---- STATE: TIMER RUNNING ----
    elif st.session_state['chronos_running']:
        remaining = st.session_state['chronos_end_time'] - datetime.now()
        total_secs = max(0, int(remaining.total_seconds()))
        mins = total_secs // 60
        secs = total_secs % 60
        
        # JavaScript countdown for smooth ticking
        end_ts = int(st.session_state['chronos_end_time'].timestamp() * 1000)
        components.html(f"""
        <div style="text-align:center; font-family:'Courier New',monospace; padding:40px 0;">
            <div id="timer" style="font-size:96px; font-weight:900; color:#00FFFF;
                 letter-spacing:8px; text-shadow:0 0 40px rgba(0,255,255,0.6);">
                {mins:02d}:{secs:02d}
            </div>
            <div style="color:#666; font-size:14px; margin-top:10px;">
                {st.session_state['chronos_duration']}분 세션 진행 중
            </div>
            <div id="progress-bar" style="margin:30px auto; width:80%; height:4px;
                 background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
                <div id="progress-fill" style="height:100%; background:linear-gradient(90deg,#00FFFF,#FFD700);
                     border-radius:2px; transition:width 1s linear;"></div>
            </div>
        </div>
        <script>
            const endTime = {end_ts};
            const totalDuration = {st.session_state['chronos_duration'] * 60 * 1000};
            function tick() {{
                const now = Date.now();
                const diff = Math.max(0, endTime - now);
                const m = Math.floor(diff / 60000);
                const s = Math.floor((diff % 60000) / 1000);
                document.getElementById('timer').textContent =
                    String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
                const elapsed = totalDuration - diff;
                const pct = Math.min(100, (elapsed / totalDuration) * 100);
                document.getElementById('progress-fill').style.width = pct + '%';
                if (diff <= 0) {{
                    document.getElementById('timer').textContent = '00:00';
                    document.getElementById('timer').style.color = '#FFD700';
                    document.getElementById('timer').style.textShadow = '0 0 60px rgba(255,215,0,0.8)';
                    clearInterval(interval);
                }}
            }}
            tick();
            const interval = setInterval(tick, 1000);
        </script>
        """, height=250)
        
        # Stop/Cancel Buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("⏹️ 타이머 완료 (지금 도킹)", use_container_width=True, type="primary"):
                st.session_state['chronos_running'] = False
                st.session_state['chronos_finished'] = True
                st.rerun()
        with col3:
            if st.button("❌ 취소", use_container_width=True):
                st.session_state['chronos_running'] = False
                st.session_state['chronos_end_time'] = None
                st.rerun()
        
        # Auto-refresh to detect timer end
        time.sleep(2)
        st.rerun()
    
    # ---- STATE: IDLE (Setup Timer) ----
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px 0;">
            <div style="font-size:96px; font-weight:900; color:rgba(255,255,255,0.15);
                 font-family:'Courier New',monospace; letter-spacing:8px;">
                00:00
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 세션 시간 선택")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔥 25분 (Pomodoro)", use_container_width=True):
                st.session_state['chronos_duration'] = 25
                st.session_state['chronos_end_time'] = datetime.now() + timedelta(minutes=25)
                st.session_state['chronos_running'] = True
                st.rerun()
        with col2:
            if st.button("⚡ 60분 (Deep Work)", use_container_width=True):
                st.session_state['chronos_duration'] = 60
                st.session_state['chronos_end_time'] = datetime.now() + timedelta(minutes=60)
                st.session_state['chronos_running'] = True
                st.rerun()
        with col3:
            custom_mins = st.number_input("커스텀 (분)", min_value=1, max_value=180, value=45, key="custom_timer")
            if st.button("🚀 시작", use_container_width=True):
                st.session_state['chronos_duration'] = custom_mins
                st.session_state['chronos_end_time'] = datetime.now() + timedelta(minutes=custom_mins)
                st.session_state['chronos_running'] = True
                st.rerun()
        
        # Recent Chronos History
        st.markdown("---")
        st.markdown("#### 최근 도킹 기록")
        all_logs = logic.load_logs()
        docked = [l for l in all_logs if l.get("duration") and l.get("linked_constitutions")][:5]
        
        if docked:
            for log in docked:
                dur = log.get('duration', 0)
                content = log.get('content', '')[:60]
                date_str = log.get('created_at', '')[:10]
                st.markdown(f"- **{dur}분** | {content}... _{date_str}_")
        else:
            st.caption("아직 도킹 기록이 없습니다. 첫 세션을 시작하세요.")


# ============================================================
# MODE 3: THE UNIVERSE  [PRESERVED + NEW v5 Soul Finviz Tab]
# ============================================================
elif st.session_state['mode'] == "universe":
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>{icons.get_icon("orbit", size=40)} THE UNIVERSE</h1>
        <p style="color: #6b7280;">Contemplation. Timeless. See the whole.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # [UPDATED v5] Add Soul Finviz tab
    tab_cosmos, tab_finviz, tab_rhythm, tab_pulse = st.tabs([
        "🌌 Constellation", "📊 Soul Finviz", "⏳ Rhythm of Will", "📈 The Pulse"
    ])
    
    with tab_cosmos:
        zoom = logic.get_zoom_level()
        if zoom < 1.0:
            st.warning("Streak이 끊겼습니다. 우주가 멀어집니다...")
        
        graph_html = logic.generate_graph_html(zoom_level=zoom)
        components.html(graph_html, height=650, scrolling=False)
        
        st.markdown(f"""
        <div style="display:flex;gap:20px;justify-content:center;margin-top:15px;margin-bottom:30px;">
            <span style="color:#FFD700;">{icons.get_icon("star", size=16)} Constitution</span>
            <span style="color:#00FF7F;">{icons.get_icon("activity", size=16)} Apology</span>
            <span style="color:#FFFFFF;">{icons.get_icon("sparkles", size=16)} Fragment</span>
            <span style="color:#00FFFF;">{icons.get_icon("link", size=16)} Apology Link</span>
            <span style="color:#FF4500;">{icons.get_icon("link", size=16)} Manual Constellation</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Telescope
        st.markdown(f"### {icons.get_icon('telescope')} Telescope", unsafe_allow_html=True)
        st.caption("관측할 별을 선택하고, 새로운 별자리를 연결하세요.")
        
        logs = logic.load_logs()
        options = {
            f"[{l.get('meta_type','?')}] {l.get('content','...')[:20]}... ({l.get('created_at','')[:10]})": l['id'] 
            for l in logs
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_label = st.selectbox("관측 대상 (Source)", options=list(options.keys()), key="telescope_source")
            if selected_label:
                source_id = options[selected_label]
                log = logic.get_log_by_id(source_id)
                with st.expander("📄 상세 내용 보기", expanded=True):
                    st.markdown(f"**{log.get('meta_type')}** | {log.get('created_at')[:16]}")
                    st.markdown("---")
                    st.markdown(log.get('content'))
                    if log.get('action_plan'):
                        st.info(f"Action Plan: {log.get('action_plan')}")

        with col2:
            st.markdown(f"#### {icons.get_icon('link')} Constellation 연결", unsafe_allow_html=True)
            target_label = st.selectbox("연결 대상 (Target)", options=list(options.keys()), key="telescope_target")
            
            if st.button("별자리 연결하기", use_container_width=True, type="primary"):
                target_id = options[target_label]
                source_id = options[selected_label]
                if source_id == target_id:
                    st.error("자기 자신과 연결할 수 없습니다.")
                else:
                    success = logic.add_manual_connection(source_id, target_id)
                    if success:
                        st.success("새로운 별자리가 연결되었습니다! 🌌")
                        st.rerun()
                    else:
                        st.warning("이미 연결되어 있거나 연결할 수 없습니다.")
    
    # [NEW v5] Soul Finviz Treemap
    with tab_finviz:
        st.markdown("### 📊 Soul Finviz — Ego Portfolio", unsafe_allow_html=True)
        st.caption("당신의 영혼 포트폴리오. 어떤 헌법이 굶주리고 있는가?")
        
        finviz_data = logic.get_finviz_data()
        
        if finviz_data:
            labels = [d['content'][:30] + "..." for d in finviz_data]
            parents = ["" for _ in finviz_data]
            values = [d['size'] for d in finviz_data]
            health_scores = [d['health_score'] for d in finviz_data]
            
            # Custom text for each block
            custom_text = []
            for d in finviz_data:
                text = (
                    f"⏱ {d['total_duration']}분 투자<br>"
                    f"📝 {d['fragment_count']}개 기록<br>"
                    f"🩹 {d['apology_count']}개 사과<br>"
                    f"📅 {d['days_since_activity']}일 전 활동"
                )
                custom_text.append(text)
            
            fig = go.Figure(go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                marker=dict(
                    colors=health_scores,
                    colorscale=[
                        [0.0, "#ff0000"],    # -1.0 = Deep Red (Starving)
                        [0.25, "#ff4444"],
                        [0.5, "#888888"],    # 0.0 = Neutral 
                        [0.75, "#00cc66"],
                        [1.0, "#00ff88"]     # 1.0 = Vibrant Green
                    ],
                    cmid=0,
                    cmin=-1.0,
                    cmax=1.0,
                    line=dict(width=2, color="rgba(255,255,255,0.2)"),
                    colorbar=dict(
                        title="Health",
                        tickvals=[-1, 0, 1],
                        ticktext=["Starving", "Neutral", "Alive"]
                    )
                ),
                text=custom_text,
                textinfo="label+text",
                hovertemplate="<b>%{label}</b><br>%{text}<br>Health: %{color:.2f}<extra></extra>",
                textfont=dict(size=14, color="white")
            ))
            
            fig.update_layout(
                height=500,
                margin=dict(t=30, l=10, r=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            total_time = sum(d['total_duration'] for d in finviz_data)
            total_frags = sum(d['fragment_count'] for d in finviz_data)
            avg_health = sum(d['health_score'] for d in finviz_data) / len(finviz_data)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("총 투자 시간", f"{total_time}분")
            m2.metric("총 기록", f"{total_frags}개")
            m3.metric("평균 건강도", f"{avg_health:.2f}")
        else:
            st.info("헌법이 없습니다. Stream에서 기록을 시작하세요.")

    with tab_rhythm:
        st.markdown(f"### {icons.get_icon('activity')} The Rhythm of Will", unsafe_allow_html=True)
        st.caption("당신의 의지가 언제 무너지는지, 시간의 패턴을 분석합니다.")
        
        st.subheader("🔥 Conflict Heatmap (요일/시간별 위반 빈도)")
        heatmap_data = logic.get_temporal_patterns()
        
        if not heatmap_data.empty:
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Hour of Day", y="Day of Week", color="Conflict Count"),
                x=[str(i) for i in range(24)],
                y=heatmap_data.index,
                color_continuous_scale="Reds"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("📉 Apology Trend (최근 30일)")
        trend_data = logic.get_daily_apology_trend()
        
        if not trend_data.empty:
            fig2 = px.line(
                trend_data,
                x='date', y='count',
                markers=True, line_shape='spline',
                color_discrete_sequence=['#ff4444']
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

    with tab_pulse:
        st.markdown(f"### {icons.get_icon('activity')} The Pulse", unsafe_allow_html=True)
        st.caption("생각을 멈추지 않았다는 시각적 증명. (Fragments vs Apologies)")
        
        pulse_data = logic.get_activity_pulse()
        
        if not pulse_data.empty:
            fig_pulse = px.imshow(
                pulse_data,
                labels=dict(x="Date", y="Type", color="Count"),
                x=pulse_data.columns,
                y=pulse_data.index,
                color_continuous_scale="Viridis"
            )
            fig_pulse.update_layout(height=300)
            st.plotly_chart(fig_pulse, use_container_width=True)
        else:
            st.info("데이터가 충분하지 않습니다. 기록을 시작하세요.")
            
        st.markdown("---")
        
        st.subheader("🌊 Apology Trend (최근 30일)")
        trend_data = logic.get_daily_apology_trend()
        
        if not trend_data.empty:
            fig2 = px.line(
                trend_data, x='date', y='count',
                title='Daily Apology Count',
                markers=True, line_shape='spline'
            )
            fig2.update_traces(line_color='#FF4500')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("데이터가 없습니다.")


# ============================================================
# MODE 4: CONTROL (Narrative Kanban)  [NEW v5]
# ============================================================
elif st.session_state['mode'] == "control":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>📋 CONTROL</h1>
        <p style="color: #6b7280;">Narrative Kanban. Thoughts orbit. Then they land.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # New Card Input — Constitution Gate: every card must orbit a star
    st.markdown("#### ✨ 새로운 궤도 생성")
    constitutions = db.get_constitutions()
    const_options_kanban = {c['content'][:50]: c['id'] for c in constitutions}
    
    if not const_options_kanban:
        st.warning("헌법이 없습니다. Stream에서 먼저 헌법을 생성하세요.")
    else:
        kc1, kc2 = st.columns([2, 1])
        with kc1:
            new_thought = st.text_input("무엇이 궤도에 떠오르고 있는가?", placeholder="이 별 주변을 떠도는 생각...", key="kanban_new")
        with kc2:
            selected_star = st.selectbox("소속 헌법", options=list(const_options_kanban.keys()), key="kanban_const")
        
        can_add = new_thought.strip() and selected_star
        if st.button("🌀 궤도에 투입", use_container_width=True, disabled=not can_add) and can_add:
            logic.create_kanban_card(new_thought.strip(), constitution_id=const_options_kanban[selected_star])
            st.toast("새로운 생각이 헌법 궤도에 진입했습니다.", icon="🌀")
            st.rerun()
    
    st.markdown("---")
    
    # Get cards
    cards = logic.get_kanban_cards()
    
    # 3-Column Layout
    col_draft, col_orbit, col_landed = st.columns(3)
    
    # ---- DRAFTS Column ----
    with col_draft:
        st.markdown("""
        <div style="text-align:center; padding:8px; background:rgba(255,255,255,0.05);
             border-radius:8px; margin-bottom:15px;">
            <h4 style="margin:0;">💭 Drafts</h4>
            <p style="color:#888; font-size:12px; margin:0;">떠오르는 생각들</p>
        </div>
        """, unsafe_allow_html=True)
        
        for card in cards.get("draft", []):
            content = card.get("content", "")[:60]
            card_id = card.get("id")
            with st.container():
                st.markdown(f"""<div class="kanban-card">📝 {content}</div>""", unsafe_allow_html=True)
                if st.button("→ In Orbit", key=f"to_orbit_{card_id}", use_container_width=True):
                    logic.move_kanban_card(card_id, "orbit")
                    st.rerun()
    
    # ---- IN ORBIT Column ----
    with col_orbit:
        st.markdown("""
        <div style="text-align:center; padding:8px; background:rgba(0,255,255,0.05);
             border-radius:8px; margin-bottom:15px;">
            <h4 style="margin:0; color:#00FFFF;">🚀 In Orbit</h4>
            <p style="color:#888; font-size:12px; margin:0;">작업 중 / Chronos 연결</p>
        </div>
        """, unsafe_allow_html=True)
        
        for card in cards.get("orbit", []):
            content = card.get("content", "")[:60]
            card_id = card.get("id")
            with st.container():
                st.markdown(f"""<div class="kanban-card" style="border-color:rgba(0,255,255,0.3);">
                    🛰️ {content}</div>""", unsafe_allow_html=True)
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("← Draft", key=f"to_draft_{card_id}", use_container_width=True):
                        logic.move_kanban_card(card_id, "draft")
                        st.rerun()
                with bc2:
                    if st.button("→ Land", key=f"to_land_{card_id}", use_container_width=True, type="primary"):
                        st.session_state['docking_modal_active'] = True
                        st.session_state['docking_card_id'] = card_id
                        st.rerun()
    
    # ---- LANDED Column ----
    with col_landed:
        st.markdown("""
        <div style="text-align:center; padding:8px; background:rgba(255,215,0,0.05);
             border-radius:8px; margin-bottom:15px;">
            <h4 style="margin:0; color:#FFD700;">🏁 Landed</h4>
            <p style="color:#888; font-size:12px; margin:0;">도킹 완료</p>
        </div>
        """, unsafe_allow_html=True)
        
        for card in cards.get("landed", []):
            content = card.get("content", "")[:60]
            dur = card.get("duration", 0)
            linked = card.get("linked_constitutions", [])
            with st.container():
                dur_text = f" | ⏱{dur}분" if dur else ""
                link_text = f" | 🔗{len(linked)}" if linked else ""
                st.markdown(f"""<div class="kanban-card" style="border-color:rgba(255,215,0,0.3);">
                    ✅ {content}{dur_text}{link_text}</div>""", unsafe_allow_html=True)
    
    # ---- DOCKING MODAL (triggered by "Land" button) ----
    if st.session_state.get('docking_modal_active'):
        st.markdown("---")
        st.markdown("""
        <div style="
            background: rgba(0, 255, 255, 0.05);
            border: 2px solid #00FFFF;
            border-radius: 15px;
            padding: 25px;
            margin: 15px auto;
            max-width: 700px;
        ">
            <h3 style="color: #00FFFF; text-align:center;">⚓ DOCKING PROTOCOL</h3>
            <p style="color: #ccc; text-align:center;">이 카드를 도킹합니다. 어떤 헌법에 귀속됩니까?</p>
        </div>
        """, unsafe_allow_html=True)
        
        constitutions = db.get_constitutions()
        const_options = {c['content'][:50]: c['id'] for c in constitutions}
        
        if const_options:
            sel_consts = st.multiselect(
                "헌법 선택 (1개 이상)",
                options=list(const_options.keys()),
                key="kanban_dock_consts"
            )
            
            dock_accomplishment = st.text_input(
                "성취 요약",
                placeholder="이 궤도에서 무엇을 완수했는가?",
                key="kanban_dock_text"
            )
            
            dock_duration = st.number_input("소요 시간 (분)", min_value=0, max_value=480, value=0, key="kanban_dock_dur")
            
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("⚓ Dock", use_container_width=True, type="primary",
                             disabled=len(sel_consts) == 0):
                    const_ids = [const_options[n] for n in sel_consts]
                    logic.land_kanban_card(
                        card_id=st.session_state['docking_card_id'],
                        constitution_ids=const_ids,
                        accomplishment=dock_accomplishment,
                        duration=dock_duration
                    )
                    st.balloons()
                    st.session_state['docking_modal_active'] = False
                    st.session_state['docking_card_id'] = None
                    st.rerun()
            with dc2:
                if st.button("취소", use_container_width=True):
                    st.session_state['docking_modal_active'] = False
                    st.session_state['docking_card_id'] = None
                    st.rerun()
        else:
            st.warning("헌법이 없습니다. Stream에서 먼저 기록을 시작하세요.")


# ============================================================
# MODE 5: THE DESK  [PRESERVED FROM v4]
# ============================================================
elif st.session_state['mode'] == "desk":
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>{icons.get_icon("book-open", size=40)} THE DESK</h1>
        <p style="color: #6b7280;">Narrative creation. Cool state. Write essays.</p>
    </div>
    """, unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1.5])
    
    with left_col:
        all_logs = logic.load_logs()
        constitutions = [l for l in all_logs if l.get("meta_type") == "Constitution"]
        apologies = [l for l in all_logs if l.get("meta_type") == "Apology"]
        fragments = [l for l in all_logs if l.get("meta_type") == "Fragment"]
        
        const_map = {c["id"]: [] for c in constitutions}
        unlinked_apologies = []
        
        for apology in apologies:
            parent_id = apology.get("parent_id")
            if parent_id in const_map:
                const_map[parent_id].append(apology)
            else:
                unlinked_apologies.append(apology)
        
        if 'desk_page' not in st.session_state:
            st.session_state['desk_page'] = 1
            
        FRAGMENTS_PER_PAGE = 10
        
        def render_card(log, icon_name="star", indent=0):
            log_id = log.get("id")
            content = log.get("content", log.get("text", ""))
            created_at = log.get("created_at", "")[:10]
            is_selected = log_id in st.session_state['selected_cards']
            margin_left = f"{indent * 20}px"
            
            with st.container():
                st.markdown(f'<div style="margin-left: {margin_left}; border-left: 2px solid rgba(255,255,255,0.1); padding-left: 10px; margin-bottom: 10px;">', unsafe_allow_html=True)
                preview = content[:30] + "..." if len(content) > 30 else content
                header_text = f"[{created_at}] {preview}"
                
                with st.expander(header_text, expanded=False):
                    st.markdown(f"{icons.get_icon(icon_name)} **{log.get('meta_type')}**", unsafe_allow_html=True)
                    st.caption(f"📅 {created_at}")
                    st.markdown(content)
                    if log.get('action_plan'):
                        st.info(f"Action Plan: {log['action_plan']}")
                    if is_selected:
                        if st.button("선택 해제", key=f"btn_{log_id}", use_container_width=True):
                            st.session_state['selected_cards'].remove(log_id)
                            st.rerun()
                    else:
                        if st.button("에세이에 추가", key=f"btn_{log_id}", use_container_width=True):
                            st.session_state['selected_cards'].append(log_id)
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

        if constitutions:
            st.markdown(f"#### {icons.get_icon('star')} Constitutions", unsafe_allow_html=True)
            for const in constitutions:
                render_card(const, icon_name="star", indent=0)
                linked_apologies = const_map.get(const["id"], [])
                if linked_apologies:
                    for apology in linked_apologies:
                        render_card(apology, icon_name="activity", indent=1)

        if unlinked_apologies:
            st.markdown(f"#### {icons.get_icon('activity')} Apologies", unsafe_allow_html=True)
            for apology in unlinked_apologies:
                render_card(apology, icon_name="activity", indent=0)
        
        st.markdown(f"#### {icons.get_icon('sparkles')} Fragments", unsafe_allow_html=True)
        p_fragments, total_count = logic.get_fragments_paginated(
            page=st.session_state['desk_page'], per_page=FRAGMENTS_PER_PAGE
        )
        total_pages = max(1, (total_count + FRAGMENTS_PER_PAGE - 1) // FRAGMENTS_PER_PAGE)
        
        for frag in p_fragments:
            render_card(frag, icon_name="sparkles", indent=0)
            
        if total_pages > 1:
            st.markdown("---")
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.session_state['desk_page'] > 1:
                    if st.button("◀ 이전", use_container_width=True):
                        st.session_state['desk_page'] -= 1
                        st.rerun()
            with c2:
                st.markdown(f"<div style='text-align:center; padding-top:5px;'>Page {st.session_state['desk_page']} / {total_pages}</div>", unsafe_allow_html=True)
            with c3:
                if st.session_state['desk_page'] < total_pages:
                    if st.button("다음 ▶", use_container_width=True):
                        st.session_state['desk_page'] += 1
                        st.rerun()
    
    with right_col:
        st.markdown(f"### {icons.get_icon('pen-tool')} Essay", unsafe_allow_html=True)
        
        if st.session_state['selected_cards']:
            st.caption(f"{len(st.session_state['selected_cards'])} cards selected")
        
        essay = st.text_area("Write your narrative", height=400, placeholder="Connect your fragments into a story...")
        
        if st.button("Save Essay", use_container_width=True):
            if essay.strip():
                logic.save_log(essay)
                st.toast("Essay saved!", icon="✍️")
                st.session_state['selected_cards'] = []
                st.rerun()


# ============================================================
# Mode Indicator  [UPDATED v5]
# ============================================================
mode_names = {
    "stream": "STREAM", "chronos": "CHRONOS",
    "universe": "UNIVERSE", "control": "CONTROL", "desk": "DESK"
}
st.markdown(f"""
<div style="position:fixed;bottom:20px;right:20px;padding:8px 16px;border-radius:20px;
    font-size:12px;background:rgba(0,0,0,0.6);color:#fff;border:1px solid rgba(255,255,255,0.1);">
    {mode_names[st.session_state['mode']]}
</div>
""", unsafe_allow_html=True)
