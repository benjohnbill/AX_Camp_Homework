"""
app.py
Antigravity v5: 5-Mode Architecture - Refactored for Modularity & Robust Icons
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta, timezone
import time
import json
import re
import base64
import os

import narrative_logic as logic
import icons
from universe_3d import render_3d_universe
db = logic.db

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page Config & Initialization
# ============================================================
@st.cache_data
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return None
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def init_page_config():
    st.set_page_config(
        page_title="Antigravity",
        page_icon=":milky_way:",
        layout="wide"
    )


def _safe_startup_error(exc: Exception) -> str:
    """Format startup errors without leaking DSN-like credentials."""
    line = str(exc).strip().splitlines()[0] if str(exc) else type(exc).__name__
    line = re.sub(r"postgres(?:ql)?://[^\\s]+", "postgresql://***", line)
    return f"{type(exc).__name__}: {line}"

def init_session_state():
    if 'db_bootstrap_done' not in st.session_state:
        # Ensure required DB schema exists before any startup reads.
        db.init_database()
        st.session_state['db_bootstrap_done'] = True

    if 'diagnostics_run' not in st.session_state:
        db.inject_genesis_data(logic.get_embedding)
        logic.run_startup_diagnostics()
        st.session_state['diagnostics_run'] = True

    if 'streak_updated' not in st.session_state:
        result = db.check_streak_and_apply_penalty()
        st.session_state['streak_info'] = result['streak_info']
        # entropy_mode는 is_entropy_mode()가 DB debt_count를 실시간 조회하므로
        # 별도 trigger 플래그 불필요 — penalty 적용 시 debt가 DB에서 이미 증가됨
        st.session_state['streak_updated'] = True

    if 'mode' not in st.session_state:
        st.session_state['mode'] = "stream"

    if 'messages' not in st.session_state:
        welcome = logic.get_welcome_message()
        st.session_state.messages = [{"role": "assistant", "content": welcome}]

    if 'current_echo' not in st.session_state:
        st.session_state['current_echo'] = logic.get_current_echo()

    # [B-2] 새로고침 후 Chronos 타이머 복원 — DB에 저장된 종료 시각이 미래이면 session_state에 복원
    if 'chronos_running' not in st.session_state:
        saved_end = db.get_chronos_timer()
        if saved_end:
            st.session_state['chronos_end_time'] = saved_end
            st.session_state['chronos_running'] = True
            st.session_state['chronos_finished'] = False

    defaults = {
        'gatekeeper_dismissed': False,
        'first_input_of_session': True,
        'selected_cards': [],
        'chronos_running': False,
        'chronos_end_time': None,
        'chronos_duration': 25,
        'chronos_finished': False,
        'docking_modal_active': False,
        'docking_card_id': None,
        'interventions_checked': False,
        'desk_page': 1,
        'violation_pending': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def apply_atmosphere(entropy_mode: bool):
    current_mode = st.session_state.get('mode', 'stream')
    
    # 1. Determine Mode-Specific CSS Variables
    if entropy_mode:
        blur_val = "15px"
        overlay_color = "rgba(15, 0, 0, 0.85)" # Deep, suffocating dark tint
        text_color = "#cccccc"
    elif current_mode == "universe":
        blur_val = "1.5px"
        overlay_color = "rgba(0, 0, 0, 0.25)"  # Vivid, immersive
        text_color = "#e94560"
    else:
        # Stream, Desk, Control, Chronos
        blur_val = "12px"
        overlay_color = "rgba(0, 0, 0, 0.65)"  # Frosted glass, high readability
        text_color = "#e94560"

    # Using a placeholder starry sky image. 
    BG_IMAGE_URL = "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=2000&auto=format&fit=crop"
    
    # Seamlessly load local AI image if the user places it in the 'assets' directory
    local_bg_path = os.path.join("assets", "bg.jpg")
    bg_base64 = get_base64_of_bin_file(local_bg_path)
    if bg_base64:
        # Assuming jpeg format for the base64 MIME
        BG_IMAGE_URL = f"data:image/jpeg;base64,{bg_base64}"

    # 2. Inject Dynamic CSS
    st.markdown(f"""
        <style>
        /* Base Variables injected by Python State */
        :root {{
            --bg-blur: {blur_val};
            --bg-overlay: {overlay_color};
            --primary-text: {text_color};
        }}

        /* Wipe Streamlit's default background completely */
        .stApp, .block-container {{
            background: transparent !important; 
            background-color: transparent !important;
        }}

        /* The Unified Background Entity */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; 
            left: 0; 
            width: 100vw; 
            height: 100vh;
            
            /* Stacks the dark overlay ON TOP of the image */
            background-image: 
                linear-gradient(var(--bg-overlay), var(--bg-overlay)), 
                url('{BG_IMAGE_URL}');
                
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            
            /* Apply blur and scale up slightly to hide blurry edges */
            filter: blur(var(--bg-blur));
            transform: scale(1.05);
            
            /* Push to bottom layer */
            z-index: -1;
            
            /* Transition for smooth mode switching */
            transition: filter 0.8s ease, background-image 0.8s ease;
        }}

        /* Force foreground elements to remain visible over the background */
        * {{ color: var(--primary-text) !important; }}
        
        /* Ensure Inputs and Kanban Cards have distinct, readable backgrounds */
        input, textarea, select {{
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            background-color: rgba(20, 20, 25, 0.85) !important; 
            backdrop-filter: blur(5px);
        }}
        
        .kanban-card {{
            background: rgba(30, 30, 40, 0.85) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 10px; 
            padding: 15px; 
            margin-bottom: 10px; 
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
        }}
        .kanban-card:hover {{ transform: translateY(-2px); border-color: rgba(255,255,255,0.5) !important; }}
        </style>
    """, unsafe_allow_html=True)

# ============================================================
# API Key
# ============================================================
def render_api_key_section():
    with st.expander("OpenAI API Key", expanded=False):
        session_key = st.session_state.get("openai_api_key", "")
        has_any_key = logic.is_api_key_configured()

        if session_key:
            st.success("Using API key from this session.")
        elif has_any_key:
            st.info("Using API key from deployment secrets/env.")
        else:
            st.warning("No API key detected. Enter your key to enable AI responses.")

        entered = st.text_input(
            "API Key",
            type="password",
            value=session_key,
            placeholder="sk-...",
            key="openai_api_key_input"
        )

        c1, c2 = st.columns(2)
        if c1.button("Apply Key", use_container_width=True):
            if entered and entered.strip():
                logic.set_api_key(entered.strip())
                st.success("Session key applied.")
                st.rerun()
            else:
                st.error("Enter a valid API key.")

        if c2.button("Clear Session Key", use_container_width=True):
            st.session_state.pop("openai_api_key", None)
            st.session_state["openai_api_key_input"] = ""
            st.info("Session key cleared.")
            st.rerun()
def render_sidebar(entropy_mode: bool):
    with st.sidebar:
        st.title(f"{icons.get_icon_text('orbit')} Antigravity")
        st.caption("Existential Audit Protocol v6.0") # Version Bump
        st.divider()
        
        if entropy_mode:
            st.warning(f"{icons.get_icon_text('shield-alert')} ENTROPY ALERT")
            st.info("시스템 엔트로피가 임계치를 초과했습니다. [Gap Analysis]가 필요합니다.")
        else:
            mode = st.radio("Select Mode", ["Stream", "Chronos", "Universe", "Control", "Desk"],
                            index=["stream", "chronos", "universe", "control", "desk"].index(st.session_state['mode']))
            st.session_state['mode'] = mode.lower()
        
        st.divider()
        streak = st.session_state.get('streak_info', {})
        st.metric("Current Streak", f"{streak.get('streak', 0)} days")
        st.metric("Longest Streak", f"{streak.get('longest', 0)} days")
        
        debt = logic.get_debt_count()
        if debt > 0: st.error(f"E-Levels: {debt}")
        else: st.success("System Stable")

        st.divider()
        render_api_key_section()

# ============================================================
# MODES
# ============================================================
def render_stream_mode():
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('waves', size=40)} THE STREAM</h1><p>Atomic thoughts. Think aloud.</p></div>", unsafe_allow_html=True)
    
    if st.session_state.get('violation_pending'):
        v = st.session_state['violation_pending']
        st.warning(f"{icons.get_icon_text('shield-alert')} Alignment Error against Core: \"{v['core']['content'][:50]}...\"")
        st.info(f"Input: {v['text']}")
        
        c1, c2 = st.columns(2)
        if c1.button(f"{icons.get_icon_text('skull')} Stop & Analyze Gap"):
            db.increment_debt(1)
            del st.session_state['violation_pending']
            st.rerun()
            
        if c2.button(f"{icons.get_icon_text('zap')} Force Merge (Entropy +1)"):
            db.increment_debt(1)
            logic.save_log(v['text'])
            del st.session_state['violation_pending']
            st.toast("Forced merge. Entropy increased.", icon="🚨")
            st.rerun()
            
        if st.button(f"{icons.get_icon_text('trash')} Discard"):
            del st.session_state['violation_pending']
            st.rerun()
        return

    echo = st.session_state.get('current_echo')
    if echo:
        echo_created_at = str(echo.get('created_at') or '')
        echo_content = str(echo.get('content') or '')
        st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border-left: 3px solid #666; padding: 15px; margin-bottom: 20px; border-radius: 4px; font-style: italic; color: #aaa;">
            <small>{icons.get_icon('sparkles', size=14)} Echo from {echo_created_at[:10]}</small><br>"{echo_content}"</div>""", unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if user_input := st.chat_input("What is the single sentence that defines you right now?"):
        process_stream_input(user_input)

def process_stream_input(user_input: str):
    status, core = logic.evaluate_input_integrity(user_input)
    if status == "VIOLATION":
        st.session_state['violation_pending'] = {"text": user_input, "core": core}
        st.rerun()
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    logic.save_chat_message("user", user_input)
    logic.save_log(user_input)
    
    if st.session_state['first_input_of_session']:
        st.toast("Log captured. Meteor Effect.", icon="☄️")
        st.session_state['first_input_of_session'] = False
    else:
        related = logic.find_related_logs(user_input)
        resp = logic.generate_response(user_input, related)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        logic.save_chat_message("assistant", resp)
    
    st.session_state['current_echo'] = logic.get_current_echo(reference_text=user_input)
    st.rerun()

def render_chronos_mode():
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('timer', size=40)} CHRONOS</h1><p>Time is the currency.</p></div>", unsafe_allow_html=True)
    
    if st.session_state['chronos_running'] and st.session_state['chronos_end_time']:
        if (st.session_state['chronos_end_time'] - datetime.now(timezone.utc)).total_seconds() <= 0:
            st.session_state['chronos_running'] = False
            st.session_state['chronos_finished'] = True

    if st.session_state['chronos_finished']: render_chronos_docking()
    elif st.session_state['chronos_running']: render_chronos_timer()
    else: render_chronos_setup()

def render_chronos_timer():
    rem = st.session_state['chronos_end_time'] - datetime.now(timezone.utc)
    mins, secs = divmod(max(0, int(rem.total_seconds())), 60)
    st.markdown(f"<h1 style='text-align:center; font-size:80px; color:#00FFFF;'>{mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    if c1.button(f"{icons.get_icon_text('check-circle')} 완료", use_container_width=True):
        st.session_state['chronos_running'] = False
        st.session_state['chronos_finished'] = True
        st.rerun()
    if c2.button(f"{icons.get_icon_text('shield-alert')} 취소", use_container_width=True):
        st.session_state['chronos_running'] = False
        st.rerun()
    time.sleep(2); st.rerun()

def render_chronos_setup():
    c1, c2, c3 = st.columns(3)
    if c1.button(f"{icons.get_icon_text('flame')} 25분", use_container_width=True): start_timer(25)
    if c2.button(f"{icons.get_icon_text('target')} 60분", use_container_width=True): start_timer(60)
    mins = c3.number_input("분", 1, 180, 45)
    if c3.button(f"{icons.get_icon_text('zap')} 시작", use_container_width=True): start_timer(mins)

def start_timer(m: int):
    end_time = datetime.now(timezone.utc) + timedelta(minutes=m)
    st.session_state['chronos_duration'] = m
    st.session_state['chronos_end_time'] = end_time
    st.session_state['chronos_running'] = True
    db.set_chronos_timer(end_time)  # [B-2] DB에 영속화
    st.rerun()

def render_chronos_docking():
    st.info(f"{icons.get_icon_text('anchor')} 이 시간은 어떤 헌법에 귀속됩니까?")
    consts = db.get_constitutions()
    options = {c['content'][:50]: c['id'] for c in consts}
    
    if not options:
        st.warning("헌법이 없습니다."); return

    sel = st.multiselect("헌법 선택", list(options.keys()))
    acc = st.text_area("성취 기록 (최소 10자)")
    if st.button(f"{icons.get_icon_text('anchor')} Dock", use_container_width=True, type="primary", disabled=len(sel)==0 or len(acc)<10):
        logic.save_chronos_log(acc, [options[n] for n in sel], st.session_state['chronos_duration'])
        db.clear_chronos_timer()  # [B-2] Dock 완료 후 타이머 클리어
        st.balloons(); st.session_state['chronos_finished'] = False; st.rerun()

def render_universe_mode():
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('orbit', size=40)} SOUL ANALYTICS</h1></div>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Cosmos", "Soul Analytics", "Legacy", "Deep Space (3D)"])
    
    with t1:
        st.caption("관측할 별을 선택하고, 새로운 별자리를 연결하세요.")
        logs = logic.load_logs()
        opts = {f"[{l['meta_type']}] {l['content'][:30]}...": l['id'] for l in logs}
        sel = st.selectbox("관측 대상", list(opts.keys()))
        if sel:
            log = logic.get_log_by_id(opts[sel])
            st.info(f"**{log['meta_type']}** | {log['content']}")
            
    with t2:
        render_soul_analytics()
        
    with t3:
        st.markdown(f"### {icons.get_icon_text('layout-dashboard')} Soul Finviz (Legacy)")
        data = logic.get_finviz_data()
        if data:
            safe_data = []
            for d in data:
                if not d or not d.get("content"):
                    continue
                log_count = int(d.get("log_count", 0) or 0)
                duration = int(d.get("size", d.get("total_duration", d.get("duration", 0))) or 0)
                safe_data.append(
                    {
                        "content": str(d.get("content", "")),
                        "size": max(1, duration if duration > 0 else (log_count * 10 or 1)),
                        "health_score": float(d.get("health_score", 0.0) or 0.0),
                    }
                )

            if not safe_data:
                st.info("표시할 Core 통계가 없습니다.")
            else:
                fig = go.Figure(go.Treemap(
                    labels=[d['content'][:30] for d in safe_data],
                    parents=["" for _ in safe_data],
                    values=[d['size'] for d in safe_data],
                    marker=dict(colors=[d['health_score'] for d in safe_data], colorscale='Viridis')
                ))
                st.plotly_chart(fig, use_container_width=True)

    with t4:
        st.markdown(f"### {icons.get_icon_text('orbit')} 1st Person Explorer")
        logs = logic.load_logs()
        cores = db.get_cores()
        render_3d_universe(logs, cores)

def render_soul_analytics():
    st.markdown(f"### {icons.get_icon_text('calendar')} Willpower Density")
    density = logic.get_density_data()
    if not density.empty:
        fig1 = px.density_heatmap(density, x="date", y="intensity", nbinsx=30, nbinsy=4, color_continuous_scale="Viridis")
        fig1.update_layout(xaxis_title="Date", yaxis_title="Intensity", height=300)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("데이터가 충분하지 않습니다.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {icons.get_icon_text('skull')} Saboteur Analysis")
        saboteur = logic.get_saboteur_data()
        if not saboteur.empty:
            fig2 = px.bar(saboteur, x="count", y="tag", orientation='h', color="count", color_continuous_scale="Reds")
            st.plotly_chart(fig2, use_container_width=True)
        else: st.info("실패 기록이 없습니다.")
            
    with c2:
        st.markdown(f"### {icons.get_icon_text('activity')} Narrative Net Worth")
        nw = logic.get_net_worth_data()
        if not nw.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=nw['date'], y=nw['cum_assets'], mode='lines', name='Assets', stackgroup='one', fill='tonexty'))
            fig3.add_trace(go.Scatter(x=nw['date'], y=nw['cum_debt'], mode='lines', name='Liabilities', stackgroup='one', fill='tonexty', line=dict(color='red')))
            st.plotly_chart(fig3, use_container_width=True)
        else: st.info("데이터가 없습니다.")

def render_control_mode():
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('layout-dashboard', size=40)} CONTROL</h1></div>", unsafe_allow_html=True)
    
    cores = db.get_cores()
    options = {c['content'][:50]: c['id'] for c in cores}
    if options:
        c1, c2 = st.columns([2, 1])
        thought = c1.text_input("새로운 생각", key="kb_new")
        star = c2.selectbox("소속 Core", list(options.keys()), key="kb_const")
        if st.button(f"{icons.get_icon_text('sparkles')} 궤도 투입") and thought:
            logic.create_kanban_card(thought, options[star]); st.rerun()
    
    cards = logic.get_kanban_cards()
    cols = st.columns(3)
    labels = [("draft", "💭 Drafts"), ("orbit", "🚀 In Orbit"), ("landed", "✅ Landed")]
    for i, (status, label) in enumerate(labels):
        with cols[i]:
            st.markdown(f"#### {label}")
            for card in cards.get(status, []):
                with st.container():
                    st.markdown(f"<div class='kanban-card'>{card['content'][:60]}</div>", unsafe_allow_html=True)
                    if status == "draft" and st.button(f"{icons.get_icon_text('orbit')} Orbit", key=f"orb_{card['id']}"):
                        logic.move_kanban_card(card['id'], "orbit"); st.rerun()
                    elif status == "orbit" and st.button(f"{icons.get_icon_text('anchor')} Land", key=f"land_{card['id']}"):
                        st.session_state['docking_modal_active'] = True
                        st.session_state['docking_card_id'] = card['id']; st.rerun()

    if st.session_state['docking_modal_active']:
        render_kanban_docking(options)

def render_kanban_docking(options):
    st.divider()
    st.markdown(f"### {icons.get_icon_text('anchor')} Kanban Docking")
    sel = st.multiselect("Core 선택", list(options.keys()), key="k_dock_sel")
    acc = st.text_input("성취 요약", key="k_dock_acc")
    dur = st.number_input("시간(분)", 0, 480, 0, key="k_dock_dur")
    if st.button(f"{icons.get_icon_text('check-circle')} Confirm Dock", type="primary"):
        logic.land_kanban_card(st.session_state['docking_card_id'], [options[n] for n in sel], acc, dur)
        st.session_state['docking_modal_active'] = False; st.rerun()

def render_desk_mode():
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('book-open', size=40)} THE DESK</h1></div>", unsafe_allow_html=True)
    l, r = st.columns([1, 1.5])
    with l:
        st.markdown(f"#### {icons.get_icon_text('sparkles')} Fragments")
        frags, count = logic.get_fragments_paginated(st.session_state['desk_page'])
        for f in frags:
            with st.expander(f['content'][:40]):
                st.write(f['content'])
                if st.button(f"{icons.get_icon_text('plus-circle')} 에세이 추가", key=f"add_{f['id']}"): 
                    st.session_state['selected_cards'].append(f['id']); st.rerun()
    with r:
        st.markdown(f"#### {icons.get_icon_text('pencil')} Essay")
        essay = st.text_area("Connect your story", height=400)
        if st.button(f"{icons.get_icon_text('save')} Save Essay") and essay:
            logic.save_log(essay); st.toast("Saved!"); st.rerun()

# ============================================================
# Main Loop
# ============================================================
def main():
    init_page_config()
    try:
        init_session_state()
    except Exception as exc:
        st.error("Database initialization failed. Check DATASTORE and DATABASE_URL settings.")
        st.caption("Startup halted to prevent repeated runtime failures.")
        st.code(_safe_startup_error(exc))
        return
    is_entropy = logic.is_entropy_mode()
    apply_atmosphere(is_entropy); render_sidebar(is_entropy)
    
    if is_entropy:
        st.error(f"{icons.get_icon_text('shield-alert')} ENTROPY ALERT: SYSTEM UNSTABLE")
        st.markdown(f"### {icons.get_icon_text('skull')} 시스템 엔트로피가 임계점을 넘었습니다.")
        with st.form("gap_analysis"):
            cores = db.get_cores()
            sel = st.selectbox("관련된 Core Violation", [c['content'] for c in cores] if cores else ["Unknown"])
            st.markdown(f"#### 1. Saboteur Analysis ({icons.get_icon_text('skull')} 실패 원인)")
            tag_h = logic.get_tag_hierarchy()
            p_cat = st.radio("Root Cause", list(tag_h.keys()), horizontal=True)
            c_tags = tag_h[p_cat]
            c1, c2 = st.columns([3, 1])
            s_tag = c1.selectbox("Specific Reason", c_tags + ["➕ Create New..."])
            f_tag = s_tag if s_tag != "➕ Create New..." else c2.text_input("New Tag Name")
            st.markdown(f"#### 2. Gap Analysis ({icons.get_icon_text('book-open')} 격차 분석)")
            reason = st.text_area("분석 (100자 이상) - 왜 의지와 행동의 격차가 발생했습니까?")
            plan = st.text_area("보정 계획 (Calibration)")
            if st.form_submit_button(f"{icons.get_icon_text('zap')} 엔트로피 해소 (Repay Debt)"):
                if len(reason) < 100: st.error("분석이 너무 얕습니다.")
                else:
                    core_id = [c['id'] for c in cores if c['content']==sel][0] if cores else None
                    if core_id:
                        logic.process_gap(reason, core_id, plan, tags=[f_tag] if f_tag else [])
                        st.success("엔트로피 감소 확인. 시스템 정상화."); time.sleep(1.5); st.rerun()
                    else:
                        st.error("Core를 찾을 수 없습니다.")
        return

    if st.session_state['gatekeeper_dismissed'] and not st.session_state['interventions_checked']:
        for m in logic.run_active_intervention(): st.toast(m, icon="🔔")
        st.session_state['interventions_checked'] = True

    m = st.session_state['mode']
    if m == "stream": render_stream_mode()
    elif m == "chronos": render_chronos_mode()
    elif m == "universe": render_universe_mode()
    elif m == "control": render_control_mode()
    elif m == "desk": render_desk_mode()

if __name__ == "__main__":
    main()



