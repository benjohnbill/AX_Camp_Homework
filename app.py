"""
app.py
Antigravity v5: 5-Mode Architecture - Refactored for Modularity & Robust Icons
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta, timezone
import time
import json

import narrative_logic as logic
import icons
db = logic.db

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page Config & Initialization
# ============================================================
def init_page_config():
    st.set_page_config(
        page_title="Antigravity",
        page_icon=":milky_way:",
        layout="wide"
    )

def init_session_state():
    if 'diagnostics_run' not in st.session_state:
        db.inject_genesis_data(logic.get_embedding)
        logic.run_startup_diagnostics()
        st.session_state['diagnostics_run'] = True

    if 'streak_updated' not in st.session_state:
        result = db.check_streak_and_apply_penalty() # [Refactor] Moved to db_manager
        st.session_state['streak_info'] = result['streak_info']
        if result['penalty_applied']:
            st.session_state['entropy_mode_trigger'] = True # [Refactor]
        st.session_state['streak_updated'] = True

    if 'mode' not in st.session_state:
        st.session_state['mode'] = "stream"

    if 'messages' not in st.session_state:
        welcome = logic.get_welcome_message()
        st.session_state.messages = [{"role": "assistant", "content": welcome}]

    if 'current_echo' not in st.session_state:
        st.session_state['current_echo'] = logic.get_current_echo()

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
        'entropy_mode_trigger': False # [Refactor]
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def apply_atmosphere(entropy_mode: bool):
    # [Refactor] Entropy Mode: Desaturated, Glitchy, Cold Grey/White
    if entropy_mode:
        bg = "linear-gradient(to bottom, #2b2b2b 0%, #000000 100%)" # Cold Grey/Black
        text_color = "#a0a0a0" # Dimmed text
    else:
        # Standard: Deep Space Blue
        bg = "radial-gradient(circle at center, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
        text_color = "#e94560" # Vivd Red/Pink accent
    
    st.markdown(f"""
        <style>
        .stApp {{
            background: {bg} !important;
            color: {text_color} !important;
        }}
        /* Ensure core containers remain readable across theme/runtime differences */
        .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        section[data-testid="stSidebar"] *, [data-testid="stMarkdownContainer"] * {{
            color: {text_color} !important;
        }}
        /* Keep input widgets readable on dark atmospheric backgrounds */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stChatInput"] textarea,
        [data-testid="stSelectbox"] div,
        [data-testid="stNumberInput"] input {{
            color: #f5f7ff !important;
            background-color: rgba(255, 255, 255, 0.08) !important;
        }}
        [data-testid="stButton"] button {{
            color: #f5f7ff !important;
            border-color: rgba(255, 255, 255, 0.25) !important;
        }}
        /* Global Text Adjustment for Entropy Mode */
        {'body { filter: grayscale(100%); }' if entropy_mode else ''}
        .kanban-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px; padding: 15px; margin-bottom: 10px; transition: all 0.3s ease;
        }}
        .kanban-card:hover {{ background: rgba(255, 255, 255, 0.08); border-color: rgba(255, 255, 255, 0.3); transform: translateY(-2px); }}
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
        st.title(f"{icons.get_icon('orbit')} Antigravity")
        st.caption("Existential Audit Protocol v6.0") # Version Bump
        st.divider()
        
        if entropy_mode:
            st.warning(f"{icons.get_icon_svg('shield-alert', size=18)} ENTROPY ALERT")
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
        st.warning(f"{icons.get_icon('shield-alert')} Alignment Error against Core: \"{v['core']['content'][:50]}...\"")
        st.info(f"Input: {v['text']}")
        
        c1, c2 = st.columns(2)
        if c1.button(f"{icons.get_icon_svg('skull', size=18)} Stop & Analyze Gap"):
            db.increment_debt(1)
            del st.session_state['violation_pending']
            st.rerun()
            
        if c2.button(f"{icons.get_icon_svg('zap', size=18)} Force Merge (Entropy +1)"):
            db.increment_debt(1)
            logic.save_log(v['text'])
            del st.session_state['violation_pending']
            st.toast("Forced merge. Entropy increased.", icon="🚨")
            st.rerun()
            
        if st.button(f"{icons.get_icon_svg('trash', size=18)} Discard"):
            del st.session_state['violation_pending']
            st.rerun()
        return

    echo = st.session_state.get('current_echo')
    if echo:
        st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border-left: 3px solid #666; padding: 15px; margin-bottom: 20px; border-radius: 4px; font-style: italic; color: #aaa;">
            <small>{icons.get_icon('sparkles', size=14)} Echo from {echo.get('created_at', '')[:10]}</small><br>"{echo.get('content', '')}"</div>""", unsafe_allow_html=True)
    
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
    if c1.button(f"{icons.get_icon_svg('check-circle', size=18)} 완료", use_container_width=True):
        st.session_state['chronos_running'] = False
        st.session_state['chronos_finished'] = True
        st.rerun()
    if c2.button(f"{icons.get_icon_svg('shield-alert', size=18)} 취소", use_container_width=True):
        st.session_state['chronos_running'] = False
        st.rerun()
    time.sleep(2); st.rerun()

def render_chronos_setup():
    c1, c2, c3 = st.columns(3)
    if c1.button(f"{icons.get_icon_svg('flame', size=18)} 25분", use_container_width=True): start_timer(25)
    if c2.button(f"{icons.get_icon_svg('target', size=18)} 60분", use_container_width=True): start_timer(60)
    mins = c3.number_input("분", 1, 180, 45)
    if c3.button(f"{icons.get_icon_svg('zap', size=18)} 시작", use_container_width=True): start_timer(mins)

def start_timer(m: int):
    st.session_state['chronos_duration'] = m
    st.session_state['chronos_end_time'] = datetime.now(timezone.utc) + timedelta(minutes=m)
    st.session_state['chronos_running'] = True
    st.rerun()

def render_chronos_docking():
    st.info(f"{icons.get_icon('anchor')} 이 시간은 어떤 헌법에 귀속됩니까?")
    consts = db.get_constitutions()
    options = {c['content'][:50]: c['id'] for c in consts}
    
    if not options:
        st.warning("헌법이 없습니다."); return

    sel = st.multiselect("헌법 선택", list(options.keys()))
    acc = st.text_area("성취 기록 (최소 10자)")
    if st.button(f"{icons.get_icon_svg('anchor', size=18)} Dock", use_container_width=True, type="primary", disabled=len(sel)==0 or len(acc)<10):
        logic.save_chronos_log(acc, [options[n] for n in sel], st.session_state['chronos_duration'])
        st.balloons(); st.session_state['chronos_finished'] = False; st.rerun()

def render_universe_mode():
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('orbit', size=40)} SOUL ANALYTICS</h1></div>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Cosmos", "Soul Analytics", "Legacy"])
    
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
        st.markdown(f"### {icons.get_icon('layout-dashboard')} Soul Finviz (Legacy)")
        data = logic.get_finviz_data()
        if data:
            fig = go.Figure(go.Treemap(
                labels=[d['content'][:30] for d in data],
                parents=["" for _ in data],
                values=[d['size'] for d in data],
                marker=dict(colors=[d['health_score'] for d in data], colorscale='Viridis')
            ))
            st.plotly_chart(fig, use_container_width=True)

def render_soul_analytics():
    st.markdown(f"### {icons.get_icon('calendar')} Willpower Density")
    density = logic.get_density_data()
    if not density.empty:
        fig1 = px.density_heatmap(density, x="date", y="intensity", nbinsx=30, nbinsy=4, color_continuous_scale="Viridis")
        fig1.update_layout(xaxis_title="Date", yaxis_title="Intensity", height=300)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("데이터가 충분하지 않습니다.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {icons.get_icon('skull')} Saboteur Analysis")
        saboteur = logic.get_saboteur_data()
        if not saboteur.empty:
            fig2 = px.bar(saboteur, x="count", y="tag", orientation='h', color="count", color_continuous_scale="Reds")
            st.plotly_chart(fig2, use_container_width=True)
        else: st.info("실패 기록이 없습니다.")
            
    with c2:
        st.markdown(f"### {icons.get_icon('activity')} Narrative Net Worth")
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
        if st.button(f"{icons.get_icon_svg('sparkles', size=18)} 궤도 투입") and thought:
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
                    if status == "draft" and st.button(f"{icons.get_icon_svg('orbit', size=14)} Orbit", key=f"orb_{card['id']}"):
                        logic.move_kanban_card(card['id'], "orbit"); st.rerun()
                    elif status == "orbit" and st.button(f"{icons.get_icon_svg('anchor', size=14)} Land", key=f"land_{card['id']}"):
                        st.session_state['docking_modal_active'] = True
                        st.session_state['docking_card_id'] = card['id']; st.rerun()

    if st.session_state['docking_modal_active']:
        render_kanban_docking(options)

def render_kanban_docking(options):
    st.divider()
    st.markdown(f"### {icons.get_icon('anchor')} Kanban Docking")
    sel = st.multiselect("Core 선택", list(options.keys()), key="k_dock_sel")
    acc = st.text_input("성취 요약", key="k_dock_acc")
    dur = st.number_input("시간(분)", 0, 480, 0, key="k_dock_dur")
    if st.button(f"{icons.get_icon_svg('check-circle', size=18)} Confirm Dock", type="primary"):
        logic.land_kanban_card(st.session_state['docking_card_id'], [options[n] for n in sel], acc, dur)
        st.session_state['docking_modal_active'] = False; st.rerun()

def render_desk_mode():
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('book-open', size=40)} THE DESK</h1></div>", unsafe_allow_html=True)
    l, r = st.columns([1, 1.5])
    with l:
        st.markdown(f"#### {icons.get_icon('sparkles')} Fragments")
        frags, count = logic.get_fragments_paginated(st.session_state['desk_page'])
        for f in frags:
            with st.expander(f['content'][:40]):
                st.write(f['content'])
                if st.button(f"{icons.get_icon_svg('plus-circle', size=14)} 에세이 추가", key=f"add_{f['id']}"): 
                    st.session_state['selected_cards'].append(f['id']); st.rerun()
    with r:
        st.markdown(f"#### {icons.get_icon('pencil')} Essay")
        essay = st.text_area("Connect your story", height=400)
        if st.button(f"{icons.get_icon_svg('save', size=18)} Save Essay") and essay:
            logic.save_log(essay); st.toast("Saved!"); st.rerun()

# ============================================================
# Main Loop
# ============================================================
def main():
    init_page_config(); init_session_state()
    is_entropy = logic.is_entropy_mode()
    apply_atmosphere(is_entropy); render_sidebar(is_entropy)
    
    if is_entropy:
        st.error(f"{icons.get_icon('shield-alert')} ENTROPY ALERT: SYSTEM UNSTABLE")
        st.markdown(f"### {icons.get_icon('skull')} 시스템 엔트로피가 임계점을 넘었습니다.")
        with st.form("gap_analysis"):
            cores = db.get_cores()
            sel = st.selectbox("관련된 Core Violation", [c['content'] for c in cores] if cores else ["Unknown"])
            st.markdown(f"#### 1. Saboteur Analysis ({icons.get_icon_svg('skull', size=16)} 실패 원인)")
            tag_h = logic.get_tag_hierarchy()
            p_cat = st.radio("Root Cause", list(tag_h.keys()), horizontal=True)
            c_tags = tag_h[p_cat]
            c1, c2 = st.columns([3, 1])
            s_tag = c1.selectbox("Specific Reason", c_tags + ["➕ Create New..."])
            f_tag = s_tag if s_tag != "➕ Create New..." else c2.text_input("New Tag Name")
            st.markdown(f"#### 2. Gap Analysis ({icons.get_icon_svg('book-open', size=16)} 격차 분석)")
            reason = st.text_area("분석 (100자 이상) - 왜 의지와 행동의 격차가 발생했습니까?")
            plan = st.text_area("보정 계획 (Calibration)")
            if st.form_submit_button(f"{icons.get_icon_svg('zap', size=18)} 엔트로피 해소 (Repay Debt)"):
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



