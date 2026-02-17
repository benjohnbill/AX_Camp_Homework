"""
app.py
Antigravity v5: 5-Mode Architecture - Refactored for Modularity
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta, timezone
import time
import json

import narrative_logic as logic
import db_manager as db
import icons

import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page Config & Initialization
# ============================================================
def init_page_config():
    st.set_page_config(
        page_title="Antigravity",
        page_icon=icons.get_icon("galaxy"),
        layout="wide"
    )

def init_session_state():
    if 'diagnostics_run' not in st.session_state:
        db.inject_genesis_data(logic.get_embedding)
        logic.run_startup_diagnostics()
        st.session_state['diagnostics_run'] = True

    if 'streak_updated' not in st.session_state:
        st.session_state['streak_info'] = logic.update_streak()
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
        'desk_page': 1
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def apply_atmosphere(red_mode: bool):
    bg = "linear-gradient(to bottom, #330000 0%, #000000 100%)" if red_mode else \
         "radial-gradient(circle at center, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
    
    st.markdown(f"""
        <style>
        .stApp {{ background: {bg} !important; color: #e94560 !important; }}
        .kanban-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px; padding: 15px; margin-bottom: 10px; transition: all 0.3s ease;
        }}
        .kanban-card:hover {{ background: rgba(255, 255, 255, 0.08); border-color: rgba(255, 255, 255, 0.3); transform: translateY(-2px); }}
        </style>
    """, unsafe_allow_html=True)

# ============================================================
# Navigation
# ============================================================
def render_sidebar(red_mode: bool):
    with st.sidebar:
        st.title(f"{icons.get_icon('orbit')} Antigravity")
        st.caption("Existential Audit Protocol v5.2")
        st.divider()
        
        if red_mode:
            st.warning("⚠️ RED MODE ACTIVE")
            st.info("해명을 완료할 때까지 모든 기능이 잠깁니다.")
        else:
            mode = st.radio("Select Mode", ["Stream", "Chronos", "Universe", "Control", "Desk"],
                            index=["stream", "chronos", "universe", "control", "desk"].index(st.session_state['mode']))
            st.session_state['mode'] = mode.lower()
        
        st.divider()
        streak = st.session_state.get('streak_info', {})
        st.metric("Current Streak", f"{streak.get('streak', 0)} days")
        st.metric("Longest Streak", f"{streak.get('longest', 0)} days")
        
        debt = logic.get_debt_count()
        if debt > 0: st.error(f"Unpaid Debts: {debt}")
        else: st.success("No active debts.")

# ============================================================
# MODES
# ============================================================
def render_stream_mode():
    st.markdown("<div style='text-align:center;'><h1>🌊 THE STREAM</h1><p>Atomic thoughts. Hot state. Think aloud.</p></div>", unsafe_allow_html=True)
    
    echo = st.session_state.get('current_echo')
    if echo:
        st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border-left: 3px solid #888; padding: 15px; margin-bottom: 20px; border-radius: 4px; font-style: italic; color: #ccc;">
            <small>✨ Echo from {echo.get('created_at', '')[:10]}</small><br>"{echo.get('content', '')}"</div>""", unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if user_input := st.chat_input("What is the single sentence that defines you right now?"):
        process_stream_input(user_input)

def process_stream_input(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})
    logic.save_chat_message("user", user_input)
    logic.save_log(user_input)
    
    if st.session_state['first_input_of_session']:
        st.toast("저장됨. Meteor Effect.", icon="☄️")
        st.session_state['first_input_of_session'] = False
    else:
        related = logic.find_related_logs(user_input)
        resp = logic.generate_response(user_input, related)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        logic.save_chat_message("assistant", resp)
    
    st.session_state['current_echo'] = logic.get_current_echo(reference_text=user_input)
    st.rerun()

def render_chronos_mode():
    st.markdown("<div style='text-align:center;'><h1>⏱️ CHRONOS</h1><p>Time is the currency.</p></div>", unsafe_allow_html=True)
    
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
    st.markdown(f"<h1 style='text-align:center; font-size:80px;'>{mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    if c1.button("⏹️ 완료", use_container_width=True):
        st.session_state['chronos_running'] = False
        st.session_state['chronos_finished'] = True
        st.rerun()
    if c2.button("❌ 취소", use_container_width=True):
        st.session_state['chronos_running'] = False
        st.rerun()
    time.sleep(2); st.rerun()

def render_chronos_setup():
    c1, c2, c3 = st.columns(3)
    if c1.button("🔥 25분", use_container_width=True): start_timer(25)
    if c2.button("⚡ 60분", use_container_width=True): start_timer(60)
    mins = c3.number_input("분", 1, 180, 45)
    if c3.button("🚀 시작", use_container_width=True): start_timer(mins)

def start_timer(m: int):
    st.session_state['chronos_duration'] = m
    st.session_state['chronos_end_time'] = datetime.now(timezone.utc) + timedelta(minutes=m)
    st.session_state['chronos_running'] = True
    st.rerun()

def render_chronos_docking():
    st.info("⚓ 이 시간은 어떤 헌법에 귀속됩니까?")
    consts = db.get_constitutions()
    options = {c['content'][:50]: c['id'] for c in consts}
    
    if not options:
        st.warning("헌법이 없습니다."); return

    sel = st.multiselect("헌법 선택", list(options.keys()))
    acc = st.text_area("성취 기록 (최소 10자)")
    if st.button("⚓ Dock", use_container_width=True, type="primary", disabled=len(sel)==0 or len(acc)<10):
        logic.save_chronos_log(acc, [options[n] for n in sel], st.session_state['chronos_duration'])
        st.balloons(); st.session_state['chronos_finished'] = False; st.rerun()

def render_universe_mode():
    st.markdown("<div style='text-align:center;'><h1>🌌 THE UNIVERSE</h1></div>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Cosmos", "Finviz", "Rhythm", "Pulse"])
    
    with t1:
        st.caption("관측할 별을 선택하고, 새로운 별자리를 연결하세요.")
        logs = logic.load_logs()
        opts = {f"[{l['meta_type']}] {l['content'][:30]}...": l['id'] for l in logs}
        sel = st.selectbox("관측 대상", list(opts.keys()))
        if sel:
            log = logic.get_log_by_id(opts[sel])
            st.info(f"**{log['meta_type']}** | {log['content']}")
            
    with t2:
        st.markdown("### 📊 Soul Finviz")
        data = logic.get_finviz_data()
        if data:
            fig = go.Figure(go.Treemap(
                labels=[d['content'][:30] for d in data],
                parents=["" for _ in data],
                values=[d['size'] for d in data],
                marker=dict(colors=[d['health_score'] for d in data], colorscale='Viridis')
            ))
            st.plotly_chart(fig, use_container_width=True)

def render_control_mode():
    st.markdown("<div style='text-align:center;'><h1>📋 CONTROL</h1></div>", unsafe_allow_html=True)
    
    consts = db.get_constitutions()
    options = {c['content'][:50]: c['id'] for c in consts}
    if options:
        c1, c2 = st.columns([2, 1])
        thought = c1.text_input("새로운 생각", key="kb_new")
        star = c2.selectbox("소속 헌법", list(options.keys()), key="kb_const")
        if st.button("🌀 궤도 투입") and thought:
            logic.create_kanban_card(thought, options[star]); st.rerun()
    
    cards = logic.get_kanban_cards()
    cols = st.columns(3)
    for i, (status, label) in enumerate([("draft", "💭 Drafts"), ("orbit", "🚀 In Orbit"), ("landed", "🏁 Landed")]):
        with cols[i]:
            st.markdown(f"#### {label}")
            for card in cards.get(status, []):
                with st.container():
                    st.markdown(f"<div class='kanban-card'>{card['content'][:60]}</div>", unsafe_allow_html=True)
                    if status == "draft":
                        if st.button("→ Orbit", key=f"orb_{card['id']}"): logic.move_kanban_card(card['id'], "orbit"); st.rerun()
                    elif status == "orbit":
                        if st.button("→ Land", key=f"land_{card['id']}"): 
                            st.session_state['docking_modal_active'] = True
                            st.session_state['docking_card_id'] = card['id']; st.rerun()

    if st.session_state['docking_modal_active']:
        render_kanban_docking(options)

def render_kanban_docking(options):
    st.divider()
    st.markdown("### ⚓ Kanban Docking")
    sel = st.multiselect("헌법 선택", list(options.keys()), key="k_dock_sel")
    acc = st.text_input("성취 요약", key="k_dock_acc")
    dur = st.number_input("시간(분)", 0, 480, 0, key="k_dock_dur")
    if st.button("⚓ Confirm Dock", type="primary"):
        logic.land_kanban_card(st.session_state['docking_card_id'], [options[n] for n in sel], acc, dur)
        st.session_state['docking_modal_active'] = False; st.rerun()

def render_desk_mode():
    st.markdown("<div style='text-align:center;'><h1>📖 THE DESK</h1></div>", unsafe_allow_html=True)
    l, r = st.columns([1, 1.5])
    with l:
        st.markdown("#### Fragments")
        frags, count = logic.get_fragments_paginated(st.session_state['desk_page'])
        for f in frags:
            with st.expander(f['content'][:40]):
                st.write(f['content'])
                if st.button("에세이 추가", key=f"add_{f['id']}"): 
                    st.session_state['selected_cards'].append(f['id']); st.rerun()
    with r:
        st.markdown("#### Essay")
        essay = st.text_area("Connect your story", height=400)
        if st.button("Save Essay") and essay:
            logic.save_log(essay); st.toast("Saved!"); st.rerun()

# ============================================================
# Main Loop
# ============================================================
def main():
    init_page_config(); init_session_state()
    red = logic.is_red_mode()
    apply_atmosphere(red); render_sidebar(red)
    
    if red:
        st.error("🟥 RED PROTOCOL: VIOLATION DETECTED")
        with st.form("apology"):
            consts = db.get_constitutions()
            sel = st.selectbox("위반한 헌법", [c['content'] for c in consts])
            reason = st.text_area("해명 (100자 이상)")
            plan = st.text_area("내일의 약속")
            if st.form_submit_button("제출") and len(reason) >= 100:
                logic.process_apology(reason, [c['id'] for c in consts if c['content']==sel][0], plan)
                st.rerun()
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
