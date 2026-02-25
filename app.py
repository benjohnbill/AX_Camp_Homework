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
import universe_auth
db = logic.db

import plotly.express as px
import plotly.graph_objects as go

_ALLOWED_MODES = ("stream", "desk", "chronos", "control", "universe")

_MODE_CARD_CONFIG = (
    ("desk", "Desk", "긴 글 작성과 정리"),
    ("chronos", "Chronos", "집중 타이머와 회고"),
    ("universe", "Universe", "분석과 3D 탐색"),
    ("control", "Control", "칸반 기반 통제"),
)

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


def _query_param_value(name: str) -> str:
    raw = st.query_params.get(name, "")
    if isinstance(raw, list):
        return str(raw[0] if raw else "")
    return str(raw)


def _is_universe_embed_request() -> bool:
    return _query_param_value("embed").strip().lower() == "universe_3d"

def _sanitize_mode(mode_value: str) -> str:
    mode = str(mode_value or "").strip().lower()
    return mode if mode in _ALLOWED_MODES else "stream"


def _ensure_valid_session_mode() -> str:
    current = _sanitize_mode(st.session_state.get("mode", "stream"))
    if st.session_state.get("mode") != current:
        st.session_state["mode"] = current
    return current


def _parse_positive_int(raw: str, default: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except Exception:
        return default


def _get_runtime_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return str(value).strip()
    try:
        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value).strip()
    except Exception:
        pass
    return default


def _emit_universe_session_cookie(cookie_name: str, cookie_value: str, max_age: int) -> None:
    """
    Streamlit does not expose response Set-Cookie APIs.
    We set Secure/SameSite cookie client-side for follow-up requests.
    HttpOnly requires an upstream gateway/proxy.
    """
    cookie_literal = json.dumps(
        f"{cookie_name}={cookie_value}; Max-Age={int(max_age)}; Path=/; Secure; SameSite=None"
    )
    components.html(
        f"<script>document.cookie = {cookie_literal};</script>",
        height=0,
        scrolling=False,
    )


# ---------------------------------------------------------------------------
# Auth error copy mapping: keyed by 'code' string from gateway payload.
# Each entry: (icon_key, headline, body)
# ---------------------------------------------------------------------------
_AUTH_ERROR_COPY = {
    "missing_token": (
        "shield-alert",
        "연결이 필요합니다.",
        "우주에 입장하기 위해서는 모바일 디바이스 또는 인증된 게이트웨이를 통한 안전한 접근이 필요합니다.",
    ),
    "token_expired": (
        "clock",
        "시간이 오래 지났습니다.",
        "안전을 위해 연결이 일시적으로 해제되었습니다. 안드로이드 앱에서 다시 연결해 주세요.",
    ),
    "forbidden_audience": (
        "lock",
        "접근 권한이 없습니다.",
        "현재 사용하신 열쇠로는 이 공간의 문을 열 수 없습니다.",
    ),
    "forbidden_issuer": (
        "lock",
        "접근 권한이 없습니다.",
        "현재 사용하신 열쇠로는 이 공간의 문을 열 수 없습니다.",
    ),
    "invalid_token": (
        "zap-off",
        "기억의 흐름이 끊겼습니다.",
        "인증 정보를 확인할 수 없습니다. 앱에서 다시 접속해 연결 상태를 점검해주세요.",
    ),
    "invalid_authorization": (
        "zap-off",
        "기억의 흐름이 끊겼습니다.",
        "인증 정보를 확인할 수 없습니다. 앱에서 다시 접속해 연결 상태를 점검해주세요.",
    ),
}

_AUTH_ERROR_FALLBACK = (
    "shield-alert",
    "우주적 미아 상태입니다.",
    "접근 요청을 확인할 수 없습니다. 연결 상태를 점검해주세요.",
)


def _get_auth_error_copy(code: str) -> tuple:
    """Return (icon_key, headline, body) for a given gateway error code."""
    return _AUTH_ERROR_COPY.get(code, _AUTH_ERROR_FALLBACK)


def _render_universe_auth_error(auth_result: universe_auth.AuthResult) -> None:
    payload = dict(auth_result.payload or {})
    payload.setdefault("status", auth_result.status)
    payload.setdefault("route", "universe_3d_embed")

    code = payload.get("code", "")
    icon_key, headline, body = _get_auth_error_copy(code)

    st.markdown("### 🌌 The Universe Space")
    st.error(f"{icons.get_icon_text(icon_key)} **{headline}**")
    st.markdown(body)

    with st.expander("Technical Support (For Debugging)"):
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
    _render_full_app_navigation_action()


def _render_full_app_navigation_action() -> None:
    st.markdown("---")
    st.caption("현재 화면은 임베드 라우트(`?embed=universe_3d`)일 수 있습니다.")
    if st.button("전체 앱으로 이동 (쿼리 제거)", key="return_full_app_from_embed", use_container_width=True):
        try:
            st.query_params.clear()
        except Exception:
            for key in list(st.query_params.keys()):
                del st.query_params[key]
        st.rerun()
    st.markdown("[또는 전체 앱 열기](./)")


def _run_universe_embed_route() -> bool:
    if not _is_universe_embed_request():
        return False

    jwt_secret = _get_runtime_secret("UNIVERSE_JWT_SECRET", "")
    session_secret = _get_runtime_secret("UNIVERSE_SESSION_SECRET", "") or jwt_secret
    issuer = _get_runtime_secret("UNIVERSE_AUTH_ISSUER", universe_auth.DEFAULT_ISSUER)
    audience = _get_runtime_secret("UNIVERSE_AUTH_AUDIENCE", universe_auth.DEFAULT_AUDIENCE)
    cookie_name = (
        _get_runtime_secret("UNIVERSE_SESSION_COOKIE", universe_auth.DEFAULT_COOKIE_NAME)
        or universe_auth.DEFAULT_COOKIE_NAME
    )
    session_ttl = _parse_positive_int(
        _get_runtime_secret("UNIVERSE_SESSION_TTL_SECONDS", str(universe_auth.DEFAULT_SESSION_TTL_SECONDS)),
        universe_auth.DEFAULT_SESSION_TTL_SECONDS,
    )

    headers = st.context.headers.to_dict() if hasattr(st, "context") else {}
    cookies = st.context.cookies.to_dict() if hasattr(st, "context") else {}
    auth_result = universe_auth.authenticate_request(
        headers=headers,
        cookies=cookies,
        jwt_secret=jwt_secret,
        session_secret=session_secret,
        issuer=issuer,
        audience=audience,
        cookie_name=cookie_name,
        session_ttl_seconds=session_ttl,
    )

    if not auth_result.ok:
        _render_universe_auth_error(auth_result)
        return True

    if auth_result.session_cookie_name and auth_result.session_cookie_value:
        _emit_universe_session_cookie(
            cookie_name=auth_result.session_cookie_name,
            cookie_value=auth_result.session_cookie_value,
            max_age=auth_result.session_cookie_max_age or session_ttl,
        )

    try:
        logs = logic.load_logs()
        cores = db.get_cores()
        render_3d_universe(logs, cores)
    except Exception as exc:
        st.markdown("### Universe 3D Render Error")
        st.code(
            json.dumps(
                {
                    "status": 500,
                    "code": "render_failure",
                    "message": "Universe 3D rendering failed after authentication.",
                    "detail": _safe_startup_error(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )
        _render_full_app_navigation_action()
    return True

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
        # streak_updated is the per-session guard.
        result = db.check_streak_and_apply_penalty()
        st.session_state['streak_info'] = result['streak_info']
        st.session_state['streak_updated'] = True

    if 'mode' not in st.session_state:
        st.session_state['mode'] = "stream"
    else:
        st.session_state['mode'] = _sanitize_mode(st.session_state['mode'])

    if 'messages' not in st.session_state:
        # Keep Stream home clean like a neutral chat workspace.
        st.session_state.messages = []

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
    """
    Applies minimal essential CSS. Heavy styling is now handled by .streamlit/config.toml
    to ensure native Streamlit components function correctly without breaking.
    """
    
    # 1. Essential CSS
    st.markdown("""
        <style>
        :root {
            --app-bg: #111317;
            --app-surface: #191c22;
            --app-surface-soft: #1f232b;
            --app-border: #2b3039;
            --app-text: #f3f5f7;
            --app-muted: #9aa3b2;
            --app-accent: #10a37f;
        }

        /* Hide Streamlit default header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 1.2rem !important;
            max-width: 1200px !important;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--app-bg);
        }

        .stream-shell {
            max-width: 860px;
            margin: 0 auto;
        }

        .stream-hero-title {
            font-size: 2rem;
            line-height: 1.2;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }

        .stream-hero-sub {
            color: var(--app-muted);
            margin-bottom: 1rem;
        }

        .mode-card {
            background: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
            min-height: 90px;
        }

        .mode-card-title {
            font-weight: 650;
            margin-bottom: 0.25rem;
        }

        .mode-card-sub {
            font-size: 0.86rem;
            color: var(--app-muted);
        }

        /* Essential Kanban Card styling without breaking native boxes */
        .kanban-card {
            background-color: var(--app-surface-soft) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 8px !important; 
            padding: 16px; 
            margin-bottom: 12px; 
        }

        [data-testid="stChatMessage"] {
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
            gap: 0.6rem;
        }

        [data-testid="stChatMessageContent"] {
            background: var(--app-surface-soft);
            border: 1px solid var(--app-border);
            border-radius: 16px;
            padding: 0.85rem 1rem;
        }

        [data-testid="stChatInput"] {
            border-radius: 16px;
            border: 1px solid var(--app-border);
            background: var(--app-surface-soft);
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--app-border);
        }

        .sidebar-section-title {
            color: var(--app-muted);
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        input, textarea {
            background-color: rgba(255, 255, 255, 0.03) !important;
        }
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
        st.title("Antigravity")
        st.caption("Narrative Loop Workspace")

        if st.button("새 스트림", use_container_width=True, key="sidebar_new_stream"):
            st.session_state["mode"] = "stream"
            st.session_state["messages"] = []
            st.session_state["first_input_of_session"] = True
            st.session_state.pop("refined_memo", None)
            st.rerun()

        st.divider()
        if entropy_mode:
            st.warning(f"{icons.get_icon_text('shield-alert')} ENTROPY ALERT")
            st.info("시스템 엔트로피가 임계치를 초과했습니다. [Gap Analysis]가 필요합니다.")
        else:
            current_mode = _ensure_valid_session_mode()
            st.markdown("<div class='sidebar-section-title'>Modes</div>", unsafe_allow_html=True)
            for mode in ("stream", "desk", "chronos", "universe", "control"):
                label = mode.title()
                if mode == current_mode:
                    label = f"● {label}"
                if st.button(label, key=f"sidebar_mode_{mode}", use_container_width=True):
                    st.session_state["mode"] = mode
                    st.rerun()

        st.divider()
        st.markdown("<div class='sidebar-section-title'>Recent Inputs</div>", unsafe_allow_html=True)
        recent_user_inputs = [
            str(msg.get("content", "")).strip()
            for msg in st.session_state.get("messages", [])
            if msg.get("role") == "user" and str(msg.get("content", "")).strip()
        ]
        if not recent_user_inputs:
            st.caption("아직 입력 기록이 없습니다.")
        else:
            for snippet in reversed(recent_user_inputs[-8:]):
                item = snippet if len(snippet) <= 46 else f"{snippet[:43]}..."
                st.caption(f"• {item}")

        st.divider()
        st.markdown("<div class='sidebar-section-title'>Image To Narrative</div>", unsafe_allow_html=True)
        with st.expander("📷 사진으로 서사 쓰기", expanded=False):
            uploaded_file = st.file_uploader(
                "이미지 업로드 (메모/풍경 등)",
                type=['png', 'jpg', 'jpeg'],
                key="vision_uploader_sidebar",
            )
            if uploaded_file:
                if st.button("사진 분석 및 서사 추출", use_container_width=True, key="vision_extract_sidebar"):
                    with st.spinner("이미지에서 서사를 추출하는 중..."):
                        image_bytes = uploaded_file.read()
                        vision_result = logic.refine_image_to_narrative_with_ai(image_bytes)
                        st.session_state['refined_memo'] = vision_result

        if 'refined_memo' in st.session_state:
            st.info(st.session_state['refined_memo'])
            if st.button("스트림에 즉시 저장", key="save_refined_sidebar", use_container_width=True, type="primary"):
                logic.save_log(st.session_state['refined_memo'])
                st.toast("서사가 스트림에 기록되었습니다.", icon="☄️")
                del st.session_state['refined_memo']
                st.rerun()

        st.divider()
        streak = st.session_state.get('streak_info', {})
        c1, c2 = st.columns(2)
        c1.metric("Streak", f"{streak.get('streak', 0)}d")
        c2.metric("Best", f"{streak.get('longest', 0)}d")
        debt = logic.get_debt_count()
        if debt > 0:
            st.error(f"E-Levels: {debt}")
        else:
            st.success("System Stable")

        st.divider()
        render_api_key_section()


def render_runtime_diagnostics_badge(entropy_mode: bool) -> None:
    embed_raw = _query_param_value("embed")
    is_embed_route = _is_universe_embed_request()
    entropy_flag = os.getenv("ENABLE_ENTROPY", "")
    session_mode = _ensure_valid_session_mode()
    st.caption(
        "Diagnostics | "
        f"query.embed={embed_raw or '<empty>'} | "
        f"is_embed_route={is_embed_route} | "
        f"ENABLE_ENTROPY={entropy_flag or '<unset>'} | "
        f"is_entropy_mode={entropy_mode} | "
        f"session.mode={session_mode}"
    )


def render_ocr_fallback_entrypoint() -> None:
    st.markdown("### 📷 OCR Quick Entry")
    st.caption("사이드바가 보이지 않는 상황을 대비한 본문 OCR 진입점입니다.")
    uploaded = st.file_uploader(
        "이미지 업로드 (OCR/서사 추출)",
        type=["png", "jpg", "jpeg"],
        key="vision_uploader_main",
    )
    if uploaded and st.button("본문에서 OCR 추출 실행", key="vision_uploader_main_btn", use_container_width=True):
        with st.spinner("이미지에서 서사를 추출하는 중..."):
            image_bytes = uploaded.read()
            vision_result = logic.refine_image_to_narrative_with_ai(image_bytes)
            st.session_state["refined_memo"] = vision_result
            st.success("OCR 기반 서사 초안을 생성했습니다. 사이드바 AI Assistant 또는 Stream에서 저장할 수 있습니다.")
    st.divider()


def render_stream_mode_switch_cards() -> None:
    st.markdown("#### 워크스페이스 전환")
    cols = st.columns(4, gap="small")
    for idx, (mode, title, subtitle) in enumerate(_MODE_CARD_CONFIG):
        with cols[idx]:
            st.markdown(
                (
                    "<div class='mode-card'>"
                    f"<div class='mode-card-title'>{title}</div>"
                    f"<div class='mode-card-sub'>{subtitle}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if st.button(f"{title} 열기", key=f"stream_hub_{mode}", use_container_width=True):
                st.session_state["mode"] = mode
                st.rerun()


def render_stream_ocr_entrypoint() -> None:
    with st.expander("📷 사진으로 서사 넣기", expanded=False):
        uploaded = st.file_uploader(
            "이미지 업로드 (메모/풍경 등)",
            type=["png", "jpg", "jpeg"],
            key="stream_vision_uploader",
        )
        if uploaded and st.button("사진 분석 및 서사 추출", key="stream_vision_extract", use_container_width=True):
            with st.spinner("이미지에서 서사를 추출하는 중..."):
                image_bytes = uploaded.read()
                vision_result = logic.refine_image_to_narrative_with_ai(image_bytes)
                st.session_state["refined_memo"] = vision_result
                st.success("서사 초안을 생성했습니다. 아래에서 검토 후 저장하세요.")

    if "refined_memo" in st.session_state:
        st.info(st.session_state["refined_memo"])
        if st.button("스트림에 즉시 저장", key="stream_save_refined", use_container_width=True, type="primary"):
            logic.save_log(st.session_state["refined_memo"])
            st.toast("서사가 스트림에 기록되었습니다.", icon="☄️")
            del st.session_state["refined_memo"]
            st.rerun()


def render_stream_chat_messages() -> None:
    for msg in st.session_state.get("messages", []):
        role = str(msg.get("role", "")).strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        # Use Material icon tokens for Cloud-safe avatar parsing.
        avatar = ":material/person:" if role == "user" else ":material/auto_awesome:"
        try:
            with st.chat_message(role, avatar=avatar):
                st.markdown(content)
        except Exception:
            # Fallback to default avatar instead of crashing the full app.
            with st.chat_message(role):
                st.markdown(content)

# ============================================================
# MODES
# ============================================================
def render_stream_mode():
    if st.session_state.get('violation_pending'):
        v = st.session_state['violation_pending']
        st.warning(f"{icons.get_icon_text('shield-alert')} Alignment Error against Core: \"{v['core']['content'][:50]}...\"")
        st.info(f"Input: {v['text']}")
        
        entropy_enabled = os.getenv("ENABLE_ENTROPY", "false").lower() == "true"
        
        c1, c2 = st.columns(2)
        if entropy_enabled:
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
        else:
            if c1.button(f"{icons.get_icon_text('zap')} Merge Log (Standard)", use_container_width=True):
                logic.save_log(v['text'])
                del st.session_state['violation_pending']
                st.toast("Log merged into stream.", icon="☄️")
                st.rerun()
            
        if st.button(f"{icons.get_icon_text('trash')} Discard", use_container_width=True if not entropy_enabled else False):
            del st.session_state['violation_pending']
            st.rerun()
        return

    st.markdown("<div class='stream-hero-title'>무엇을 기록하고 싶나요?</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='stream-hero-sub'>Stream은 입력 허브입니다. 아래 카드에서 Desk/Chronos/Universe/Control로 바로 전환할 수 있습니다.</div>",
        unsafe_allow_html=True,
    )

    has_user_messages = any(
        m.get("role") == "user" and str(m.get("content", "")).strip()
        for m in st.session_state.get("messages", [])
    )
    if has_user_messages:
        with st.expander("워크스페이스 전환", expanded=False):
            render_stream_mode_switch_cards()
    else:
        render_stream_mode_switch_cards()
    st.divider()
    render_stream_ocr_entrypoint()
    st.divider()

    echo = st.session_state.get('current_echo')
    if echo and not has_user_messages:
        echo_created_at = str(echo.get('created_at') or '')
        echo_content = str(echo.get('content') or '')
        st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border-left: 3px solid #666; padding: 15px; margin-bottom: 20px; border-radius: 4px; font-style: italic; color: #aaa;">
            <small>{icons.get_icon('sparkles', size=14)} Echo from {echo_created_at[:10]}</small><br>"{echo_content}"</div>""", unsafe_allow_html=True)
    
    render_stream_chat_messages()
    
    if user_input := st.chat_input("메시지를 입력하세요..."):
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

    try:
        related = logic.find_related_logs(user_input)
        resp = logic.generate_response(user_input, related)
    except Exception:
        resp = "입력을 기록했습니다. 응답 생성 중 문제가 발생해 기본 모드로 저장만 완료했습니다."
    if str(resp).strip():
        st.session_state.messages.append({"role": "assistant", "content": resp})
        logic.save_chat_message("assistant", resp)
    
    st.session_state['current_echo'] = logic.get_current_echo(reference_text=user_input)
    # Final rerun to sync state to UI
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
    total_secs = max(0, int(rem.total_seconds()))
    mins, secs = divmod(total_secs, 60)
    
    # JavaScript countdown for smooth ticking and auto-refresh on finish
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
    </div>
    <script>
        const endTime = {end_ts};
        function tick() {{
            const now = Date.now();
            const diff = Math.max(0, endTime - now);
            const m = Math.floor(diff / 60000);
            const s = Math.floor((diff % 60000) / 1000);
            const timerEl = document.getElementById('timer');
            timerEl.textContent = String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
            if (diff <= 0) {{
                timerEl.style.color = '#FFD700';
                clearInterval(interval);
                // Trigger a refresh after a small delay once finished
                setTimeout(() => {{ window.parent.location.reload(); }}, 1000);
            }}
        }}
        tick();
        const interval = setInterval(tick, 1000);
    </script>
    """, height=250)
    
    c1, c2 = st.columns(2)
    if c1.button(f"{icons.get_icon_text('check-circle')} 완료", use_container_width=True):
        st.session_state['chronos_running'] = False
        st.session_state['chronos_finished'] = True
        db.clear_chronos_timer()
        st.rerun()
    if c2.button(f"{icons.get_icon_text('shield-alert')} 취소", use_container_width=True):
        st.session_state['chronos_running'] = False
        db.clear_chronos_timer()
        st.rerun()

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
        try:
            logs = logic.load_logs()
            cores = db.get_cores()
            render_3d_universe(logs, cores)
        except Exception as exc:
            st.error(f"{icons.get_icon_text('shield-alert')} **Deep Space 렌더링에 실패했습니다.**")
            st.caption("데이터 동기화 충돌이 발생했을 수 있습니다. (서버 측 JSON 직렬화 문제 등)")
            st.info("💡 **안내:** 이 오류는 3D 시각화에만 영향을 미치며, 좌측의 'Cosmos' 및 'Soul Analytics' 탭은 정상적으로 이용 가능합니다.")
            
            c1, c2 = st.columns([1, 3])
            if c1.button("🔄 다시 시도 (Retry)", use_container_width=True):
                st.rerun()
                
            with st.expander("Technical Details (For Debugging)"):
                st.code(_safe_startup_error(exc))

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
    if _run_universe_embed_route():
        return
    try:
        init_session_state()
    except Exception as exc:
        st.error("Database initialization failed. Check DATASTORE and DATABASE_URL settings.")
        st.caption("Startup halted to prevent repeated runtime failures.")
        st.code(_safe_startup_error(exc))
        return
    is_entropy = logic.is_entropy_mode()
    apply_atmosphere(is_entropy); render_sidebar(is_entropy)
    # render_runtime_diagnostics_badge(is_entropy)
    # render_ocr_fallback_entrypoint()
    
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
                        st.success("엔트로피 감소 확인. 시스템 정상화.")
                        # Reset session flags that might interfere
                        st.session_state['interventions_checked'] = False
                        time.sleep(1.5); st.rerun()
                    else:
                        st.error("Core를 찾을 수 없습니다.")
        return

    if st.session_state['gatekeeper_dismissed'] and not st.session_state['interventions_checked']:
        for m in logic.run_active_intervention(): st.toast(m, icon="🔔")
        st.session_state['interventions_checked'] = True

    m = _ensure_valid_session_mode()
    if m == "stream": render_stream_mode()
    elif m == "chronos": render_chronos_mode()
    elif m == "universe": render_universe_mode()
    elif m == "control": render_control_mode()
    elif m == "desk": render_desk_mode()

if __name__ == "__main__":
    main()



