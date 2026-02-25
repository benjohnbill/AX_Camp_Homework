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
import uuid

import narrative_logic as logic
import icons
from universe_3d import render_3d_universe
import universe_auth
db = logic.db

import plotly.express as px
import plotly.graph_objects as go

_ALLOWED_MODES = ("stream", "desk", "chronos", "control", "universe")

_MODE_CARD_CONFIG = (
    ("desk", "DESK", "긴 글 작성과 정리", "book-open"),
    ("chronos", "CHRONOS", "집중 타이머와 회고", "timer"),
    ("universe", "SOUL ANALYTICS", "분석과 3D 탐색", "orbit"),
    ("control", "CONTROL", "칸반 기반 통제", "layout-dashboard"),
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
        layout="wide",
        initial_sidebar_state="expanded"
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


def _new_stream_id() -> str:
    return f"stream-{uuid.uuid4().hex[:12]}"


def _normalize_message_rows(rows: list) -> list:
    out = []
    for row in rows or []:
        role = str((row or {}).get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = str((row or {}).get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _load_messages_for_stream(stream_id: str, limit: int = 200) -> list:
    rows = logic.load_chat_stream_messages(stream_id=stream_id, limit=limit)
    return _normalize_message_rows(rows)


def _to_stream_title(text: str, max_len: int = 40) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return "제목 없는 스트림"
    return clean[:max_len] if len(clean) > max_len else clean


def _display_stream_title(raw_title: str) -> str:
    title = str(raw_title or "").strip()
    if not title:
        return "제목 없는 스트림"
    lowered = title.lower()
    if lowered == "legacy stream":
        return "레거시 스트림"
    if lowered == "untitled stream":
        return "제목 없는 스트림"
    return title


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

    if "active_stream_id" not in st.session_state:
        streams = logic.load_chat_streams(limit=1)
        if streams:
            st.session_state["active_stream_id"] = str(streams[0].get("stream_id") or "").strip() or _new_stream_id()
        else:
            st.session_state["active_stream_id"] = _new_stream_id()

    if "messages" not in st.session_state:
        st.session_state["messages"] = _load_messages_for_stream(st.session_state["active_stream_id"])

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
        'workspace_dock_open': False,
        'profile_settings_open': False,
        'sidebar_open': True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # If we loaded an existing Stream from DB, we are no longer at first input.
    if st.session_state.get("messages"):
        st.session_state["first_input_of_session"] = False

def apply_atmosphere(entropy_mode: bool):
    """
    Applies minimal essential CSS. Heavy styling is now handled by .streamlit/config.toml
    to ensure native Streamlit components function correctly without breaking.
    """
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #111317;
            --app-surface: #191c22;
            --app-surface-soft: #1f232b;
            --app-border: #2b3039;
            --app-text: #f3f5f7;
            --app-muted: #9aa3b2;
            --app-accent: #10a37f;
            --sidebar-width: 240px;
        }

        header {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 1.0rem !important;
            max-width: min(1180px, 92vw) !important;
            transition: max-width 0.3s ease, padding 0.3s ease;
        }

        [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 50% 0%, #1a1e26 0%, #111317 60%); }

        /* Sidebar: Python-controlled toggle, hide native buttons */
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        [data-testid="stSidebarCollapseButton"]   { display: none !important; }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--app-border);
        }

        .stream-shell {
            width: min(100%, 1160px);
            margin: 0 auto;
        }

        .stream-empty-center {
            width: min(100%, 720px);
            margin: 0 auto;
            padding-top: 15vh; /* Restore comfortable central offset */
            padding-bottom: 10vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .stream-hero-title {
            /* Position: fixed removed to restore central grouping */
            font-size: clamp(1.5rem, 1.2rem + 1.5vw, 2.1rem);
            line-height: 1.3;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 2.5rem;
            text-align: center;
            color: var(--app-text);
        }

        /* Group container for OCR and Switcher */
        .empty-action-block {
            width: 100%;
            background: transparent;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            /* margin-top removed to restore tight grouping with title */
        }

        /* Fixed Toggle Button Container for Sandwich structure */
        .workspace-dock-container {
            background: var(--app-surface-soft);
            border: 1px solid var(--app-border);
            border-radius: 12px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            animation: slideUp 0.2s ease-out;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            position: fixed;
            bottom: 82px; /* Fixed above chat input */
            width: min(100%, 720px);
            left: 50%;
            transform: translateX(-50%);
            z-index: 1001;
        }

        .fixed-toggle-area {
            position: fixed;
            bottom: 84px;
            left: calc(50% - 340px); /* Positioned relative to the 720px center */
            z-index: 1002;
        }

        @media (max-width: 1000px) {
            .fixed-toggle-area { left: 1rem; transform: none; }
        }

        .block-container {
            padding-top: 1.0rem !important;
            padding-bottom: 7rem !important;
            max-width: min(1180px, 92vw) !important;
            transition: max-width 0.3s ease, padding 0.3s ease;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes sparkle-glow {
            0%, 100% { opacity: 1; filter: drop-shadow(0 0 3px #10a37f88); }
            50%       { opacity: 0.78; filter: drop-shadow(0 0 7px #10a37fcc); }
        }
        .icon-sparkles { animation: sparkle-glow 2.8s ease-in-out infinite; }

        /* Toggle button styling - RESTRICTED to its container to avoid affecting DESK card */
        .fixed-toggle-area .stButton button {
            border-radius: 12px !important;
            height: 48px !important; /* Match chat input height roughly */
            font-size: 1.4rem !important;
            font-weight: 300 !important;
            background: var(--app-surface-soft) !important;
            border: 1px solid var(--app-border) !important;
            color: var(--app-muted) !important;
            transition: all 0.2s ease;
        }

        .fixed-toggle-area .stButton button:hover {
            border-color: var(--app-accent) !important;
            color: var(--app-text) !important;
        }

        .stButton > button[kind="secondary"] {
            border-radius: 12px !important;
            font-size: 0.88rem !important;
            padding: 0.4rem 0.8rem !important;
        }

        /* Slim Sidebar Stream Card styling with Ellipsis and Fixed Height */
        [data-testid="stSidebar"] .stButton button {
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.4rem 0.75rem !important;
            font-size: 0.85rem !important;
            border-radius: 8px !important;
            margin-bottom: 0.15rem !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            color: var(--app-text) !important;
            transition: all 0.2s ease;
            
            /* Enforce one-line with ellipsis */
            height: 38px !important;
            overflow: hidden !important;
            white-space: nowrap !important;
            display: block !important;
            text-overflow: ellipsis !important;
        }

        /* Ensure the icon stays at the left */
        [data-testid="stSidebar"] .stButton button div[data-testid="stMarkdownContainer"] p {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        [data-testid="stSidebar"] .stButton button:hover {
            border-color: var(--app-border) !important;
            background: var(--app-surface-soft) !important;
        }

        input, textarea { background-color: rgba(255, 255, 255, 0.03) !important; }
        @media (max-width: 1080px) {
            .block-container {
                max-width: min(980px, 96vw) !important;
            }
        }
        @media (max-width: 820px) {
            .block-container {
                max-width: 98vw !important;
                padding-left: 0.65rem !important;
                padding-right: 0.65rem !important;
            }
        }
        </style>
        """
        ,
        unsafe_allow_html=True,
    )
    if not st.session_state.get("sidebar_open", True):
        st.markdown(
            """<style>
            [data-testid="stSidebar"] { display: none !important; }
            .block-container {
                max-width: min(1380px, 96vw) !important;
                padding-left: 1.2rem !important;
                padding-right: 1.2rem !important;
            }
            *:has(#nl-open-anchor) + [data-testid="stButton"] button {
                position: fixed !important;
                top: 0.62rem !important;
                left: 0.68rem !important;
                z-index: 99999 !important;
                border-radius: 999px !important;
                width: 2rem !important;
                height: 2rem !important;
                min-height: 0 !important;
                padding: 0 !important;
                background: var(--app-surface-soft, #1f232b) !important;
                border: 1px solid var(--app-border, #2b3039) !important;
                color: var(--app-muted, #9aa3b2) !important;
                font-size: 1.1rem !important;
            }
            *:has(#nl-open-anchor) + [data-testid="stButton"] button:hover {
                border-color: var(--app-accent, #10a37f) !important;
                color: var(--app-text, #f3f5f7) !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )


# ============================================================
# API Key
# ============================================================
def render_api_key_section():
    session_key = st.session_state.get("openai_api_key", "")
    display_name = str(st.session_state.get("profile_display_name", "")).strip()
    profile_settings_open = bool(st.session_state.get("profile_settings_open", False))

    st.markdown("<div class='sidebar-section-title'>Account</div>", unsafe_allow_html=True)
    
    # Header: Name and Settings Toggle (ChatGPT Style)
    col_name, col_btn = st.columns([4, 1])
    with col_name:
        st.markdown(f"**{display_name if display_name else '사용자'}**")
    with col_btn:
        # Use simple label for toggle consistency
        if st.button("⚙️", key="profile_settings_toggle", help="설정"):
            st.session_state["profile_settings_open"] = not profile_settings_open
            st.rerun()

    # Settings Content (Visible only when toggle is active)
    if st.session_state.get("profile_settings_open"):
        with st.container(border=False):
            st.caption("Profile Settings")
            # 1. Display Name Change (Inside Settings only)
            new_name = st.text_input(
                "이름",
                value=display_name,
                placeholder="이름 입력",
                key="name_input_field",
            )
            # 2. API Key (Inside Settings)
            new_key = st.text_input(
                "API Key",
                type="password",
                value=session_key,
                placeholder="sk-...",
                key="key_input_field",
            )
            
            c1, c2 = st.columns(2)
            if c1.button("Save", key="save_account_settings", use_container_width=True, type="primary"):
                if new_key: logic.set_api_key(new_key.strip())
                st.session_state["profile_display_name"] = str(new_name).strip()
                st.session_state["profile_settings_open"] = False # Auto-close after save
                st.rerun()
            if c2.button("Reset", key="reset_account_settings", use_container_width=True):
                st.session_state.pop("openai_api_key", None)
                st.session_state["profile_display_name"] = ""
                st.rerun()




def render_sidebar(entropy_mode: bool):
    sidebar_open = st.session_state.get("sidebar_open", True)
    with st.sidebar:
        if not sidebar_open:
            return

        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button("⟨", key="sb_close_btn", help="사이드바 닫기"):
                st.session_state["sidebar_open"] = False
                st.rerun()

        st.markdown("### Narrative Loop")

        if st.button("새 스트림", use_container_width=True, key="sidebar_new_stream"):
            st.session_state["mode"] = "stream"
            st.session_state["active_stream_id"] = _new_stream_id()
            st.session_state["messages"] = []
            st.session_state["first_input_of_session"] = True
            st.session_state.pop("refined_memo", None)
            st.rerun()

        if entropy_mode:
            st.warning(f"{icons.get_icon_text('shield-alert')} ENTROPY ALERT")
            st.info("시스템 엔트로피가 임계치를 초과했습니다. [Gap Analysis]가 필요합니다.")

        streak = st.session_state.get('streak_info', {})
        st.caption(f"Streak {streak.get('streak', 0)}d")

        st.divider()
        st.markdown("<div class='sidebar-section-title'>스트림</div>", unsafe_allow_html=True)
        with st.container(border=False, height=430):
            st.markdown("<div class='sidebar-stream-scroll'>", unsafe_allow_html=True)
            streams = logic.load_chat_streams(limit=80)
            active_stream_id = str(st.session_state.get("active_stream_id") or "").strip()
            if not streams:
                st.caption("저장된 스트림이 없습니다.")
            else:
                for stream in streams:
                    stream_id = str(stream.get("stream_id") or "").strip()
                    if not stream_id:
                        continue
                    
                    title = _display_stream_title(stream.get("title"))
                    
                    # Prefix with message icon as a fixed indicator
                    label = f"💬 {title}"
                    if stream_id == active_stream_id:
                        label = f"● {label}"
                    
                    count = int(stream.get("message_count") or 0)
                    updated = str(stream.get("updated_at") or "")
                    meta = f"{count}개 메시지 · {updated[:16] if updated else ''}"
                    
                    # Full title is passed to the button; CSS handles the '...' truncation
                    if st.button(label, key=f"sidebar_stream_{stream_id}", use_container_width=True, help=meta):
                        st.session_state["mode"] = "stream"
                        st.session_state["active_stream_id"] = stream_id
                        st.session_state["messages"] = _load_messages_for_stream(stream_id)
                        st.session_state["first_input_of_session"] = len(st.session_state["messages"]) == 0
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Force account section to the bottom
        st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div class='sidebar-account-dock'>", unsafe_allow_html=True)
        render_api_key_section()
        st.markdown("</div>", unsafe_allow_html=True)


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


def render_stream_mode_switch_cards(show_heading: bool = True, key_prefix: str = "main") -> None:
    if show_heading:
        # Heading removed for a minimalist, intuitive design as requested
        pass
    
    # Strictly 4 columns for a single horizontal row (ChatGPT style)
    cols = st.columns(4, gap="small")
    for idx, (mode, title, subtitle, icon_key) in enumerate(_MODE_CARD_CONFIG):
        with cols[idx % 4]:
            # Consistent formatting: System Icon + Bold Title
            label = f"{icons.get_icon_text(icon_key)} **{title}**"
            if st.button(label, key=f"stream_hub_{key_prefix}_{mode}", use_container_width=True, help=subtitle):
                st.session_state["mode"] = mode
                st.rerun()


def render_stream_ocr_entrypoint(expanded: bool = False) -> None:
    with st.expander("📷 사진으로 서사 넣기", expanded=expanded):
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
            text = st.session_state["refined_memo"]
            del st.session_state["refined_memo"]
            _save_and_respond(text)
            st.toast("서사가 스트림에 기록되었습니다.", icon="☄️")
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
                text = v['text']
                del st.session_state['violation_pending']
                _save_and_respond(text)
                st.toast("Forced merge. Entropy increased.", icon="🚨")
                st.rerun()
        else:
            if c1.button(f"{icons.get_icon_text('zap')} Merge Log (Standard)", use_container_width=True):
                text = v['text']
                del st.session_state['violation_pending']
                _save_and_respond(text)
                st.rerun()
            
        if st.button(f"{icons.get_icon_text('trash')} Discard", use_container_width=True if not entropy_enabled else False):
            del st.session_state['violation_pending']
            st.rerun()
        return

    has_messages = any(
        m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
        for m in st.session_state.get("messages", [])
    )

    if not has_messages:
        # 1. EMPTY STATE: Large Title + OCR + Workspace Cards (Row of 4)
        st.markdown("<div class='stream-empty-center'>", unsafe_allow_html=True)
        st.markdown(f"<div class='stream-hero-title'>{icons.get_icon('sparkles', size=32)}<br>무엇을 기록하고 싶나요?</div>", unsafe_allow_html=True)
        
        # Action block starts
        st.markdown("<div class='empty-action-block'>", unsafe_allow_html=True)
        render_stream_ocr_entrypoint(expanded=False)
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)        
        render_stream_mode_switch_cards(show_heading=True, key_prefix="empty")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # In Empty State, chat input is still visible at the bottom, but NO [+] toggle
        if user_input := st.chat_input("메시지를 입력하세요..."):
            process_stream_input(user_input)
            
    else:
        # 2. CHAT STATE: Messages + [+] Toggle + Optional Dock + Input
        render_stream_chat_messages()

        # Render Dock ONLY if open via toggle
        if st.session_state.get("workspace_dock_open", False):
            st.markdown("<div class='workspace-dock-container'>", unsafe_allow_html=True)
            render_stream_mode_switch_cards(show_heading=False, key_prefix="dock")
            st.markdown("</div>", unsafe_allow_html=True)

        # OCR panel: always rendered as expander (client-side toggle, no rerun)
        render_stream_ocr_entrypoint(expanded=False)

        # [+] Toggle Button and Chat Input area
        st.markdown("<div class='fixed-toggle-area'>", unsafe_allow_html=True)
        col_dock, col_new = st.columns([1, 2])
        with col_dock:
            btn_label = "×" if st.session_state.get("workspace_dock_open", False) else "+"
            if st.button(btn_label, key="workspace_dock_toggle", help="워크스페이스 전환", use_container_width=True):
                st.session_state["workspace_dock_open"] = not st.session_state.get("workspace_dock_open", False)
                st.rerun()
        with col_new:
            if st.button("✚ 새 스트림", key="new_stream_quick", help="새 대화 시작", use_container_width=True):
                st.session_state["mode"] = "stream"
                st.session_state["active_stream_id"] = _new_stream_id()
                st.session_state["messages"] = []
                st.session_state["first_input_of_session"] = True
                st.session_state.pop("refined_memo", None)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if user_input := st.chat_input("메시지를 입력하세요..."):
            process_stream_input(user_input)

def _save_and_respond(user_input: str):
    """Core stream processing: append to chat, generate response, persist."""
    active_stream_id = str(st.session_state.get("active_stream_id") or "").strip()
    if not active_stream_id:
        active_stream_id = _new_stream_id()
        st.session_state["active_stream_id"] = active_stream_id

    is_first_user_in_stream = not any(
        str(msg.get("role") or "").strip().lower() == "user"
        and str(msg.get("content") or "").strip()
        for msg in st.session_state.get("messages", [])
    )

    st.session_state.messages.append({"role": "user", "content": user_input})
    user_meta = {"stream_id": active_stream_id}
    if is_first_user_in_stream:
        user_meta["stream_title"] = _to_stream_title(user_input)
    logic.save_chat_message("user", user_input, metadata=user_meta)
    logic.save_log(user_input)

    if st.session_state.get('first_input_of_session'):
        st.toast("Log captured. Meteor Effect.", icon="☄️")
        st.session_state['first_input_of_session'] = False

    try:
        related = logic.find_related_logs(user_input)
        resp = logic.generate_response(user_input, related)
    except Exception:
        resp = "입력을 기록했습니다. 응답 생성 중 문제가 발생해 기본 모드로 저장만 완료했습니다."
    if str(resp).strip():
        st.session_state.messages.append({"role": "assistant", "content": resp})
        logic.save_chat_message("assistant", resp, metadata={"stream_id": active_stream_id})

    st.session_state['current_echo'] = logic.get_current_echo(reference_text=user_input)


def process_stream_input(user_input: str):
    status, core = logic.evaluate_input_integrity(user_input)
    if status == "VIOLATION":
        st.session_state['violation_pending'] = {"text": user_input, "core": core}
        st.rerun()
        return
    _save_and_respond(user_input)
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
    st.markdown(f"<div style='text-align:center;'><h1>{icons.get_icon('book-open', size=40)} DESK</h1></div>", unsafe_allow_html=True)
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
    if not st.session_state.get("sidebar_open", True):
        st.markdown('<span id="nl-open-anchor"></span>', unsafe_allow_html=True)
        if st.button("⟩", key="main_sb_open"):
            st.session_state["sidebar_open"] = True
            st.rerun()
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



