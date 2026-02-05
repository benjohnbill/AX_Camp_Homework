"""
app.py
Narrative Loop - Streamlit 메인 앱
서사 밀도 그래프 (Contribution Graph) 포함
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import narrative_logic as logic

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Narrative Loop",
    page_icon="🔄",
    layout="wide"
)

# ============================================================
# 커스텀 스타일 (미니멀 & 차분한 톤)
# ============================================================
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* 카드 스타일 */
    .echo-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        backdrop-filter: blur(10px);
    }
    
    .echo-card h4 {
        color: #e94560;
        margin-bottom: 16px;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .echo-card p {
        color: #eaeaea;
        font-size: 16px;
        line-height: 1.8;
    }
    
    .echo-card .quote {
        color: #9ca3af;
        font-size: 14px;
        font-style: italic;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin-bottom: 12px;
    }
    
    .mode-mirroring {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .mode-nietzsche {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .waiting-message {
        color: #6b7280;
        font-style: italic;
        text-align: center;
        padding: 60px 20px;
    }
    
    /* 헤더 스타일 */
    .main-header {
        text-align: center;
        padding: 20px 0 40px 0;
    }
    
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: 300;
        letter-spacing: 4px;
    }
    
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
    }
    
    /* 섹션 타이틀 */
    .section-title {
        color: #9ca3af;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    /* 그래프 타이틀 */
    .graph-title {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 서사 밀도 그래프 생성 함수
# ============================================================
def create_narrative_density_chart(logs: list) -> go.Figure:
    """GitHub 잔디 스타일의 서사 밀도 히트맵 생성"""
    
    # 최근 12주(84일) 데이터 준비
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=83)
    
    # 날짜 범위 생성
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 로그에서 날짜별 카운트 집계
    date_counts = {}
    for log in logs:
        log_date = datetime.fromisoformat(log['timestamp']).date()
        if start_date <= log_date <= end_date:
            date_str = str(log_date)
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
    
    # 주 단위로 데이터 구성 (7행 x 12열)
    weeks = 12
    days_of_week = 7
    
    z_data = []
    hover_text = []
    
    for day in range(days_of_week):
        row = []
        hover_row = []
        for week in range(weeks):
            idx = week * 7 + day
            if idx < len(date_range):
                current_date = date_range[idx]
                date_str = str(current_date.date())
                count = date_counts.get(date_str, 0)
                row.append(count)
                hover_row.append(f"{date_str}: {count}개")
            else:
                row.append(0)
                hover_row.append("")
        z_data.append(row)
        hover_text.append(hover_row)
    
    # Plotly 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        text=hover_text,
        hoverinfo='text',
        colorscale=[
            [0, '#1a1a2e'],      # 0개: 어두운 배경
            [0.25, '#3d1a3d'],   # 낮음
            [0.5, '#6b1d4a'],    # 중간
            [0.75, '#a61e4d'],   # 높음
            [1, '#e94560']       # 최대: 메인 강조색
        ],
        showscale=False,
        xgap=3,
        ygap=3
    ))
    
    # 레이아웃 설정 (미니멀)
    fig.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            autorange='reversed'
        )
    )
    
    return fig


# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    # 서사 밀도 그래프 (최상단)
    st.markdown('<p class="graph-title">📊 서사의 밀도</p>', unsafe_allow_html=True)
    
    logs = logic.load_logs()
    
    if logs:
        density_chart = create_narrative_density_chart(logs)
        st.plotly_chart(density_chart, use_container_width=True, config={'displayModeBar': False})
    else:
        st.caption("기록이 쌓이면 여기에 서사의 밀도가 나타납니다.")
    
    st.divider()
    
    # 서사 통계
    st.markdown("### 📈 서사 통계")
    st.metric(label="축적된 서사 조각", value=f"{len(logs)}개")
    
    st.divider()
    
    # 개발자 도구
    st.markdown("### ⚙️ 개발자 도구")
    if st.button("🗑️ 데이터 초기화", type="secondary"):
        logic.save_logs([])
        if 'last_echo' in st.session_state:
            del st.session_state['last_echo']
        st.success("모든 데이터가 초기화되었습니다.")
        st.rerun()

# ============================================================
# 메인 헤더
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔄 NARRATIVE LOOP</h1>
    <p>당신의 과거가 현재에게 질문합니다</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 메인 레이아웃 (2 컬럼)
# ============================================================
left_col, right_col = st.columns([1, 1], gap="large")

# ------------------------------------------------------------
# Left Column: 오늘의 기록
# ------------------------------------------------------------
with left_col:
    st.markdown('<p class="section-title">✍️ 오늘의 기록</p>', unsafe_allow_html=True)
    
    input_text = st.text_area(
        label="당신의 생각을 기록하세요",
        height=300,
        key="input_text",
        placeholder="오늘 떠오른 생각, 고민, 다짐을 자유롭게 적어보세요...",
        label_visibility="collapsed"
    )
    
    tags_input = st.text_input(
        label="태그",
        placeholder="예: #의지 #계획 #성찰",
        key="tags_input"
    )
    
    submit_button = st.button(
        "🔗 기록 및 서사 연결",
        type="primary",
        use_container_width=True
    )

# ------------------------------------------------------------
# 버튼 클릭 시 처리
# ------------------------------------------------------------
if submit_button:
    if not input_text.strip():
        st.warning("기록할 내용을 입력해주세요.")
    else:
        with st.spinner("서사를 연결하는 중..."):
            # 태그 파싱
            tags = [tag.strip() for tag in tags_input.replace("#", " #").split("#") if tag.strip()]
            
            # 1. 로그 저장
            logic.save_log(input_text, tags)
            
            # 2. 에코 생성 (Hybrid Search: 태그 전달)
            echo_response, mode, past_log = logic.generate_echo(input_text, tags)
            
            # 3. 세션에 저장
            st.session_state['last_echo'] = {
                "response": echo_response,
                "mode": mode,
                "past_log": past_log
            }
        
        st.success("✨ 기록이 서사에 통합되었습니다.")
        st.rerun()

# ------------------------------------------------------------
# Right Column: 서사의 메아리
# ------------------------------------------------------------
with right_col:
    st.markdown('<p class="section-title">🪞 서사의 메아리</p>', unsafe_allow_html=True)
    
    if 'last_echo' in st.session_state:
        echo_data = st.session_state['last_echo']
        mode = echo_data['mode']
        response = echo_data['response']
        past_log = echo_data.get('past_log')
        
        # 모드에 따른 배지
        if mode == "mirroring":
            mode_badge = '<span class="mode-badge mode-mirroring">🪞 거울 모드</span>'
            mode_title = "과거의 당신이 묻습니다"
        else:
            mode_badge = '<span class="mode-badge mode-nietzsche">🔥 니체 모드</span>'
            mode_title = "철학자가 묻습니다"
        
        # 과거 인용문 표시 (거울 모드일 때만)
        quote_html = ""
        if mode == "mirroring" and past_log:
            past_date = past_log['timestamp'][:10]
            past_text = past_log['text'][:100] + "..." if len(past_log['text']) > 100 else past_log['text']
            quote_html = f'<p class="quote">📜 "{past_text}" — {past_date}</p>'
        
        st.markdown(f"""
        <div class="echo-card">
            {mode_badge}
            <h4>{mode_title}</h4>
            <p>{response}</p>
            {quote_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="waiting-message">
            <p>🌙</p>
            <p>당신의 기록을 기다리고 있습니다...</p>
        </div>
        """, unsafe_allow_html=True)
