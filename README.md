# Antigravity (Narrative Loop)

**AI-Powered Self-Reflection & Narrative Tracking System**

> "기록은 파편이지만, 연결된 기록은 서사가 된다."
> "Records are fragments, but connected records become a narrative."

Antigravity는 개인의 생각, 감정, 원칙을 기록하고 AI가 이를 과거의 기록과 연결하여 깊은 자아 성찰을 돕는 **AI 기반 성찰 플랫폼**입니다. 단순한 일기장을 넘어, 사용자의 '내적 헌법'과 실제 행동 사이의 간극을 추적하고 성장을 위한 루프를 형성합니다.

---

## 🌌 프로젝트 철학: Narrative Loop

이 프로젝트는 **'나를 데이터로 보는 기술'**과 **'성찰을 돕는 인문학적 접근'**을 결합합니다.
- **Fragmentation to Constellation:** 흩어진 생각(Fragment)들을 별자리(Universe)처럼 연결합니다.
- **Self-Correction Loop:** 내 원칙(Constitution)을 어겼을 때 이를 직면하고 해명(Apology)하는 과정을 통해 행동을 교정합니다.
- **Temporal Connection:** 과거의 나와 현재의 나를 연결하여 일관된 성찰을 유도합니다.

---

## 🚀 주요 기능 (5-Mode Architecture)

### 1. [Waves] Stream Mode (채팅 & 기록)
- **실시간 서사 기록:** 사용자가 문장을 입력하면 'Fragment'로 즉시 저장됩니다.
- **Echo System:** AI "문학적 천문학자"가 현재의 생각과 가장 유사한 과거의 기록을 찾아 인용하며 응답합니다.
- **Silent Zoom:** 첫 입력은 조용히 저장(Meteor)되며, 대화가 깊어질수록 AI의 개입이 정교해집니다.

### 2. [Orbit] Universe Mode (시각화)
- **3-Body Hierarchy:** [Star] Constitution(원칙) - [Activity] Apology(해명) - [Sparkles] Fragment(기록)의 구조를 그래프로 시각화합니다.
- **인터랙티브 그래프:** PyVis를 사용하여 기록 간의 관계와 군집을 탐색할 수 있습니다.
- **Soul Finviz:** 나의 감정 상태나 기록 밀도를 히트맵과 트리맵으로 분석합니다.

### 3. [Droplet] Red Protocol (자기 교정)
- **위반 탐지:** 설정된 '헌법'에 위배되는 행동이나 생각이 감지되면 "Red Mode"가 활성화됩니다.
- **해명 프로세스:** 레드 모드에서는 일반 대화가 제한되며, 위반에 대한 진솔한 해명과 내일의 약속을 제출해야만 해제됩니다.

### 4. [Moon] Gatekeeper (시작 점검)
- **Contradiction Detection:** 앱 실행 시 어제의 약속을 지켰는지, 최근 기록 중 모순된 생각은 없는지 AI가 점검합니다.
- **Dream Report:** 수면 중이나 공백기 동안 쌓인 데이터의 패턴을 분석하여 보고합니다.

### 5. [Chronos & Control] 시간 관리 및 칸반
- **Chronos Timer:** 몰입 시간을 측정하고, 해당 시간이 어떤 원칙(Constitution)에 기여했는지 태깅합니다.
- **Narrative Kanban:** 생각의 카드들을 `Draft → Orbit → Landed`의 상태로 관리하여 아이디어를 구체화합니다.

---

## 🛠 기술 스택

| 분류 | 기술 |
|------|------|
| **Frontend** | Streamlit, Plotly, PyVis |
| **Backend** | Python 3.12+ |
| **AI Layer** | OpenAI GPT-4o-mini, text-embedding-3-small |
| **Database** | SQLite (Local), PostgreSQL/Supabase (Remote) |
| **Search** | Hybrid Search (Vector Similarity + SQL FTS) |
| **DevOps** | Python-dotenv, Pytest |

---

## 📂 프로젝트 구조

```
Narrative_Loop/
├── app.py                 # 메인 Streamlit 애플리케이션 (5-Mode UI)
├── narrative_logic.py     # AI 엔진, 임베딩 및 성찰 로직
├── db_manager.py          # 멀티 DB 라우터 (SQLite/Postgres)
├── db_manager_sqlite.py   # SQLite 전용 핸들러
├── db_manager_postgres.py # Postgres/Supabase 전용 핸들러
├── icons.py               # 시스템 UI 아이콘 관리
├── sql/                   # 데이터베이스 스키마 및 마이그레이션 SQL
├── tools/                 # DB 마이그레이션 및 인덱스 빌드 도구
├── data/                  # 로컬 DB 및 JSON 로그 저장소
└── tests/                 # 시스템 안정성 검증을 위한 테스트 코드
```

---

## ⚙️ 설치 및 설정

### 1. 환경 설정
```bash
git clone <repository_url>
cd Narrative_Loop
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 및 DB 설정
`.streamlit/secrets.toml` 또는 환경 변수에 다음 정보를 입력합니다:
```toml
OPENAI_API_KEY = "your-key"
DATASTORE = "sqlite" # 또는 "postgres"
# Postgres 사용 시
DATABASE_URL = "postgresql://..."
```

### 2-1. Supabase Schema Bootstrap (Phase 1)
Supabase SQL Editor에서 아래 파일 내용을 순서대로 붙여넣어 실행하세요.
```sql
-- 1) sql/1_create_logs_table.sql
-- 2) sql/2_create_system_tables.sql
```

검증 체크:
- `logs`, `user_stats`, `chat_history`, `connections` 테이블 생성 확인
- `user_stats`에 `id=1` 기본 row 존재 확인
- `idx_logs_embedding_ivfflat`, `idx_logs_content_trgm` 인덱스 생성 확인

### 3. 실행
```bash
streamlit run app.py
```

---

## 🔒 보안 및 라이선스
- 본 프로젝트는 개인의 민감한 성찰 데이터를 다룹니다. `.streamlit/secrets.toml` 및 `data/` 폴더의 보안에 유의하세요.
- 라이선스: 개인 학습 및 성찰 목적의 비상업적 이용을 권장합니다.

---
Built with Gravity & Love for Deep Self-Reflection
