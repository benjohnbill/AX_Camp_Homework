# agent.md — Antigravity / Narrative Loop 에이전트 운영 헌법

> **이 문서를 읽는 에이전트에게:**
> 코드를 "수정"하는 것이 아니라 "철학과 로직을 정렬"하는 것이다.
> 변경 전 반드시 이 문서를 끝까지 읽어라.

---

## 0. 현재 시스템 상태 (2026-02 기준)

| 항목            | 상태                                                     |
| --------------- | -------------------------------------------------------- |
| **운영 DB**     | Supabase (PostgreSQL) — `DATASTORE=postgres`             |
| **Fallback DB** | SQLite (`data/narrative.db`) — 로컬 개발/롤백 전용       |
| **AI 모델**     | `gpt-4o-mini` (응답) + `text-embedding-3-small` (임베딩) |
| **런타임**      | Streamlit (Single Page App, 5-Mode)                      |
| **배포**        | Streamlit Cloud                                          |

> **GEMINI.md는 구버전입니다.** 해당 파일의 "SQLite 기반" 기술은 현재 상태를 반영하지 않습니다.
> 이 `agent.md`가 유일한 진실의 원천(Single Source of Truth)입니다.

---

## 1. 아키텍처 지도 — 파일별 역할

```
app.py                   ← [UI Layer] Streamlit 렌더링, 세션 상태, 5개 모드 라우팅
narrative_logic.py       ← [Mind Layer] EvidenceGateway + PolicyEngine (AI/비즈니스 로직)
db_manager.py            ← [Memory Layer] 멀티 DB 라우터 — DATASTORE 값에 따라 분기
db_manager_postgres.py   ← [Backend] Supabase/Postgres CRUD (현재 운영 경로)
db_manager_sqlite.py     ← [Backend] SQLite CRUD (롤백/로컬 개발 전용)
db_router.py             ← [Entry] 환경변수를 읽어 db_manager_postgres 또는 db_manager_sqlite를 반환
icons.py                 ← [UI] 시스템 아이콘 상수 모음 (이모지/기호)
tools/                   ← [DevOps] 마이그레이션·검증 스크립트 모음
tests/                   ← [QA] pytest 기반 안정성 테스트
sql/                     ← [Schema] Supabase 테이블 생성 SQL
data/                    ← [Storage] 로컬 DB 및 JSON 보조 로그
```

### 핵심 2-클래스 구조 (`narrative_logic.py`)

```
EvidenceGateway   ← I/O 전담 (DB 조회, OpenAI API 호출, 임베딩 생성)
                     결정을 내리지 않음. 오직 "증거"를 수집함.

PolicyEngine      ← 순수 로직 전담 (Silence 판정, Violation 탐지)
                     DB에 직접 접근하지 않음. DTO를 받아 결정만 반환.
```

> **원칙:** 이 두 클래스의 역할 경계를 절대 섞지 마라.
> `EvidenceGateway`에 if-else 의사결정 로직을 추가하거나,
> `PolicyEngine`에서 DB를 직접 호출하는 코드는 아키텍처 위반이다.

---

## 2. DB 라우팅 규칙

```python
# db_router.py 핵심 로직 (의사코드)
DATASTORE = os.getenv("DATASTORE") or st.secrets.get("DATASTORE", "sqlite")

if DATASTORE == "postgres":
    from db_manager_postgres import *   # ← 현재 운영 경로
else:
    from db_manager_sqlite import *     # ← fallback
```

### 환경변수 설정 위치

| 환경                  | 설정 위치                        |
| --------------------- | -------------------------------- |
| 로컬 개발             | `.streamlit/secrets.toml`        |
| 운영(Streamlit Cloud) | Streamlit Cloud Secrets 대시보드 |

### 필수 환경변수

```toml
DATASTORE = "postgres"
DATABASE_URL = "postgresql://postgres.<project_ref>:...@*.pooler.supabase.com:6543/postgres"
OPENAI_API_KEY = "sk-..."
```

> **주의:** `DATABASE_URL`에서 Session Pooler 사용 시 username은 반드시
> `postgres.<project_ref>` 형식이어야 한다. `postgres`만 쓰면 인증 실패.

---

## 3. 데이터 스키마 — 3-Body Hierarchy

```
Core (Constitution / 헌법)   ← [Star] 사용자의 핵심 가치/원칙
  └── Gap (Apology / 해명)   ← [Activity] 원칙 위반에 대한 해명 + 내일의 약속
        └── Log (Fragment)   ← [Sparkles] 일상 기록/생각의 파편
```

### Supabase 테이블 목록

| 테이블         | 역할                                           |
| -------------- | ---------------------------------------------- |
| `logs`         | 모든 기록 저장 (Log, Core, Gap, Kanban 통합)   |
| `chat_history` | AI 대화 기록                                   |
| `connections`  | 로그 간 연결 관계 (Universe 그래프용)          |
| `user_stats`   | 스트릭/debt/회복 포인트 (항상 `id=1` 단일 row) |

### `logs` 테이블 핵심 컬럼

```
id, content, meta_type, embedding(BLOB/bytea),
emotion, dimension, keywords(JSON), tags(JSON),
linked_constitutions(JSON), duration, kanban_status,
debt_repaid, quality_score, reward_points, created_at
```

---

## 4. 5-Mode 동작 규칙

### Mode 1: Stream (채팅 & 기록)

- **핵심 함수:** `process_stream_input()` in `app.py`
- **DB 쓰기:** `create_log()` → `logs` 테이블 + `chat_history`
- **AI 경로:** `EvidenceGateway.get_embedding_safe()` → `hybrid_search()` → `generate_response()`
- **첫 입력:** "Silent Zoom" — AI 응답 없이 조용히 저장 (Meteor 저장)
- **두 번째 입력부터:** 유사 과거 로그를 컨텍스트로 AI 응답 생성

### Mode 2: Chronos (타이머)

- **핵심 함수:** `render_chronos_timer()`, `render_chronos_docking()`
- **DB 쓰기:** 타이머 종료 후 Docking 시 `create_log(meta_type="Log", duration=초)`
- **연결:** Docking 시 특정 Constitution(`core_id`)과 연결

### Mode 3: Universe (시각화)

- **핵심 함수:** `render_universe_mode()`, `render_soul_analytics()`
- **DB 읽기만:** `get_all_logs()`, `get_connection_counts()`, `get_logs_for_analytics()`
- **렌더링:** PyVis 그래프 + Plotly 히트맵/트리맵
- **이 모드에서는 쓰기 없음**

### Mode 4: Control (칸반)

- **핵심 함수:** `render_control_mode()`, `render_kanban_docking()`
- **DB 쓰기:** `create_log(kanban_status="draft"|"orbit"|"landed")`
- **상태 전이:** `draft → orbit → landed`

### Mode 5: Desk (에세이 작성)

- **핵심 함수:** `render_desk_mode()`
- **DB 읽기+쓰기:** `get_fragments_paginated()` 조회 후 에세이 `create_log(meta_type="Log")`

---

## 5. 레드 프로토콜 & Streak 시스템

### 레드 프로토콜 (Red Protocol)

```
트리거: PolicyEngine.detect_violation() → True 반환 시
상태: st.session_state["entropy_mode"] = True
제약: 일반 대화 차단, 해명(Gap) + 내일의 약속 제출 필수
해제: create_gap() 호출 후 debt_repaid로 debt 감소
```

**절대 규칙:** `entropy_mode`가 True일 때는 `create_log()` 일반 경로를 막아야 한다.
`PolicyEngine.detect_violation()`을 우회하는 코드를 추가하지 마라.

### Streak & Debt 시스템

```python
# db_manager.py 핵심 함수들
check_streak_and_apply_penalty()  # 로그인 시 자동 호출 — streak 끊기면 debt +1
update_streak()                   # 로그 작성 시 streak 갱신
increment_debt(amount)            # debt 증가
decrement_debt(amount)            # debt 감소 (Gap 제출 시)
get_current_streak()              # 현재 streak 조회
```

- `user_stats` 테이블의 `id=1` row가 항상 존재해야 한다.
- `debt_count > 0` → `entropy_mode = True` 조건 성립

---

## 6. 에이전트 작업 원칙 (황금률)

### 작업 전 반드시 확인

- [ ] `DATASTORE` 환경변수가 `postgres`로 설정되어 있는가?
- [ ] 수정할 함수가 `EvidenceGateway`(I/O)인지 `PolicyEngine`(로직)인지 명확한가?
- [ ] DB 스키마 변경 시 `sql/` 폴더에 마이그레이션 SQL이 준비되어 있는가?
- [ ] Supabase `user_stats(id=1)` row가 존재하는가?

### 절대 금지 사항

1. **`PolicyEngine`에서 DB 직접 호출 금지** — 반드시 `EvidenceGateway`를 통해야 함
2. **`narrative_logic.py`에서 `st.session_state` 직접 수정 금지** — UI 레이어(app.py)의 역할
3. **`entropy_mode` 강제 해제 금지** — Gap 제출 + debt 감소 절차 없이 해제 불가
4. **`DATABASE_URL` 로그 출력 금지** — 인증 정보 마스킹 필수
5. **SQLite 파일(`data/narrative.db`) 운영 DB로 취급 금지** — fallback 전용

### 코드 수정 후 검증 순서

```powershell
# 1. Postgres 연결 인증 확인
python tools/preflight_postgres_auth.py

# 2. 스키마 부트스트랩 검증
python tools/check_supabase_phase1.py

# 3. 스모크 테스트
python tools/check_postdeploy_smoke.py --strict-postgres

# 4. 데이터 무결성 검증
python tools/check_data_integrity.py --expect-postgres --max-dup-chat 0 --report-json data/integrity_latest.json

# 5. 유닛 테스트
python -m pytest -q tests/
```

**원스텝 Gate 실행 (권장):**

```powershell
python tools/run_agent_a_gate.py
```

---

## 7. 검색 & 임베딩 시스템

```python
# narrative_logic.py 핵심 검색 함수

hybrid_search(query_text, top_k=10, alpha=0.7)
# alpha=0.7: 벡터 유사도 70% + 키워드 매칭 30% 가중치

# 내부 흐름:
# 1. EvidenceGateway.get_embedding_safe(text) → 벡터 생성
# 2. ann_query(embedding) → Annoy 인덱스 ANN 검색
# 3. simple_keyword_search(text) → SQLite FTS5 or Postgres LIKE
# 4. RRF(Reciprocal Rank Fusion) → 최종 랭킹 병합
```

- 임베딩은 `numpy bytes` (`BLOB` in SQLite / `bytea` in Postgres)로 저장
- Annoy 인덱스: `data/` 폴더에 캐시, `load_or_build_index()`로 초기화
- 임베딩 없이도 키워드 검색만으로 fallback 동작 가능

---

## 8. 흔한 버그 패턴 & 방지법

| 버그 패턴                                        | 원인                                            | 방지법                                              |
| ------------------------------------------------ | ----------------------------------------------- | --------------------------------------------------- |
| `AttributeError: check_streak_and_apply_penalty` | 함수가 `narrative_logic.py`에 있다고 가정       | 항상 `db_manager.py`에 있어야 함                    |
| Streamlit 검은 화면                              | `init_session_state()` 미실행 또는 DB 연결 실패 | `EvidenceGateway.check_system_health()` 리턴값 확인 |
| `password authentication failed`                 | `DATABASE_URL` 비밀번호 불일치 또는 username 오류 | 비밀번호 재발급 후 반영, `postgres.<project_ref>` 확인 |
| 임베딩 shape 불일치                              | 모델 변경 시 기존 BLOB과 차원 불일치            | 임베딩 모델 변경 시 인덱스 재빌드(`build_index.py`) |
| Postgres `id=1` not found                        | `user_stats` 초기 row 없음                      | `check_supabase_phase1.py`로 확인 후 SQL 재실행     |

---

## 9. 롤백 절차

운영 중 Postgres 장애 발생 시:

```toml
# .streamlit/secrets.toml 또는 Streamlit Cloud Secrets
DATASTORE = "sqlite"   # postgres → sqlite 변경
```

Streamlit Cloud에서 앱 재시작 후 로컬 `data/narrative.db`로 fallback.
롤백 시각과 사유를 `handoff.txt`에 기록할 것.

---

## 10. 참고 문서

| 문서                             | 용도                         |
| -------------------------------- | ---------------------------- |
| `SUPABASE_RUNBOOK.md`            | 배포 전후 표준 점검 절차     |
| `README.md`                      | 프로젝트 개요 및 설치 가이드 |
| `PROJECT_ANALYSIS_KR.md`         | 초보자용 데이터 플로우 설명  |
| `handoff.txt`                    | 세션 간 인계 메모            |
| `sql/1_create_logs_table.sql`    | Supabase 테이블 생성 SQL     |
| `sql/2_create_system_tables.sql` | Supabase 보조 테이블 SQL     |
