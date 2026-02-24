# agent.md — Narrative_Loop 통합 운영 문서 (SSOT)

이 문서는 Narrative_Loop의 단일 진실 원천이다.  
모든 AI 도구(Codex CLI, Antigravity, Android Studio, Gemini 3.1 Pro)는 이 문서를 우선 기준으로 작업한다.

---

## 0) 프로젝트 핵심 목적

이 프로젝트는 생산성 앱이 아니라, 다음 목적의 자기서사 도구다.

1. 과거 기록을 근거로 현재 감정을 회고한다.
2. 스스로 원칙(Core) 기반의 결정을 선언한다.
3. 반복 기록을 통해 "자기결정" 비중을 높인다.

행동의 질서도(프로젝트 기준):

1. 무의식적 행동
2. 충동/우발적 행동
3. 결정을 미루며 시간을 보내는 행동
4. 불확실성 속에서도 스스로 결정한 행동

목표는 1~3의 비중을 줄이고 4의 비중을 늘리는 것이다.

---

## 1) 현재 시스템 스냅샷 (2026-02)

- Runtime UI: Streamlit (`app.py`)
- Core Logic: `narrative_logic.py`
  - `EvidenceGateway` (I/O)
  - `PolicyEngine` (판단 로직)
- DB Router: `db_router.py`
- DB Backend:
  - 운영: `db_manager_postgres.py` (`DATASTORE=postgres`)
  - fallback: `db_manager_sqlite.py` (로컬/롤백 전용)
- Main DB: Supabase PostgreSQL
  - `logs`, `chat_history`, `connections`, `user_stats`

중요 경계:

- Streamlit은 UI/상태 관리용이다.
- 모바일 앱의 외부 POST 수신 API 서버 역할은 Streamlit이 아니라 별도 인제스트 계층이 담당한다.

---

## 2) 현재 코드 구조 요약

### UI 라우팅 (`app.py`)

- `process_stream_input()`
- `render_stream_mode()`
- `render_chronos_mode()`
- `render_universe_mode()`
- `render_control_mode()`
- `render_desk_mode()`
- `main()`

### 로직 핵심 (`narrative_logic.py`)

- Search: `hybrid_search()`, `find_related_logs()`
- Response: `generate_response()`
- Write: `save_log()`, `save_chronos_log()`, `create_kanban_card()`, `land_kanban_card()`
- Policy: `evaluate_input_integrity()`, `process_gap()`
- Ops: `run_startup_diagnostics()`

---

## 3) 필수 아키텍처 원칙

1. `EvidenceGateway`는 I/O만 담당한다.
2. `PolicyEngine`은 DB를 직접 호출하지 않는다.
3. `narrative_logic.py`에서 UI 상태(`st.session_state`) 직접 조작 금지.
4. `DATABASE_URL`, API Key 등 민감값은 로그/문서에 절대 노출하지 않는다.
5. SQLite는 운영 DB가 아니라 fallback/로컬 개발용이다.

---

## 4) 멀티도구 역할 분담

### Codex CLI (기획/통합)

- 철학 정렬, 아키텍처 결정, 통합 설계, 문서 기준 확정.
- 에이전트 산출물 검토 및 충돌 해소.

### Antigravity (Backend + Streamlit)

- 인제스트 API 계층 구현/운영.
- `narrative_logic.py` 재사용 연결.
- Supabase 저장 규약/무결성/운영 점검 책임.

### Android Studio (모바일)

- CameraX, OCR 입력 UX, 네트워크 재시도, 응답 렌더링 책임.
- 키를 클라이언트에 두지 않고 토큰 기반 인증만 사용.

### Gemini 3.1 Pro (UI 전문)

- 내러티브 UX 개선.
- 숙제화 없는 카피/인터랙션 설계.
- Streamlit 화면 개선안 제안(구현은 Antigravity와 연동).

---

## 5) Android OCR 통합 기준 워크플로우

기준 파이프라인:

1. Android에서 촬영(CameraX)
2. 인제스트 API에 업로드
3. OCR(Gemini Vision) 수행
4. 정규화 텍스트를 기존 파이프라인으로 연결
5. `save_log()` 계열 저장 + `find_related_logs()` + `generate_response()`
6. Android에 응답 반환
7. 동일 로그가 Streamlit Universe/Desk 회고에 반영

---

## 6) 공통 API 계약 (초안)

### `POST /v1/ocr/ingest`

Request fields:

- `user_id` (string)
- `image_base64` (string)
- `client_ts` (ISO8601 string)
- `session_id` (string)
- `mode_hint` (`stream` | `desk` | `auto`)
- `manual_override_text` (optional string)

Response fields:

- `request_id`
- `ocr_text_raw`
- `ocr_text_normalized`
- `confidence` (0~1)
- `saved_log_id`
- `ai_response`
- `related_log_ids`
- `warnings`

Error codes:

- `400`, `401`, `422`, `429`, `500`

---

## 7) 데이터 저장 규칙

- `meta_type`: `Log`
- `content`: 최종 확정 텍스트
- `dimension`: 필요 시 `handwriting` 반영
- `tags`: `source:android_ocr`, `input:handwritten` 계열 태깅
- `keywords`: 기존 메타 추출 결과 유지

원칙:

- 외부 소스 구분 가능해야 한다.
- 검색/회고 경로에서 동일한 1급 로그로 취급한다.

---

## 8) 품질 게이트

실행 기준:

```powershell
python tools/preflight_postgres_auth.py
python tools/check_supabase_phase1.py
python tools/check_postdeploy_smoke.py --strict-postgres
python tools/check_data_integrity.py --expect-postgres --max-dup-chat 0 --report-json data/integrity_latest.json
python -m pytest -q tests/
```

원스텝:

```powershell
python tools/run_agent_a_gate.py
```

---

## 9) 보안 기준

1. OpenAI/Gemini/DB 키는 서버에만 저장한다.
2. Android 앱에는 키를 넣지 않는다.
3. 로그 출력 시 DSN/키 마스킹을 기본으로 한다.
4. OCR 이미지 원본은 최소 보관 원칙을 따른다.

---

## 10) 인계 포맷 (모든 도구 공통)

각 작업 완료 보고는 아래 4블록으로 고정한다.

1. `What changed` (변경 사항 3~7줄)
2. `Validation` (검증 명령/결과)
3. `Risks` (남은 리스크)
4. `Next 3 actions` (다음 액션 3개)

---

## 11) 문서 정책

이전의 분산 문서를 통합해 본 파일을 기준으로 운영한다.  
도구별 실행 문서는 아래 3개를 사용한다.

- `Antigravity_agent.md`
- `Android_Studio_agent.md`
- `Gemini-3.1-Pro_agent.md`
