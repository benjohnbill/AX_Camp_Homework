# Agent.md — Narrative_Loop 통합 운영 문서 (SSOT)

이 문서는 Narrative_Loop의 단일 진실 원천이다.  
모든 AI 도구(Codex CLI, Antigravity, Android Studio, frontend_ide)는 이 문서를 우선 기준으로 작업한다.

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

## 0.1) 자율 루프 (Autonomous Ralph Loop) 운영 원칙

CT의 과부하를 방지하고 에이전트 간 병렬 협업을 극대화하기 위해 '자율 루프' 아키텍처를 채택하며, **관제 부관(Adjutant - Analyst CLI)** 레이어를 통해 지휘관의 통제력을 유지한다.

1. **관제 부관 (Adjutant Oversight)**:
   - 모든 워커와 CT의 산출물(`result.json`, `handoff.json`)은 부관의 검수 과정을 거친다.
   - 부관은 시스템 L0/L1 원칙에 따라 게이트를 검증하고 지휘관에게 '현상-원인-대책' 보고를 수행한다.
   - 지휘관의 최종 승인(Approval)이 필요한 항목은 부관이 실행을 차단하고 보고한다.

2. **자율 제안 (Pull-based Tasking)**:
   - 워커는 CT의 지시를 기다리지 않고, `integration_status.md`와 `INBOX.md`를 스스로 스캔하여 수행 가능한 작업을 찾아 `task.json` 초안(Proposal)을 제안한다.
   - CT는 제안된 `task.json`을 검토하고 승격(Approve)하는 관리자 역할을 수행한다.

3. **신경망 동기화 (Sync-Dispatcher)**:
   - 부관은 모든 워커의 산출물을 실시간 취합하여 공용 게시판(`integration_status.md`)을 갱신하고, 다음 워커의 실행 트리거를 관리한다.

3. **안전 정지 및 에스컬레이션 (Safety Stop Rules)**:
   - 다음 상황에서 워커는 자율 루프를 즉시 중단하고 CT에게 보고(Escalation)해야 한다.
     - **규정 정지**: '사용자 승인 필요 항목' 수행 직전.
     - **반복 실패**: 동일 오류 3회 이상 발생 시.
     - **권한 초과**: `DOMAIN_MAP.md`를 벗어나는 결정이 필요할 때.
   - 보고를 받은 CT는 즉시 사용자에게 개입을 요청한다.

---

시스템 우선 규칙:
- 본 문서는 프로젝트 SSOT이지만 시스템 헌법 문서와 충돌할 수 없다.
- 충돌 시 시스템 문서가 우선한다:
  - `02_Core_Resources/01_Agent_Orchastration_System/SYSTEM_BLUEPRINT.md`
  - `02_Core_Resources/01_Agent_Orchastration_System/SYSTEM_AGENT_POLICY.md`
  - `02_Core_Resources/01_Agent_Orchastration_System/SYSTEM_HANDOFF_CONSTITUTION.md`
  - Optional on demand: `SYSTEM_HANDOFF_MIGRATION_POLICY.md`, `SYSTEM_SKILL_GOVERNANCE_POLICY.md`, `SYSTEM_MCP_POLICY.md`, `SYSTEM_REMOTE_POLICY.md`

MVP 고정 경계(현재 단계):
- 지금 고정: 권한 계층, handoff canonical(JSON), DDD-lite 도메인 경계, gate mode split, skill intake 통제.
- 프로젝트 중 개선: skill 승격(candidate->pilot->core), 도메인 내부 모델 정밀화, 실행 스크립트 확대.

Core-first loading (zero-context CT):
1. `SYSTEM_BLUEPRINT.md`
2. `SYSTEM_AGENT_POLICY.md`
3. `SYSTEM_HANDOFF_CONSTITUTION.md`
4. this `Agent.md`
5. `DOMAIN_MAP.md`
6. `orchestration/contracts/*.schema.json`

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
- Backend retrieval/context 정책 기준: `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md`
- Documentation harness 정책 기준: `Harness_Policy.md`
- MCP 최소 연결 설정: `.mcp.json` (Phase 1 read-only)
- Python runtime policy:
  - This repository is OneDrive-synced. Project-local `venv/.venv` is not canonical.
  - Runtime root is resolved dynamically: `$env:LIFE_VENV_ROOT` -> default `$env:USERPROFILE\.venvs_hub` (fallback `C:\venvs_hub` for non-ASCII profile environments).
  - Canonical runtime venv naming rule: `<resolved_vroot>\Narrative_Loop.venv`
  - Compatibility alias naming rule: `<resolved_vroot>\narrative_loop` (junction may point to canonical venv).
  - Docs/logs must record canonical venv name + resolved venv root for the current machine.
  - Optional override: `$env:LIFE_VENV_ROOT`
  - Detailed runbook: `LOCAL_ENV_SETUP.md`
  - Bootstrap command: `.\tools\bootstrap_env.ps1 -Recreate -InstallPreCommit`

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

### 2.1) Domain-Driven Ownership (Lite)

도메인 소유권은 "파일 위치"보다 "업무 책임의 완결성"을 기준으로 판정한다.

- `Narrative Core`: Codex(통합 정책) + Antigravity(구현) 공동 책임
- `Ingest/Auth Gateway`: Antigravity 1차 책임, Android는 클라이언트 계약 책임
- `Retrieval/Context Enrichment`: Antigravity 1차 책임
- `Runtime UI/UX`: frontend_ide(설계+구현, 운영 주체 Antigravity) 책임
- `Governance/Orchestration`: Codex 1차 책임

파일 경로는 소유권 판정의 보조 기준(거점)이며, 최종 판정은 `DOMAIN_MAP.md`와
Control Tower 통합 게이트를 따른다.

---

## 3) 필수 아키텍처 원칙

1. `EvidenceGateway`는 I/O만 담당한다.
2. `PolicyEngine`은 DB를 직접 호출하지 않는다.
3. `narrative_logic.py`에서 UI 상태(`st.session_state`) 직접 조작 금지.
4. `DATABASE_URL`, API Key 등 민감값은 로그/문서에 절대 노출하지 않는다.
5. SQLite는 운영 DB가 아니라 fallback/로컬 개발용이다.
6. 하이브리드 검색/컨텍스트 enrichment 정책 충돌 시 `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md` 기준을 우선 적용한다.
7. 문서 간 충돌 발생 시 권한/판정 규칙은 `Harness_Policy.md` 기준을 따른다.

---

## 4) 멀티도구 역할 분담

### Codex CLI (기획/통합)

- 철학 정렬, 아키텍처 결정, 통합 설계, 문서 기준 확정.
- 에이전트 산출물 검토 및 충돌 해소.

### Antigravity (Backend + Streamlit)

- 인제스트 API 계층 구현/운영.
- `narrative_logic.py` 재사용 연결.
- Supabase 저장 규약/무결성/운영 점검 책임.
- 하이브리드 검색(RRF), context split 저장, sync/async enrichment 워크플로우 구현 책임.

### Android Studio (모바일)

- CameraX, OCR 입력 UX, 네트워크 재시도, 응답 렌더링 책임.
- 키를 클라이언트에 두지 않고 토큰 기반 인증만 사용.

### Frontend IDE (UI/UX + 구현)

- 내러티브 UX 설계와 `app.py` 구현을 단일 트랙으로 수행.
- 숙제화 없는 카피/인터랙션 설계와 회귀 검증을 함께 책임.
- 기존 Gemini UI 제안 역할은 frontend_ide에 통합한다.

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
.\tools\project_python.ps1 tools/preflight_postgres_auth.py
.\tools\project_python.ps1 tools/check_supabase_phase1.py
.\tools\project_python.ps1 tools/check_postdeploy_smoke.py --strict-postgres
.\tools\project_python.ps1 tools/check_data_integrity.py --expect-postgres --max-dup-chat 0 --report-json data/integrity_latest.json
.\tools\project_python.ps1 tools/check_docs_contract.py --mode warn
.\tools\project_python.ps1 tools/check_skill_registry.py --mode warn
.\tools\project_python.ps1 tools/validate_contracts.py
.\tools\project_python.ps1 tools/run_risk_closure_gate.py
.\tools\project_python.ps1 -m pytest -q tests/
```

원스텝:

```powershell
.\tools\project_python.ps1 tools/run_agent_a_gate.py
```

Push/Merge gate (blocking):

```powershell
.\tools\project_python.ps1 tools/check_docs_contract.py --mode strict
.\tools\project_python.ps1 tools/check_skill_registry.py --mode strict
.\tools\project_python.ps1 tools/run_agent_a_gate.py --policy-mode strict
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

Canonical artifact:
- `orchestration/handoff/*.handoff.json` (schema-valid, machine-readable)
- `handoff.txt`는 선택 브리핑 요약 아티팩트로 유지 가능

---

## 11) 문서 정책

이전의 분산 문서를 통합해 본 파일을 기준으로 운영한다.  
도구별 실행 문서는 아래를 사용한다.

- `Antigravity_Agent.md`
- `Android_Studio_Agent.md`
- `Gemini-3.1-Pro_Agent.md` (legacy alias: frontend_ide 통합 가이드)

보안/운영 참고 문서:

- `DEBUG_TOKEN_GOVERNANCE.md` (디버그 JWT 운영 규칙 및 책임 분담)
- `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md` (하이브리드 검색/컨텍스트 저장 실행 기준)
- `Harness_Policy.md` (문서 하니스 권한 체계/동기화/자동검증/주기 루프 기준)
- `MCP_USAGE_POLICY.md` (MCP 최소 연결, 호출 예산, 중단 규칙 기준)
- `SKILL_PROMOTION_POLICY.md` (`SKILL.md` 승격 조건 및 추천 트리거 기준)
- `LOCAL_ENV_SETUP.md` (OneDrive 동기화 환경에서의 로컬 venv 복구 기준)
- `skills/integration-status-sync/SKILL.md` (`integration_status.md` 증적 동기화 파일럿 스킬)
- `orchestration/README.md` (`task/result/handoff` 계약 및 샘플 아티팩트 기준)
- `DOMAIN_MAP.md` (DDD-lite 도메인 경계 기준)

