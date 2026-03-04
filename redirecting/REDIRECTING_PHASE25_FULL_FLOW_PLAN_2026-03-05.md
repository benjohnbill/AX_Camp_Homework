---
doc_type: redirecting_phase_plan
phase: 2.5
owner: control_tower
authority_level: planning
last_updated: 2026-03-05
sync_with:
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - redirecting/REDIRECTING_PHASE2_CLOSURE_2026-03-04.md
  - redirecting/REDIRECTING_PHASE3_COMPLETE_2026-03-03.md
  - app.py
  - gateway_fastapi.py
change_triggers:
  - phase25_scope_changed
  - full_flow_contract_changed
sunset_condition: Replace when Phase 2.5 gate is closed and Phase 3 kickoff is approved.
---
# Redirecting Phase 2.5: Full Flow Completion Plan (2026-03-05)

## 0) 목적
- Phase 2 데모 기준 완료와 별개로, 원 기획의 핵심 사용자 흐름을 실제 제품 흐름으로 완성한다.
- 대상 핵심 흐름:
  - `Plan-first: Frog -> Time-Box -> Pomodoro -> Reflection`
  - `Focus-first: Pomodoro -> Retro Time-Box -> Reflection`
  - `OCR -> Session Link -> Reflection Curation`

## 1) 왜 Phase 2.5인가
1. 현재 상태는 데모 범위(축약판) 기준 완료이며, 풀 플로우는 일부 미완이다.
2. 이 갭은 아키텍처 전환(Phase 3) 없이도 현재 스택(Streamlit + gateway)에서 먼저 닫을 수 있다.
3. 따라서 Phase 3로 바로 점프하지 않고, Phase 2.5에서 제품 UX 계약을 먼저 완성한다.

## 2) Scope (Must)
1. Plan-first 경로 실구현
   - Frog 입력 단계
   - Time-Box 편집/확정 단계
   - 확정 후 Focus 실행
   - Reflection 필수 3필드 저장
2. Focus-first 경로 실구현
   - 즉시 Focus 시작
   - 종료 후 Retro Time-Box 입력(또는 Skip)
   - Reflection 합류
3. OCR 연동 실구현
   - OCR 업로드 시 세션 연결(우선순위 규칙 적용)
   - Reflection에서 실제 세션 연결 evidence 1~2개 큐레이션
4. Journal/Promote/Core 경로 정합성
   - Journal 저장
   - Journal -> Session 승격
   - Core 수동 승격
5. Universe 회고 정합성 유지
   - 7일 replay + 3-tier + CTA/Skip 계속 유지

## 3) Scope (Not in Phase 2.5)
1. 프론트엔드 런타임 분리/전면 스택 교체
2. 고급 드래그 컴포넌트 완성형 도입
3. 3D 고급 물리/대규모 필터
4. 자동 Core 승격

## 4) Lane Plan
### 4.1 Backend (P0)
1. 세션 흐름 API 보강:
   - `POST /v1/execution/session/{id}/frog`
   - `POST /v1/execution/session/{id}/timebox/draft`
   - `POST /v1/execution/session/{id}/timebox/retro`
   - `POST /v1/execution/session/{id}/evidence/link` (또는 동등 계약)
2. 기존 `start/commit/focus/reflect/journal/promote/core`와 상태 전이 일관성 확보.
3. rule-fallback 유지: AI 지연/실패가 코어 흐름 차단 금지.

### 4.2 Frontend (P0)
1. 상태머신을 기획 계약과 동일하게 노출:
   - `start_choice -> frog -> timebox_edit -> timebox_commit -> focus_running -> reflection -> done`
   - `start_choice -> focus_running -> retro_timebox -> reflection -> done`
2. `Plan Start` 클릭 시 Control 모드 우회가 아니라 Plan-first 단계로 진입.
3. Reflection evidence를 placeholder가 아닌 실제 세션 evidence 목록으로 교체.
4. Stream은 보조 진입점으로 유지하되 기본 흐름 방해 금지.

### 4.3 Android (P1)
1. OCR 업로드 시 세션 연결 필드(`session_id` 또는 규약 필드) 전달.
2. 토큰/인증 연속성 유지 상태에서 OCR -> 저장 -> 세션 반영 흐름 검증.
3. 증거는 Step A/B bridge 규격으로 canonical 경로 반영.

## 5) Iteration Plan
1. Iteration 1 (계약 정렬)
   - API/상태 전이 계약 확정 + 스키마 테스트 추가.
2. Iteration 2 (UI 풀플로우)
   - Plan-first/Focus-first 단계형 UI 완성 + Reflection 연동.
3. Iteration 3 (OCR 연계 + 회귀)
   - OCR-세션 링크 + 큐레이션 실데이터 반영 + E2E 회귀.
4. Iteration 4 (Gate)
   - 3대 시나리오 실측 PASS 후 close.

## 6) Acceptance Criteria (Gate)
1. AC-01: Plan-first 전체 단계 1회 완주 PASS.
2. AC-02: Focus-first + Retro Time-Box 경로 1회 완주 PASS.
3. AC-03: OCR 업로드가 세션에 연결되고 Reflection 큐레이션에 노출 PASS.
4. AC-04: Journal -> Promote -> Core 수동 승격 PASS.
5. AC-05: AI 실패/지연 상황에서도 세션 완료까지 비차단 PASS.
6. AC-06: Universe 7-day replay/3-tier/CTA/Skip 회귀 PASS.

## 7) Risks
1. Streamlit rerun으로 단계 draft 유실 가능성.
2. OCR 연동 시 세션 매칭 실패/경합 가능성.
3. Android bridge 누락으로 CT 집계 불일치 가능성.

## 8) Controls
1. 단계 저장은 `st.form` submit 경계에서만 commit.
2. OCR link 우선순위 규칙을 서버 단에서 강제.
3. Android 결과는 Step B canonical mirror 없으면 PASS 금지.

## 9) Exit Rule
1. Phase 2.5 close 전에는 Phase 3 kickoff 금지.
2. Phase 2.5 close 이후에만 Phase 3(아키텍처 고도화) 진입 여부를 결정한다.

