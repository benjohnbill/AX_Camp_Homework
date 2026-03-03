---
doc_type: redirecting_phase_execution_plan
phase: 2
owner: control_tower
authority_level: operational
last_updated: 2026-03-04
sync_with:
  - redirecting/REDIRECTING_PHASE2_MID_MVP_2026-03-03.md
  - redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md
  - redirecting/REDIRECTING_ASYNC_WORKER_PLAN_2026-03-03.md
  - orchestration/handoff/latest.handoff.json
change_triggers:
  - phase2_scope_changed
  - phase2_acceptance_changed
sunset_condition: Replace when Phase 2 close handoff is published.
---
# Redirecting Phase 2 Execution Plan (2026-03-04)

## 0) Entry Condition
- Phase 1 gate PASS is published in canonical handoff/result.
- Phase 2 starts with bounded demo scope only.

## 1) Phase 2 Scope Lock
1. 목표: `추천/회고 품질 강화 데모`.
2. 3D는 `7일 read-only replay + CTA 1개 + Skip`에 한정.
3. 자동 Core 승격/고급 drag 편집/프론트 스택 전환은 제외.

## 2) Lane Plan

### 2.1 Backend (P0)
1. `ai_jobs` 최소 lifecycle 보강 (`queued/running/succeeded/failed`).
2. `auto_tag_extraction`, `similar_session_linking`, `next_action_recommendation`만 운영.
3. `session insights/week insight`는 AI 지연 시 rule fallback 즉시 응답.
4. 핵심: AI 지연/실패가 코어 루프를 절대 차단하지 않아야 함.

### 2.2 Frontend (P0)
1. Plan-first / Focus-first / Journal 경로를 데모 기준으로 연결 유지.
2. Reflection evidence 1~2장 큐레이션 + `1줄 의미/Skip` 추가.
3. 3D/회고 화면에 7일 고정 + 3-tier 표시:
   - Tier1 `session_completed`
   - Tier2 `session_interrupted`
   - Tier3 `supporting_evidence`
4. buffering 대응:
   - 화면 계산량 분리
   - rerun 유발 상태 변이 최소화
   - 장시간 호출은 polling/비차단 조회로 전환

### 2.3 Android (P1)
1. OCR 업로드 흐름을 Phase 2 데모에서도 재사용 가능하게 유지.
2. Universe 진입/인증 연속성 점검.
3. 미구현 구간은 명시적으로 `Phase 3 candidate`로 보고.
4. 독립 레포 작업은 artifact bridge 프로토콜 Step A/B를 고정 적용.

## 3) Delivery Sequence
1. Iteration 1: backend insight/fallback + frontend evidence curation 동시.
2. Iteration 2: frontend 3-tier replay + android continuity check.
3. Iteration 3: cross-lane demo runbook 고정 및 gate verdict.

## 4) Acceptance Contract
1. 코어 루프 비차단 보장.
2. 3-tier 위계가 일관되게 시각 표현됨.
3. 3D 종료 CTA/Skip 모두 동작.
4. 발표 문구는 `회고용 3D v1` 범위를 넘지 않음.

## 5) Risks and Controls
1. Async backlog 증가:
   - low-priority job 제한 + fallback 고정.
2. Streamlit rerun 체감 저하:
   - 7일 payload 상한 + 페이지 분리 + form submit 경계 강화.
3. Android separate-repo drift:
   - bridge protocol 강제 + schema-native 템플릿 도입.

## 6) Worker Output Rules
1. Fast lane: L1 업데이트 필수.
2. Slow lane: schema-valid `result.json`, `handoff.json` 필수.
3. narrative-only 보고는 완료 판정 불가.
