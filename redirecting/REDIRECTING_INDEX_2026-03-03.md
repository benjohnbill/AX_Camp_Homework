---
doc_type: redirecting_index
owner: product
authority_level: planning
last_updated: 2026-03-04
sync_with:
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - app.py
  - narrative_logic.py
  - gateway_fastapi.py
change_triggers:
  - redirecting_scope_changed
  - api_contract_changed
  - worker_contract_changed
sunset_condition: Replace when redirecting plan is frozen into implementation tickets.
---
# Redirecting Index (2026-03-03)

## 목적
- `MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md`를 구현 단계로 분해한다.
- Streamlit UI 제약, 비동기 워커 분리, 데이터/API 계약을 독립 문서로 분할한다.
- Codex CLI가 문서 순서대로 구현할 수 있도록 실행 순서를 고정한다.

## 문서 목록
1. [REDIRECTING_COMPONENT_PLAN_2026-03-03.md](./REDIRECTING_COMPONENT_PLAN_2026-03-03.md)
2. [REDIRECTING_ASYNC_WORKER_PLAN_2026-03-03.md](./REDIRECTING_ASYNC_WORKER_PLAN_2026-03-03.md)
3. [REDIRECTING_DATA_API_CONTRACT_2026-03-03.md](./REDIRECTING_DATA_API_CONTRACT_2026-03-03.md)
4. [REDIRECTING_ROLLOUT_MIGRATION_PLAN_2026-03-03.md](./REDIRECTING_ROLLOUT_MIGRATION_PLAN_2026-03-03.md)
5. [REDIRECTING_KEEP_KILL_DECISION_2026-03-03.md](./REDIRECTING_KEEP_KILL_DECISION_2026-03-03.md)
6. [REDIRECTING_3D_UNIVERSE_V2_BACKLOG_2026-03-03.md](./REDIRECTING_3D_UNIVERSE_V2_BACKLOG_2026-03-03.md)
7. [REDIRECTING_PHASE1_FAST_MVP_2026-03-03.md](./REDIRECTING_PHASE1_FAST_MVP_2026-03-03.md)
8. [REDIRECTING_PHASE2_MID_MVP_2026-03-03.md](./REDIRECTING_PHASE2_MID_MVP_2026-03-03.md)
9. [REDIRECTING_PHASE3_COMPLETE_2026-03-03.md](./REDIRECTING_PHASE3_COMPLETE_2026-03-03.md)
10. [REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md](./REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md)
11. [REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md](./REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md)
12. [REDIRECTING_PHASE3_DEMO_CHECKLIST_2026-03-03.md](./REDIRECTING_PHASE3_DEMO_CHECKLIST_2026-03-03.md)
13. [REDIRECTING_PHASE2_EXECUTION_PLAN_2026-03-04.md](./REDIRECTING_PHASE2_EXECUTION_PLAN_2026-03-04.md)
14. [REDIRECTING_PHASE2_CLOSURE_2026-03-04.md](./REDIRECTING_PHASE2_CLOSURE_2026-03-04.md)

## 구현 순서 (권장)
1. 데이터/API 계약 확정.
2. 비동기 워커 스켈레톤 및 job lifecycle 구현.
3. Streamlit 단계형 UI와 컴포넌트 연결.
4. 롤아웃/플래그/검증 절차로 점진 전환.
5. Phase 1 Fast MVP 기준으로 우선 출시.
6. Phase 2 Mid MVP에서 3D 회고/비동기 최소 고도화.
7. Phase 3 Complete에서 아키텍처 승격.

## 범위 명시
- 본 redirecting 패키지는 "설계 문서"이며, 코드 변경은 별도 실행 단계에서 진행한다.
- 현재 운영 경로(기존 Stream/Chronos/Universe)는 유지하고 feature flag로 병행 전환한다.
