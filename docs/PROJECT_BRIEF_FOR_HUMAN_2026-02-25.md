---
doc_type: project_brief
owner: control_tower
authority_level: reference
last_updated: 2026-03-02
sync_with:
  - docs/CT_BASELINE_2026-03-02.md
  - docs/CYCLE06_POSTCHECK_PRODUCT_CHECKLIST.md
  - orchestration/handoff/latest.handoff.json
  - integration_status.md
change_triggers:
  - cycle_kickoff
  - cycle_close
  - major_scope_change
sunset_condition: Replace at next cycle closure with a newer human-readable brief.
---
# Narrative_Loop 프로젝트 브리프 (사람용, 2026-03-02 갱신)

## Quick Links
- [CT Baseline](./CT_BASELINE_2026-03-02.md)
- [Cycle06 Post-Check Product Checklist](./CYCLE06_POSTCHECK_PRODUCT_CHECKLIST.md)
- [Master Plan (Cycle 04-06)](./MASTER_PLAN_CYCLE04_06.md)
- [Session Bootstrap Protocol](./SESSION_BOOTSTRAP_PROTOCOL.md)
- [Docs Index](./README.md)

## 한 줄 요약
Cycle04~06 운영 안정화는 완료됐고, 현재는 Cycle07 kickoff로 전환하여 기획서 기준 `부분충족/미충족` 항목을 실증 기반으로 `충족`으로 바꾸는 단계입니다.

## 1) 현재 상태(사실 기준)
- Cycle06 close는 canonical artifact로 종료 완료.
- Cycle06 post-check 결과:
  - Core Loop: PASS
  - Cycle04~06 Ops: PASS
  - Short-Term: CONDITIONAL
  - Overall(중기/장기 제외): 부분충족
- 따라서 Cycle07의 목표는 신규 확장보다 checklist closure입니다.

## 2) Cycle07에서 반드시 닫아야 할 항목
- Key User Scenario:
  - SC-01 외부 텍스트 OCR + 감상 저장
  - SC-02 손글씨 OCR 인식/정정/저장
- Architecture/Implementation:
  - AR-03 동기/비동기 저장 분리 경로 검증 심화
  - AR-04 pgvector + SQLite fallback 경로 운영 검증 심화
  - AR-05 Android OCR -> Auth -> Backend -> Streamlit E2E 증거 강화
- Short-Term:
  - ST-01 CameraX 전용 UI(또는 동등 수준) 검증
  - ST-02 모바일 로컬 캐싱 강화 검증
  - ST-03 오프라인/저지연 연속성 검증

## 3) 운영 방식(유지)
- CT가 task/dispatch/acceptance를 고정.
- Worker(backend/frontend/android)가 lane별 증적 JSON 제출.
- 판정 우선순위:
  1. `orchestration/handoff/latest.handoff.json`
  2. `orchestration/task.json`
  3. 최신 `orchestration/results/*.result.json`
  4. `integration_status.md`
- 설명 문서보다 canonical JSON 판정을 우선 적용.

## 4) Cycle07 kickoff 활성화 상태
- Active trace: `trace-narrative_loop-20260302-cycle07`
- Active task: `T-narrative_loop-20260302-cycle07-kickoff`
- Dispatch:
  - `orchestration/dispatch/20260302-cycle07-kickoff.worker-prompts.json`
- Lane tasks:
  - `orchestration/tasks/20260302T213500Z.backend-cycle07.task.json`
  - `orchestration/tasks/20260302T213500Z.frontend-cycle07.task.json`
  - `orchestration/tasks/20260302T213500Z.android-cycle07.task.json`

## 5) 지금 팀이 해야 할 일
1. 각 lane에서 checklist 항목별 measurable acceptance를 먼저 고정.
2. narrative 보고가 아니라 schema-valid result + 실행 로그/스크린샷/디바이스 증적으로 제출.
3. iteration-1 aggregation에서 항목별 status delta(부분충족->충족 가능성)를 판정.

## 6) 리스크(현시점)
- 범위 확장 리스크:
  - checklist closure 대신 신규 기능 확장으로 새는 경우.
- 증적 품질 리스크:
  - 스크린샷 중심 보고만 있고 정량/재현 증거가 부족한 경우.
- 운영 일관성 리스크:
  - lane별 파일명/계약/스키마 불일치가 재발하는 경우.

## 관련 문서
- [CT_BASELINE_2026-03-02.md](./CT_BASELINE_2026-03-02.md)
- [CYCLE06_POSTCHECK_PRODUCT_CHECKLIST.md](./CYCLE06_POSTCHECK_PRODUCT_CHECKLIST.md)
- [MASTER_PLAN_CYCLE04_06.md](./MASTER_PLAN_CYCLE04_06.md)
- [SESSION_BOOTSTRAP_PROTOCOL.md](./SESSION_BOOTSTRAP_PROTOCOL.md)
