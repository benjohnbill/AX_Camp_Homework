---
doc_type: worker_inbox
owner: control_tower
authority_level: L2
last_updated: 2026-03-02
sync_with:
  - orchestration/android.current.json
  - orchestration/dispatch/20260302-cycle05-iteration2.worker-prompts.json
  - orchestration/tasks/20260302T174500Z.android-cycle05-iteration2.task.json
change_triggers:
  - cycle05_kickoff_published
  - android_cycle05_result_submitted
sunset_condition: Replace when cycle05 iteration handoff updates the active android lane directive.
review_by: 2026-03-04
---

# CT Inbox for Android (Cycle05 Iteration1)

## 1) Status Update
- Current focus: cycle05 iteration2 Android resilience execution.
- Active trace/task: `trace-narrative_loop-20260302-cycle05` / `T-narrative_loop-20260302-android-cycle05-iteration2`.
- Canonical pointers:
  - `orchestration/android.current.json`
  - `orchestration/task.json`
  - `orchestration/handoff/latest.handoff.json`

## 2) Required Execution
1. 공통 지시(동일 문구): metric evidence 포함, schema-valid result 제출, blocker 발생 시 L1에서 원인/완화 먼저 보고.
2. 공통 지시(동일 문구): 무단 기능/스키마/API/권한 확장 금지.
3. android 실행: physical+emulator 동일 윈도우에서 create/save/re-open/re-query/universe 경로 검증.
4. android 산출: `worker=android_ide` result JSON + device_id/timestamp 포함 증거 경로.

## 3) Reporting Contract
1. L1 update first using `orchestration/templates/chat_l1_worker_update.md`.
2. Result JSON second with:
   - evidence paths
   - pass/fail checks
   - blocker root cause + mitigation when blocked
3. Do not expand feature scope in cycle05 iteration1.

## 4) Scope Guard
- Allowed: reliability checks, stabilization fixes, evidence refresh.
- Not allowed: new feature development, permission/scope expansion, backend contract redesign without approval.
