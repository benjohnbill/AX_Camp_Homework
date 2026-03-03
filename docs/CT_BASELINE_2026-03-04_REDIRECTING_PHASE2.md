---
doc_type: ct_baseline
owner: control_tower
authority_level: operational
last_updated: 2026-03-04
sync_with:
  - orchestration/handoff/latest.handoff.json
  - orchestration/task.json
  - redirecting/REDIRECTING_PHASE2_EXECUTION_PLAN_2026-03-04.md
  - redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md
  - orchestration/ANDROID_EXTERNAL_REPO_ARTIFACT_BRIDGE_2026-03-04.md
change_triggers:
  - phase2_kickoff_changed
  - phase2_gate_changed
sunset_condition: Replace when Phase 2 close handoff is published.
---
# CT Baseline (Redirecting Phase2 As-Of 2026-03-04)

## 1) Snapshot Anchor
- Baseline timestamp: `2026-03-04T14:00:00Z`
- Current execution state: `Redirecting Phase 2 kickoff active`
- Canonical kickoff handoff:
  - `orchestration/handoff/latest.handoff.json`
  - `orchestration/handoff/20260304T140000Z.T-narrative_loop-20260304-redirecting-phase2-kickoff.handoff.json`

## 2) Source-of-Truth Priority
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. latest `orchestration/results/*.result.json`
4. `integration_status.md`
5. `redirecting/*.md`, `docs/*.md`

## 3) Phase 2 Kickoff Package
1. `orchestration/task.json`
2. `orchestration/dispatch/20260304-redirecting-phase2-kickoff.worker-prompts.json`
3. `orchestration/dispatch/20260304-redirecting-phase2-l2-directives.md`
4. `orchestration/tasks/20260304T140000Z.backend-redirecting-phase2.task.json`
5. `orchestration/tasks/20260304T140000Z.frontend-redirecting-phase2.task.json`
6. `orchestration/tasks/20260304T140000Z.android-redirecting-phase2.task.json`
7. `orchestration/results/20260304T140000Z.T-narrative_loop-20260304-redirecting-phase2-kickoff.result.json`
8. `orchestration/handoff/20260304T140000Z.T-narrative_loop-20260304-redirecting-phase2-kickoff.handoff.json`

## 4) Scope Lock
1. 목표: 추천/회고 품질 강화 데모.
2. 3D 범위: 7일 read-only replay + CTA 1개 + Skip.
3. 제외: 자동 Core 승격, 고급 drag 편집, 프론트 스택 전환.

## 5) Android Separate-Repo Rule
- Android lane은 artifact bridge protocol 강제:
  - Step A: `android/NarrativeLoopMobile/orchestration/*`
  - Step B: canonical `orchestration/*` mirror + schema validation
- Step B 완료 전 PASS 판정 금지.

## 6) CT Immediate Next 3 Actions
1. Phase 2 L2 directive를 각 worker에 dispatch.
2. iteration-1 산출물 수집 후 schema validation 실행.
3. Phase 2 iteration-1 aggregate handoff 발행.
