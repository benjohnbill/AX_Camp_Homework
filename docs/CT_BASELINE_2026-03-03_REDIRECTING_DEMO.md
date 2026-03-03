---
doc_type: ct_baseline
owner: control_tower
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - orchestration/handoff/latest.handoff.json
  - orchestration/task.json
  - redirecting/REDIRECTING_INDEX_2026-03-03.md
  - redirecting/REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md
  - redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md
  - integration_status.md
change_triggers:
  - phase_gate_changed
  - handoff_updated
  - worker_dispatch_changed
sunset_condition: Replace when redirecting Phase 2 demo close handoff is published.
---
# CT Baseline (Redirecting Demo As-Of 2026-03-03)

## 1) Snapshot Anchor
- Baseline timestamp: `2026-03-03T12:00:00Z`
- Current execution state: `Redirecting Phase1/Phase2 demo kickoff active`
- Canonical kickoff handoff:
  - `orchestration/handoff/latest.handoff.json`
  - `orchestration/handoff/20260303T120000Z.T-narrative_loop-20260303-redirecting-phase12-kickoff.handoff.json`

## 2) Source-of-Truth Priority
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. latest `orchestration/results/*.result.json`
4. `integration_status.md`
5. `redirecting/*.md`, `docs/*.md`

## 3) Redirecting Kickoff Package (Published)
1. `orchestration/task.json`
2. `orchestration/dispatch/20260303-redirecting-phase12-kickoff.worker-prompts.json`
3. `orchestration/tasks/20260303T120000Z.backend-redirecting-phase12.task.json`
4. `orchestration/tasks/20260303T120000Z.frontend-redirecting-phase12.task.json`
5. `orchestration/tasks/20260303T120000Z.android-redirecting-phase12.task.json`
6. `orchestration/results/20260303T120000Z.T-narrative_loop-20260303-redirecting-phase12-kickoff.result.json`
7. `orchestration/handoff/20260303T120000Z.T-narrative_loop-20260303-redirecting-phase12-kickoff.handoff.json`
8. this document `docs/CT_BASELINE_2026-03-03_REDIRECTING_DEMO.md`

## 4) Active Worker Pointers
- Backend: `orchestration/tasks/20260303T120000Z.backend-redirecting-phase12.task.json`
- Frontend: `orchestration/tasks/20260303T120000Z.frontend-redirecting-phase12.task.json`
- Android: `orchestration/tasks/20260303T120000Z.android-redirecting-phase12.task.json`
- Shared dispatch: `orchestration/dispatch/20260303-redirecting-phase12-kickoff.worker-prompts.json`

## 5) Phase Gates
- Gate P1 (must pass first):
  - Focus/Reflection loop demo completion within 3 minutes.
  - OCR failure/delay must not block session completion.
  - Gap/Entropy flow remains disabled in demo path.
- Gate P2 (can start only after P1 pass):
  - Reflection evidence curation (1~2 images + meaning/skip).
  - 3-tier weekly replay (`session_completed`, `session_interrupted`, `supporting_evidence`).
  - Async insight fallback remains non-blocking.

## 6) Guardrails
- No deploy/schema migration/permission elevation/destructive command without explicit user approval.
- No coercive dashboard UX default.
- No uncontrolled expansion to Phase 3 architecture in this cycle.
- Canonical JSON verdict overrides markdown narrative.
- Android separate-repo mode must follow artifact bridge protocol:
  - `orchestration/ANDROID_EXTERNAL_REPO_ARTIFACT_BRIDGE_2026-03-04.md`

## 7) CT Immediate Next 3 Actions
1. Dispatch lane prompts and collect Phase 1 evidence first.
2. Publish P1 pass/block verdict artifact with explicit blocker roots.
3. Start only bounded P2 scope after P1 pass.
