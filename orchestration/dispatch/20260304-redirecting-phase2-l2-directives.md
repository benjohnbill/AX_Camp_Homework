---
doc_type: ct_directives
owner: control_tower
authority_level: operational
last_updated: 2026-03-04
sync_with:
  - orchestration/tasks/20260304T140000Z.backend-redirecting-phase2.task.json
  - orchestration/tasks/20260304T140000Z.frontend-redirecting-phase2.task.json
  - orchestration/tasks/20260304T140000Z.android-redirecting-phase2.task.json
  - orchestration/dispatch/20260304-redirecting-phase2-kickoff.worker-prompts.json
sunset_condition: Replace when Phase 2 iteration-1 aggregate is published.
---
# Redirecting Phase2 L2 Directives

## Backend
```text
[L2_CT_DIRECTIVE]
target_worker: backend_cli
trace_id: trace-narrative_loop-20260304-redirecting-phase2
task_file: orchestration/tasks/20260304T140000Z.backend-redirecting-phase2.task.json
priority: P0
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.
scope:
- Harden ai_jobs minimum lifecycle and keep behavior deterministic.
- Guarantee session/week insight immediate response through rule fallback when AI jobs are delayed/failed.
- Preserve all Phase 1 non-blocking core loop guarantees and compatibility paths.
exit_criteria:
1) Async fallback evidence proves no core-loop blocking.
2) Schema-valid result/handoff with reproducible evidence is published.
required_output:
1) L1 update (fast lane)
2) schema-valid result.json + handoff.json (slow lane)
```

## Frontend
```text
[L2_CT_DIRECTIVE]
target_worker: frontend_ide
trace_id: trace-narrative_loop-20260304-redirecting-phase2
task_file: orchestration/tasks/20260304T140000Z.frontend-redirecting-phase2.task.json
priority: P0
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.
scope:
- Implement reflection evidence curation (1~2 items, meaning/skip).
- Implement 7-day 3-tier replay with Tier1 session_completed, Tier2 session_interrupted, Tier3 supporting_evidence.
- Keep replay read-only with single CTA + Skip and avoid rerun-heavy regressions.
exit_criteria:
1) Demo runbook proves curation and 3-tier replay behavior.
2) Core loop remains non-blocking and stable under phase2 changes.
required_output:
1) L1 update (fast lane)
2) schema-valid result.json + handoff.json (slow lane)
```

## Android
```text
[L2_CT_DIRECTIVE]
target_worker: android_ide
trace_id: trace-narrative_loop-20260304-redirecting-phase2
task_file: orchestration/tasks/20260304T140000Z.android-redirecting-phase2.task.json
priority: P1
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.
scope:
- Validate OCR path reuse and auth continuity for phase2 demo.
- Validate universe entry path and report phase3-candidate gaps explicitly if not implemented.
- Follow external-repo artifact bridge protocol (Step A local, Step B canonical mirror).
exit_criteria:
1) Bridge-compatible result/handoff and reproducible evidence are published.
2) Any unresolved scope is marked blocked with root cause + mitigation.
required_output:
1) L1 update (fast lane)
2) schema-valid result.json + handoff.json (slow lane)
```
