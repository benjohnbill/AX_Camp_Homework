---
doc_type: ct_directives
owner: control_tower
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - orchestration/task.json
  - orchestration/dispatch/20260303-redirecting-phase12-kickoff.worker-prompts.json
  - orchestration/tasks/20260303T120000Z.backend-redirecting-phase12.task.json
  - orchestration/tasks/20260303T120000Z.frontend-redirecting-phase12.task.json
  - orchestration/tasks/20260303T120000Z.android-redirecting-phase12.task.json
sunset_condition: Replace after Phase1 gate verdict publication.
---
# Redirecting Phase12 L2 Directives

## Backend
```text
[L2_CT_DIRECTIVE]
target_worker: backend_cli
trace_id: trace-narrative_loop-20260303-redirecting-phase12
task_file: orchestration/tasks/20260303T120000Z.backend-redirecting-phase12.task.json
priority: P0
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.
scope:
- Phase 1 first: implement minimum execution loop API path (session start/focus end/reflect/today or compatibility equivalent) and prove demo completion <=3 minutes.
- Keep OCR ingest and existing /v1/narrative compatibility; OCR or AI delay/failure must not block reflection completion.
- Only after P1 evidence pass, implement Phase 2 minimum ai_jobs lifecycle (queued/running/succeeded/failed) plus insight fallback.
exit_criteria:
1) Phase 1 backend evidence published with reproducible command outputs and non-blocking behavior proof.
2) Phase 2 backend evidence (if started) shows fallback response when AI job is delayed/failed.
required_output:
1) L1 update (fast lane)
2) schema-valid result.json (slow lane)
```

## Frontend
```text
[L2_CT_DIRECTIVE]
target_worker: frontend_ide
trace_id: trace-narrative_loop-20260303-redirecting-phase12
task_file: orchestration/tasks/20260303T120000Z.frontend-redirecting-phase12.task.json
priority: P0
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.
scope:
- Phase 1 first: expose Plan Start + Focus Now entry parity and complete Focus->Reflection save path with stable streamlit behavior.
- Minimize rerun churn using st.form boundaries and reduced state keys; do not re-enable Gap/Entropy UX in demo path.
- Only after P1 gate pass, add Phase 2 evidence curation UI (1-2 images + meaning/skip) and 3-tier weekly replay representation.
exit_criteria:
1) Phase 1 demo runbook evidence shows deterministic completion and no blocking modal path.
2) Phase 2 UI evidence (if started) shows 3-tier replay and still preserves non-blocking core loop.
required_output:
1) L1 update (fast lane)
2) schema-valid result.json (slow lane)
```

## Android
```text
[L2_CT_DIRECTIVE]
target_worker: android_ide
trace_id: trace-narrative_loop-20260303-redirecting-phase12
task_file: orchestration/tasks/20260303T120000Z.android-redirecting-phase12.task.json
priority: P1
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.
scope:
- Phase 1 first: align OCR ingest endpoint/body/auth contract with backend and provide verified request/response evidence.
- Keep Android role explicit as auxiliary evidence-input channel if full native focus/reflection parity is not in scope.
- Only after P1 gate pass, run limited Phase 2 continuity checks (auth/replay entry compatibility) and report blocked reasons precisely if any.
exit_criteria:
1) Phase 1 Android evidence proves OCR contract compatibility and stable upload path.
2) Phase 2 evidence (if started) includes continuity check or explicit blocked root-cause with mitigation.
required_output:
1) L1 update (fast lane)
2) schema-valid result.json (slow lane)
```
