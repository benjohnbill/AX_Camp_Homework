---
doc_type: ct_directive
owner: control_tower
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - android/NarrativeLoopMobile/agent.md
  - orchestration/tasks/20260303T120000Z.android-redirecting-phase12.task.json
  - redirecting/REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md
  - redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md
  - orchestration/ANDROID_EXTERNAL_REPO_ARTIFACT_BRIDGE_2026-03-04.md
sunset_condition: Replace after android phase1 schema-valid artifacts are published.
---
# Android Redrive Directive (Redirecting Phase12)

```text
[L2_CT_DIRECTIVE]
target_worker: android_ide
trace_id: trace-narrative_loop-20260303-redirecting-phase12
task_id: T-narrative_loop-20260303-android-redirecting-phase12
priority: P1
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.

bootstrap_read_mandatory:
1) orchestration/task.json
2) orchestration/handoff/latest.handoff.json
3) orchestration/tasks/20260303T120000Z.android-redirecting-phase12.task.json
4) android/NarrativeLoopMobile/agent.md
5) redirecting/REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md
6) redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md

redrive_reason:
- Prior android update was narrative-only and missing canonical slow-lane artifacts for CT aggregation.
- CT requires schema-valid result/handoff JSON for phase verdict.
- Separate-repo artifact bridge is active; follow bridge protocol exactly.

scope:
- Phase 1 only: OCR ingest contract alignment and evidence publication.
- Keep Android role as auxiliary input channel for demo.
- Do not start Phase 2 before explicit CT Phase 1 PASS.

mandatory_checks:
1) Endpoint and method: POST /v1/ocr/ingest (multipart image/file compatibility).
2) Auth: Authorization Bearer header path verified.
3) Evidence: one success-case request/response path with reproducible log reference.
4) If separate repo is used, generate in local orchestration path first, then mirror to canonical path per bridge protocol.
5) If mirroring is not possible, submit blocked with root cause + mitigation.

required_output:
1) orchestration/results/<timestamp>.L1-android-redirecting-phase12-update.md
2) orchestration/results/<timestamp>.T-narrative_loop-20260303-android-redirecting-phase12.result.json
3) orchestration/handoff/<timestamp>.T-narrative_loop-20260303-android-redirecting-phase12.handoff.json

schema_contract:
- result must pass orchestration/contracts/result.schema.json
- handoff must pass orchestration/contracts/handoff.schema.json

exit_criteria:
1) schema-valid result/handoff artifacts exist and are readable in repo paths.
2) OCR ingest + auth evidence is concrete and reproducible.
3) Any unresolved item is marked blocked with root cause and mitigation.
```
