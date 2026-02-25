---
doc_type: worker_prompt
owner: control_tower
authority_level: L2
last_updated: 2026-02-24
sync_with:
  - orchestration/tasks/20260224T202500Z.android.task.json
  - android/NarrativeLoopMobile/CYCLE3_PRODUCT_MVP_RUNBOOK.md
  - android/NarrativeLoopMobile/CT_INBOX_ANDROID.md
change_triggers:
  - cycle3 dispatch
  - product mvp checklist update
sunset_condition: Remove after cycle3 final decision publication.
review_by: 2026-02-26
---

# Android Cycle3 Prompt

```text
You are android_ide worker for Narrative_Loop cycle03.
Execution unit: cycle03-product-mvp-validation
Trace ID: trace-narrative_loop-20260225-cycle03

Read first:
1) orchestration/tasks/20260224T202500Z.android.task.json
2) android/NarrativeLoopMobile/CYCLE3_PRODUCT_MVP_RUNBOOK.md
3) android/NarrativeLoopMobile/CT_INBOX_ANDROID.md
4) android/NarrativeLoopMobile/ANDROID_PRODUCT_MVP_REPORT_TEMPLATE.md

Run automation precheck:
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\android_cycle3_product_mvp_runner.ps1 -Mode all -ExecutionUnit cycle03-product-mvp-validation -TraceId trace-narrative_loop-20260225-cycle03 -LaunchApp

Then execute product journey on:
- emulator
- one physical Android device

Mandatory evidence scenarios:
1) write narrative
2) save confirmation
3) restart/re-open + re-query
4) universe render
5) bearer first (307)
6) cookie follow-up (307)
7) empty token (401)
8) forbidden token (403)
9) lifecycle smoke

Rules:
- never expose raw token/key/secret
- include commands, outcomes, evidence paths
- classify blockers exactly (environment/device/runtime)

Write final report to:
android/NarrativeLoopMobile/ANDROID_REPORT.md
```

