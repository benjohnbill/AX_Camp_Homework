---
doc_type: worker_prompt
owner: control_tower
authority_level: L2
last_updated: 2026-02-25
sync_with:
  - android/NarrativeLoopMobile/CT_INBOX_ANDROID.md
  - android/NarrativeLoopMobile/ANDROID_REPORT_TEMPLATE.md
change_triggers:
  - execution unit change
  - checklist change
sunset_condition: Remove after Android worker can consume orchestration dispatch JSON directly.
review_by: 2026-03-01
---

# Android Quick Prompt

Copy and paste this prompt to Android Studio worker, then only replace placeholders.

```text
You are android_ide worker for Narrative_Loop.
Execution unit: [CYCLE03_UNIT_ID]
Date: [DATE_YYYY-MM-DD]

You can only read this directory:
D:\dev\Narrative_Loop\android\NarrativeLoopMobile

Read first:
1) CT_INBOX_ANDROID.md
2) CT_ANDROID_FEEDBACK.md
3) Android_Studio_agent.md
4) ANDROID_REPORT_TEMPLATE.md
5) CYCLE3_PRODUCT_MVP_RUNBOOK.md

Objective:
Verify Cycle 03 Product Journey MVP (Write/Save/Re-query/Universe) on emulator and physical device.

Required evidence:
1) Auth contract stability (Bearer->Cookie transition).
2) Narrative lifecycle: Write log -> Save -> Re-open app -> Search/Query same log.
3) Universe mobile-path UX (3D load + narrative context).
4) 401/403 narrative-first copy verification.
5) Physical device stability (lifecycle + network retry).

Important:
- Use tools/android_cycle3_product_mvp_runner.ps1 if possible.
- Physical device evidence is mandatory for closure.
- Sanitize all evidence and report artifacts.

Write your report to:
ANDROID_REPORT.md

Report must keep exactly these 4 blocks:
1) What changed
2) Validation (command + outcome + evidence path)
3) Risks (severity included)
4) Next 3 actions

Rules:
- Never expose plain token/key/secret.
- Keep evidence paths inside android/NarrativeLoopMobile/evidence/.
```
