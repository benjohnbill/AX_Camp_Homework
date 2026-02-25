---
doc_type: worker_prompt
owner: control_tower
authority_level: L2
last_updated: 2026-02-25
sync_with:
  - android/NarrativeLoopMobile/CT_INBOX_ANDROID.md
  - android/NarrativeLoopMobile/CT_ANDROID_FEEDBACK.md
  - android/NarrativeLoopMobile/ANDROID_REPORT_TEMPLATE.md
change_triggers:
  - retry unit opened
  - acceptance criteria changed
sunset_condition: Remove after cycle03 android runtime evidence passes.
review_by: 2026-03-01
---

# Android Retry Prompt (Paste to Android Studio)

```text
You are android_ide worker for Narrative_Loop.
Execution unit: cycle03-android-product-mvp-01
Date: 2026-02-25

You can only read:
D:\dev\Narrative_Loop\android\NarrativeLoopMobile

Read in order:
1) CT_INBOX_ANDROID.md
2) CT_ANDROID_FEEDBACK.md
3) Android_Studio_agent.md
4) ANDROID_REPORT_TEMPLATE.md
5) CYCLE3_PRODUCT_MVP_RUNBOOK.md

Objective:
Complete Cycle 03 Product Journey MVP verification on both emulator and physical device.

Required scenarios:
1) Auth lifecycle (Bearer->Cookie) stability proof.
2) Narrative write/save/re-query flow.
3) Universe mobile-path UX stability.
4) 401/403 friendly UX narrative-first copy verification.
5) Lifecycle smoke (background/foreground/resume) on physical device.

Important:
- Use tools/android_cycle3_product_mvp_runner.ps1 if available.
- Physical device evidence is mandatory for Cycle 03 closure.
- Sanitize all evidence and report artifacts.

Write output to:
ANDROID_REPORT.md

Use exactly these 4 blocks:
1) What changed
2) Validation (command + outcome + evidence path)
3) Risks (severity included)
4) Next 3 actions

Security:
- Never expose plain token/key/secret.
- Keep evidence paths inside android/NarrativeLoopMobile/evidence/.
```
