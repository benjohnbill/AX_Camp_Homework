---
doc_type: worker_prompt
owner: control_tower
authority_level: L2
last_updated: 2026-02-24
sync_with:
  - android/NarrativeLoopMobile/CT_INBOX_ANDROID.md
  - android/NarrativeLoopMobile/ANDROID_REPORT_TEMPLATE.md
change_triggers:
  - execution unit change
  - checklist change
sunset_condition: Remove after Android worker can consume orchestration dispatch JSON directly.
review_by: 2026-02-26
---

# Android Quick Prompt

Copy and paste this prompt to Android Studio worker, then only replace placeholders.

```text
You are android_ide worker for Narrative_Loop.
Execution unit: [EXECUTION_UNIT_ID]
Date: [DATE_YYYY-MM-DD]

You can only read this directory:
D:\OneDrive\바탕 화면\Life_System\01_Active_Projects\08_AX 코딩 아카데미\Narrative_Loop\android\NarrativeLoopMobile

Read first:
1) CT_INBOX_ANDROID.md
2) CT_ANDROID_FEEDBACK.md
3) Android_Studio_agent.md
4) ANDROID_REPORT_TEMPLATE.md
5) TEMP_ANDROID_PROGRESS_REPORT.md (reference only; not runtime closure)

Target runtime URL:
https://ax-camp-universe-gateway-staging.onrender.com/gateway/universe_3d

Backend precondition (already verified by CT):
- /debug/token with admin key = 200 issued
- gateway health/auth-coded behavior is stable
- /debug/token can still reject forbidden audience minting (policy allowlist)

Do on-device E2E and collect evidence for:
1) first bearer request success
2) cookie follow-up success
3) clear auth session before negative-case tests
4) 401 friendly UX
5) 403 friendly UX
6) lifecycle stability (tab switch, background->foreground)

403 rule:
- If `/debug/token` rejects forbidden audience minting, generate forbidden-audience JWT locally with valid signature/issuer and wrong aud claim.

Important:
- Code review only is not accepted.
- Runtime evidence is mandatory for each checklist item.
- TEMP report can be used as implementation context only, not as completion evidence.

Write your report to:
ANDROID_REPORT.md

Report must keep exactly these 4 blocks:
1) What changed
2) Validation (command + outcome + evidence path)
3) Risks (severity included)
4) Next 3 actions

Rules:
- Never expose plain token/key/secret.
- Keep evidence paths inside android/NarrativeLoopMobile/.
- Include canonical root and alias root availability status.
- If blocked, classify blocker as environment/device/runtime.
```
