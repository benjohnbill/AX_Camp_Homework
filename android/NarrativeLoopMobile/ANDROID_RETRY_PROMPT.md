---
doc_type: worker_prompt
owner: control_tower
authority_level: L2
last_updated: 2026-02-24
sync_with:
  - android/NarrativeLoopMobile/CT_INBOX_ANDROID.md
  - android/NarrativeLoopMobile/CT_ANDROID_FEEDBACK.md
  - android/NarrativeLoopMobile/ANDROID_REPORT_TEMPLATE.md
change_triggers:
  - retry unit opened
  - acceptance criteria changed
sunset_condition: Remove after cycle02 android runtime evidence passes.
review_by: 2026-02-26
---

# Android Retry Prompt (Paste to Android Studio)

```text
You are android_ide worker for Narrative_Loop.
Execution unit: cycle02-android-e2e-recovery-02
Date: 2026-02-24

You can only read:
D:\OneDrive\바탕 화면\Life_System\01_Active_Projects\08_AX 코딩 아카데미\Narrative_Loop\android\NarrativeLoopMobile

Read in order:
1) CT_INBOX_ANDROID.md
2) CT_ANDROID_FEEDBACK.md
3) Android_Studio_agent.md
4) ANDROID_REPORT_TEMPLATE.md
5) TEMP_ANDROID_PROGRESS_REPORT.md (reference only)

Target URL:
https://ax-camp-universe-gateway-staging.onrender.com/gateway/universe_3d

Backend precondition (verified):
- /debug/token with admin key returns 200 issued
- `/debug/token` may reject `aud=forbidden-audience` with `403 forbidden_audience` by policy

Important:
- Code review only is not accepted.
- You must provide real device runtime evidence.
- TEMP report is preparation evidence only and cannot close the checklist.

Required scenarios:
1) first bearer request success
2) cookie follow-up success
3) clear auth session state (logout/clear cookies) before negative-case tests
4) 401 friendly UX runtime evidence
5) 403 friendly UX runtime evidence
6) lifecycle smoke runtime evidence (tab switch + background/foreground + resume)

403 token rule:
- Do not depend on `/debug/token` to mint forbidden audience token.
- If forbidden audience issuance is blocked, mint a local JWT using project runtime secret/issuer and wrong audience claim.
- Keep token values masked in report/log artifacts.

Path workaround (if Gradle module detection fails on non-ASCII path):
- subst X: "D:\OneDrive\바탕 화면\Life_System\01_Active_Projects\08_AX 코딩 아카데미\Narrative_Loop\android\NarrativeLoopMobile"
- run from X: `gradlew.bat -q projects` then `gradlew.bat :app:tasks`
- if build/install is needed, continue from X: with `gradlew.bat :app:assembleDebug` or install command
- cleanup: `subst X: /d`

Write output to:
ANDROID_REPORT.md

Use exactly these 4 blocks:
1) What changed
2) Validation (command + outcome + evidence path)
3) Risks (severity included)
4) Next 3 actions

Security:
- Never expose plain token/key/secret.
- Keep evidence paths inside android/NarrativeLoopMobile/.
- Include blocker class if failed (`environment_blocked`, `device_blocked`, `runtime_blocked`).
```
