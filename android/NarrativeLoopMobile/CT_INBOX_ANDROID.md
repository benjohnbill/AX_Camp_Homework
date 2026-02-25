---
doc_type: worker_inbox
owner: control_tower
authority_level: L2
last_updated: 2026-02-24
sync_with:
  - android/NarrativeLoopMobile/ANDROID_REPORT.md
  - android/NarrativeLoopMobile/ANDROID_REPORT_TEMPLATE.md
  - android/NarrativeLoopMobile/ANDROID_PRODUCT_MVP_REPORT_TEMPLATE.md
  - android/NarrativeLoopMobile/CYCLE3_PRODUCT_MVP_RUNBOOK.md
  - android/NarrativeLoopMobile/ANDROID_QUICK_PROMPT.md
  - android/NarrativeLoopMobile/CT_ANDROID_FEEDBACK.md
  - android/NarrativeLoopMobile/ANDROID_RETRY_PROMPT.md
  - android/NarrativeLoopMobile/TEMP_ANDROID_PROGRESS_REPORT.md
  - integration_status.md
  - agent.md
change_triggers:
  - cycle change
  - auth contract change
  - acceptance criteria change
  - backend blocker status changed
  - autonomous loop dispatch update
sunset_condition: Remove after Android worker can read root orchestration artifacts directly.
review_by: 2026-02-26
---

# CT Inbox for Android

## 1) Purpose
This file is the Android worker inbox for cycle execution when Android Studio works only inside `NarrativeLoopMobile`.

## 2) Execution Unit (Current)
- Unit ID: `cycle03-product-mvp-validation`
- Date: `2026-02-24`
- Goal: confirm product MVP from end-user perspective on both emulator and physical device.

## 2.1) Current Backend Precondition (Fact-Checked)
- Backend fixed HTTPS probe currently shows:
  - `POST /debug/token` with admin key -> `200 issued`
  - `GET /healthz` -> `200`
  - `GET /gateway/session` (no auth) -> `401 missing_token`
  - `GET /gateway/universe_3d` (no auth) -> `401 missing_token`
- Source of truth: `data/staging_auth_probe_latest.json`
- Implication: Android must now prove full product journey (not only auth contract).

## 2.2) Temporary Android Progress Report Policy
- `TEMP_ANDROID_PROGRESS_REPORT.md` is treated as preparation-only evidence (`pending_user_validation`).
- It does not close the runtime checklist by itself.
- Runtime closure is accepted only through `ANDROID_REPORT.md` with explicit device-level proofs.

## 3) Access Constraint
- Visible scope: `D:\OneDrive\바탕 화면\Life_System\01_Active_Projects\08_AX 코딩 아카데미\Narrative_Loop\android\NarrativeLoopMobile`
- Android worker must not rely on files outside the scope above.

## 4) Required E2E Checklist
Run on both:
- emulator (API 36 class)
- at least one physical Android device

Required scenarios:
1. Write narrative entry and confirm save success.
2. Restart/re-open flow and re-query written narrative.
3. Universe entry/render success.
4. First bearer request success to gateway universe URL.
5. Cookie follow-up request success.
6. 401 UX behavior with friendly user-facing message.
7. 403 UX behavior with friendly user-facing message.
8. Lifecycle smoke: tab switch, background -> foreground, and resume stability.

### 4.2) Token Issuance Rule for 403 Scenario
- `POST /debug/token` is admin-protected and audience-allowlisted.
- If request body uses `aud=forbidden-audience`, service may return:
  - `403 forbidden_audience` (expected policy behavior)
- Therefore, 403 UX runtime scenario must use a locally signed JWT with:
  - valid signature (`UNIVERSE_JWT_SECRET`)
  - valid issuer (`UNIVERSE_AUTH_ISSUER`)
  - intentionally wrong audience (for example `forbidden-audience`)
- Do not log or expose plain token values in report artifacts.

## 4.1) Execution Steps (Follow in Order)
1. Run automation precheck first:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\android_cycle3_product_mvp_runner.ps1 -Mode all -ExecutionUnit cycle03-product-mvp-validation -TraceId trace-narrative_loop-20260225-cycle03 -LaunchApp`
2. Confirm emulator and physical device are both ADB-visible.
2. Build/install debug app from `android/NarrativeLoopMobile`.
   - If non-ASCII path causes Gradle module resolution issues, use temporary drive mapping workaround:
     - `subst X: "D:\OneDrive\바탕 화면\Life_System\01_Active_Projects\08_AX 코딩 아카데미\Narrative_Loop\android\NarrativeLoopMobile"`
     - run Gradle from `X:` (for example: `gradlew.bat -q projects`, `gradlew.bat :app:tasks`, `gradlew.bat :app:assembleDebug`)
     - cleanup: `subst X: /d`
3. Open app and navigate to runtime path used for gateway validation (`Universe` and `Debug` flows).
4. Run mandatory scenarios in this exact order:
   1) write narrative + save confirm
   2) restart/re-open + re-query confirm
   3) universe render confirm
   4) valid bearer scenario
   5) cookie follow-up scenario
   6) clear auth session state (token + cookie) before negative-case testing
   7) invalid/empty token scenario for 401 UX
   8) forbidden audience/permission scenario for 403 UX
   9) lifecycle smoke scenario
5. Repeat steps 3-4 on physical device after emulator run.
6. Capture evidence for each scenario before moving to the next one.

## 5) Runtime Target
- `https://ax-camp-universe-gateway-staging.onrender.com/gateway/universe_3d`

## 6) Evidence Rules
1. Record only sanitized logs and screenshots.
2. Never expose plain token, key, or secret values.
3. Evidence paths must be inside `android/NarrativeLoopMobile/`.
4. Include both of these path notes in report:
   - Canonical root: `android/NarrativeLoopMobile`
   - Alias root in this machine: `C:\Users\LG\AndroidStudioProjects\NarrativeLoopMobile` (if unavailable, explicitly mark as unavailable).
5. Code review or static logic verification alone is not accepted for checklist closure.
6. Each checklist item must include at least one runtime proof (device action + observed result + evidence path).
7. If scenario is blocked, report exact blocker class:
   - `environment_blocked` (IDE/Gradle path issue)
   - `device_blocked` (ADB/device unavailable)
   - `runtime_blocked` (app/runtime behavior mismatch)
8. Include one-line pass/fail/blocked table for all eight mandatory scenarios.

## 6.1) Minimum Evidence Bundle
- Screenshot evidence for each mandatory scenario (emulator + physical device).
- Logcat snippet showing request/response outcome near the scenario timestamp.
- ADB connectivity proof (device list or emulator online status).
- Automation report:
  - `data/evidence/android_cycle3_product_mvp_latest.json`
  - `android/NarrativeLoopMobile/evidence/android_cycle3_product_mvp_latest.md`
- Runtime root note:
  - Canonical root always required.
  - Alias root availability required (explicit yes/no).

## 7) Reporting Contract
- Write output to `ANDROID_REPORT.md` using `ANDROID_PRODUCT_MVP_REPORT_TEMPLATE.md`.
- Keep these 4 sections exactly:
1. What changed
2. Validation (command + outcome + evidence path)
3. Risks (severity included)
4. Next 3 actions
- Add mandatory scenario status summary at the end:
  - Write narrative: pass/fail/blocked
  - Save confirmation: pass/fail/blocked
  - Restart/re-open + re-query: pass/fail/blocked
  - Universe render: pass/fail/blocked
  - First bearer request: pass/fail/blocked
  - Cookie follow-up: pass/fail/blocked
  - 401 UX: pass/fail/blocked
  - 403 UX: pass/fail/blocked
  - Lifecycle smoke: pass/fail/blocked

## 8) Completion Rule
Unit is complete only when all checklist items in Section 4 have explicit runtime evidence and result status.

## 9) Autonomous Loop / Safety Stop
- Per `agent.md` autonomous loop policy, Android worker may propose follow-up task after report submission.
- Safety stop is mandatory before approval-required changes or when same runtime blocker repeats 3+ times.
- If safety stop triggered, mark report and inbox update as:
  - `HOLD: Human Intervention Required`
