---
doc_type: execution_runbook
owner: control_tower
authority_level: L2
last_updated: 2026-02-24
sync_with:
  - android/NarrativeLoopMobile/ANDROID_REPORT.md
  - android/NarrativeLoopMobile/ANDROID_REPORT_TEMPLATE.md
  - android/NarrativeLoopMobile/evidence/
  - orchestration/task.json
change_triggers:
  - cycle3 start
  - product mvp checklist update
  - auth contract update
sunset_condition: Remove after cycle3 final decision is published.
review_by: 2026-02-26
---

# Cycle3 Product MVP Runbook

## Goal
Confirm product MVP from user perspective on both emulator and physical device.

Required flow:
1. write narrative
2. save success
3. app restart and re-open
4. written narrative re-query success
5. universe entry/render success
6. auth UX fallback (401/403) and lifecycle stability

## 0) Preconditions
- Shell opened at repository root.
- Runtime secrets are set in current shell (`UNIVERSE_JWT_SECRET`, `UNIVERSE_AUTH_ISSUER`, `DEBUG_TOKEN_ADMIN_KEY`).
- Android target visible in `adb devices`.

## 1) Run automation precheck
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\android_cycle3_product_mvp_runner.ps1 `
  -Mode all `
  -ExecutionUnit cycle03-product-mvp-validation `
  -TraceId trace-narrative_loop-20260225-cycle03 `
  -LaunchApp `
  -JsonOut data/evidence/android_cycle3_product_mvp_latest.json `
  -MarkdownOut android/NarrativeLoopMobile/evidence/android_cycle3_product_mvp_latest.md
```

Expected precheck status:
- `valid_first=307`
- `cookie_follow_up=307`
- `forbidden=403`
- `no_auth=401`
- `online_device_count >= 1`

## 2) Manual user journey verification
Fill the generated checklist file:
- `android/NarrativeLoopMobile/evidence/android_cycle3_product_mvp_latest.md`

Attach:
- screenshot paths
- adb/logcat snippets
- write/save/re-query outcomes

## 3) Report update
Update:
- `android/NarrativeLoopMobile/ANDROID_REPORT.md`

Mandatory final table should include both:
- auth contract scenarios (307/401/403)
- product journey scenarios (write/save/re-query/universe/lifecycle)

## 4) CT artifacts
After report closure:
1. publish worker `result.json`
2. publish cycle `result.json`
3. publish canonical `handoff.json`
4. refresh `orchestration/handoff/latest.handoff.json`

