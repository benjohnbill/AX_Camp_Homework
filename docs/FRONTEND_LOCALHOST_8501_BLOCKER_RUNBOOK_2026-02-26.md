---
doc_type: runbook
owner: control_tower
authority_level: operational
last_updated: 2026-02-26
sync_with:
  - integration_status.md
  - orchestration/tasks/20260225T145437Z.frontend-ui-manual-check.task.json
  - orchestration/results/20260226T000639Z.T-narrative_loop-20260225-frontend-ui-manual-check.result.json
change_triggers:
  - frontend_runtime_blocker_detected
  - streamlit_startup_failure
sunset_condition: Archive after frontend manual UI check turns PASS and pre-cycle4 gate is closed.
---
# Frontend Localhost 8501 Blocker Runbook

## Symptom
- Browser shows `ERR_CONNECTION_REFUSED` at `http://localhost:8501`.
- Frontend worker result recorded as blocked:
  - `orchestration/results/20260226T000639Z.T-narrative_loop-20260225-frontend-ui-manual-check.result.json`

## Goal
Restore local Streamlit runtime and complete one-shot manual UI validation:
- Root URL shell visibility (sidebar/mode/OCR)
- Embed route detect/escape
- Diagnostics badge values

## Step 1) Verify port listener state
```powershell
netstat -ano | findstr LISTENING | findstr :8501
```

- If a PID is returned:
  - inspect process:
    ```powershell
    Get-Process -Id <PID>
    ```
  - if it is stale/rogue, terminate:
    ```powershell
    Stop-Process -Id <PID> -Force
    ```

## Step 2) Clean stale Python/Streamlit processes
```powershell
Get-Process -Name streamlit,python -ErrorAction SilentlyContinue
```

- Terminate only stale processes related to the current workspace.

## Step 3) Start Streamlit with explicit host/port and capture logs
```powershell
.\tools\project_python.ps1 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false *> data/evidence/<TS>_frontend_streamlit_8501_startup.log
```

- Confirm startup line includes local URL binding.
- If startup fails, preserve full log as blocker evidence.

## Step 4) If 8501 still fails, use fallback port (document explicitly)
```powershell
.\tools\project_python.ps1 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false *> data/evidence/<TS>_frontend_streamlit_8502_startup.log
```

- Use `http://localhost:8502` for manual validation.
- Record that fallback port was used.

## Step 5) Re-run one-shot manual check
- Root URL validation: sidebar, mode selector, OCR entrypoints visible.
- Embed URL validation: `?embed=universe_3d` and return-to-full-app action works.
- Diagnostics badge validation:
  - `query.embed`
  - `is_embed_route`
  - `ENABLE_ENTROPY`
  - `is_entropy_mode`
  - `session.mode`

## Required Evidence Outputs
- `data/evidence/<TS>_frontend_manual_browser_ui_check.md`
- `data/evidence/<TS>_frontend_root_shell_visible.png`
- `data/evidence/<TS>_frontend_embed_route_screen.png`
- `data/evidence/<TS>_frontend_embed_escape_success.png`
- startup log:
  - `data/evidence/<TS>_frontend_streamlit_8501_startup.log`
  - or fallback `..._8502_startup.log`

## Result Contract
- Publish one schema-valid result:
  - `orchestration/results/<TS>.T-narrative_loop-20260225-frontend-ui-manual-check.result.json`
- If blocked, include:
  - exact command
  - error line
  - port used
  - remediation attempted

## Notes
- This blocker is environment/runtime class and is separate from Android physical-device blocker.
- Do not start cycle4 feature expansion while pre-cycle4 gate remains blocked.
- Project default disables onboarding prompt via `.streamlit/config.toml` (`browser.gatherUsageStats=false`).

## Recurrence Prevention Policy (Fixed)
Use the rules below for every frontend manual validation run:

1. Browser profile isolation
- Keep validation in a dedicated browser profile (or InPrivate).
- Do not treat primary daily-use profile as canonical validation environment.

2. URL policy
- Default validation URL must be root:
  - `https://benjohnbill-ax-camp-homework.streamlit.app/`
- Use embed route only for dedicated embed checks:
  - `https://benjohnbill-ax-camp-homework.streamlit.app/?embed=universe_3d`
- After embed check, return to root URL and confirm normal shell visibility.

3. Session sanity check (before test)
- Confirm query string is empty on root URL.
- Confirm browser zoom is 100% and window width is sufficient for sidebar visibility.
- Disable extension interference for the run (adblock/script-injection/dark-mode override class).

4. Data reset rule for profile-specific anomalies
- If InPrivate works but normal profile fails, classify as browser-profile data issue first.
- Clear only site data for the target domain and rerun validation.

5. Evidence metadata requirements
- Every frontend manual-check artifact must record:
  - `browser_profile` (`normal|inprivate|test-profile`)
  - `url_used`
  - `query_params`
  - `extensions_state` (`on|off|partial`)
- This metadata is mandatory to distinguish app regressions from browser-state anomalies.
