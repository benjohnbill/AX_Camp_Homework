---
doc_type: execution_checklist
owner: control_tower
authority_level: operational
last_updated: 2026-03-01
sync_with:
  - PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md
  - CYCLE4_PREP_NON_EXPANSION_2026-02-26.md
  - orchestration/handoff/latest.handoff.json
  - integration_status.md
change_triggers:
  - precycle4_blocker_status_changed
  - cycle4_kickoff_ready
sunset_condition: Archive after pre-cycle4 gate passes and cycle4 kickoff package is accepted.
---
# Pre-Cycle4 -> Cycle4 Execution Checklist (2026-03-01)

## 0) Source Priority (Fixed)
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. latest `orchestration/results/*.result.json`
4. `integration_status.md`

## 1) Pre-Cycle4 Closure (Must Finish First)

### Step A. Parallel lane run
1. Android lane:
   - `adb devices -l` confirms emulator + physical are both `device`.
   - Collect same-window full journey evidence (Write/Save/Re-open/Re-query/Universe).
2. Frontend lane:
   - Run one-shot GUI checklist for root/embed/diagnostics/OCR.
   - Capture required screenshots/report evidence.
3. CT lane:
   - Keep canonical trace/task alignment.
   - Prepare result/handoff re-aggregation payload.

### Step B. Lane artifact publication
1. Publish latest Android result artifact (`...android-precycle4.result.json`).
2. Publish latest Frontend result artifact (`...frontend-ui-manual-check.result.json`).
3. Validate both with `tools/validate_contracts.py`.

### Step C. CT gate re-aggregation
1. Publish latest gate result artifact (`...precycle4-gate.result.json`).
2. Publish latest gate handoff artifact (`...precycle4-gate.handoff.json`).
3. Update `orchestration/handoff/latest.handoff.json`.
4. Sync `integration_status.md` facts only.

### Step D. Pre-cycle4 pass criteria
All below are required:
1. Android same-window emulator + physical evidence is attached.
2. Frontend GUI checklist evidence is attached.
3. Critical-path blockers (`auth/storage/universe/android`) are zero.
4. Gate result and handoff are schema-valid.

If any criterion fails: keep `blocked`, record root cause, rerun only failed lane(s).

## 2) Cycle4 Kickoff (Only After Pre-Cycle4 PASS)
Publish mandatory five:
1. `orchestration/task.json`
2. `orchestration/dispatch/*.worker-prompts.json`
3. `orchestration/results/*cycle4-kickoff*.result.json`
4. `orchestration/handoff/*cycle4-kickoff*.handoff.json` + `latest.handoff.json`
5. `docs/CT_BASELINE_<date>.md`

## 3) Safety Rules
1. Do not start cycle4 feature implementation before pre-cycle4 PASS.
2. Do not override canonical JSON verdicts with markdown-only claims.
3. Keep MCP/skill capability expansion locked during pre-cycle4 close.

