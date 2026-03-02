---
name: feature-lock-audit
description: Re-aggregate pre-cycle feature-lock gate status from canonical artifacts and publish consistent result/handoff/status updates. Use when pre-cycle blockers are rerun or gate decision needs refresh.
---

# SKILL: Feature Lock Audit

## Goal
Standardize pre-cycle gate re-aggregation so CT can refresh gate verdict quickly and consistently.

## Source Priority (Must Follow)
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. latest `orchestration/results/*.result.json`
4. `integration_status.md`
5. `docs/PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md`

## Scope
- In:
  - collect lane verdicts (backend/frontend/android)
  - refresh CT gate result artifact
  - refresh canonical handoff and `latest.handoff.json`
  - sync `integration_status.md` facts only
- Out:
  - cycle4 feature implementation
  - API/schema capability expansion
  - overriding JSON verdicts by markdown narrative

## Procedure
1. Resolve current trace/task from canonical handoff and task contract.
2. Load newest lane artifacts and classify each lane:
   - `pass`
   - `partial`
   - `blocked`
3. Enforce gate hard rules from pre-cycle doc:
   - android same-window emulator + physical evidence requirement
   - critical path blockers (`auth/storage/universe`) block gate pass
4. Emit CT gate result JSON with explicit evidence paths.
5. Emit handoff JSON and update `latest.handoff.json`.
6. Update `integration_status.md` to mirror canonical JSON decision.

## Validation Commands
1. `.\tools\project_python.ps1 tools/validate_contracts.py --file <new_result.json> --file <new_handoff.json> --file orchestration/handoff/latest.handoff.json`
2. Optional strict governance check:
   - `.\tools\project_python.ps1 tools/check_docs_contract.py --mode strict`
   - `.\tools\project_python.ps1 tools/check_skill_registry.py --mode strict`

## Output Contract
- Required:
  - `orchestration/results/<TS>.T-...-precycle4-gate.result.json`
  - `orchestration/handoff/<TS>.T-...-precycle4-gate.handoff.json`
  - `orchestration/handoff/latest.handoff.json`
  - updated `integration_status.md`
- Status language must match canonical JSON (`success/partial/blocked/failed` mapping).

