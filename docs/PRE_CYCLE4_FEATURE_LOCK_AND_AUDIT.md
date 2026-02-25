---
doc_type: hardening_gate
owner: control_tower
authority_level: operational
last_updated: 2026-02-25
sync_with:
  - orchestration/task.json
  - orchestration/results
  - orchestration/handoff/latest.handoff.json
  - MASTER_PLAN_CYCLE04_06.md
change_triggers:
  - cycle4_kickoff
  - major_feature_change
  - regression_detected
sunset_condition: Archive after Cycle 04 kickoff handoff is accepted and gate checklist is fully passed.
---
# Pre-Cycle4 Feature Lock And Audit

## Goal
Lock current functionality and operational safety before starting Cycle 4.

## Coverage Rule
This audit covers full active resources, not only `Stream/Desk/Chronos/Control/Universe` mode visibility.

## Audit Matrix

| Area | Scope | Evidence/Code Anchor | Owner | Status |
|---|---|---|---|---|
| Web mode routing | Mode selection and render dispatch | `app.py` (`render_stream_mode`, `render_desk_mode`, `render_chronos_mode`, `render_control_mode`, `render_universe_mode`) | frontend_ide | [ ] |
| Write/Save/Re-query loop | Save and related-search behavior | `narrative_logic.py` (`save_log`, `find_related_logs`) | frontend_ide | [ ] |
| Chronos persistence | Timer set/get/clear lifecycle | `app.py`, `db_manager.py`, `tests/test_interface_parity.py` | frontend_ide | [ ] |
| Universe render path | 3D payload serialization and render resilience | `universe_3d.py`, `tests/test_universe_3d_serialization.py` | frontend_ide | [ ] |
| Auth gateway contract | `200/401/403/307` behavior and cookie/session bridge | `gateway_fastapi.py`, `universe_auth.py`, `tests/test_gateway_fastapi.py`, `tests/test_universe_auth.py` | backend_cli | [ ] |
| Korean query rewrite | Query normalization/rewrite quality | `korean_query_rewrite.py`, `tests/test_korean_query_rewrite.py` | backend_cli | [ ] |
| Storage interface parity | DB abstraction parity and regressions | `db_manager*.py`, `tests/test_interface_parity.py` | backend_cli | [ ] |
| Android product journey | Write/save/re-open/query/universe on emulator + physical device | `android/NarrativeLoopMobile/ANDROID_REPORT.md` and latest evidence docs | android_ide | [ ] |
| Android navigation/mode wiring | Home/Desk/Chronos/Universe fragment navigation health | `android/NarrativeLoopMobile/app/src/main/res/navigation/mobile_navigation.xml` + fragment files | android_ide | [ ] |
| Contract governance | task/result/handoff schema and trace/task alignment | `tools/validate_contracts.py`, `orchestration/contracts/*.schema.json` | control_tower | [ ] |
| Docs/governance sync | Baseline + status docs aligned with canonical JSON | `integration_status.md`, `CT_BASELINE_2026-02-25.md` | control_tower | [ ] |

## Mandatory Test Set
Run at least:
- `.\tools\project_python.ps1 -m pytest -q tests/test_interface_parity.py tests/test_universe_auth.py tests/test_universe_3d_serialization.py`
- `.\tools\project_python.ps1 -m pytest -q tests/test_gateway_fastapi.py tests/test_korean_query_rewrite.py`
- `.\tools\project_python.ps1 tools/validate_contracts.py`

## Exit Criteria (All Required)
- [ ] All audit matrix rows have owner, evidence path, and explicit verdict (`pass/partial/blocked`).
- [ ] No unresolved `blocked` item in auth/storage/universe critical path.
- [ ] Android evidence includes both emulator and physical-device rerun logs for the same cycle window.
- [ ] Contract validation passes on all updated cycle artifacts.
- [ ] Cycle 4 kickoff package is published (task/dispatch/result/handoff/baseline).

## Failure Policy
If any exit criterion fails:
1. Do not start Cycle 4 feature expansion.
2. Publish a blocking CT handoff with root cause and mitigation.
3. Re-run only failed matrix sections with new evidence timestamps.

## Deliverables
- Updated `orchestration/results/*.result.json` for hardening run.
- Updated `orchestration/handoff/latest.handoff.json` with gate decision.
- Updated `CT_BASELINE_<date>.md` reflecting lock verdict.
