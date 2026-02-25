---
doc_type: ct_baseline
owner: control_tower
authority_level: operational
last_updated: 2026-02-25
sync_with:
  - orchestration/handoff/latest.handoff.json
  - orchestration/task.json
  - integration_status.md
  - MASTER_PLAN_CYCLE04_06.md
change_triggers:
  - cycle_close
  - handoff_updated
  - worker_dispatch_changed
sunset_condition: Replace when next cycle baseline is published after CT final handoff.
---
# CT Baseline (As-Of 2026-02-25)

This document is the minimum context package for a contextless CT.

## Quick Links
- [Master Plan (Cycle 04-06)](./MASTER_PLAN_CYCLE04_06.md)
- [Pre-Cycle4 Feature Lock And Audit](./PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md)
- [Project Brief For Human](./PROJECT_BRIEF_FOR_HUMAN_2026-02-25.md)
- [Docs Index](./README.md)

## 1) Snapshot Anchor
- Baseline timestamp: `2026-02-25T23:59:59Z`
- Final cycle state: `Cycle 03 accepted/pass`
- Primary evidence:
  - `orchestration/handoff/latest.handoff.json`
  - `orchestration/results/20260225T060000Z.T-narrative_loop-20260225-backend-cycle03.result.json`
  - `orchestration/results/20260225T055000Z.T-narrative_loop-20260225-frontend-cycle03.result.json`
  - `android/NarrativeLoopMobile/ANDROID_REPORT.md`

## 2) Source-of-Truth Priority
When documents conflict, apply this order:
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. `orchestration/results/*.result.json` (latest timestamp)
4. `integration_status.md`
5. Human planning references (for direction only), including `D:\OneDrive\바탕 화면\Life_System\00_Inbox\2021117038 조진근.md`

## 3) Current Verified Functional Surface
- Web app (`app.py`):
  - `Stream` mode: write/save/re-query flow
  - `Desk` mode: long-form writing and save
  - `Chronos` mode: timer + persistence hooks
  - `Universe` mode: auth-gated 3D render path
  - `Control` mode: admin/risk controls
- Backend/auth contracts:
  - gateway auth status semantics (`200/401/403/307`) are documented as stable in cycle03 evidence.
- Android app:
  - Product journey evidence exists for write/save/query/universe and device+emulator verification.

## 4) Operating Model (Fixed)
- Topology:
  - CT (`control_tower`) defines cycle acceptance rules.
  - Workers (`backend_cli`, `frontend_ide`, `android_ide`) execute scoped tasks.
- Execution contract:
  - Inputs: `orchestration/task.json` + `orchestration/tasks/*.task.json` + dispatch prompt package
  - Outputs: `orchestration/results/*.result.json` + `orchestration/handoff/*.handoff.json`
- Governance:
  - JSON artifacts are authoritative for pass/fail.
  - Markdown is explanatory and must not override canonical JSON verdicts.

## 5) Immediate Next Gate (Pre-Cycle4)
Cycle 4 must not begin until `PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md` exit criteria are all checked:
- Full feature/resource coverage audit completed (not only mode visibility).
- Critical user journeys rerun with reproducible evidence.
- Task/result/handoff contract validation completed for updated artifacts.
- Cycle4 kickoff package published.

## 6) CT Bootstrap Prompt (Copy/Paste)
Use this as first instruction for a contextless CT:

```text
You are control_tower for Narrative_Loop.
Baseline date is 2026-02-25 23:59:59Z.
Trust source order:
1) orchestration/handoff/latest.handoff.json
2) orchestration/task.json
3) latest orchestration/results/*.result.json
4) integration_status.md

Do not start Cycle 4 work directly.
First execute Pre-Cycle4 hardening gate from docs/PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md.
Then publish cycle4 kickoff artifacts:
- orchestration/task.json
- orchestration/dispatch/<cycle4>.worker-prompts.json
- orchestration/results/<cycle4-kickoff>.result.json
- orchestration/handoff/<cycle4-kickoff>.handoff.json
- docs/CT_BASELINE_<date>.md
```

## 7) Cycle Close Output Set (Mandatory 5)
At the end of each cycle, update all five:
1. `orchestration/task.json`
2. `orchestration/dispatch/*.worker-prompts.json`
3. `orchestration/results/*.result.json`
4. `orchestration/handoff/latest.handoff.json`
5. `CT_BASELINE_<date>.md` (same cycle timestamp family)

## 8) Related Documents
- [MASTER_PLAN_CYCLE04_06.md](./MASTER_PLAN_CYCLE04_06.md)
- [PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md](./PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md)
- [PROJECT_BRIEF_FOR_HUMAN_2026-02-25.md](./PROJECT_BRIEF_FOR_HUMAN_2026-02-25.md)
