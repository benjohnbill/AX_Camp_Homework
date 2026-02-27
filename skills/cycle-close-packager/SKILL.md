---
name: cycle-close-packager
description: Package and validate mandatory cycle-close artifacts for CT handoff. Use when closing a cycle or preparing a contextless CT restart package.
---

# SKILL: Cycle Close Packager

## Goal
Standardize cycle-close packaging so CT can publish a complete, reproducible handoff set with minimal rework.

## Required Inputs
1. `docs/CT_BASELINE_2026-02-25.md`
2. `docs/MASTER_PLAN_CYCLE04_06.md`
3. `orchestration/handoff/latest.handoff.json`
4. `orchestration/task.json`
5. Latest `orchestration/results/*.result.json`
6. `integration_status.md`

## Mandatory Output Set (Five)
1. `orchestration/task.json`
2. `orchestration/dispatch/*.worker-prompts.json`
3. `orchestration/results/*.result.json`
4. `orchestration/handoff/latest.handoff.json`
5. `docs/CT_BASELINE_<date>.md`

## Packaging Procedure
1. Resolve cycle trace ID and timestamp basis from latest canonical handoff/result artifacts.
2. Verify each mandatory output exists and belongs to the same cycle timestamp family.
3. Check consistency fields:
   - trace ID continuity (`task`, `result`, `handoff`)
   - worker coverage (`backend_cli`, `frontend_ide`, `android_ide` when required by scope)
   - status consistency (pass/blocked language matches canonical JSON)
4. Reconcile markdown summaries (`integration_status.md`, baseline docs) so they do not override JSON verdicts.
5. Publish cycle-close summary using L3 template:
   - `orchestration/templates/chat_l3_ct_summary.md`

## Safety Rules
1. If any mandatory artifact is missing, stop close and mark `blocked` with root cause.
2. If evidence is contradictory, trust canonical JSON in this order:
   - `orchestration/handoff/latest.handoff.json`
   - `orchestration/task.json`
   - latest `orchestration/results/*.result.json`
   - `integration_status.md`
3. Do not perform capability expansion (new MCP servers or external skill activation) during cycle close.

## Validation Commands
1. `.\tools\project_python.ps1 tools/validate_contracts.py`
2. `.\tools\project_python.ps1 tools/check_docs_contract.py --mode strict`
3. `.\tools\project_python.ps1 tools/check_skill_registry.py --mode strict`
4. `.\tools\project_python.ps1 tools/run_agent_a_gate.py --policy-mode strict`
