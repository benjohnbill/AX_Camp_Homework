---
doc_type: playbook
owner: control_tower
authority_level: operational
last_updated: 2026-02-25
sync_with:
  - CT_BASELINE_2026-02-25.md
  - SESSION_BOOTSTRAP_PROTOCOL.md
  - orchestration/dispatch/20260225-precycle4.worker-prompts.json
  - orchestration/antigravity.current.json
  - orchestration/backend.current.json
  - orchestration/android.current.json
change_triggers:
  - chat_cli_operation_change
  - worker_reporting_format_change
  - ct_load_shedding_rule_change
sunset_condition: Replace when chat-trigger constraints are removed and event-driven orchestration is introduced.
---
# Chat CLI Message Protocol (CT Single)

## Goal
Reduce Control Tower (CT) overload in a chat-triggered runtime where CT cannot poll autonomously without an external scheduler.

## Scope
- Applies to CT and workers (`frontend_ide`, `backend_cli`, `android_ide`) in chat-based operation.
- Keeps canonical verdict authority unchanged:
  1. `orchestration/handoff/latest.handoff.json`
  2. `orchestration/task.json`
  3. latest `orchestration/results/*.result.json`
  4. `integration_status.md`

## Core Model
- Single CT model:
  - One CT handles final decisions.
  - Load is reduced by message-layer discipline, not by adding a second CT in chat mode.
- Two-lane data handling:
  - Fast lane (chat summary, non-canonical)
  - Slow lane (JSON artifacts, canonical)

## Fast Lane (L1)
- Purpose: quick status signal and escalation.
- Format: max 12 lines.
- Required keys:
  - `worker`
  - `task_id`
  - `status` (`running|partial|blocked|success`)
  - `blocker_class` (`none|auth|storage|universe|android_runtime|other`)
  - `evidence_top3` (up to 3 paths)
  - `next_3`
- Template:
  - `orchestration/templates/chat_l1_worker_update.md`
- Rule:
  - L1 never overrides canonical JSON artifacts.

## Slow Lane (L2/L3 Artifacts)
- Purpose: final decision evidence.
- Required artifacts:
  - `orchestration/results/*.result.json`
  - `orchestration/handoff/*.handoff.json` (CT-side final/aggregate)
- Worker output rule:
  - Submit L1 summary first for rapid triage.
  - Submit schema-valid `result.json` for acceptance.

## CT Response Layers
- L2 (CT directive):
  - Short execution instruction to one worker.
  - Template: `orchestration/templates/chat_l2_ct_directive.md`
- L3 (CT cycle summary):
  - Aggregated decision summary for cycle/session boundary.
  - Template: `orchestration/templates/chat_l3_ct_summary.md`

## Escalation Policy
- Immediate CT interruption only when:
  - `status=blocked` AND `blocker_class` in (`auth`, `storage`, `universe`)
- Otherwise:
  - Worker continues and reports next L1 checkpoint or final `result.json`.

## Worker Channel Lock
- CT->worker communication uses single-pointer channel files:
  - `orchestration/antigravity.current.json` (frontend_ide / Antigravity)
  - `orchestration/backend.current.json` (backend_cli)
  - `orchestration/android.current.json` (android_ide)
- Additional contextual notes must stay inside channel files (`notes`, `assumptions`), not scattered across ad-hoc markdown comments.

## Non-Goals
- This protocol does not introduce a fully autonomous non-chat CT loop.
- This protocol does not replace task/result/handoff contracts.
