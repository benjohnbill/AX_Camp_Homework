---
doc_type: playbook
owner: control_tower
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - CT_BASELINE_2026-03-03_REDIRECTING_DEMO.md
  - redirecting/REDIRECTING_INDEX_2026-03-03.md
  - redirecting/REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md
  - redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md
  - CHAT_CLI_MESSAGE_PROTOCOL_2026-02-25.md
change_triggers:
  - bootstrap_flow_change
  - session_boundary_policy_change
  - handoff_contract_change
sunset_condition: Keep active as canonical session-boundary bootstrap protocol.
---
# Session Bootstrap Protocol

## Purpose
Standardize handoff when a chat/session ends due to context limits, independent of cycle closure.

## Scope
- This protocol is for session boundary transitions.
- It does not replace cycle-level pass/fail handoff logic.
- In chat-triggered mode, use L3 summary format from `docs/CHAT_CLI_MESSAGE_PROTOCOL_2026-02-25.md` for concise carry-over.

## Source Priority
When resuming in a new session, trust this order:
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. latest `orchestration/results/*.result.json`
4. `integration_status.md`
5. `docs/*.md` explanatory context

## Session-Close Prompt (Copy/Paste)
Use this at the end of a session:

```text
You are control_tower finishing this session.
Generate a session-close package without changing governance rules.

Read in order:
1) docs/CT_BASELINE_2026-03-03_REDIRECTING_DEMO.md
2) redirecting/REDIRECTING_INDEX_2026-03-03.md
3) redirecting/REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md
4) redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md
5) orchestration/handoff/latest.handoff.json
6) orchestration/task.json

Output must include:
- What is done in this session (facts only)
- What remains next (top 3 actions)
- Risks/blockers
- Any changes to MCP/skill policy (or explicit "no change")
- Files updated and why
```

## Session-Open Prompt (Copy/Paste)
Use this in a new session:

```text
You are control_tower resuming from prior session.
Use docs/README.md as entrypoint and follow links.

Hard constraints:
- Do not override canonical JSON verdicts with markdown narrative.
- Keep MCP set unchanged unless explicit approval exists in handoff.
- Do not promote external skills beyond candidate without checksum+strict gate.
- Do not start Phase 2 work before explicit Phase 1 gate pass is published.

Return:
1) Current state summary
2) Immediate gate status (Phase 1 / Phase 2)
3) Next 3 executable actions
4) Any conflicts between docs and canonical JSON
```

## Required Outputs At Session Boundary
- Session-close answer must explicitly list:
  - timestamp basis
  - source files read
  - unresolved blockers
  - handoff target (next CT/worker)

## Relationship To Cycle Deliverables
- Cycle close still requires the mandatory five outputs defined in:
  - `docs/CT_BASELINE_2026-03-03_REDIRECTING_DEMO.md`
  - `orchestration/contracts/*.schema.json`
- Session close is additive, not a substitute.
