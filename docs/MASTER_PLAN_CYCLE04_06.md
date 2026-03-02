---
doc_type: master_plan
owner: control_tower
authority_level: policy
last_updated: 2026-02-26
sync_with:
  - CT_BASELINE_2026-02-25.md
  - PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md
  - orchestration/task.json
  - orchestration/handoff/latest.handoff.json
change_triggers:
  - cycle_transition
  - strategy_shift
  - major_risk_update
sunset_condition: Replace by a new master plan when Cycle 06 close handoff is finalized.
---
# Master Plan: Cycle 04-06

## Quick Links
- [CT Baseline](./CT_BASELINE_2026-02-25.md)
- [Session Bootstrap Protocol](./SESSION_BOOTSTRAP_PROTOCOL.md)
- [Pre-Cycle4 Feature Lock And Audit](./PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md)
- [Project Brief For Human](./PROJECT_BRIEF_FOR_HUMAN_2026-02-25.md)
- [Docs Index](./README.md)

## 1) Purpose
Define a decision-complete execution plan from Cycle 04 through Cycle 06, with clear separation of:
- `Committed (execution)`
- `Exploratory (validation)`
- `Speculative (idea backlog for Cycle 7+)`

Baseline product vision source:
- `D:\OneDrive\Desktop\Life_System\00_Inbox\2021117038 조진근.md`

## 2) Planning Ratio (Locked)
- Committed: `70%`
- Exploratory: `20%`
- Speculative: `10%`

## 3) Current Stage
- Cycle 03 is treated as accepted baseline (`2026-02-25T23:59:59Z`).
- Cycle 04 begins only after Pre-Cycle4 hardening gate passes.

## 3.1) MCP/Skill Governance Lock
- MCP lock:
  - Keep only approved read-only servers in `.mcp.json`.
  - Do not add new MCP servers in Cycle 04 unless CT records explicit approval in canonical handoff.
- Skill lock:
  - External skills remain `candidate` by default (`skills/approved_skills_registry.json`).
  - No promotion to `pilot/core` without checksum and strict registry gate.
  - Internal operation-skill plan for Cycle 04-06:
    1. `integration-status-sync` (active pilot refresh)
    2. `cycle-close-packager` (active pilot bootstrap)
    3. `feature-lock-audit` (candidate)
    4. `evidence-redaction-validator` (candidate)
  - External first-pilot preparation target:
    - `frontend-design` remains `candidate` until checksum, rollback path, and strict gate are complete.

## 4) Cycle 04 (Phase 1: Stabilization + Native Strengthening)
### Committed
- OCR flow hardening and correction UX stabilization.
- Write/Save/Re-open/Re-query/Universe user-journey non-regression.
- Android navigation reliability across Home/Stream/Desk/Chronos/Universe.
- Cycle artifacts and governance sync stabilization.
- Integration-status-sync skill refresh to remove stale `INBOX.md` assumptions and align with current channel pointer + CT inbox structure.
- Cycle-close-packager pilot bootstrap for mandatory five artifact packaging.

### Exploratory
- CameraX transition spike: compare system intent vs dedicated camera flow on quality and latency.
- Offline tolerance spike for key writing/timer experiences.

### Speculative
- Advanced OCR semantic tagging ideas (no production commitment this cycle).

### Definition of Done
- `PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md` exit criteria all passed.
- Cycle04 kickoff and close artifacts are schema-valid.
- Critical path blocker count is zero for auth/storage/universe.
- MCP/skill policy checks pass with no unauthorized expansion.

## 5) Cycle 05 (Phase 1->2 Bridge: Retrieval Quality + Reflection UX)
### Committed
- RRF/hybrid retrieval quality uplift with measurable before/after comparisons.
- Re-query consistency improvements for Korean query variations.
- Universe entry resilience and user guidance consistency (401/403 friendly path preserved).

### Exploratory
- Early intervention signal framework (detect tension between current action and prior principles).
- Knowledge-graph candidate schema and visualization technical feasibility.

### Speculative
- Persona-tailored reflection prompts and proactive guidance heuristics.

### Definition of Done
- Retrieval quality metrics and scenario benchmarks are published.
- Re-query regression suite remains green.
- Cycle05 final handoff explicitly records production-ready and deferred parts.

## 6) Cycle 06 (Operationalization: Repeatable CT/Worker System)
### Committed
- Formalize reusable cycle-closing runbook for contextless CT handoff.
- Enforce cycle-close artifact set of five (task/dispatch/results/latest handoff/CT baseline).
- Lock governance checks for doc+JSON consistency with no critical mismatch.

### Exploratory
- Evaluate lightweight automation for coverage matrix generation and drift alerts.
- Evaluate phase2 intervention data contracts (without releasing behavior changes).

### Speculative
- Cross-user/social architecture sketches under privacy-governed assumptions.

### Definition of Done
- New CT can restart execution from docs + canonical JSON without oral context.
- Cycle06 close package is complete and reproducible.
- Cycle07+ backlog is clearly separated and prioritized.

## 7) Cycle Execution Skeleton (Every Cycle)
1. Kickoff: lock goals, scope, acceptance in task/dispatch.
2. Worker execution: backend/frontend/android scoped tasks + evidence.
3. CT aggregation: merge evidence and publish canonical handoff decision.
4. Baseline sync: publish updated CT baseline and human brief.
5. Retrospective: capture blocker classes, carryovers, and next-cycle priorities.

## 7.1) Session Boundary Rule (Non-Cycle)
- If a session ends due to context/token constraints, CT must use:
  - `docs/SESSION_BOOTSTRAP_PROTOCOL.md`
- Session-close output is mandatory for continuity, but it does not replace cycle-close deliverables.

## 8) Cycle Close Deliverables (Mandatory)
1. `orchestration/task.json`
2. `orchestration/dispatch/*.worker-prompts.json`
3. `orchestration/results/*.result.json`
4. `orchestration/handoff/latest.handoff.json`
5. `CT_BASELINE_<date>.md`

## 9) Cycle 7+ Idea Backlog (Not Committed)
- Selective narrative sharing model with privacy controls.
- Collaborative feedback model under explicit consent and de-identification.
- Long-horizon insight extraction from multi-year narrative logs.

These are idea-stage items and must not be treated as committed work before Cycle 06 close review.
