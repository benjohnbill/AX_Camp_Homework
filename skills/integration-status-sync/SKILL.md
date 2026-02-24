---
name: integration-status-sync
description: Update `integration_status.md` from backend/mobile evidence while preserving `Agent.md` and `Harness_Policy.md` boundaries. Use when integration progress, validation evidence, risks, and next actions must be synchronized into one status report.
---

# Integration Status Sync

## 1) Load Required Context

Read these files before drafting updates:
1. `Agent.md`
2. `Harness_Policy.md`
3. `integration_status.md`
4. Relevant latest report(s) from Android and backend agents

## 2) Enforce Status-Only Boundary

1. Keep `integration_status.md` as evidence/status output only.
2. Do not redefine architecture, policy, or authority model in status sections.
3. If policy conflict is found, reference `Harness_Policy.md` and move details to policy docs.

## 3) Normalize Incoming Evidence

1. Extract concrete facts only:
- command and result
- commit id / branch
- endpoint URL
- pass/fail evidence
2. Reject unverified claims:
- mark as pending when no command output, no commit, or no traceable proof exists.

## 4) Update Workflow

Apply updates in this order:
1. `Overall Progress`
2. `Completed (Fact-Checked)`
3. `In Progress`
4. `Open Gaps / Risks`
5. `Latest Validation Snapshot`
6. `Changelog`

## 5) Output Contract

Always report with four blocks:
1. `What changed`
2. `Validation`
3. `Risks`
4. `Next 3 actions`

If evidence is insufficient, return:
- `Blocked: missing evidence`
- `Required evidence (top 3)`

## 6) Quality Checks

Before finalizing:
1. Ensure no section introduces new policy language.
2. Ensure progress percentage changed only with new evidence.
3. Ensure every new completed item has at least one explicit trace.
