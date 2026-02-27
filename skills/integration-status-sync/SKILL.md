---
name: integration-status-sync
description: Synchronize validated orchestration evidence into integration_status.md and dispatch the next worker through channel pointers and CT inbox files. Use when new result/handoff artifacts arrive or before running CT heartbeat.
---

# SKILL: Integration Status Sync

## Goal
Keep `integration_status.md` aligned with canonical JSON evidence and publish the next execution signal.

## Source Priority (Must Follow)
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. Latest `orchestration/results/*.result.json`
4. `integration_status.md`
5. `android/NarrativeLoopMobile/ANDROID_REPORT.md` (supporting evidence only)

## Required Inputs
1. `Agent.md` (authority)
2. `Harness_Policy.md` (governance)
3. Source-priority files listed above

## Procedure
1. Collect new result/handoff artifacts since the last board update timestamp.
2. Validate claims from schema-valid JSON first; treat markdown as supplemental context only.
3. Update `integration_status.md` sections:
   - `Completed (Fact-Checked)`: passed items with evidence paths.
   - `In Progress`: active items with next action.
   - `Open Gaps / Risks`: blocked/failed items with root cause.
   - `Latest Validation Snapshot`: newest verdict with artifact paths.
4. Keep wording factual. If proof is missing, write `Blocked: missing evidence`.
5. Record trace/task IDs and UTC timestamp in the status timeline/changelog.

## Dispatch Rule
1. Choose target worker from open blocker class:
   - auth/storage/retrieval -> `backend_cli`
   - runtime UI/UX -> `frontend_ide`
   - mobile/device/e2e -> `android_ide`
2. Update one channel pointer JSON:
   - `orchestration/backend.current.json`
   - `orchestration/antigravity.current.json`
   - `orchestration/android.current.json`
3. Update matching CT inbox markdown signal:
   - frontend/backend scope: `CT_INBOX_ANTIGRAVITY.md` and `CT_INBOX_GEMINI_UI.md` (legacy alias)
   - android scope: `android/NarrativeLoopMobile/CT_INBOX_ANDROID.md`

## Output Contract
1. Fast lane: `orchestration/templates/chat_l1_worker_update.md`
2. Slow lane: schema-valid `orchestration/results/*.result.json` and `orchestration/handoff/*.handoff.json`
3. Never override canonical JSON verdicts with markdown narrative.
