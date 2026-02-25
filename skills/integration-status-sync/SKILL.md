# SKILL: Integration Status Sync (Universal Guide)

- **Owner**: Any active agent in the loop
- **Trigger**: Execution of `.\tools\ralph_heartbeat.ps1` or explicit task
- **Goal**: Synchronize all agent results into `integration_status.md` and dispatch the next task.

## 1) Load Required Context
Read these files before updating:
1. `Agent.md` (Authority)
2. `Harness_Policy.md` (Governance)
3. `integration_status.md` (Current State)
4. `orchestration/results/*.result.json` (New Evidence)
5. `android/NarrativeLoopMobile/ANDROID_REPORT.md` (Mobile Evidence)

## 2) Update Integration Status Board
1. **Analyze Evidence**: Scan `result.json` and `ANDROID_REPORT.md` for completed items.
2. **Move Items**: If proof is verified (commands, logs, commits), move items from `In Progress` to `Completed (Fact-Checked)`.
3. **Log Risks**: If a task failed or hit a "Safety Stop", move it to `Open Gaps / Risks`.
4. **Maintain Timeline**: Record the change in the `Changelog` with timestamp and evidence ID.

## 3) The Dispatcher (Next-Loop Strategy)
Based on the updated board, decide who should work next:
- **Backend (Codex)**: If auth/retrieval gaps remain.
- **Frontend (Antigravity)**: If UI/UX improvements or API connections are needed.
- **Android**: If mobile integration or E2E validation is required.

## 4) Update INBOX.md (The Signal)
Update (or create) the target agent's inbox with the following format:
- **[ACTIVE]** (Status flag for the agent to notice)
- **Context**: Why this task is assigned now.
- **Next Action**: Specific implementation or validation goal.
- **Contract**: Reference to `task.json` if applicable.

## 5) Output Contract (JSON-First)
- All status updates must align with the `result.json` schema.
- If evidence is missing, mark as `Blocked: missing evidence`.

---
*Note: This is a universal protocol for all agents participating in the Ralph Loop.*
