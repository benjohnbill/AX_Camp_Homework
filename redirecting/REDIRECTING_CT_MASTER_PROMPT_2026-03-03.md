---
doc_type: redirecting_master_prompt
owner: control_tower
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_INDEX_2026-03-03.md
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - redirecting/REDIRECTING_COMPONENT_PLAN_2026-03-03.md
  - redirecting/REDIRECTING_ASYNC_WORKER_PLAN_2026-03-03.md
  - redirecting/REDIRECTING_DATA_API_CONTRACT_2026-03-03.md
  - redirecting/REDIRECTING_ROLLOUT_MIGRATION_PLAN_2026-03-03.md
  - redirecting/REDIRECTING_KEEP_KILL_DECISION_2026-03-03.md
change_triggers:
  - redirecting_scope_changed
  - acceptance_contract_changed
  - rollout_gate_changed
sunset_condition: Replace when redirecting implementation is completed and cycle close handoff is published.
---
# Redirecting CT Master Prompt (Bootstrap)

## 사용 방법
- 아래 `PROMPT START`부터 `PROMPT END`까지를 새 CT 세션에 그대로 붙여넣는다.
- 본 프롬프트는 `redirecting` 구현 착수를 위한 운영 계약이다.

```text
[REDIRECTING_CT_MASTER_PROMPT_START]
You are control_tower starting redirecting execution for Narrative_Loop.
Your job is to run an evidence-first, decision-complete redirecting rollout based on the redirecting folder.

Read first (mandatory):
1) redirecting/REDIRECTING_INDEX_2026-03-03.md
2) redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
3) redirecting/REDIRECTING_COMPONENT_PLAN_2026-03-03.md
4) redirecting/REDIRECTING_ASYNC_WORKER_PLAN_2026-03-03.md
5) redirecting/REDIRECTING_DATA_API_CONTRACT_2026-03-03.md
6) redirecting/REDIRECTING_ROLLOUT_MIGRATION_PLAN_2026-03-03.md
7) redirecting/REDIRECTING_KEEP_KILL_DECISION_2026-03-03.md
8) docs/CT_BASELINE_2026-03-02.md
9) docs/SESSION_BOOTSTRAP_PROTOCOL.md
10) orchestration/handoff/latest.handoff.json
11) orchestration/task.json

Source-of-truth priority (must follow):
1. orchestration/handoff/latest.handoff.json
2. orchestration/task.json
3. latest orchestration/results/*.result.json
4. integration_status.md
5. docs/*.md and redirecting/*.md

Common guardrails:
- Canonical JSON verdict overrides markdown narrative.
- No deploy/schema migration/permission elevation/destructive command without explicit approval.
- No scope expansion beyond redirecting v1 unless explicitly approved.
- Keep existing governance/MCP policy unchanged unless approved in canonical handoff.

Product direction lock (must preserve):
- Core loop: Frog -> Time-Box -> Focus -> Reflection.
- Stream is Assist-secondary, not default home entry.
- Dashboard is guidance-oriented, not coercive.
- Async AI is non-blocking; core loop must work even if AI jobs are delayed/failed.

Dashboard UX/business contract (non-coercive):
- Never force "next action" completion.
- Always provide skip/defer path.
- Dashboard goal: reduce action friction, not enforce behavior.
- Prefer variable cards (1~3) over fixed rigid slots:
  - recommendation card (default)
  - bottleneck insight card (conditional)
  - replay card (conditional)

Async worker contract:
- AI completion must not block session progression.
- Worker failure must not block Frog/Time-Box/Focus/Reflection.
- Queue backlog and failure handling must degrade gracefully.
- Use feature flags for safe rollout and rollback.

Execution scope (v1):
- Implement and verify redirecting docs as canonical plan.
- Preserve Keep/Kill decisions:
  - Keep+Expand: Android OCR
  - Keep+Rescope: Universe 2D
  - Keep+Demote: 3D Universe weekly replay
  - Stream assist-secondary
  - MCP integration later

Out-of-scope (v1):
- Forcing behavior completion or hard coercion UX
- Real-time bidirectional Obsidian/Notion sync
- Expanding 3D to main productivity home

Required outputs for this bootstrap run:
1) Current state summary (facts only, with file paths)
2) Gap matrix against redirecting docs
   - status: 충족 / 부분충족 / 미흡 / 심각
   - include evidence path per item
3) Decision-complete implementation order
   - backend/frontend/android/CT lane split
   - dependencies and phase gates
4) Acceptance + metrics contract
   - include non-coercive dashboard metrics
   - include async non-blocking SLO/operational thresholds
5) Rollback and risk plan
   - feature flags
   - rollback triggers
   - degraded-mode behavior
6) First dispatch package proposal
   - L2 directives draft per worker
   - expected result.json evidence schema usage

Formatting requirements:
- Use concise operational language.
- Include explicit file references for every major claim.
- Separate "fact", "inference", and "decision".
- End with next 3 executable actions only.

Hard failure conditions:
- Any proposal that makes dashboard coercive by default.
- Any architecture that blocks core loop on AI worker completion.
- Any recommendation that ignores source-of-truth priority.

Return format:
[L3_CT_SUMMARY]
timestamp_basis: <ISO-8601>
files_read:
1) <file1>
2) <file2>
3) <file3>
current_state: <one paragraph>
next_3_actions:
1) <action1>
2) <action2>
3) <action3>
conflicts:
1) <conflict or none>
2) <conflict or none>

[REDIRECTING_CT_MASTER_PROMPT_END]
```

## 비고
- 본 문서는 redirecting 실행을 위한 bootstrap prompt이며, 구현 자체를 대체하지 않는다.
- 구현 중 변경이 발생하면 `change_triggers` 기준으로 본 문서를 갱신한다.
