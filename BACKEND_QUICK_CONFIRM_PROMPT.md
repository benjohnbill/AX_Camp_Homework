---
doc_type: worker_prompt
owner: control_tower
authority_level: L2
last_updated: 2026-02-25
sync_with:
  - orchestration/task.json
  - data/staging_auth_probe_latest.json
change_triggers:
  - cycle change
  - deploy contract change
sunset_condition: Remove when backend worker channel is fully automated.
review_by: 2026-02-26
---

# Backend Quick Confirm Prompt

Copy and paste this prompt to the backend execution environment that has deploy credentials.

```text
You are backend_cli worker for Narrative_Loop cycle03.
Date: 2026-02-25

Objective:
Validate Advanced Retrieval Metrics and fixed HTTPS auth contract stability.

Current baseline:
- Fixed HTTPS auth probe report exists:
  data/staging_auth_probe_latest.json
- Gateway auth routes are responsive.
- Strict gate is active.

Required work:
1) Verify staging auth contract remains stable:
   tools/probe_staging_auth_contract.py --json-report data/staging_auth_probe_latest.json
2) Collect Advanced Retrieval Metrics (RRF + split context):
   tools/eval_korean_retrieval.py
3) Capture backend strict-gate health:
   tools/run_agent_a_gate.py --policy-mode strict
4) Re-verify bearer->cookie probe:
   - first bearer: 307 + set-cookie
   - cookie follow-up: 307 + auth-source cookie
   - wrong audience: 403 forbidden_audience

Output format (4 blocks):
1) What changed
2) Validation (command + outcome + evidence path)
3) Risks (severity included)
4) Next 3 actions

Security rules:
- Never print plain secrets/tokens.
- Use masked values and fingerprints only.
```

