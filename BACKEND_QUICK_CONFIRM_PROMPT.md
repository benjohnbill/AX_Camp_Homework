---
doc_type: worker_prompt
owner: control_tower
authority_level: L2
last_updated: 2026-02-24
sync_with:
  - orchestration/task.json
  - orchestration/results/20260224T155926Z.T-narrative_loop-20260224-backend-cycle02.result.json
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
You are backend_cli worker for Narrative_Loop cycle02.
Date: 2026-02-24

Objective:
Close backend blockers for final CT acceptance by attaching deploy/secrets-sync/restart evidence.

Current baseline:
- Fixed HTTPS auth probe report already exists:
  data/staging_auth_probe_latest.json
- Gateway auth routes are responsive.
- Blocking point: /debug/token with admin key is still unauthorized_admin in current evidence.

Required work:
1) Execute staging secrets sync for:
   - UNIVERSE_JWT_SECRET
   - UNIVERSE_SESSION_SECRET
   - DEBUG_TOKEN_ADMIN_KEY
2) Restart staging services and capture sanitized logs.
3) Re-run endpoint evidence:
   - POST /debug/token (with admin key): expect 200 code=issued
   - GET /healthz: expect 200
   - GET /gateway/session (no auth): expect 401 missing_token
   - GET /gateway/universe_3d (no auth): expect 401 missing_token
4) Re-run bearer->cookie probe:
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

