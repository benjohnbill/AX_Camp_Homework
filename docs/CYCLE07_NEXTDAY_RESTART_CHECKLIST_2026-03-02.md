---
doc_type: restart_checklist
owner: control_tower
authority_level: operational
last_updated: 2026-03-02
sync_with:
  - orchestration/handoff/latest.handoff.json
  - integration_status.md
  - docs/CT_BASELINE_2026-03-02.md
change_triggers:
  - infra_blocked_checkpoint
  - next_day_restart
sunset_condition: Replace after cycle07 infra blocker is resolved and close is revalidated.
---
# Cycle07 Next-Day Restart Checklist

## 0) Recommended Location
- Keep this file at: `docs/CYCLE07_NEXTDAY_RESTART_CHECKLIST_2026-03-02.md`
- Rationale: `docs/` is the canonical bootstrap area for new CT sessions.

## 1) Read Order (Must)
1. `orchestration/handoff/latest.handoff.json`
2. `integration_status.md`
3. `docs/CT_BASELINE_2026-03-02.md`
4. `docs/CYCLE06_POSTCHECK_PRODUCT_CHECKLIST.md`
5. `docs/MASTER_PLAN_CYCLE04_06.md`

## 2) Current State Snapshot
- State: `Cycle07 infra-hotfix BLOCKED`
- Primary blocker class: `network_integrity / staging_infrastructure`
- Working hypothesis: Android traffic is failing at TLS/HTTP2 edge path before stable app-layer response.

## 3) Evidence Path Freeze (Fixed)
- Backend infra reachability:
  - `orchestration/results/20260302T105614Z.T-narrative_loop-20260302-backend-cycle07-infra-reachability.result.json`
- Backend infra hotfix:
  - `orchestration/results/20260302T110346Z.T-narrative_loop-20260302-backend-cycle07-infra-hotfix.result.json`
- Frontend pathway support:
  - `orchestration/results/20260302T200500Z.T-narrative_loop-20260302-frontend-pathway-support.result.json`
- Frontend infra support:
  - `orchestration/results/20260302T201500Z.T-narrative_loop-20260302-frontend-cycle07-infra-support.result.json`
- Android blocker (runtime):
  - `orchestration/results/20260302T233000Z.T-narrative_loop-20260302-android-cycle07.result.json`
- Android blocker (infra hotfix):
  - `orchestration/results/20260302T235900Z.T-narrative_loop-20260302-android-cycle07-infra-hotfix.result.json`

## 4) Day-2 First Actions
1. Retrieve Render/edge logs for `2026-03-02T22:30:00Z~23:59:59Z`.
2. Focus routes only:
   - `/v1/ocr/ingest`
   - `/gateway/session`
   - `/gateway/universe_3d`
3. Confirm one root cause with evidence:
   - WAF deny/challenge/rate limit, or
   - TLS handshake drop, or
   - HTTP/2 stream/protocol reset.
4. Re-run Android physical+emulator same-window after infra adjustment.

## 5) Exit Criteria to Unblock
1. Backend publishes a log-backed infra verdict with one confirmed root cause.
2. Android confirms TLS/HTTP2 failure no longer reproduces.
3. Android `Camera OCR + Universe` both PASS on physical + emulator.
4. CT republishes close only after 1~3 are all satisfied.

## 6) Quick Start Commands (New Device)
```bash
git pull
python tools/validate_contracts.py --file orchestration/handoff/latest.handoff.json
```

