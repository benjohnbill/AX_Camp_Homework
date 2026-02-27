# Codex Reasoning Policy (Narrative_Loop)

policy_version: 2026.02.27-r1
router_compat: 1.x
Last updated: 2026-02-27
Scope: Codex CLI reasoning mode recommendation in this project

## 1) Default

- Default mode: `medium`
- Principle: start low, escalate only when unresolved risk remains.

## 2) Decision Matrix (0-3 each)

Score each dimension and sum:

1. Scope coupling
- 0: single-file local
- 1: multi-file same module
- 2: cross-module contract touch
- 3: cross-domain/service boundary

2. Concurrency/state risk
- 0: none
- 1: simple async/retry
- 2: coordinated workers/state transitions
- 3: distributed consistency/race risk

3. Security/compliance impact
- 0: no security surface
- 1: basic validation
- 2: auth/session/token handling
- 3: key/PII/permission boundary change

4. Blast radius
- 0: local-only dev
- 1: non-critical internal
- 2: user-facing behavior
- 3: gate/release/production critical

5. Uncertainty
- 0: known pattern
- 1: minor unknown
- 2: competing interpretations in docs/code
- 3: root cause unclear across artifacts

6. Verification burden
- 0: unit/single check
- 1: integration check
- 2: gate/evidence reconciliation
- 3: strict gate or audit-critical validation

### Mode Mapping

- `medium`: total 0-5
- `high`: total 6-11
- `xhigh`: total >=12

## 3) Hard Override to `xhigh`

Use `xhigh` immediately when any one is true:

- Auth/token/permission scope changes
- Canonical JSON handoff/result conflict or strict gate failure analysis
- Deployment, schema, or destructive-path approval workflow
- Cross-domain ownership ambiguity that requires CT escalation

## 4) Downgrade Rule

- If blocker is localized and risk drops, downgrade:
  - `xhigh -> high` after root-cause isolation
  - `high -> medium` after boundary/security/gate risks clear

## 5) Mismatch Interaction Rule

Per user turn:

1. Determine `recommended_mode`.
2. If `current_mode != recommended_mode`, ask once:
   - "`<recommended_mode>`로 전환할까요? (사유: <short_reason>)"
3. If equal, proceed immediately.

## 6) Narrative_Loop Specific Triggers

Recommend at least `high` for:

- Pre-Cycle gate re-aggregation
- `integration_status.md` blocked-lane reconciliation
- multi-worker contract alignment (`task/result/handoff`)

Recommend `xhigh` for:

- `DEBUG_TOKEN_GOVERNANCE.md` scope change
- handoff JSON precedence disputes
- strict-mode gate failure for merge/release decisions

## 7) References

- `./agent.md`
- `./Harness_Policy.md`
- `./DOMAIN_MAP.md`
- `./DEBUG_TOKEN_GOVERNANCE.md`
- `./docs/CT_BASELINE_2026-02-25.md`
