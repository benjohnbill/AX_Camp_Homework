# SKILL_PROMOTION_POLICY.md

## Purpose
- This document defines when a recurring workflow should be promoted into a reusable `SKILL.md`.
- Actual skill creation is on hold by default.
- Promotion is triggered only when objective conditions are met.
- Controlled pilot skills are allowed without changing the global default state.
- skills.sh style external skills are allowed only through controlled intake and review.
- System-level baseline is defined in `02_Core_Resources/01_Agent_Orchastration_System/SYSTEM_SKILL_GOVERNANCE_POLICY.md`.

## Intake Safety Gate (Pre-Install)
Before installing any external skill, all below must pass:
1. Source trust:
- official vendor or verified maintainer preferred.
2. Adoption signal:
- meaningful usage/download signal or equivalent project references.
3. Content review:
- inspect `SKILL.md` manually for destructive commands, secret leakage risk, and policy conflicts.
4. Rollback readiness:
- installation/removal path must be documented before pilot starts.
5. Registry entry:
- skill must exist in `skills/approved_skills_registry.json`.
6. Checksum rule:
- for `pilot` or `core` status, `checksum_sha256` must be present and valid.

If any gate fails, classify as `blocked` and do not install.

## Discovery and Intake Flow (Control Tower)
1. Discovery is allowed:
- Control Tower may request/search new skills (for example with `find-skills`).
- Discovery output is recorded as `candidate` only.

2. Discovery is not installation:
- No skill is installed directly from discovery output.
- Installation starts only after registry entry + intake safety gate pass.

3. New project default:
- Start from `HOLD`.
- Move to `candidate` first, then evaluate for `pilot`.

## Curated Pilot Catalog (Initial)
The following are curated candidates, not auto-enabled defaults:
1. `find-skills`
2. `vercel-react-best-practices`
3. `web-design-guidelines`
4. `remotion-best-practices`
5. `frontend-design`

Registry path:
- `skills/approved_skills_registry.json` is the single intake registry for this project.

## Decision States
1. `HOLD`
- No immediate skill creation.
- Keep operating with project `.md` guidance.

2. `CANDIDATE`
- A workflow shows repeated usage or repeated confusion.
- Add it to a watchlist in regular status updates.

3. `RECOMMEND`
- Codex actively recommends creating a skill in chat.
- Recommendation includes scope, expected ROI, and draft skill name.

4. `PROMOTE`
- Skill creation is approved and implementation starts.

## Promotion Criteria (Objective)

### A. Repetition Trigger (required)
- The same workflow appears 3 or more times within recent 14 days,
  OR
- The same instruction pack is sent to 2 or more tools repeatedly (for example Antigravity + Android).

### B. Determinism/Risk Trigger (required, one of below)
- The workflow has 5 or more ordered steps where omission can break behavior/security.
- The workflow contains security-sensitive handling (token, secret, auth contract) and has non-trivial failure impact.
- The workflow requires fixed validation commands/checklists to avoid regressions.

### C. Stability Trigger (required)
- Core contract is stable for 2 consecutive update cycles
  (API fields, auth contract, data model, or acceptance checks are no longer frequently changing).

## Recommendation Rule
- Codex recommends skill promotion when **A + B + C are all satisfied**.
- If A and B are satisfied but C is not, Codex marks it as `CANDIDATE` and waits for stability.

## Candidate -> Pilot Checklist (One Page)

All items are required:
1. Registry readiness:
- entry exists in `skills/approved_skills_registry.json`
- source is in allowlist

2. Content safety:
- `SKILL.md` manually reviewed for destructive commands, secret leakage, and policy conflicts

3. Checksum readiness:
- `checksum_sha256` is filled and validated

4. Rollback readiness:
- uninstall and fallback path documented

5. Gate readiness:
- `.\tools\project_python.ps1 tools/check_skill_registry.py --mode strict` passes in CI context

## Recommendation Output Format (when triggered)
When promotion is recommended, Codex provides:
1. `Why now` (matched criteria evidence)
2. `Proposed skill name`
3. `Scope In / Scope Out`
4. `Initial skill structure`
- `SKILL.md`
- optional `references/`
- optional `scripts/`
5. `Expected maintenance cost and ROI`

## Pilot Measurement (for CANDIDATE/RECOMMEND)

For each pilot cycle, record:
1. `Cycle time` (minutes): time from raw report to publish-ready status update.
2. `Correction rounds` (count): number of revision loops after first draft.
3. `Evidence miss count` (count): completed claims removed due to missing proof.
4. `Policy drift count` (count): policy-like statements accidentally inserted in status docs.

Promotion confidence increases only when these metrics trend stable/improving for 2 consecutive cycles.

## Current Default
- Status: `HOLD` (no immediate skill creation).
- Codex monitors incoming Android/Antigravity reports and applies this policy continuously.
- Active internal pilot skills:
  1. `skills/integration-status-sync/SKILL.md`
  2. `skills/cycle-close-packager/SKILL.md`
- External pilot-prep focus:
  - `frontend-design` remains `candidate` until checksum, rollback path, and strict gate are complete.
