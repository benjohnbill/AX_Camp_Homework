---
doc_type: design_note
owner: control_tower
authority_level: operational
last_updated: 2026-03-01
sync_with:
  - skills/feature-lock-audit/SKILL.md
  - docs/PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md
  - integration_status.md
change_triggers:
  - precycle4_blocker_status_changed
  - gate_reaggregation_frequency_increase
sunset_condition: Replace when feature-lock-audit skill reaches pilot with executable script automation.
---
# Feature-Lock-Audit Skill Design (2026-03-01)

## Why This Skill
- Pre-cycle gate re-aggregation is repetitive and error-prone.
- Current flow depends on repeated manual JSON synthesis.
- Existing internal skills cover status sync and cycle close, but not dedicated pre-cycle gate orchestration.

## Target Problem
Automate and standardize CT-side pre-cycle gate refresh without changing scope boundaries.

## Design Boundaries
- Must keep canonical precedence:
  1. `orchestration/handoff/latest.handoff.json`
  2. `orchestration/task.json`
  3. latest `orchestration/results/*.result.json`
  4. `integration_status.md`
- Must not start cycle feature expansion.
- Must not override canonical JSON with markdown-only claims.

## Proposed Skill Shape
1. Skill entry:
   - `skills/feature-lock-audit/SKILL.md`
2. Initial mode:
   - procedure-first (document-driven), no mutation script yet
3. Planned next mode:
   - add deterministic helper script in `tools/` for result/handoff scaffold

## Integration with Existing Skills
- `integration-status-sync`:
  - consumes final gate verdict and syncs board/inbox pointers
- `cycle-close-packager`:
  - consumes stabilized gate artifacts for cycle-close packaging
- `feature-lock-audit`:
  - sits between lane rerun outputs and status/cycle-close layers

## Acceptance Criteria for Pilot Upgrade
1. Two consecutive reruns produce schema-valid artifacts without manual field patching.
2. Gate verdict consistency matches canonical evidence every run.
3. Manual CT re-aggregation time is reduced measurably.

## Immediate Consistency Reinforcement Applied
1. `skills/environment-sync/SKILL.md` normalized to standard skill frontmatter.
2. Runtime naming/path rules aligned with project SSOT (`Narrative_Loop.venv` + `narrative_loop` alias).
3. New candidate skill skeleton added at `skills/feature-lock-audit/SKILL.md`.

