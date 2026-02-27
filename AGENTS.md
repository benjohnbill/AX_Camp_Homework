# AGENTS.md - Thin Router (Narrative_Loop)

## Purpose

This file is the always-on router for agent behavior in this project.
Keep this file short. Put detailed thresholds in tool-specific policy files.

## Always-On Mode Check (Per Turn)

Before final output on every user turn:
1. Compute a recommended reasoning mode from the active policy file.
2. Compare `current_mode` vs `recommended_mode`.
3. If different, ask once: "`<recommended_mode>`로 전환할까요?"
4. If same, execute directly without asking.

## Agent Routing

- If runtime agent is Codex CLI: read `./Codex.md` for detailed mode policy.
- If runtime agent is Gemini CLI and `./GEMINI.md` exists: read `./GEMINI.md`.
- If agent-specific policy file is missing: continue with safest default (`medium`) and report missing file.

## Minimal Read Strategy

- Default read set: `./agent.md`, `./Codex.md`
- Add only when needed:
  - Security/token/auth: `./DEBUG_TOKEN_GOVERNANCE.md`
  - Authority/doc conflict: `./Harness_Policy.md`
  - Domain boundary: `./DOMAIN_MAP.md`

## Core Constraints

- System constitution docs override project-local docs on conflict.
- Canonical acceptance evidence is JSON handoff/result artifacts.
- Approval-required actions follow existing governance docs.

## Budget Guard

- Keep per-turn policy loading minimal.
- Do not re-load long docs unless task risk changes or conflict appears.
