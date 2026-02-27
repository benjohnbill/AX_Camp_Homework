# AGENTS.md - Thin Router (Narrative_Loop)

router_version: 1.0
required_codex_policy_version: 2026.02.27-r1

## Always-On Rule (Per Turn)

1. Load agent-specific policy (`Codex.md` for Codex CLI).
2. Compare `current_mode` vs `recommended_mode`.
3. If mismatch, ask once: "`<recommended_mode>`로 전환할까요? (사유: <short_reason>)".
4. If match, execute directly.

## Routing

- Codex CLI -> `./Codex.md`
- Gemini CLI -> `./GEMINI.md` (if exists)
- Missing policy file -> fallback `medium`, then report missing file.

## Version Lock

- Read `policy_version` from `./Codex.md`.
- If `policy_version != required_codex_policy_version`, report version drift and suggest sync.

## Constraints

- System constitution > project docs on conflict.
- Canonical acceptance evidence: JSON handoff/result/task artifacts.
- Keep this router short; detailed thresholds stay in `Codex.md`.
