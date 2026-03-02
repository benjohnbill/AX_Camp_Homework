---
name: environment-sync
description: Detect and report local runtime environment drift for Narrative_Loop before execution. Use when a session starts, after pull/clone, or when Python/venv/dependency mismatch symptoms appear.
---

# SKILL: Environment Sync

## Goal
Keep local runtime alignment with project runtime policy before feature or gate execution.

## Source Priority
1. `agent.md`
2. `LOCAL_ENV_SETUP.md`
3. `requirements.txt`
4. `MULTI_DEVICE_MIGRATION_GUIDE.md`

## Runtime Naming Rules (Must Follow)
- Canonical venv name: `Narrative_Loop.venv`
- Compatibility alias: `narrative_loop`
- Resolved root: `$env:LIFE_VENV_ROOT` or default `%USERPROFILE%\.venvs_hub` (fallback `C:\venvs_hub` for non-ASCII profile environments)

## Procedure
1. Verify project python resolver works:
   - `.\tools\project_python.ps1 --version`
2. Verify canonical/alias venv paths exist under resolved venv root.
3. Compare `requirements.txt` against installed packages in the active project python.
4. Check `.env` required keys against `MULTI_DEVICE_MIGRATION_GUIDE.md` mandatory list.
5. Report drift with three classes only:
   - `missing_venv`
   - `dependency_drift`
   - `missing_env_keys`

## Ask-Then-Execute Rule
When drift is detected, report first and ask for approval before mutation.

Allowed remediation examples after approval:
- `.\tools\bootstrap_env.ps1 -Recreate -InstallPreCommit`
- `.\tools\project_python.ps1 -m pip install -r requirements.txt`

## Output Contract
- Emit concise drift report with:
  - detected class
  - evidence command/log path
  - proposed remediation command
- Do not silently install or mutate environment.

