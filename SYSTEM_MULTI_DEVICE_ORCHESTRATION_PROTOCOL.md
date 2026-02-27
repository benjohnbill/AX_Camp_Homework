# SYSTEM: Multi-Device Orchestration Protocol (MDOP)

> **Status**: CORE SYSTEM PROTOCOL (v1.1)
> **Mandate**: Ensure seamless transition and environment parity across multiple devices using dynamic path resolution.

## 1. Zero-Trust & Dynamic Path Policy
The Agent must never hardcode absolute machine-specific paths (e.g., `C:\Users\John\...`). All shared resources must be resolved dynamically.

### A. Dynamic Venv Hub (`\.venvs_hub`)
- **Rule**: Standardize virtual environments in a root-relative path named `\.venvs_hub`.
- **Implementation**: Environments should be stored as `\.venvs_hub\[project-name]`.
- **Benefit**: This allows the project to be on any drive (D:, E:, etc.) while keeping the venv separate from the Git source tree but consistently reachable.

## 2. The "Pre-flight" Routine (Session Start & Pull)
Every time a session begins or `git pull` is executed, the Agent MUST:

1. **Path Resolution**: Verify the existence of `\.venvs_hub\[project-name]`.
2. **Dependency Parity**: Compare `requirements.txt` hash. If drift is detected, run `pip install`.
3. **Secret Parity**: Scan `.env` against the mandatory list in `MULTI_DEVICE_MIGRATION_GUIDE.md`.
4. **Health Check**: Run `.	ools\bootstrap_env.ps1` to confirm all layers (OS, Venv, Secrets, DB) are synchronized.

## 3. Rules for Guide Creation/Updates
When the Agent creates or modifies a `MIGRATION_GUIDE.md`:
1. **Use Relative Logic**: Always provide commands that work regardless of the drive letter.
2. **Categorize Secrets**: Clearly mark [Core/Global] vs [Local/Optional] secrets.
3. **Automate Questions**: Include a section for "Questions for the User" if setup cannot be fully automated.

---
*Last Updated: 2026-02-27. This protocol is the foundational law for all cross-device Agent behavior.*
