# Multi-Device Migration & Environment Bootstrap Guide (v1.1)

> **[CRITICAL INSTRUCTION FOR AI AGENT]**
> You are initializing this project in a new environment. Your goal is to reach a "Ready to Develop" state with minimal user friction. Read this entire guide before executing any command.

## 1. Phase 0: System & Policy Check
Before creating any environment, ensure the system allows script execution.

1. **Check PowerShell Execution Policy**:
   - Run: `Get-ExecutionPolicy -Scope CurrentUser`
   - If it is `Restricted`, ask the user: *"I need to run local scripts. May I set the execution policy to RemoteSigned for the current user? (Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)"*
2. **Verify Python**: Ensure Python 3.9+ is available in the PATH.

## 2. Phase 1: Dependency & Git Hook Setup
Automate the boring parts.

1. **Create/Link Virtual Environment**:
   - Note: We use a shared hub at `\.venvs_hub` to keep environments out of the project tree.
   ```powershell
   # Create venv in the dynamic hub path
   if (!(Test-Path "\.venvs_hub\narrative-loop")) {
       python -m venv "\.venvs_hub\narrative-loop"
   }
   # Activate from the hub
   & "\.venvs_hub\narrative-loop\Scripts\Activate.ps1"
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Setup Git Hooks (Pre-commit)**:
   - Run: `pip install pre-commit` (if not in requirements)
   - Run: `pre-commit install`
   - *Reason*: This ensures your future commits follow the project's linting/security standards.

## 3. Phase 2: Secret Management (The "Ask" Phase)
The `.env` file is missing. **Stop and ask the user for these specific values.**

### [Mandatory - Core AI & Auth]
- `GEMINI_API_KEY`: For LLM features.
- `UNIVERSE_JWT_SECRET`: For internal token generation.

### [Conditional - Database Selection]
Ask the user: *"Are we using local SQLite or Supabase/Postgres for this session?"*
- If **Supabase**: Request `SUPABASE_URL` and `SUPABASE_KEY`.
- If **Local SQLite**: No extra keys needed (verify `narrative_loop.db` later).

### [Action for AI]
Create the `.env` file using the provided values. **Do not log or print these secrets.**

## 4. Phase 3: Project Bootstrap & Health Check
Run the internal tools to verify everything is wired correctly.

1. **Run Bootstrap**: `.\tools\bootstrap_env.ps1`
2. **Database Preflight**:
   - For SQLite: `python tools/migrate_db.py`
   - For Postgres: `python tools/preflight_postgres_auth.py`
3. **Core Test**: `pytest tests/test_gateway_fastapi.py`

## 5. Phase 4: Final Handover Report
Once finished, provide a summary to the user:
```markdown
### 🚀 Environment Setup Complete
- [x] Venv created & Requirements installed
- [x] Pre-commit hooks active
- [x] .env configured (Secrets stored safely)
- [x] DB connection verified ([SQLite/Postgres])
- [x] Core tests passed

**Current Project State**: [Read integration_status.md and provide a 1-sentence summary of where we left off.]
```

---
*Last Refined: 2026-02-27. Use this protocol whenever a clean-slate setup is detected.*
