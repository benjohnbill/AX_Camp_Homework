# Supabase Operations Runbook

## Scope
This runbook standardizes deploy checks for Supabase-backed production runs.

## Preconditions
- Run commands from project root: `Narrative_Loop/`
- `DATASTORE=postgres`
- `DATABASE_URL` is set
- App secrets include `OPENAI_API_KEY` (if AI response is required)
- If env vars are missing in shell, checks will auto-read `.streamlit/secrets.toml`

## Quick Gate (Recommended)
Run all goal-A checks in one command:

```bash
python tools/run_agent_a_gate.py
```

Equivalent steps include:
- preflight auth (`tools/preflight_postgres_auth.py`)
- bootstrap schema check (`tools/check_supabase_phase1.py`)
- smoke check (`tools/check_postdeploy_smoke.py --strict-postgres`)
- integrity check (`tools/check_data_integrity.py --expect-postgres --max-dup-chat 0`)

## Pre-Deploy Checklist
1. `python tools/preflight_postgres_auth.py`
2. `python tools/check_supabase_phase1.py`
3. `python -m pytest -q tests/test_phase4_phase5_stability.py`
4. `python tools/check_data_integrity.py --expect-postgres --max-dup-chat 0 --report-json data/integrity_pre.json`
5. Verify branch is clean and pushed.

## Deploy Steps
1. Deploy latest `main` to Streamlit Cloud.
2. Confirm secrets in Streamlit Cloud:
   - `DATASTORE=postgres`
   - `DATABASE_URL=...`
   - `OPENAI_API_KEY=...`
3. Restart app once after secrets change.

## Post-Deploy Smoke
1. Run local smoke script:
   - `python tools/check_postdeploy_smoke.py --strict-postgres`
2. Run integrity gate:
   - `python tools/check_data_integrity.py --expect-postgres --max-dup-chat 0 --report-json data/integrity_post.json`
3. Browser smoke (manual):
   - Create one stream log.
   - Send one follow-up message (AI response path).
   - Create/move one kanban card.
4. Confirm Supabase row growth:
   - `logs` increased
   - `chat_history` increased
   - `connections` unchanged or increased depending on action

## Auth Failure Triage
If you see `password authentication failed` or `circuit breaker open`:
1. Run `python tools/preflight_postgres_auth.py`.
2. Confirm `DATASTORE` is exactly `postgres` (not a URL).
3. Confirm `DATABASE_URL` is Session Pooler URI (`*.pooler.supabase.com:6543`).
4. For Session Pooler URI, username must be `postgres.<project_ref>`.
5. Reset DB password in Supabase, then update all `DATABASE_URL` locations (local env + Streamlit secrets).

## Failure Handling
1. If app fails to load:
   - Check `DATABASE_URL`, then `DATASTORE`.
   - Run `python tools/check_supabase_phase1.py`.
2. If writes are missing:
   - Run `python tools/check_postdeploy_smoke.py`.
   - Run `python tools/check_data_integrity.py --expect-postgres --max-dup-chat 0`.
   - Verify DB role permissions on `logs/chat_history/connections`.
3. If AI responses fail:
   - Confirm `OPENAI_API_KEY`.
   - Confirm app still saves logs even when AI fails.

## Rollback Procedure
1. Set `DATASTORE=sqlite` in Streamlit secrets.
2. Restart app.
3. Verify app boots and can create logs locally.
4. Record rollback timestamp and reason.

## Evidence to Store After Every Deploy (Required 4)
- Git commit hash
- Result of `check_supabase_phase1.py`
- Result of `check_postdeploy_smoke.py --strict-postgres`
- Manual smoke checklist status

Recommended additional evidence:
- Result JSON from `check_data_integrity.py`
