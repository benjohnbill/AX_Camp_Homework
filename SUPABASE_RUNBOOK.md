# Supabase Operations Runbook

## Scope
This runbook standardizes deploy checks for Supabase-backed production runs.

## Preconditions
- `DATASTORE=postgres`
- `DATABASE_URL` is set
- App secrets include `OPENAI_API_KEY` (if AI response is required)

## Pre-Deploy Checklist
1. `python tools/check_supabase_phase1.py`
2. `python -m pytest -q tests/test_phase4_phase5_stability.py`
3. Verify branch is clean and pushed.

## Deploy Steps
1. Deploy latest `main` to Streamlit Cloud.
2. Confirm secrets in Streamlit Cloud:
   - `DATASTORE=postgres`
   - `DATABASE_URL=...`
   - `OPENAI_API_KEY=...`
3. Restart app once after secrets change.

## Post-Deploy Smoke
1. Run local smoke script:
   - `python tools/check_postdeploy_smoke.py`
2. Browser smoke (manual):
   - Create one stream log.
   - Send one follow-up message (AI response path).
   - Create/move one kanban card.
3. Confirm Supabase row growth:
   - `logs` increased
   - `chat_history` increased
   - `connections` unchanged or increased depending on action

## Failure Handling
1. If app fails to load:
   - Check `DATABASE_URL`, then `DATASTORE`.
   - Run `python tools/check_supabase_phase1.py`.
2. If writes are missing:
   - Run `python tools/check_postdeploy_smoke.py`.
   - Verify DB role permissions on `logs/chat_history/connections`.
3. If AI responses fail:
   - Confirm `OPENAI_API_KEY`.
   - Confirm app still saves logs even when AI fails.

## Rollback Procedure
1. Set `DATASTORE=sqlite` in Streamlit secrets.
2. Restart app.
3. Verify app boots and can create logs locally.
4. Record rollback timestamp and reason.

## Evidence to Store After Every Deploy
- Git commit hash
- Result of `check_supabase_phase1.py`
- Result of `check_postdeploy_smoke.py`
- Manual smoke checklist status
