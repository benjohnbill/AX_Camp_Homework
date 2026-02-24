# Integration Status: Android <-> Backend

Last updated: 2026-02-24 (cycle02 dispatch for end-of-day blocker closure)
Maintainer: Codex (integration coordinator)
Projects:
- Backend/Web: Narrative_Loop (Streamlit + Antigravity)
- Mobile: NarrativeLoopMobile (`android/NarrativeLoopMobile` in the same GitHub repository; Android Studio can open a local workspace alias path)
Documentation governance baseline: `Harness_Policy.md`

## 1) Overall Progress
- Current progress: 90%
- Status: MVP integration is working in staging, but auth contract hardening is still in progress.

### MVP Governance Risks (Current)
- Local Python runtime instability can delay local gate feedback, so CI strict results remain the final gate.
- Candidate skills in registry are not pilot-ready until `checksum_sha256` is filled and verified.

### 100% Completion Definition
Integration is considered 100% only when all items below are done:
1. First request auth: Android sends `Authorization: Bearer <JWT>` and backend accepts it.
2. Session transition: backend issues cookie-based session for follow-up requests.
3. Contract verification: 401/403 behavior is validated end-to-end with evidence.
4. Governance verification: token issuance/rotation policy is operating in staging.
5. Stability verification: Android tab switching/background-foreground smoke tests pass with no critical issue.
6. Backend retrieval-context track reaches done state (RRF + split context + mixed sync/async + worker safety + acceptance checks).

## 2) Completed (Fact-Checked)
- Android `Universe3DFragment` loads Universe URL with Bearer header.
- Android has WebView lifecycle handling (`onPause`, `onResume`, `onDestroyView`).
- Android debug token UI exists in `HomeFragment` (Save/Clear/Logout).
- Android logout path clears both TokenStore and WebView cookies.
- Android debug URL is configured via `BuildConfig.UNIVERSE_URL` in debug build.
- Backend `universe_auth.py` exists:
  - JWT signature and claim validation (`iss`, `aud`, `exp`, `user_id`)
  - request authentication (cookie first, bearer fallback)
  - session token issuance
- Backend `debug_token_server.py` exists:
  - `POST /debug/token`
  - admin key protection (`X-Debug-Admin-Key`)
  - staging/dev-only issuance gate
  - stable JSON errors and no-store headers
- Token governance document exists: `DEBUG_TOKEN_GOVERNANCE.md`.
- Backend retrieval/context decision baseline is documented in `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md`.
- Documentation harness policy exists: `Harness_Policy.md`.
- Secret rotation workflow now supports 3-key rotation in one run:
  - `UNIVERSE_JWT_SECRET`
  - `UNIVERSE_SESSION_SECRET`
  - `DEBUG_TOKEN_ADMIN_KEY`
- Auth workflow commit reported: `59926b4` pushed to `origin/main`.
- Temporary public HTTPS verification for `/debug/token` completed:
  - success path `200 code=issued`
  - failure path `401 code=unauthorized_admin`
- Strict auth gateway implementation branch started and pushed:
  - branch: `feat/gateway-strict-auth-contract`
  - commit: `ab93df1`
- `gateway_fastapi.py` added with strict contract behavior:
  - first request via Bearer validation
  - session cookie issuance (`HttpOnly; Secure; SameSite=None; Path=/`)
  - subsequent cookie-only session path
  - real `401/403` JSON failures
- Gateway/deployment test baseline added:
  - `tests/test_gateway_fastapi.py` included
  - reported test run: `17 passed`
- Android updated `Universe3DFragment`:
  - `onResume` now forces `webView.reload()` for minimal session revalidation
  - existing 401/403 handler remains in place

## 3) In Progress
- Hardening auth contract to strict web semantics across all hops.
- Aligning cookie-based session flow between Android WebView and backend runtime behavior.
- Streamlit Cloud secrets sync and restart (latest rotated values) is pending.
- `debug_token_server.py` persistent staging deployment (fixed HTTPS host) is pending.
- Strict contract gateway is implemented in branch, but fixed-HTTPS staging deployment is pending.
- Android `UNIVERSE_URL` cutover from Streamlit URL to gateway URL is pending.
- Stage A handoff migration (`*.handoff.json` + `handoff.txt`) is pending adoption in all cycles.
- Context hygiene hardening (`.geminiignore` + cache noise exclusion) is in progress.
- Policy gate split rollout (`Local WARN / Push FAIL`) is in progress.
- Skill supply-chain hardening (approved registry + checksum rule) is in progress.
- Cross-device runtime rollout (`tools/bootstrap_env.ps1`, `tools/project_python.ps1`, machine-local `.venvs_hub`) is in progress.

Priority order (gate-first):
1. Antigravity:
   - Apply 3 rotated secrets to Streamlit Cloud staging and restart.
   - Deploy `debug_token_server.py` to fixed HTTPS staging host (replace temporary tunnel).
   - Deploy strict-contract FastAPI gateway and provide E2E-ready URL.
2. Android:
   - Switch `UNIVERSE_URL` to deployed gateway URL.
   - Run gateway cutover checklist (first bearer load, cookie-only follow-up, 401/403 UX, lifecycle stability).
   - Replace reload-only refresh with lightweight auth status strategy after cutover.
3. Android refactors (deferred until gateway E2E pass):
   - `LoginActivity` introduction.
   - `BaseWebViewFragment` commonization.

## 4) Open Gaps / Risks
- Streamlit runtime limitation:
  - true `HttpOnly` cookie issuance and strict HTTP status control may require an upstream auth gateway/proxy.
- Upstream gateway code exists, but fixed HTTPS staging deployment is still blocked by environment/permission.
- If Cloud secrets and local rotated secrets diverge, Android auth can continue to fail.
- `/?embed=universe_3d` dark-screen symptom remains and may block effective UX validation.
- Token lifecycle:
  - refresh strategy is not fully automated yet on Android.
- Android `onResume -> reload()` is intentionally minimal but can add unnecessary network/battery overhead.
- Human error risk:
  - accidental hardcoded token commit remains possible without automated guardrail.
- Local runtime variance:
  - `python` command behavior differs by shell/machine, which can defer local policy gate execution.

## 5) Current Contract Snapshot
- Universe URL (debug):
  - `https://benjohnbill-ax-camp-homework.streamlit.app/?embed=universe_3d`
- Debug token endpoint (backend service):
  - `POST /debug/token`
  - Header: `X-Debug-Admin-Key: <admin_key>`
  - JSON body: `user_id`, `aud` (optional), `ttl_minutes` (optional)
  - Temporary public URL (session-bound): `https://social-things-crash.loca.lt/debug/token`
- Target fixed staging URLs (prepared, not yet active):
  - debug token: `https://ax-camp-debug-token-staging.onrender.com/debug/token`
  - gateway universe: `https://ax-camp-universe-gateway-staging.onrender.com/gateway/universe_3d`

## 6) Backend Retrieval-Context Track

Locked decisions:
- Rank fusion: `RRF`
- Context storage: split (`content` and `context_text`)
- Context generation: mixed mode (short sync / long async)
- Runtime: UI path and heavy backend path separation

Implemented:
- Policy source document exists and role ownership is fixed:
  - `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md` (policy)
  - `Antigravity_Agent.md` (execution)
  - `Agent.md` (SSOT linkage)
- Korean short-term retrieval baseline is implemented:
  - rule-based rewrite module: `korean_query_rewrite.py`
  - evaluation dataset: `data/korean_retrieval_eval.json`
  - evaluation tool: `tools/eval_korean_retrieval.py`
  - local risk gate runner: `tools/run_risk_closure_gate.py`

In Progress:
- Data model lifecycle fields rollout (`context_status`, `context_version`, `context_source_hash`, retry metadata)
- Async worker safety path (idempotency + stale handling + bounded retry)
- Retrieval safeguards (`exclude_ids`, context fallback verification)
- Korean short-term boosts tuning (dictionary coverage + metric threshold hardening)

Blockers / Missing Evidence:
- Latest backend reports focus on auth/gateway; retrieval/context rollout proof is still missing
- Staging/production Korean retrieval metrics are still missing (current evidence is local synthetic evaluation only)

## 7) Latest Validation Snapshot
- Android side reported:
  - `app:assembleDebug` and sync completed
  - `onResume` reload behavior added and tab/background stability verified
  - gateway cutover checklist is ready, but real gateway URL cutover test is pending
- Frontend side reported:
  - `orchestration/results/20260224T102000Z.T-narrative_loop-20260224-frontend.result.json` updated with `app.py` 401/403 friendly warning UX evidence
  - `python -m pytest -q tests/test_universe_auth.py` and `tools/check_postdeploy_smoke.py` were reported as PASS in the frontend result artifact
  - CT follow-up dependency: verify same UX path with real Android->gateway E2E traffic before final accept
- Backend side reported:
  - `python -m pytest -q tests/test_gateway_fastapi.py tests/test_debug_token_server.py tests/test_universe_auth.py` -> `17 passed`
  - `python tools/run_agent_a_gate.py` -> PASS (A-0~A-4)
  - `python -m pytest -q tests` -> `34 passed`
  - `python tools/run_risk_closure_gate.py` -> PASS (`R-1` docs, `R-2` gateway local E2E, `R-3` Korean retrieval eval)
  - `python tools/run_agent_a_gate.py` can fail in this environment when outbound DB auth is blocked (`WinError 10013`)
  - local runtime strict contract evidence:
    - first bearer request -> `307` + `Set-Cookie` issued
    - cookie-only follow-up -> `307` + `X-Auth-Source: cookie`
    - auth failure -> `401 missing_token`
    - claim/permission failure -> `403 forbidden_audience`
  - fixed HTTPS deployment and cloud-secret sync remain pending
  - 2026-02-24 CT cycle strict gate result: `run_agent_a_gate.py --policy-mode strict` failed at docs strict gate (`A-1.1`), so cycle status is `blocked`
  - canonical cycle artifacts published:
    - `orchestration/results/20260224T100600Z.T-narrative_loop-20260224-cycle01.result.json`
    - `orchestration/handoff/20260224T100600Z.T-narrative_loop-20260224-cycle01.handoff.json`
    - `orchestration/handoff/latest.handoff.json` updated
  - 2026-02-24 strict recovery: `tools/check_docs_contract.py` sync-drift detection logic was corrected (diff+status union, case-insensitive matching), and strict gate was revalidated as PASS
  - updated canonical cycle artifacts:
    - `orchestration/results/20260224T101200Z.T-narrative_loop-20260224-cycle01.result.json` (`status=partial`)
    - `orchestration/handoff/20260224T101200Z.T-narrative_loop-20260224-cycle01.handoff.json`
    - `orchestration/handoff/latest.handoff.json` updated to strict-pass snapshot
  - 2026-02-24 worker execution sequence completed (backend -> android -> frontend) and worker result artifacts collected:
    - `orchestration/results/20260224T101700Z.T-narrative_loop-20260224-backend.result.json` (`status=partial`)
    - `orchestration/results/20260224T101800Z.T-narrative_loop-20260224-android.result.json` (`status=partial`)
    - `orchestration/results/20260224T102000Z.T-narrative_loop-20260224-frontend.result.json` (`status=partial`)
  - 2026-02-24 CT final decision for cycle01: `blocked` (strict gate pass but mandatory runtime/deploy evidence incomplete)
  - final decision artifacts:
    - `orchestration/results/20260224T102500Z.T-narrative_loop-20260224-cycle01.result.json` (`status=blocked`)
    - `orchestration/handoff/20260224T102500Z.T-narrative_loop-20260224-cycle01.handoff.json`
    - `orchestration/handoff/latest.handoff.json` updated to final blocked snapshot
  - 2026-02-24 cycle02 dispatch issued for same-day blocker closure:
    - `orchestration/task.json` switched to `trace-narrative_loop-20260224-cycle02`
    - `orchestration/dispatch/20260224-cycle02.worker-prompts.json` published
    - worker tasks published:
      - `orchestration/tasks/20260224T121500Z.backend.task.json`
      - `orchestration/tasks/20260224T121500Z.android.task.json`
      - `orchestration/tasks/20260224T121500Z.frontend.task.json`

## 8) Update Protocol (Operating Rule)
- User collects fresh reports from Android Studio and Antigravity.
- Codex updates this file by moving items between:
  - `Completed`
  - `In Progress`
  - `Open Gaps / Risks`
- Codex separately tracks both auth integration and backend retrieval-context track evidence.
- Policy/status 충돌이 발생하면 `Harness_Policy.md` authority model을 기준으로 판정한다.
- Progress percentage is updated only with concrete validation evidence.
- When all 100% completion conditions are met, Codex creates final handover doc:
  - `integration_handover_v1.md`
  - and reports completion immediately.

## 9) Changelog
- 2026-02-20: Initial draft created and synchronized across backend/mobile workspaces.
- 2026-02-20: Added Backend Retrieval-Context Track and linked playbook-based completion criteria.
- 2026-02-20: Synced latest auth-rotation report (3-key rotation, temporary HTTPS `/debug/token`, pending Cloud sync/persistent staging).
- 2026-02-20: Applied gate-first execution priority (Antigravity auth rollout first, Android refactors deferred).
- 2026-02-20: Synced strict gateway implementation status (`ab93df1`), Android `onResume` refresh update, and deployment blockers.
- 2026-02-21: Python environment rebuilt via parallel swap (`venv_new` -> `venv`), old env archived as `venv_backup_2026-02-21_065537`.
- 2026-02-21: `python tools/check_docs_contract.py --mode warn` revalidated in new env (`FAIL=0`, `WARN=0`).
- 2026-02-20: Added Korean rewrite/eval tooling and risk closure gate (`check_gateway_e2e.py`, `eval_korean_retrieval.py`, `run_risk_closure_gate.py`) with local PASS evidence.
- 2026-02-20: Fixed SQLite parity helpers in `db_manager.py` (`get_all_embeddings`, `save_embedding`, `ensure_fts`) and revalidated `pytest tests` (`34 passed`).
- 2026-02-23: Added Stage A handoff migration and context-hygiene alignment items.
- 2026-02-23: Added docs gate mode split and strict CI policy gate workflow.
- 2026-02-23: Added skill supply-chain registry and checksum-based promotion guard.
- 2026-02-23: Added MVP governance risk summary (local runtime + candidate checksum readiness).
- 2026-02-23: Added machine-local venv bootstrap scripts and wrapper command policy for OneDrive-safe runtime.
- 2026-02-24: Locked user decisions (B items), dual-write exit criteria, and final CT two-day execution workflow.
- 2026-02-24: Implemented Step 5 artifacts (`result.json` + canonical `handoff.json`) and recorded blocked decision due to strict gate failure.
- 2026-02-24: Fixed docs strict-gate false positive in `tools/check_docs_contract.py` and revalidated strict gate PASS.
- 2026-02-24: Executed worker sequence, collected 3 worker results, and issued final CT blocked decision with canonical handoff update.
- 2026-02-24: Integrated Android Studio project files into `android/NarrativeLoopMobile` under this repository and reclassified `C:\Users\LG\AndroidStudioProjects\NarrativeLoopMobile` as a local workspace alias path.
- 2026-02-24: Synced frontend report update (`orchestration/results/20260224T102000Z.T-narrative_loop-20260224-frontend.result.json`) with explicit 401/403 friendly UX validation notes.
- 2026-02-24: Issued cycle02 CT dispatch package for end-of-day blocker closure (`orchestration/task.json`, `orchestration/dispatch/20260224-cycle02.worker-prompts.json`, and 3 worker task files).

## 10) User Decisions (Locked, 2026-02-24)

### 10.1 USER_STARTER_PLAYBOOK Alignment Snapshot
- Project name: `Narrative_Loop`
- Goals:
  - Stabilize Android-Backend auth contract (Bearer -> HttpOnly session -> 401/403) on staging E2E.
  - Operate Korean retrieval quality with measurable metrics (`recall@k`, `precision@k`, `acceptance`).
  - Standardize CT-Worker execution and handoff via task/result/handoff JSON canonical flow to strict gate.
- Out of scope:
  - Full OpenSearch migration in this cycle.
  - Korean cross-encoder reranker production rollout in this cycle.
  - Large Android UI refactor (`LoginActivity`/`BaseWebViewFragment`) in this cycle.
  - Always-on paid LLM rewrite pipeline in this cycle.
- Non-negotiables:
  - No secret/API key/DB credential plain-text exposure.
  - No deploy/schema change/permission elevation/destructive command without approval.
  - Canonical handoff is `*.handoff.json`; `handoff.txt` is optional briefing only.
  - Gate split is mandatory (local warn, push/merge strict).
  - Query rewrite starts with free rule-based + synonym dictionary strategy.
- Roles:
  - User (final approval), CT/Codex, Antigravity, Android Studio Agent, Gemini 3.1 Pro.
- Integrations:
  - API: `gateway_fastapi.py`, `debug_token_server.py`, `/v1/ocr/ingest`
  - DB: Supabase PostgreSQL (`DATASTORE=postgres`)
  - MCP: `.mcp.json` -> `docs-search`, `db-observer`, `deploy-health` (read-only)
  - Skill: `skills/integration-status-sync/SKILL.md` pilot
  - Contracts: `orchestration/contracts/task.schema.json`, `result.schema.json`, `handoff.schema.json`
- Approval-needed changes:
  - deploy: User (CT pre-gate verification required)
  - schema change: User + Antigravity technical review
  - permission elevation: User
  - destructive command execution: User (explicit one-time approval)

### 10.2 Stage A Dual-Write Exit Criteria (`handoff.txt` Optional Path)
- Target date: `2026-02-26`
- Exit conditions:
  1. Latest 3 consecutive cycles pass handoff schema validation for all `orchestration/handoff/*.handoff.json` via `tools/validate_contracts.py`.
  2. Accept/blocked decisions and merge approvals are made from JSON evidence only.
  3. Operational continuity remains stable across 2 consecutive cycles without `handoff.txt`.
- Exception:
  - `handoff.txt` is allowed only for external briefing and must not introduce facts absent from canonical JSON.
- If unmet:
  - Keep Stage B (optional summary), extend 7 days, then re-validate with gap-closure evidence.

### 10.3 Front-Matter WARN -> FAIL Promotion
- Scope:
  - All newly created markdown docs.
  - Priority L1-L3 policy/ops docs linked to `Agent.md`, `Harness_Policy.md`, `integration_status.md`.
- Dates:
  - WARN end date: `2026-02-25`
  - FAIL effective date: `2026-02-26`
- FAIL criteria:
  - Missing required front-matter keys (`doc_type`, `owner`, `authority_level`, `last_updated`, `sync_with`, `change_triggers`, `sunset_condition`) in new/changed docs.
  - Missing `sunset_condition` or `review_by` in temporary docs.
  - Strict-mode docs contract check failure.
- Grace policy:
  - Legacy docs are temporarily exempt until `2026-03-31`; any newly changed legacy doc is immediately subject to the rule.
- Owner:
  - Codex (CT), final approval by User.

### 10.4 Workflow Contract (`task -> result -> handoff`)
- Start conditions:
  - USER_STARTER_PLAYBOOK values locked
  - Project-local policy aligned
  - Contract schema files exist
  - Runtime environment ready
- Step 1 (CT):
  - Publish `task.json` with `trace_id`, goal, in/out scope, approval-required items, acceptance criteria, and priority.
- Step 2 (Worker):
  - Execute implementation/validation and publish `result.json` with changed files, commands, exit codes, key logs, risks, and rollback points.
  - Worker must include execution root in evidence:
    - backend/frontend/docs: this `Narrative_Loop` repository
    - android (canonical): `android/NarrativeLoopMobile` (same `Narrative_Loop` repository)
    - android (workspace alias): `C:\Users\LG\AndroidStudioProjects\NarrativeLoopMobile` (Android Studio local workspace path)
  - Python runtime references must use dynamic notation (`LIFE_VENV_ROOT` + `Narrative_Loop.venv`) instead of device-specific absolute paths.
- Step 3 (CT):
  - Approve only schema-valid `*.handoff.json` and then update `latest.handoff.json`.
  - Validate worker evidence includes repository root and runtime notation compliance.
- Step 4 (Gate):
  - Run local warn gates during development and enforce strict gates before push/merge.
- Step 5 (Decision):
  - `accept`: schema pass + strict gate pass + approvals satisfied + evidence complete.
  - `blocked`: strict failure or missing approval or insufficient evidence.
- Evidence storage:
  - canonical handoff: `orchestration/handoff/*.handoff.json`
  - validation reports: `data/*_latest.json`
  - status aggregation: this file changelog and section updates
- SLA target:
  - CT first triage in 2 hours, and remediation plan within 24 hours after blocked transition.

### 10.5 Locked 100% Acceptance Checklist
- [ ] Android first request passes Authorization: Bearer validation.
- [ ] Cookie session transition works on follow-up requests.
- [ ] 401/403 failure scenarios have E2E evidence.
- [ ] Streamlit/Backend secrets sync and restart evidence is complete.
- [ ] Validation is complete on fixed HTTPS staging URLs.
- [ ] Strict gate pass evidence is complete (push/merge basis).
- [ ] Canonical handoff JSON passes schema validation.
- [ ] No unresolved high/critical risk remains (or approved mitigation plan exists).
- Final approver: User
- Target completion date: `2026-02-26`
- Note:
  - Per `SYSTEM_HANDOFF_MIGRATION_POLICY`, Stage A is not a system-wide mandatory requirement.
  - This project controls `handoff.txt` retain/retire decision locally while preserving JSON canonical governance.

## 11) Final CT Workflow (2026-02-24 to 2026-02-25, Forced Execution)

### 11.1 Day 1: 2026-02-24 (Today) - Gate Readiness + Deployment Prerequisites
1. CT
   - Freeze scope and publish the cycle `task.json` with approval gates and acceptance criteria.
2. Antigravity
   - Apply rotated secrets to staging, restart services, and submit restart evidence.
   - Deploy fixed HTTPS `debug_token_server` and strict `gateway_fastapi` staging endpoints.
3. Android Agent
   - Prepare gateway cutover checklist in `android/NarrativeLoopMobile` (or local Android Studio workspace alias path) and ensure debug build points to gateway-ready config branch.
4. CT
   - Verify `result.json` evidence completeness and reject incomplete reports before end-of-day.
5. Gate checkpoint (end of day)
   - Minimum pass condition: fixed HTTPS endpoints reachable + auth smoke evidence attached.
   - Failure condition: missing deployment evidence or unresolved approval item -> status stays `blocked`.

### 11.2 Day 2: 2026-02-25 (Tomorrow) - E2E Contract Validation + Strict Gate Decision
1. Android Agent
   - In `android/NarrativeLoopMobile` (or local Android Studio workspace alias path), switch `UNIVERSE_URL` to deployed gateway URL and execute E2E flow (first bearer, cookie follow-up, 401/403 UX, lifecycle stability).
2. Antigravity
   - Run backend validation set and provide command outputs tied to gateway/auth contract.
3. CT
   - Validate contract artifacts (`task/result/handoff`) and ensure schema-valid canonical handoff update.
4. CT + User approval
   - Run strict push/merge gate evidence review and produce final decision:
     - `accept` if strict pass + full evidence + approvals satisfied.
     - `blocked` if any strict failure, missing approval, or incomplete evidence.
5. End-of-day deliverables
   - Updated `integration_status.md` sections (Completed/In Progress/Risks).
   - New canonical handoff JSON (`orchestration/handoff/*.handoff.json`) and refreshed `latest.handoff.json` if accepted.

### 11.3 Non-Negotiable Decision Rule
- No strict-pass, no accept.
- No canonical JSON handoff, no completion claim.
- No user approval on approval-required changes, no deployment finalization.

