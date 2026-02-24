# Integration Status: Android <-> Backend

Last updated: 2026-02-20 (gateway branch + Android resume refresh)
Maintainer: Codex (integration coordinator)
Projects:
- Backend/Web: Narrative_Loop (Streamlit + Antigravity)
- Mobile: NarrativeLoopMobile (Android Studio)

## 1) Overall Progress
- Current progress: 90%
- Status: MVP integration is working in staging, but auth contract hardening is still in progress.

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
  - `Antigravity_agent.md` (execution)
  - `agent.md` (SSOT linkage)

In Progress:
- Data model lifecycle fields rollout (`context_status`, `context_version`, `context_source_hash`, retry metadata)
- Async worker safety path (idempotency + stale handling + bounded retry)
- Retrieval safeguards (`exclude_ids`, context fallback verification)
- Korean short-term boosts (synonym dictionary + rule rewrite)

Blockers / Missing Evidence:
- Latest backend reports focus on auth/gateway; retrieval/context rollout proof is still missing
- No evaluation report yet for Korean retrieval metrics (recall@k / precision@k / user acceptance)

## 7) Latest Validation Snapshot
- Android side reported:
  - `app:assembleDebug` and sync completed
  - `onResume` reload behavior added and tab/background stability verified
  - gateway cutover checklist is ready, but real gateway URL cutover test is pending
- Backend side reported:
  - `python -m pytest -q tests/test_gateway_fastapi.py tests/test_debug_token_server.py tests/test_universe_auth.py` -> `17 passed`
  - `python tools/run_agent_a_gate.py` -> PASS (A-0~A-4)
  - local runtime strict contract evidence:
    - first bearer request -> `307` + `Set-Cookie` issued
    - cookie-only follow-up -> `307` + `X-Auth-Source: cookie`
    - auth failure -> `401 missing_token`
    - claim/permission failure -> `403 forbidden_audience`
  - fixed HTTPS deployment and cloud-secret sync remain pending

## 8) Update Protocol (Operating Rule)
- User collects fresh reports from Android Studio and Antigravity.
- Codex updates this file by moving items between:
  - `Completed`
  - `In Progress`
  - `Open Gaps / Risks`
- Codex separately tracks both auth integration and backend retrieval-context track evidence.
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
