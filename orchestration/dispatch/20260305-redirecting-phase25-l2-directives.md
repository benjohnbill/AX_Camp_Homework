# Redirecting Phase2.5 L2 Directives (2026-03-05)

## Common Guard
`stabilization-only + full-flow restoration. no Phase 3 scope expansion. every claim requires timestamped evidence path + schema-valid result.`

## Backend
`[L2_CT_DIRECTIVE]`
- worker: `backend_cli`
- task_id: `T-narrative_loop-20260305-backend-redirecting-phase25`
- objective: Full-flow backend contract completion for Plan-first/Focus-first/OCR link.
- must:
  1. Implement or map contract-equivalent endpoints:
     - `POST /v1/execution/session/{id}/frog`
     - `POST /v1/execution/session/{id}/timebox/draft`
     - `POST /v1/execution/session/{id}/timebox/retro`
     - `POST /v1/execution/session/{id}/evidence/link`
  2. Keep `start/commit/focus/reflect/journal/promote/core` transitions deterministic.
  3. Add regression tests for AI delay/failure non-blocking completion.
- deliverables:
  - `orchestration/results/<TS>.T-narrative_loop-20260305-backend-redirecting-phase25.result.json`
  - `data/evidence/<TS>_backend_phase25_contract_tests.log`
  - `data/evidence/<TS>_backend_phase25_transition_tests.log`

## Frontend
`[L2_CT_DIRECTIVE]`
- worker: `frontend_ide`
- task_id: `T-narrative_loop-20260305-frontend-redirecting-phase25`
- objective: Real full staged workflow in Streamlit (not demo shortcut).
- must:
  1. `Plan Start` -> `frog` stage (no control-mode bypass).
  2. Implement staged path:
     - `frog -> timebox_edit -> timebox_commit -> focus_running -> reflection -> done`
  3. Implement focus-first path:
     - `focus_running -> retro_timebox -> reflection -> done`
  4. Replace placeholder reflection evidence with session-linked data.
- deliverables:
  - `orchestration/results/<TS>.T-narrative_loop-20260305-frontend-redirecting-phase25.result.json`
  - `data/evidence/<TS>_frontend_phase25_flow_walkthrough.md`
  - `data/evidence/<TS>_frontend_phase25_reflection_evidence_ui.png`

## Android
`[L2_CT_DIRECTIVE]`
- worker: `android_ide`
- task_id: `T-narrative_loop-20260305-android-redirecting-phase25`
- objective: OCR session-link runtime validation + bridge compliant artifacts.
- must:
  1. Ensure OCR upload sends session link field aligned with backend contract.
  2. Verify token continuity and universe regression safety remain pass.
  3. Run physical + emulator same-window full path verification.
  4. Mirror Step A artifacts to canonical Step B paths.
- deliverables:
  - `orchestration/results/<TS>.T-narrative_loop-20260305-android-redirecting-phase25.result.json`
  - `android/NarrativeLoopMobile/evidence/<TS>_android_phase25_ocr_session_link.log`
  - `android/NarrativeLoopMobile/evidence/<TS>_android_phase25_dual_device.md`

## CT Gate
`[CT_ACCEPTANCE_CRITERIA]`
1. AC-01 Plan-first full completion PASS.
2. AC-02 Focus-first + retro timebox PASS.
3. AC-03 OCR -> session link -> reflection curation PASS.
4. AC-04 Journal -> Promote -> Core manual promote PASS.
5. AC-05 AI failure/delay non-blocking PASS.
6. AC-06 Universe replay regression PASS.

