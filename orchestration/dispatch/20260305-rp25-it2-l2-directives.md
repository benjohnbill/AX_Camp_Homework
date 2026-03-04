# Redirecting Phase2.5 Iteration-2 L2 Directives

## Common Guard
`stabilization-only. no Phase3 expansion. every claim must include timestamped evidence path + schema-valid result.`

## Backend
`[L2_CT_DIRECTIVE]`
- worker: `backend_cli`
- task_id: `T-nl-20260305-rp25-it2-backend`
- objective: Keep backend green and provide contract support for Android E2E closure.
- must:
  1. Re-run phase2.5 transition/non-blocking guards.
  2. Confirm save/refine/reflect/evidence-link payload contract for Android runtime.
  3. Publish schema-valid support result.

## Frontend
`[L2_CT_DIRECTIVE]`
- worker: `frontend_ide`
- task_id: `T-nl-20260305-rp25-it2-frontend`
- objective: Hold phase2.5 staged-flow baseline and verify no regression while Android closes gaps.
- must:
  1. Re-run flow helper + replay tests.
  2. Confirm session-linked reflection evidence path remains intact.
  3. Publish schema-valid support result.

## Android
`[L2_CT_DIRECTIVE]`
- worker: `android_ide`
- task_id: `T-nl-20260305-rp25-it2-android`
- objective: Move lane from partial -> success by full runtime E2E closure.
- must:
  1. Implement CreateNarrative `AI Refine` and `Save Narrative` real API handling with retry/error UX.
  2. Implement stage chains:
     - Plan-first: `start(plan) -> frog -> timebox/draft -> commit -> focus/start/end -> reflect`
     - Focus-first: `start(focus_now) -> focus/end -> timebox/retro -> reflect`
  3. Ensure OCR `image_event_id` is passed to `reflect evidence_links` as real IDs.
  4. Implement Desk minimum read path for persisted result visibility.
  5. Run SC-A/SC-B/SC-C/SC-D with physical+emulator same-window proof.
- deliverables:
  - `android/NarrativeLoopMobile/evidence/<TS>_android_phase25_e2e_walkthrough.md`
  - `android/NarrativeLoopMobile/evidence/<TS>_android_phase25_e2e_logcat.log`
  - `orchestration/results/<TS>.T-narrative_loop-20260305-android-redirecting-phase25.result.json` (status=success)
  - Step A/B mirrored canonical artifacts

## CT Gate
`[CT_ACCEPTANCE_CRITERIA]`
1. Android lane status is `success` with SC-A/SC-B/SC-C/SC-D proof.
2. Backend/frontend support reruns are non-regressed.
3. All artifacts are schema-valid.
4. Only then publish Phase2.5 close.

