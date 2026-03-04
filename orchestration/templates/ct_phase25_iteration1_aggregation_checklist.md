# CT Phase2.5 Iteration-1 Aggregation Checklist

## 0) Purpose
- Use this checklist to aggregate backend/frontend/android iteration-1 reports for Redirecting Phase 2.5.
- Do not approve close at this stage; decide `pass_to_it2` or `blocked`.

## 1) Required Inputs
- Backend result JSON:
  - `orchestration/results/<TS>.T-narrative_loop-20260305-backend-redirecting-phase25.result.json`
- Frontend result JSON:
  - `orchestration/results/<TS>.T-narrative_loop-20260305-frontend-redirecting-phase25.result.json`
- Android result JSON:
  - `orchestration/results/<TS>.T-narrative_loop-20260305-android-redirecting-phase25.result.json`
- Current pointers:
  - `orchestration/task.json`
  - `orchestration/handoff/latest.handoff.json`

## 2) Contract Validation
1. Run schema check on all 3 lane results.
2. Confirm `trace_id` is `trace-narrative_loop-20260305-redirecting-phase25`.
3. Confirm each lane `task_id` matches its phase25 lane task.
4. Reject narrative-only reports without evidence paths.

## 3) AC Mapping (Iteration-1)
- AC-01 Plan-first full completion
  - backend transition evidence + frontend flow walkthrough
- AC-02 Focus-first + retro timebox
  - frontend flow evidence + backend retro endpoint/contract proof
- AC-03 OCR -> session link -> reflection curation
  - android payload/runtime proof + frontend reflection evidence source proof
- AC-04 Journal -> Promote -> Core
  - backend endpoint test evidence (journal/promote/core)
- AC-05 AI delay/failure non-blocking
  - backend fallback regression logs
- AC-06 Universe replay regression
  - frontend replay evidence + android universe continuity

## 4) Decision Rule
- `pass_to_it2`:
  - At least 4/6 AC are evidence-backed pass and no critical blocker.
- `blocked`:
  - Any critical blocker, or AC-01/02/03 missing hard evidence.

## 5) CT Outputs (Mandatory)
- Iteration-1 aggregate result:
  - `orchestration/results/<TS>.T-narrative_loop-20260305-redirecting-phase25-iteration1-aggregate.result.json`
- Iteration-1 aggregate handoff:
  - `orchestration/handoff/<TS>.T-narrative_loop-20260305-redirecting-phase25-iteration1-aggregate.handoff.json`
- Pointer update:
  - `orchestration/handoff/latest.handoff.json`

## 6) L3 Summary Snippet
```text
[CT_L3_SUMMARY]
trace_id: trace-narrative_loop-20260305-redirecting-phase25
iteration: 1
decision: pass_to_it2 | blocked
ac_pass: AC-01, AC-02, ...
ac_gap: AC-xx ...
critical_blocker: none | <blocker_class>
next_3:
1) ...
2) ...
3) ...
```

