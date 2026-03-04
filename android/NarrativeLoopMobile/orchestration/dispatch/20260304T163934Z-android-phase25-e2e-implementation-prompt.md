# CT Prompt: Android Phase2.5 E2E Completion (20260304T163934Z)

[TO_ANDROID_IDE]
목표:
- Android에서 실제 사용자 E2E를 완결한다.
- 범위: 촬영 OCR -> session link -> Plan-first/Focus-first stage 연동 -> reflect/저장 결과 확인.

현재 사실:
- PASS: OCR session-link payload, auth/universe regression guard, dual-device same-window runtime.
- GAP: CreateNarrative save/refine 미구현, Plan/Focus stage API 연동 미구현, Desk 조회 미구현.

필수 구현:
1. `CreateNarrativeFragment`
- `AI Refine` 버튼: `/v1/narrative/refine` 호출 + UI 반영 + 예외/재시도 처리.
- `Save Narrative` 버튼: 사용자 텍스트를 세션 컨텍스트와 함께 저장(계약 endpoint)하고 성공/실패 명확 표시.

2. Plan-first / Focus-first stage 흐름 연결
- Plan-first: `start(plan)` -> `frog` -> `timebox/draft` -> `commit` -> `focus/start/end` -> `reflect`.
- Focus-first: `start(focus_now)` -> `focus/end` -> `timebox/retro` -> `reflect`.
- stage 상태를 UI에서 확인 가능하게 노출.

3. OCR->Reflection curation 실연동
- OCR 결과의 `image_event_id`를 유지하여 reflect 시 `evidence_links`에 실제 ID 전달.
- placeholder 데이터 사용 금지.

4. Desk 결과 확인
- 최소 read path 구현(오늘 session 또는 최근 로그/세션)로 저장 결과를 사용자가 확인 가능하게 한다.

검증 시나리오(필수):
- SC-A: Plan-first 완주 1회 (OCR 포함)
- SC-B: Focus-first+retro 완주 1회
- SC-C: OCR image_event가 reflect evidence_links에 실제 연결되는지 확인
- SC-D: physical+emulator same-window에서 핵심 플로우 최소 1단계씩 재현

산출물(필수):
1. `android/NarrativeLoopMobile/evidence/<TS>_android_phase25_e2e_walkthrough.md`
2. `android/NarrativeLoopMobile/evidence/<TS>_android_phase25_e2e_logcat.log`
3. `orchestration/results/<TS>.T-narrative_loop-20260305-android-redirecting-phase25.result.json` (schema-valid)
4. Step A/B bridge 준수: Android local -> canonical mirror 동일본 제출

판정 규칙:
- narrative 설명만으로 완료 판정 금지.
- evidence path + 재현 커맨드 + schema-valid result가 모두 있어야 success.
