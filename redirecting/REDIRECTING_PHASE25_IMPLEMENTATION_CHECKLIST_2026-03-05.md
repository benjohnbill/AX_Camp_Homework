---
doc_type: redirecting_phase_checklist
phase: 2.5
track: full_flow
owner: backend_frontend_android
authority_level: operational
last_updated: 2026-03-05
sync_with:
  - redirecting/REDIRECTING_PHASE25_FULL_FLOW_PLAN_2026-03-05.md
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
sunset_condition: Replace after Phase 2.5 Go/No-Go is closed.
---
# Phase 2.5 Implementation Checklist (2026-03-05)

## 0) Scope Lock
- [ ] 데모 축약범위가 아닌 `풀 플로우 복원`을 목표로 고정.
- [ ] Phase 3 항목(아키텍처 분리/고급 UI) 사전 진입 금지.

## 1) Backend
- [ ] `session/{id}/frog` API 구현.
- [ ] `session/{id}/timebox/draft` API 구현.
- [ ] `session/{id}/timebox/retro` API 구현.
- [ ] OCR 업로드와 세션 링크 규칙(API 또는 동등 계약) 구현.
- [ ] `start/commit/focus/reflect` 상태 전이 일관성 테스트 추가.
- [ ] AI 실패/지연 fallback 비차단 회귀 테스트 PASS.

## 2) Frontend (Streamlit)
- [ ] Plan Start 진입이 Control 모드 우회 없이 `frog` 단계로 연결.
- [ ] `frog -> timebox_edit -> commit -> focus -> reflection` 단계 노출.
- [ ] `focus -> retro_timebox -> reflection` 경로 노출.
- [ ] Reflection evidence가 실제 세션 evidence 데이터로 표시.
- [ ] placeholder evidence 옵션 제거(또는 demo 전용 플래그 격리).
- [ ] rerun 시 draft 유실 방지 검증.

## 3) Android
- [ ] OCR 업로드에 세션 링크 필드 전달.
- [ ] 토큰 저장 상태에서 OCR -> 세션 연결 검증.
- [ ] Universe/인증 회귀 PASS 유지.
- [ ] Step A 산출물을 Step B canonical path로 미러링.

## 4) Core Scenarios
- [ ] SC-A Plan-first 완주: Frog -> Time-Box -> Focus -> Reflection.
- [ ] SC-B Focus-first 완주: Focus -> Retro Time-Box -> Reflection.
- [ ] SC-C OCR 완주: OCR 업로드 -> 세션 링크 -> Reflection 큐레이션.

## 5) Gate Verdict
- [ ] AC-01 Plan-first PASS
- [ ] AC-02 Focus-first PASS
- [ ] AC-03 OCR-Reflection Link PASS
- [ ] AC-04 Journal->Promote->Core PASS
- [ ] AC-05 Non-blocking under AI delay/failure PASS
- [ ] AC-06 Universe replay regression PASS

## 6) Result Artifacts (Mandatory)
- [ ] `orchestration/results/<TS>.T-narrative_loop-20260305-backend-redirecting-phase25.result.json`
- [ ] `orchestration/results/<TS>.T-narrative_loop-20260305-frontend-redirecting-phase25.result.json`
- [ ] `orchestration/results/<TS>.T-narrative_loop-20260305-android-redirecting-phase25.result.json`
- [ ] `orchestration/results/<TS>.T-narrative_loop-20260305-redirecting-phase25-close.result.json`
- [ ] `orchestration/handoff/<TS>.T-narrative_loop-20260305-redirecting-phase25-close.handoff.json`

