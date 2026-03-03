---
doc_type: redirecting_phase_checklist
phase: 2
track: demo
owner: frontend_backend_android
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_PHASE2_MID_MVP_2026-03-03.md
  - redirecting/REDIRECTING_KEEP_KILL_DECISION_2026-03-03.md
sunset_condition: Replace after Phase 2 demo Go/No-Go is closed.
---
# Phase 2 Demo Checklist (2026-03-03)

## 0) Scope Lock
- [x] Phase 2 목표를 `추천/회고 품질 강화 데모`로 고정한다.
- [x] 3D는 `주간 read-only replay + CTA 1개 + Skip` 범위로 제한한다.
- [x] 자동 Core 승격/고급 드래그 편집은 제외한다.

## 1) Backend
- [x] `ai_jobs` 최소 lifecycle(queued/running/succeeded/failed)을 구현한다.
- [x] `auto_tag_extraction`, `similar_session_linking`, `next_action_recommendation` 최소 job만 연결한다.
- [x] worker 지연/실패 시 rule-based fallback으로 응답을 유지한다.
- [x] insight API가 AI 결과 유무와 관계없이 즉시 응답하도록 보장한다.

## 2) Frontend (Streamlit)
- [x] Plan-first, Focus-first, Journal 경로를 모두 데모 가능한 수준으로 연결한다.
- [x] Reflection에서 evidence 1~2장 큐레이션과 `1줄 의미/Skip` 액션 제공.
- [x] 3D 회고 진입 화면에서 7일 고정 타임라인을 노출.
- [x] 3-tier 시각 위계를 반영한다:
  - [x] Tier 1 `session_completed`
  - [x] Tier 2 `session_interrupted`
  - [x] Tier 3 `supporting_evidence`

## 3) Android
- [x] OCR 업로드 흐름을 데모에서 재사용 가능한 API로 정렬한다.
- [x] Universe 진입 경로와 인증 흐름이 데모 환경에서 깨지지 않는지 점검한다.
- [x] Android 미구현 구간은 발표에서 "Phase 3 후보"로 명확히 고지한다.

## 4) Streamlit Buffering Mitigation
- [x] 3D payload는 최근 7일 + 상한 노드 수로 제한한다.
- [x] 불필요한 rerun을 유발하는 버튼/상태 변이를 정리한다.
- [x] 장시간 blocking 호출을 비동기 상태 조회로 전환한다.
- [x] 타이머/회고/추천 화면을 분리해 한 화면 계산량을 줄인다.

## 5) Demo Runbook
- [x] 시연 시나리오 1: Plan-first 완결 세션 생성 -> 3D에서 Tier 1 확인.
- [x] 시연 시나리오 2: 중단 세션 생성 -> 3D에서 Tier 2 확인.
- [x] 시연 시나리오 3: OCR evidence 연결 -> 3D/Reflection에서 Tier 3 확인.
- [x] 시연 시나리오 4: AI job 실패 상황에서도 코어 루프 진행 확인.

## 6) Go/No-Go
- [x] 코어 루프 비차단 동작 유지.
- [x] 3-tier 위계가 시각적으로 일관되게 표현됨.
- [x] 3D 종료 CTA + Skip 둘 다 정상 동작.
- [x] 발표에서 "회고용 3D v1" 범위를 과장 없이 설명 가능.

## 7) Closure Note (2026-03-04)
- 비차단 리스크 1건을 명시한다: backend reflection projection 비동기 경로에서 Streamlit `ScriptRunContext` warning이 로그에 노출됨.
- 해당 이슈는 데모 진행을 차단하지 않으며, Phase 3 하드닝 항목으로 이관한다.
