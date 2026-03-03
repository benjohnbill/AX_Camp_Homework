---
doc_type: redirecting_phase_checklist
phase: 1
track: demo
owner: frontend_backend_android
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_PHASE1_FAST_MVP_2026-03-03.md
  - redirecting/REDIRECTING_DATA_API_CONTRACT_2026-03-03.md
sunset_condition: Replace after Phase 1 demo Go/No-Go is closed.
---
# Phase 1 Demo Checklist (2026-03-03)

## 0) Scope Lock
- [ ] 데모 목표를 `코어 루프 안정 시연`으로 고정한다.
- [ ] 데모 범위 밖 항목(드래그 Time-Box, 독립 worker, 자동 Core 승격)을 명시적으로 제외한다.
- [ ] Gap/Entropy는 UI 진입을 차단하거나 완전 비활성화한다.

## 1) Backend
- [ ] `session/start`, `focus/start`, `focus/end`, `reflect`, `today` 최소 API를 정의/연결한다.
- [ ] `journal/entry`, `journal/{id}/promote`, `core/promote`를 수동 흐름으로 연결한다.
- [ ] OCR 업로드는 저장 성공을 우선 보장하고, 해석 실패가 루프를 막지 않게 한다.
- [ ] 기존 `/v1/narrative`, `/v1/ocr/ingest` 호환 경로를 유지한다.

## 2) Frontend (Streamlit)
- [ ] 홈에서 `Plan Start`와 `Focus Now`를 노출한다.
- [ ] Focus 시작 -> 종료 -> Reflection 3필수 저장 흐름을 1경로로 완성한다.
- [ ] 모든 입력을 `st.form` submit 단위로 묶는다.
- [ ] `st.rerun()` 호출 지점을 최소화한다.
- [ ] 세션 상태 키를 최소 집합으로 제한한다 (`session_id`, `flow_stage`, `entry_mode`, `reflection_draft`).

## 3) Android
- [ ] 데모에서 사용할 API 계약(특히 OCR ingest body 형식)을 서버와 일치시킨다.
- [ ] Focus/Reflection 루프를 Android에서 직접 시연하지 않으면 발표 시 역할을 "보조 입력 채널"로 명시한다.
- [ ] 인증 토큰/헤더 경로를 데모 리허설 환경에서 검증한다.

## 4) Streamlit Buffering Mitigation
- [ ] 무거운 섹션(3D/대시보드)은 lazy 렌더링한다.
- [ ] OCR/AI 처리는 blocking spinner 최소화, 결과 지연 시 안내 문구로 대체한다.
- [ ] 페이지당 DB write 횟수를 1 submit 1 write 원칙으로 제한한다.
- [ ] 대량 데이터 조회를 피하고 today/최근 N개만 로드한다.

## 5) Demo Runbook
- [ ] 시연 시나리오 1: Focus-first -> Reflection 저장.
- [ ] 시연 시나리오 2: Journal 저장 -> Session 승격 -> Core 수동 승격.
- [ ] 시연 시나리오 3: OCR 업로드 성공 + 해석 지연/실패 시 fallback.
- [ ] 실패 대응 문구(네트워크 지연/AI timeout) 사전 준비.

## 6) Go/No-Go
- [ ] 3분 내 코어 루프 완료 가능.
- [ ] OCR 실패 시에도 코어 루프 완료 가능.
- [ ] 데모 중 강제 UX/오류 모달로 흐름이 막히지 않음.
- [ ] 발표 문구를 `Demo MVP`로 고정(Production-ready 표현 금지).

