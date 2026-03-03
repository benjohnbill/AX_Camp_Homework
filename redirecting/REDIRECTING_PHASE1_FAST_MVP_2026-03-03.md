---
doc_type: redirecting_phase_plan
phase: 1
owner: product_frontend_backend
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - redirecting/REDIRECTING_DATA_API_CONTRACT_2026-03-03.md
  - app.py
  - gateway_fastapi.py
change_triggers:
  - phase_scope_changed
  - acceptance_changed
sunset_condition: Replace when Phase 1 acceptance is passed and Phase 2 starts.
---
# Redirecting Phase 1: Fast MVP (2026-03-03)

체크리스트 문서:
- `redirecting/REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md`

## 0) 목표
- 시간 제약 하에서 실제 사용 가능한 최소 실행 루프를 우선 출시한다.
- Streamlit 한계를 고려해 UI 복잡도를 낮추고, 서버 계약을 최소로 고정한다.

## 1) 핵심 원칙
1. 코어 루프는 단순하고 반드시 완료 가능해야 한다.
2. 비동기/고급 UI 실패가 코어 루프를 막으면 안 된다.
3. 자동화보다 수동 확정(사용자 의사결정)을 우선한다.

## 2) 포함 범위 (Must)
1. `Focus-first -> Reflection` 기본 완료 루프.
2. `Plan-first` 최소 입력(Frog + 간단 Time-Box 리스트 에디터).
3. Journal 저장 + Journal -> Session 수동 승격.
4. Core 수동 승격 (`POST /v1/core/promote`).
5. OCR 업로드 저장(원본 + 메타)과 Reflection 시점 최소 연결.
6. Today 조회 및 세션 재진입.

## 3) 제외 범위 (Not in Phase 1)
1. 드래그형 Time-Box 컴포넌트.
2. 독립 worker 서비스 분리 배포.
3. 자동 Core 후보/자동 승격.
4. 3D 고급 상호작용/고급 클러스터 분석.

## 4) 구현 단위
1. Frontend (Streamlit):
   - Start 선택(Plan-first / Focus-first / Journal).
   - Form submit 기반 단계 진행.
   - Reflection 3필수 저장.
2. Backend (API):
   - session/focus/reflect/today/journal/promote/core-promote 최소 엔드포인트.
   - 기본 검증 규칙과 호환성 유지.
3. Data:
   - `execution_sessions`, `execution_blocks`(리스트형), `journal_entries`, `image_events`, `core_entries`.
   - logs 투영 저장 유지.

## 5) 수용 기준
1. 3분 내 Focus 시작 -> Reflection 저장이 가능해야 한다.
2. OCR 지연/실패와 무관하게 세션 완료가 가능해야 한다.
3. Journal은 저장 가능하되 승격 전 코어 완료율에 포함되지 않아야 한다.
4. Core는 수동 승격 API에서만 생성되어야 한다.

## 6) 리스크/대응
1. Streamlit rerun 입력 유실:
   - 대응: `st.form` 단위 저장과 단계별 최소 상태키 유지.
2. API 확장 과부하:
   - 대응: 필수 엔드포인트만 우선 구현, 나머지는 Phase 2로 이월.

## 7) Phase Gate (1 -> 2)
1. Focus 완료율/Reflection 작성률 기본 지표 확보.
2. 코어 루프 회귀 이슈 없이 1주 운영.
3. 세션 데이터 품질이 추천 입력으로 활용 가능한 수준 확보.
