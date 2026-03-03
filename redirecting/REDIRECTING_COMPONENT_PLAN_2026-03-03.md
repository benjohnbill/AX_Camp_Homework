---
doc_type: redirecting_component_plan
owner: frontend
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - app.py
  - universe_3d.py
change_triggers:
  - home_ia_changed
  - component_contract_changed
sunset_condition: Replace when Time-Box UI contract is frozen and implemented.
---
# Redirecting Component Plan (2026-03-03)

## 0) 목표
- Streamlit 기반에서 `Start -> Frog -> Time-Box -> Focus -> Reflection` 단계형 UX를 안정적으로 구현한다.
- 드래그형 Time-Box를 컴포넌트로 붙이되, 실패 시 폴백 UI를 즉시 제공한다.
- rerun 특성으로 인한 입력 손실/버벅임을 방지한다.

## 1) 현재 제약 (사실 기준)
1. `app.py`는 사용자 입력 후 동기 저장/응답 생성 흐름이 메인이다.
2. Streamlit은 상호작용마다 rerun이 발생한다.
3. `components.html` 사용 경험은 이미 존재한다.

## 2) 핵심 원칙
1. 단계 상태를 `st.session_state["flow_stage"]` 단일 키로 고정한다.
2. 드래그 이벤트마다 DB를 갱신하지 않는다.
3. Time-Box는 "로컬 draft"와 "서버 committed"를 분리한다.
4. UI는 "작은 rerun" 중심으로 구성한다.

## 3) 상태머신 설계

### 3.1 단계 정의
- `idle`: 시작 전
- `frog`: 오늘의 Frog 입력
- `timebox_edit`: Time-Box 설계
- `timebox_commit`: 확정 검토
- `focus_running`: Pomodoro 실행
- `reflection`: 회고 입력
- `done`: 일일 세션 종료

### 3.2 전이 규칙
1. `idle -> frog`: `시작` 버튼 클릭.
2. `frog -> timebox_edit`: Frog 유효성 통과.
3. `timebox_edit -> timebox_commit`: 블록 유효성 통과 + 저장.
4. `timebox_commit -> focus_running`: 사용자 확정.
5. `focus_running -> reflection`: 타이머 종료 또는 수동 완료.
6. `reflection -> done`: 필수 3필드 저장 완료.

### 3.3 세션 키
- `flow_stage`
- `active_execution_session_id`
- `timebox_draft_json`
- `timebox_committed_at`
- `focus_preset`
- `focus_end_at`
- `reflection_draft`

## 4) 화면 구성 계약

### 4.1 홈 화면
1. 첫 진입은 `시작` 단일 버튼만 노출.
2. 상단 고정: 현재 단계와 진행률(예: 2/6).
3. 메인 패널: 단계별 단일 카드.
4. 보조 패널: AI Assist (명시 호출형).

### 4.2 단계별 렌더링
1. `frog`: 입력 + 최소 실행단위 선택.
2. `timebox_edit`: 드래그 캘린더 또는 폴백 리스트 에디터.
3. `timebox_commit`: 변경 요약 + 확정 버튼.
4. `focus_running`: 타이머 + 중단/완료 버튼.
5. `reflection`: 3필수 + 자유서술.

## 5) 드래그형 Time-Box 컴포넌트 설계

### 5.1 옵션
1. FullCalendar 기반 커스텀 Streamlit 컴포넌트.
2. Schedule-X 기반 커스텀 Streamlit 컴포넌트.

### 5.2 필수 이벤트 계약
- `on_block_create`
- `on_block_update`
- `on_block_delete`
- `on_range_select`
- `on_validation_error`

### 5.3 입력/출력 payload
- 입력: `date`, `timezone`, `blocks[]`, `preset`, `readonly`.
- 출력: `blocks[]`, `dirty`, `validation_errors[]`, `stats`.

### 5.4 폴백 UI
- 컴포넌트 실패 시 table 기반 block editor로 자동 전환.
- 폴백에서도 동일 JSON 스키마를 유지한다.

## 6) rerun/성능 대응
1. 블록 변경은 `local draft`에만 반영하고 `확정` 때만 DB write.
2. `st.form` 기반 submit으로 이벤트를 묶는다.
3. 무거운 탭(3D, 대시보드)은 lazy rendering.
4. 캐시는 읽기 쿼리에만 적용하고 쓰기 후 명시 invalidate.

## 7) 파일 단위 변경 계획 (프론트)
1. `app.py`
   - 단계 상태머신 추가.
   - 기존 stream-first 진입을 start-first로 변경.
   - `render_timebox_component()` 추가.
2. `icons.py`
   - 신규 단계 아이콘(시작/블록/회고) 추가.
3. 신규: `frontend_components/timebox_component/`
   - React/TS 컴포넌트(FullCalendar 또는 Schedule-X) 구현.

## 8) 실패 모드 및 대응
1. 컴포넌트 로딩 실패:
   - 즉시 폴백 에디터 노출.
2. rerun 중 draft 유실:
   - session_state 초기에 draft 복원.
3. 타임존 오차:
   - UTC 저장 + 로컬 렌더링 분리.
4. 큰 블록 데이터로 렌더링 저하:
   - 당일/익일 범위만 기본 로드.

## 9) Acceptance (컴포넌트)
1. 시작 화면에서 채팅 입력창이 기본 노출되지 않는다.
2. Time-Box 편집 중 페이지 rerun이 발생해도 draft가 유실되지 않는다.
3. 확정 전에는 DB에 실행 블록 write가 발생하지 않는다.
4. 컴포넌트 장애 시 폴백 에디터로 작업이 계속 가능하다.
5. Focus 전환 후 reflection까지 최소 1회 완료 가능하다.

