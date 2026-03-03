---
doc_type: redirecting_rollout_plan
owner: control_tower
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_COMPONENT_PLAN_2026-03-03.md
  - redirecting/REDIRECTING_ASYNC_WORKER_PLAN_2026-03-03.md
  - redirecting/REDIRECTING_DATA_API_CONTRACT_2026-03-03.md
change_triggers:
  - phase_gate_changed
  - feature_flag_changed
sunset_condition: Replace after redirecting rollout is completed or cancelled.
---
# Redirecting Rollout/Migration Plan (2026-03-03)

## 0) 목표
- 기존 서비스 다운타임 없이 단계적으로 redirecting을 적용한다.
- 실패 시 즉시 기존 UX로 롤백 가능한 구조를 유지한다.

## 1) 전환 전략
1. Big-bang 전환 금지.
2. Feature flag 기반 병행 운영.
3. `새 플로우 10% -> 30% -> 100%` 점진 확대.

## 2) Feature flags
- `FF_EXECUTION_HOME_ENABLED`
- `FF_TIMEBOX_COMPONENT_ENABLED`
- `FF_ASYNC_WORKER_ENABLED`
- `FF_REFLECTION_REQUIRED`
- `FF_ASSIST_SECONDARY_MODE`

## 3) 단계별 계획

### Phase 0: 준비
1. 신규 테이블/인덱스 migration.
2. read-only 조회 API 준비.
3. 기존 로그 파이프라인 영향 분석.

### Phase 1: 백엔드 선행
1. execution session API 추가.
2. ai_jobs enqueue 로직 추가.
3. worker 프로세스 배포 (flag off).

### Phase 2: UI 병행
1. 홈에서 새 플로우 진입 버튼만 추가.
2. 구 Stream을 기본 유지한 채 새 플로우 내부 QA.
3. Time-Box 컴포넌트 장애 시 폴백 검증.
4. 이 단계에서는 "채팅 기본 비노출"을 요구하지 않는다 (최종 Acceptance 대상 아님).

### Phase 3: 제한 공개
1. 내부 사용자 + 소수 베타(10%).
2. 핵심 지표 모니터링.
3. blocker 발생 시 즉시 flag rollback.

### Phase 4: 기본 전환
1. 홈 기본을 새 플로우로 변경.
2. Stream은 Assist 섹션으로 이동.
3. 기존 플로우는 백업 경로로 유지.
4. 이 단계부터 최종 Acceptance(홈 기본에서 채팅 비노출)를 적용한다.

## 4) 검증 체크리스트
1. 세션 생성/커밋/회고 API 정상.
2. 포커스 종료 후 job enqueue 정상.
3. worker down 상황에서 코어 플로우 유지.
4. Time-Box 컴포넌트 오류 시 폴백 동작.
5. 기존 `/v1/narrative` 경로 비회귀.

## 5) 성공 지표 (초기 2주)
1. 새 플로우 시작률 >= 60%.
2. Focus 완료율 >= 45%.
3. Reflection 작성률 >= 70%.
4. D1 재방문률 +10%p 개선.
5. Stream 진입 비율은 줄어도 세션 완료율은 상승.

## 6) 롤백 조건
1. 새 플로우 오류율 급증.
2. 완료율이 기존 대비 유의미하게 하락.
3. worker 적체로 p95 지연 임계 초과.

## 7) 롤백 절차
1. `FF_EXECUTION_HOME_ENABLED=false`.
2. `FF_TIMEBOX_COMPONENT_ENABLED=false`.
3. `FF_ASYNC_WORKER_ENABLED=false`.
4. 기존 stream/chronos 경로로 즉시 복귀.

## 8) 구현 단위 태스크 (파일 기준)
1. `sql/` 신규 migration 파일 추가.
2. `db_manager_postgres.py` 신규 repository 함수 추가.
3. `gateway_fastapi.py` session/job endpoint 추가.
4. 신규 `worker_ai.py` + `worker_jobs.py` 추가.
5. `app.py` 단계형 UI 및 flag 분기 추가.
6. `tests/` API/worker/component smoke 테스트 추가.

## 9) 운영 핸드오프
1. CT가 phase gate 문서/결과 JSON을 고정한다.
2. backend/frontend/android lane이 각자 evidence를 제출한다.
3. `redirecting` 폴더의 문서를 canonical implementation prompt로 사용한다.
