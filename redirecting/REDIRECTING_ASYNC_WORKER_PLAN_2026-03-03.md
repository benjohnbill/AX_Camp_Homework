---
doc_type: redirecting_async_worker_plan
owner: backend
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - narrative_logic.py
  - gateway_fastapi.py
  - db_manager_postgres.py
change_triggers:
  - ai_job_contract_changed
  - ingestion_latency_changed
sunset_condition: Replace when async worker architecture is production-ready.
---
# Redirecting Async Worker Plan (2026-03-03)

## 0) 목표
- UI 요청 경로에서 무거운 AI 작업을 분리한다.
- Streamlit rerun 경로를 가볍게 유지한다.
- `계획 확정/포커스 종료/회고 저장` 이벤트에서 비동기 분석 결과를 안정적으로 적재한다.
- `OCR evidence 업로드`와 `journal 승격 보조정규화`도 비동기로 처리한다.

## 1) 현재 상태 (사실)
1. `save_log`는 동기 경로다.
2. `/v1/narrative`도 동기 저장이다.
3. 별도 worker 프로세스 서비스는 현재 배포 구성에 없다.

## 2) 목표 아키텍처
1. API/UI는 "명령 수집 + job enqueue"만 수행.
2. 워커는 "job claim -> AI 처리 -> 결과 저장"만 수행.
3. 대시보드/Assist는 "결과 조회"만 수행.

## 3) 워커 토폴로지

### 3.1 최소 구성 (MVP)
1. `gateway_fastapi` (API)
2. `ai_worker` (독립 프로세스)
3. `ai_jobs` 테이블 (큐 역할)

### 3.2 권장 실행
- Render 기준:
  - Web service: existing gateway
  - Background worker: new worker service

## 4) Job 계약

### 4.1 Job type
- `auto_tag_extraction`
- `similar_session_linking`
- `weekly_insight_summary`
- `next_action_recommendation`
- `ocr_parse_and_summary`
- `journal_auto_tagging`

### 4.2 상태
- `queued`
- `running`
- `succeeded`
- `failed`
- `stale`
- `cancelled`

### 4.3 필수 필드
- `job_id`
- `job_type`
- `entity_type` (`execution_session|log`)
- `entity_id`
- `payload_json`
- `status`
- `attempt`
- `max_attempts`
- `idempotency_key`
- `created_at`
- `updated_at`
- `run_after`
- `last_error`

## 5) 트리거 정책
1. `session_commit` 완료 시:
   - `auto_tag_extraction`
2. `focus_end` 완료 시:
   - `similar_session_linking`
   - `next_action_recommendation`
3. `reflection_saved` 완료 시:
   - `weekly_insight_summary` (하루 1회 제한)
4. `evidence_uploaded` 완료 시:
   - `ocr_parse_and_summary`
5. `journal_saved` 완료 시:
   - `journal_auto_tagging`

## 6) 안정성 설계

### 6.1 Idempotency
- 동일 `entity_id + job_type + source_hash` 조합은 중복 실행 금지.

### 6.2 Retry
- 지수 백오프: 30s, 120s, 600s.
- `max_attempts` 초과 시 `failed` 고정.

### 6.3 Stale 판정
- 엔티티 최신 hash가 job payload hash와 다르면 `stale`.

### 6.4 Locking
- claim 시 `FOR UPDATE SKIP LOCKED` (postgres).
- sqlite fallback은 단일 worker 모드로 제한.

## 7) 보안/비용 정책
1. OpenAI 호출은 worker에서만 수행.
2. 모델/토큰 상한은 job type별로 다르게 설정.
3. PII 가능 필드는 마스킹 후 로그 저장.
4. worker 로그에 원문 텍스트 전체 출력 금지.

## 8) API 경계
1. UI/API는 AI 결과를 기다리지 않는다.
2. 응답은 `accepted + job_ref`를 반환.
3. 조회 엔드포인트:
   - `GET /v1/jobs/{job_id}`
   - `GET /v1/execution/session/{id}/insights`
4. OCR 실패/지연이어도 evidence upload API는 성공으로 반환한다.

## 9) 파일 단위 변경 계획 (백엔드)
1. `gateway_fastapi.py`
   - enqueue 엔드포인트 추가.
   - 기존 동기 AI 경로 최소화.
2. 신규: `worker_ai.py`
   - claim/process/update 루프.
3. 신규: `worker_jobs.py`
   - job repository와 lifecycle 함수.
4. `db_manager_postgres.py`
   - `ai_jobs` CRUD.
5. `narrative_logic.py`
   - 동기 호출 함수를 worker-friendly 함수로 분리.

## 10) 운영 관측
1. `job_success_rate`
2. `p95_job_latency`
3. `queue_depth`
4. `stale_rate`
5. `retry_rate`

## 11) 장애 대응
1. worker down:
   - UI는 degraded banner 노출.
   - core flow(Frog->Focus->Reflection)는 계속 허용.
2. OpenAI 장애:
   - job status `failed` 기록 + 재시도.
   - 기본 rule-based 추천으로 임시 대체.
3. queue 적체:
   - insight generation 주기 축소.
   - low-priority job 일시 중단.
4. OCR provider 장애:
   - image event는 `uploaded/inbox` 유지.
   - reflection 큐레이션은 원본/썸네일만으로 동작.

## 12) Acceptance (워커)
1. 핵심 사용자 플로우에서 AI 응답 대기 없이 단계 전환이 가능해야 한다.
2. 동일 이벤트 중복 전송 시 분석 결과가 중복 생성되지 않아야 한다.
3. worker 장애 상황에서도 세션 저장/타이머/회고는 정상 동작해야 한다.
4. job 처리 결과가 session insight 조회에 반영되어야 한다.
5. OCR 지연/실패가 세션 진행/회고 완료를 막지 않아야 한다.

