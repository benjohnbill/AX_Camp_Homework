---
doc_type: redirecting_data_api_contract
owner: backend
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - sql/1_create_logs_table.sql
  - sql/2_create_system_tables.sql
  - gateway_fastapi.py
change_triggers:
  - schema_changed
  - endpoint_changed
sunset_condition: Replace when v1 migration SQL and OpenAPI spec are finalized.
---
# Redirecting Data/API Contract (2026-03-03)

## 0) 목적
- Time-Box 기반 플로우를 위한 최소 데이터 모델과 API 계약을 고정한다.
- 기존 `logs` 호환성을 유지하며 신규 세션 모델을 추가한다.

## 1) 데이터 모델 (MVP)

### 1.1 execution_sessions
```sql
CREATE TABLE IF NOT EXISTS execution_sessions (
    id TEXT PRIMARY KEY,
    session_date DATE NOT NULL,
    flow_stage TEXT NOT NULL DEFAULT 'idle',
    entry_mode TEXT NOT NULL DEFAULT 'plan',
    frog_title TEXT,
    frog_why TEXT,
    plan_status TEXT NOT NULL DEFAULT 'draft',
    focus_preset TEXT NOT NULL DEFAULT '50_10',
    focus_total_minutes INTEGER NOT NULL DEFAULT 0,
    focus_started_at TIMESTAMPTZ,
    focus_ended_at TIMESTAMPTZ,
    reflection_good TEXT,
    reflection_hard TEXT,
    reflection_next_action TEXT,
    reflection_free_text TEXT,
    manual_tags JSONB DEFAULT '[]'::jsonb,
    auto_tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 1.2 execution_blocks
```sql
CREATE TABLE IF NOT EXISTS execution_blocks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES execution_sessions(id) ON DELETE CASCADE,
    block_title TEXT NOT NULL,
    block_goal TEXT,
    block_why TEXT,
    inbox_note TEXT,
    starts_at TIMESTAMPTZ NOT NULL,
    ends_at TIMESTAMPTZ NOT NULL,
    order_index INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 1.3 image_events
```sql
CREATE TABLE IF NOT EXISTS image_events (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES execution_sessions(id) ON DELETE SET NULL,
    storage_uri TEXT NOT NULL,
    thumbnail_uri TEXT,
    capture_source TEXT NOT NULL DEFAULT 'web',
    ocr_status TEXT NOT NULL DEFAULT 'queued',
    ocr_text TEXT,
    ai_summary TEXT,
    user_meaning TEXT,
    link_status TEXT NOT NULL DEFAULT 'inbox',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 1.4 journal_entries
```sql
CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY,
    entry_text TEXT NOT NULL,
    next_action TEXT NOT NULL,
    manual_tags JSONB DEFAULT '[]'::jsonb,
    auto_tags JSONB DEFAULT '[]'::jsonb,
    promoted_session_id TEXT REFERENCES execution_sessions(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 1.5 core_entries
```sql
CREATE TABLE IF NOT EXISTS core_entries (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    promoted_by TEXT NOT NULL,
    promoted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 1.6 ai_jobs
```sql
CREATE TABLE IF NOT EXISTS ai_jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    idempotency_key TEXT NOT NULL,
    run_after TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_jobs_idempotency_key
ON ai_jobs (idempotency_key);
```

## 2) 기존 logs와 연결
1. execution_sessions 완료 시 요약 텍스트를 `logs(meta_type='Log')`에 투영 저장.
2. 기존 하이브리드 검색은 `logs`를 그대로 사용.
3. 고급 검색에서는 session_id를 metadata로 연결 가능하게 확장.

## 3) API 계약

### 3.1 Start
- `POST /v1/execution/session/start`
- request:
```json
{
  "session_date": "2026-03-03",
  "entry_mode": "plan"
}
```
- response:
```json
{
  "status": "ok",
  "session_id": "sess_...",
  "flow_stage": "frog",
  "entry_mode": "plan"
}
```

`entry_mode`:
- `plan` -> `flow_stage=frog`
- `focus_now` -> `flow_stage=focus_running`

### 3.2 Save Frog
- `POST /v1/execution/session/{session_id}/frog`
- request:
```json
{
  "frog_title": "오늘 핵심 제안서 초안 완성",
  "frog_why": "이번 주 마감 리스크 해소"
}
```

### 3.3 Draft Time-Box
- `POST /v1/execution/session/{session_id}/timebox/draft`
- request:
```json
{
  "blocks": [
    {
      "id": "blk_1",
      "title": "제안서 구조 작성",
      "goal": "섹션 5개 완료",
      "why": "논리 구조 고정",
      "inbox_note": "[[proposal]] [[deadline]]",
      "starts_at": "2026-03-03T09:00:00Z",
      "ends_at": "2026-03-03T10:00:00Z",
      "order_index": 1
    }
  ],
  "manual_tags": ["proposal", "deadline"]
}
```

### 3.4 Commit Plan
- `POST /v1/execution/session/{session_id}/commit`
- 동작:
  - `plan_status=committed`
  - `flow_stage=focus_running`
  - `auto_tag_extraction` job enqueue

### 3.5 Focus Start/End
- `POST /v1/execution/session/{session_id}/focus/start`
- `POST /v1/execution/session/{session_id}/focus/end`
- end 시 job enqueue:
  - `similar_session_linking`
  - `next_action_recommendation`

### 3.5b Retro Time-Box
- `POST /v1/execution/session/{session_id}/timebox/retro`
- 목적:
  - Focus-first 경로에서 종료 후 사후 블록을 저장한다.
  - 저장 또는 Skip 상태를 모두 허용한다.

### 3.6 Reflection
- `POST /v1/execution/session/{session_id}/reflect`
- request:
```json
{
  "reflection_good": "첫 블록에서 핵심 논리를 빠르게 정리했다.",
  "reflection_hard": "중간에 메신저 알림으로 집중이 깨졌다.",
  "reflection_next_action": "내일 오전 9시 알림 차단 후 50분 집중 블록 시작",
  "reflection_free_text": "...",
  "evidence_links": [
    {
      "image_event_id": "img_1",
      "decision": "linked",
      "user_meaning": "핵심 도표 완성 시점"
    },
    {
      "image_event_id": "img_2",
      "decision": "skipped"
    }
  ]
}
```

### 3.7 Insight
- `GET /v1/execution/session/{session_id}/insights`
- response:
```json
{
  "status": "ok",
  "job_status": {
    "auto_tag_extraction": "succeeded",
    "next_action_recommendation": "running"
  },
  "auto_tags": ["proposal", "focus", "deadline"],
  "insights": {
    "similar_pattern": "...",
    "next_action": "..."
  }
}
```

### 3.8 Today Session
- `GET /v1/execution/session/today`
- 목적:
  - 오늘 세션 진입/복귀를 빠르게 지원한다.
  - 오늘 세션이 없으면 생성 없이 `not_found`를 반환한다.
- response (exists):
```json
{
  "status": "ok",
  "session": {
    "id": "sess_...",
    "session_date": "2026-03-03",
    "flow_stage": "timebox_edit",
    "plan_status": "draft"
  }
}
```
- response (not found):
```json
{
  "status": "not_found"
}
```

### 3.9 Week Insight (v1-lite)
- `GET /v1/execution/insight/week?anchor_date=2026-03-03`
- v1-lite 원칙:
  - 기본은 rule-based 집계만 제공한다.
  - AI worker 결과가 있으면 `insight_source=ai`, 없으면 `insight_source=rule`로 반환한다.
  - AI 결과 대기 때문에 응답을 지연시키지 않는다.
- response:
```json
{
  "status": "ok",
  "week_range": {
    "start": "2026-02-25",
    "end": "2026-03-03"
  },
  "insight_source": "rule",
  "metrics": {
    "sessions_started": 5,
    "focus_completed": 3,
    "reflection_written": 4
  },
  "top_blockers": ["interruptions", "context_switch"],
  "recommended_next_action": "오전 첫 블록에서 메신저 알림 차단"
}
```

### 3.10 Job Status
- `GET /v1/jobs/{job_id}`
- 목적:
  - 비동기 작업 상태를 polling으로 조회한다.
  - UI는 job 완료를 기다리지 않고, 상태만 확인해 점진 반영한다.
- response:
```json
{
  "status": "ok",
  "job": {
    "id": "job_...",
    "job_type": "next_action_recommendation",
    "entity_type": "execution_session",
    "entity_id": "sess_...",
    "state": "running",
    "attempt": 1,
    "max_attempts": 3,
    "updated_at": "2026-03-03T10:10:00Z",
    "last_error": null
  }
}
```

### 3.11 Evidence Upload (OCR 비동기)
- `POST /v1/execution/session/{session_id}/evidence/upload`
- 동작:
  - 원본 이미지는 스토리지에 즉시 저장한다.
  - `image_events`에 메타를 기록하고 OCR job을 enqueue한다.
  - OCR 완료를 기다리지 않고 `accepted`를 반환한다.

### 3.12 Journal
- `POST /v1/journal/entry`
- request:
```json
{
  "entry_text": "오늘 계획 없이 바로 몰입했는데...",
  "next_action": "내일 오전 같은 시간 25분 몰입 재시도",
  "manual_tags": ["focus", "experiment"]
}
```
- `POST /v1/journal/{entry_id}/promote`
- 목적:
  - 자유 감상을 execution session으로 승격한다.
  - 승격된 시점부터 코어 루프 지표 집계에 포함한다.

### 3.13 Core Promote (수동 승격 전용)
- `POST /v1/core/promote`
- request:
```json
{
  "source_type": "execution_session",
  "source_id": "sess_...",
  "title": "아침 첫 50분은 알림 차단 후 시작",
  "body": "이번 주 실행/회고에서 반복 확인된 개인 실행 원칙",
  "promoted_by": "user_..."
}
```
- 규칙:
  - v1에서 Core 승격은 사용자 직접 호출만 허용한다.
  - AI/worker는 Core 자동 승격을 수행하지 않는다.
  - AI는 요약/태그 보조만 제공한다.

## 4) 검증 규칙
1. `frog_title`은 비어 있으면 안 된다.
2. Time-Box 블록은 `starts_at < ends_at`이어야 한다.
3. Reflection 3필수(`good/hard/next_action`)가 모두 채워져야 `done`.
4. 동일 `idempotency_key` job은 한 번만 큐에 들어간다.
5. Journal 저장 시 `next_action`은 비어 있으면 안 된다.
6. OCR 실패/지연이어도 image_events 생성과 session 진행은 성공해야 한다.
7. Core 승격은 수동 API 호출로만 생성되어야 한다.

## 5) 캐시/저장 경계
1. 읽기 캐시는 허용.
2. 쓰기 API는 항상 DB 트랜잭션 우선.
3. Commit/Reflect 후 관련 읽기 캐시는 무효화.
4. `today/week` 조회는 캐시 가능하되 TTL은 짧게 유지한다.

## 6) backward compatibility
1. 기존 `/v1/narrative`, `/v1/ocr/ingest`는 유지.
2. 기존 앱이 새 endpoint를 사용하지 않아도 동작해야 한다.
3. 신규 UI는 세션 endpoint를 우선 사용한다.
4. 기존 OCR ingest는 `image_events` 파이프라인에 점진 연결한다.
5. 기존 Core/LegacyLog 데이터는 조회 호환성을 유지하되, 신규 Core 생성은 `/v1/core/promote`로 단일화한다.
