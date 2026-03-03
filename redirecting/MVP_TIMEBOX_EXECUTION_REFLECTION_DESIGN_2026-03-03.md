---
doc_type: product_mvp_design
owner: product
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - C:/Users/LG/OneDrive/바탕 화면/Life_System/01_Active_Projects/08_AX 코딩 아카데미/Project_Narrative_Loop/260227_실제_기획안.md
  - docs/PROJECT_BRIEF_FOR_HUMAN_2026-02-25.md
  - app.py
  - narrative_logic.py
  - gateway_fastapi.py
change_triggers:
  - home_ia_changed
  - loop_contract_changed
  - metrics_contract_changed
sunset_condition: Replace when v1 product IA and data contract are frozen.
---
# MVP 설계서: Time-Box 기반 실행-성찰 비서 (2026-03-03)

## 0) 목적
- 현재 `Stream(채팅)` 중심 UX를 `행동 설계 + 실행 + 성찰 데이터 루프` 중심 UX로 전환한다.
- 기존 코드 자산(Chronos, 하이브리드 검색, Universe 분석, 키워드 시스템)을 폐기하지 않고 재배치한다.
- 제품 포지셔닝을 `AI 채팅앱`이 아니라 `서사 기반 실행 시스템`으로 명확히 한다.

## 1) 결론 요약 (의사결정)
1. 메인 화면은 `Plan Start`와 `Focus Now`를 동등하게 노출한다.
2. 시작 경로는 2개다: `Plan-first`(Frog -> Time-Box)와 `Focus-first`(즉시 몰입).
3. 실행 모드는 `Pomodoro(기본 50/10, 25/5 선택 가능)`를 공통으로 사용한다.
4. 두 경로 모두 종료 시 `Reflection` 구조화 입력으로 합류한다.
5. `자유 감상`은 허용하되, 기본은 journal로 저장하고 세션 승격 시 코어 지표에 반영한다.
6. `Core 승격`은 v1에서 사용자 수동 확정만 허용한다 (AI 자동 승격 없음).
7. 채팅(Stream)은 기본값에서 제외하고 `Assist 도구`로 하향 배치한다.

## 2) 기존 대비 개선점
- 기존: 빈 채팅 입력 중심 -> 신규: 시작 버튼 중심.
- 기존: 기록이 남아도 행동 전환이 약함 -> 신규: 계획 확정 -> 실행 -> 성찰의 폐루프.
- 기존: 대시보드가 관찰 중심 -> 신규: 다음 행동 결정을 위한 추천 중심.
- 기존: AI가 전면 대화 중심 -> 신규: 백그라운드 분석 + 필요 시 호출.

## 3) 비즈니스/경쟁 관점

### 3.1 To-Do 대비 차별점
- To-Do는 체크 완료가 끝점이지만, 본 제품은 `왜/어떻게/무엇을 배웠는지`까지 저장한다.
- 단순 할일 리스트가 아니라 `실행 맥락 + 성찰`이 누적되는 개인 데이터 자산을 만든다.

### 3.2 ChatGPT 대비 차별점
- ChatGPT는 대화 생성이 핵심이고, 본 제품은 `행동 결과 데이터 축적`이 핵심이다.
- 사용자가 매일 축적하는 것은 채팅 로그가 아니라 `계획-실행-회고 단위 세션`이다.

### 3.3 Obsidian 대비 차별점
- Obsidian은 지식 연결 중심, 본 제품은 시간축 행동 연결 중심.
- 수동 [[키워드]]와 자동 키워드를 결합해 `생각의 연결`과 `행동의 연결`을 동시에 확보한다.

## 4) 기존 기획 의도와의 정합성
- 기획 핵심 루프 `기록 -> 저장 -> 재조회 -> 회고`와 완전히 정합적이다.
- 변경점은 기능 제거가 아니라 진입점 재배치다.
  - `Stream`: 메인 -> 보조.
  - `Chronos`: 보조 -> 메인 실행 엔진.
  - `Universe`: 사후 관찰 -> 다음 행동 추천 근거.

## 5) 사용자 플로우 (MVP Contract: Duale Struktur)

### 5.0 시작 경로
- `Plan-first`: 의도된 계획을 먼저 선언하고 실행으로 진입.
- `Focus-first`: 사전 계획 없이 즉시 몰입하고 종료 후 의미를 역구성.
- `Journal`: 타이머 없이 자유 감상을 먼저 기록하고 필요 시 세션으로 승격.

### Plan-first Step 1) Frog 선택
- 질문: `오늘 반드시 해야 할 일 1개는 무엇인가요?`
- 입력: 텍스트 1개 (필수), 최소 실행 단위(예: 15분) 선택.
- 저장: `frog_title`, `frog_why`(선택), `difficulty_score`(선택).

### Plan-first Step 2) Time-Box 설계
- 사용자는 하루 블록을 직접 배치한다.
- 각 블록: `title`, `start`, `end`, `goal`, `why`, `inbox_note`.
- 옆 패널 AI Assist:
  - 일정 충돌 점검.
  - 과거 유사 블록 기반 현실성 경고.
  - 집중 블록 추천.

### Plan-first Step 3) 계획 확정 (Commit)
- `확정` 시 세션 계획 버전을 고정한다.
- 사용자는 선택적으로 [[키워드]]를 수동 부여한다.
- 백그라운드 AI는 자동 키워드 3개를 추출한다.
- 저장 구조:
  - `manual_tags[]`
  - `auto_tags[]`
  - `plan_version_status = committed`

### 공통 Step 4) Focus 실행 (Pomodoro)
- Time-Box -> Focus 모드 전환 시 확인 모달 노출.
- 기본 룰: `50/10` (옵션: 25/5, 90/20).
- 실행 중에는 계획 수정 비권장.
- MVP에서 수정은 `최대 1회 재진입`까지 허용.

### 공통 Step 5) Reflection 기록
- 타이머 종료 후 반드시 회고 입력:
  1. 무엇이 잘 됐나
  2. 무엇이 막혔나
  3. 다음 행동 1개
- 자유서술 추가 칸 제공.
- 저장 데이터는 다음 추천/검색/Universe 분석의 핵심 입력으로 사용.

### Focus-first 보강 Step) Retro Time-Box
- Focus-first로 시작한 세션은 Focus 종료 후 선택적으로 `retro block`을 입력한다.
- retro block은 방금 수행한 활동의 정체성을 사후 부여하는 최소 블록이다.
- 사용자는 Skip할 수 있으나, 작성 시 주간 분석/리플레이 품질이 올라간다.

### OCR Evidence Step) 증거 캡처/큐레이션
- 사용자는 Focus 중/종료 직후 선택적으로 사진을 업로드할 수 있다.
- 이미지는 즉시 영구 저장(스토리지 + 메타)되며, OCR/요약은 비동기로 처리한다.
- 연결 우선순위: `active focus session -> today latest session -> inbox`.
- Reflection 시점에는 세션 관련 이미지 1~2장만 큐레이션하여 제시한다.
- 사용자는 각 이미지에 `1줄 의미`를 남기거나 `Skip`할 수 있다.

### Journal Step) 자유 감상 및 세션 승격
- 타이머 없이 자유 감상을 먼저 기록할 수 있다.
- journal 저장 시 `next_action` 1줄을 필수로 둔다.
- 사용자는 이후 `세션으로 승격`을 눌러 execution session으로 전환할 수 있다.
- 승격 전 journal은 코어 완료율 지표에서 분리 집계한다.

### Core Step) Core 수동 승격
- Core는 `execution session` 또는 `journal`에서 사용자가 직접 승격 버튼으로 확정한다.
- v1에서 AI는 Core 후보를 자동 생성/자동 확정하지 않는다.
- AI는 요약/태깅 보조만 수행하며, 최종 승격 결정권은 사용자에게 둔다.

## 6) UI 정보구조 (새 홈 IA)

### 6.1 홈 카드 (순서 고정)
1. `Start Today` (Plan Start / Focus Now 동등 CTA)
2. `오늘의 Frog + Time-Box Planner`
3. `Focus Now`
4. `Session Reflection`
5. `Journal` (자유 감상, 승격 가능)
6. `My Week Insight` (Universe 요약 카드)
7. `Assist` (구 Stream)

### 6.2 원칙
- 첫 화면에서 자유 채팅창을 보이지 않는다.
- 모든 카드 끝에는 행동 버튼이 있어야 한다.
- 대시보드는 그래프보다 `다음 1개 행동` CTA를 우선 표기한다.

## 7) AI 동작 정책 (UI/백엔드 분리)

### 7.1 기본 정책
- UI에서 AI는 조용해야 한다 (명시 요청 시 전면 응답).
- 백엔드 AI는 상시 동작한다 (키워드, 유사도, 패턴 탐지).

### 7.2 백그라운드 사용처
- 자동 키워드 추출.
- 유사 세션 탐색(성공/실패 패턴).
- 주간 요약 및 다음 실행 제안.
- 대시보드 추천 문구 생성.

### 7.3 전면 호출 트리거
- 사용자가 `AI로 일정 개선` 클릭.
- 사용자가 `이번 주 실패 패턴 분석` 클릭.
- 사용자가 `다음 실행 1개 추천` 클릭.

## 8) 데이터 모델 변경안 (MVP)

### 8.1 신규 테이블: `execution_sessions`
- `id` (PK, uuid)
- `created_at`
- `session_date`
- `entry_mode` (`plan|focus_now|journal_promoted`)
- `frog_title`
- `frog_why`
- `plan_json` (time-box blocks)
- `plan_status` (`draft|committed|running|done`)
- `focus_preset` (`25_5|50_10|90_20`)
- `focus_total_minutes`
- `manual_tags` (json/text)
- `auto_tags` (json/text)
- `reflection_good`
- `reflection_hard`
- `reflection_next_action`
- `reflection_free_text`

### 8.2 신규 테이블: `image_events` (OCR 증거)
- `id` (PK, uuid)
- `session_id` (nullable, FK)
- `storage_uri`
- `thumbnail_uri`
- `capture_source` (`android|web`)
- `ocr_status` (`queued|running|succeeded|failed`)
- `ocr_text`
- `ai_summary`
- `user_meaning`
- `link_status` (`linked|inbox|skipped`)
- `created_at`
- `updated_at`

### 8.3 신규 테이블: `journal_entries`
- `id` (PK, uuid)
- `entry_text`
- `next_action`
- `manual_tags`
- `auto_tags`
- `promoted_session_id` (nullable)
- `created_at`
- `updated_at`

### 8.4 logs와 연결
- `logs`는 유지한다.
- `execution_sessions` 종료 시 주요 텍스트를 `Log`로 투영 저장(검색 호환 목적).
- `linked_constitutions`는 기존 필드를 재사용한다.

### 8.5 태그 정책
- 수동 [[키워드]]: 사용자 의도(의미).
- 자동 키워드: 모델 추출(일관성).
- 검색시 수동 태그 가중치 우선.

## 9) API 변경안 (gateway_fastapi 기준)

### 9.1 신규/변경 엔드포인트
- `POST /v1/execution/session/start`
- `POST /v1/execution/session/{session_id}/timebox/retro`
- `POST /v1/execution/session/{session_id}/frog`
- `POST /v1/execution/session/{session_id}/timebox/draft`
- `POST /v1/execution/session/{session_id}/commit`
- `POST /v1/execution/session/{session_id}/focus/start`
- `POST /v1/execution/session/{session_id}/focus/end`
- `POST /v1/execution/session/{session_id}/reflect`
- `POST /v1/execution/session/{session_id}/evidence/upload`
- `POST /v1/journal/entry`
- `POST /v1/journal/{entry_id}/promote`
- `POST /v1/core/promote` (source_type: `execution_session|journal`)
- `GET /v1/execution/session/{session_id}/insights`
- `GET /v1/execution/session/today`
- `GET /v1/execution/insight/week` (v1-lite: rule-based 우선, AI 결과는 비차단 결합)

### 9.2 기존 엔드포인트 유지
- `/v1/ocr/ingest`, `/v1/narrative/refine`, `/v1/narrative/vision` 유지.
- `/v1/narrative`는 하위 호환으로 유지하되 신규 UI는 세션 단위 API 우선 사용.

## 10) 모드 재배치 (app.py 기준)
- 유지 모드: `stream`, `desk`, `chronos`, `control`, `universe`.
- UX 우선순위 변경:
  1. `chronos` + 신규 planner 뷰를 홈 기본으로 결합.
  2. `stream`은 Assist 섹션으로 이동.
  3. `universe`는 주간/패턴 분석 요약 카드와 연결.

## 11) 대시보드 설계 원칙

### 11.1 사용자 화면 지표 (보여줄 것)
- 이번 주 집중 세션 수.
- 약속 대비 완료율.
- 반복 방해요인 Top 3.
- 다음 주 1개 규칙 추천.

### 11.2 내부 운영 지표 (제품팀용)
- 세션 시작률.
- 포모도로 완료율.
- D1/D7 재방문률.
- Reflection 작성률.
- 추천 CTA 클릭 후 실제 실행 전환률.

## 12) MVP 범위/비범위

### 12.1 MVP 포함
- Plan-first와 Focus-first 동등 진입.
- Frog 1개 선정(Plan-first 경로).
- Time-Box 편집/확정 + Focus-first retro block.
- Focus 실행/완료.
- Reflection 저장.
- Journal 자유 감상 + 세션 승격.
- Core 수동 승격(사용자 직접 확정).
- 자동/수동 키워드.
- OCR evidence 업로드 + 비동기 처리 + reflection 큐레이션.
- 주간 추천 1개.

### 12.2 MVP 제외
- 다중 생산성 방법 전체 지원(3/3/3, Ivy Lee, GTD, Matrix 등).
- 복잡한 협업/소셜.
- 고급 캘린더 외부 연동.
- AI 자동 Core 승격/자동 Core 제안 큐.

## 13) 리스크 및 대응
1. 선택마비 리스크:
   - 대응: 방법론 다중 노출 금지, 기본 1개 플로우 고정.
2. 타이머 앱 전락 리스크:
   - 대응: 종료 회고 + 다음 행동 입력 강제.
3. 채팅 복귀 리스크:
   - 대응: Stream을 보조로 강등, 홈에서 숨김.
4. 데이터 품질 리스크:
   - 대응: 회고 입력 최소 필드 3개 고정 + 자동 태그 보강.

## 14) 수용 기준 (Acceptance)
1. 홈 첫 진입에서 채팅창이 기본 노출되지 않는다.
2. 사용자는 3분 내 `Plan-first` 또는 `Focus-first`로 실행 시작이 가능해야 한다.
3. Focus 종료 후 Reflection 3필드 저장 전에는 세션 완료 처리되지 않는다.
4. OCR 업로드는 OCR 성공 여부와 무관하게 저장/연결되어야 한다.
5. Reflection에서 세션 관련 evidence 1~2개를 제시하고 Skip 경로를 제공해야 한다.
6. Journal은 저장 가능하되, 승격 전에는 코어 완료율 지표에서 분리 집계된다.
7. 저장된 세션 데이터가 Universe 요약 카드 1개 이상에 반영된다.
8. Assist(Stream)는 접근 가능하지만 기본 플로우를 방해하지 않는다.

## 15) 구현 순서 (권장)
1. 데이터 스키마/세션 API 추가.
2. 홈 IA 교체 + Frog/Planner/Focus/Reflection 카드 구현.
3. 기존 Chronos 재사용 연결.
4. 자동 키워드/유사도 추천을 세션 기반으로 연결.
5. Universe 요약 카드 1차 버전 연결.

## 16) 최종 포지셔닝 문장
`Narrative_Loop는 생각을 기록하는 앱이 아니라, 기록을 내일의 실행으로 바꾸는 서사 기반 실행 비서다.`
