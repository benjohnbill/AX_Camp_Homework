---
doc_type: redirecting_phase_plan
phase: 2
owner: product_frontend_backend
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_PHASE1_FAST_MVP_2026-03-03.md
  - redirecting/REDIRECTING_ASYNC_WORKER_PLAN_2026-03-03.md
  - redirecting/REDIRECTING_KEEP_KILL_DECISION_2026-03-03.md
change_triggers:
  - phase_scope_changed
  - hierarchy_contract_changed
sunset_condition: Replace when Phase 2 acceptance is passed and Phase 3 starts.
---
# Redirecting Phase 2: Mid MVP (2026-03-03)

체크리스트 문서:
- `redirecting/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md`

## 0) 목표
- Fast MVP 위에서 추천 품질과 회고 품질을 강화한다.
- 3D Universe를 회고 전용으로 실사용 가능한 수준까지 연결한다.

## 1) 핵심 원칙
1. 코어 루프는 계속 non-blocking으로 유지한다.
2. 3D는 편집기가 아니라 회고/전환 보조 도구로 제한한다.
3. 세션 완결 위계(Tier1/2/3)를 운영 지표와 동일하게 적용한다.

## 2) 포함 범위 (Must)
1. Plan-first 강화:
   - Time-Box 리스트 에디터 안정화.
   - Focus-first 종료 후 retro time-box 연결.
2. 비동기 최소 도입:
   - `ai_jobs` 기반 최소 job lifecycle.
   - 자동 태그/유사 세션/다음 행동 추천 비차단 반영.
3. 3D 회고 v1:
   - 최근 7일 고정 타임라인 리플레이.
   - 3-tier 위계 렌더링:
     - Tier 1 `session_completed`
     - Tier 2 `session_interrupted`
     - Tier 3 `supporting_evidence`
   - 종료 CTA: 다음 블록 1개 생성 + Skip.
4. OCR 큐레이션:
   - Reflection에서 세션 관련 evidence 1~2장 노출.
   - 1줄 의미/Skip 확정.

## 3) 제외 범위 (Not in Phase 2)
1. 드래그형 Time-Box 고급 컴포넌트.
2. 3D 고급 물리/동적 재정렬/대규모 필터.
3. AI 자동 Core 후보/자동 승격.
4. 프론트엔드 스택 전환.

## 4) 구현 단위
1. Frontend:
   - 단계 흐름 안정화 + 3D 회고 진입/종료 CTA.
   - Insight 카드에서 비차단 추천 표시.
2. Backend:
   - 최소 async job 상태관리.
   - session insights/week insight 신뢰도 개선.
3. Data:
   - 세션 이벤트 생성 규칙 고정(완결/중단/보조).
   - 로그 투영과 검색 연결 강화.

## 5) 수용 기준
1. AI job 지연/실패 시에도 코어 루프 100% 진행 가능.
2. 3D 회고에서 3-tier 위계와 7일 타임라인이 일관되게 렌더링.
3. 3D 종료 CTA 클릭/Skip 둘 다 정상 동작.
4. OCR evidence가 Reflection 입력 품질을 저해하지 않음.

## 6) 리스크/대응
1. 비동기 적체:
   - 대응: low-priority job 일시 중단, rule-based fallback 유지.
2. 3D 과다복잡화:
   - 대응: 읽기 중심 리플레이와 단일 CTA 규칙 고정.

## 7) Phase Gate (2 -> 3)
1. 3D 회고 주간 사용 패턴과 CTA 전환률 안정화.
2. 큐 적체/실패율이 운영 임계 내 유지.
3. 코어 루프 지표와 추천 지표가 동시에 개선.
