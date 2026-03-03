---
doc_type: redirecting_phase_checklist
phase: 3
track: demo_plus
owner: architecture_platform
authority_level: operational
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_PHASE3_COMPLETE_2026-03-03.md
  - redirecting/REDIRECTING_ROLLOUT_MIGRATION_PLAN_2026-03-03.md
sunset_condition: Replace when architecture transition is completed.
---
# Phase 3 Demo Checklist (2026-03-03)

## 0) Scope Lock
- [ ] Phase 3는 `아키텍처 전환 데모`로 정의한다.
- [ ] Phase 2 경로를 fallback으로 유지한다.
- [ ] 코어 루프 안정성 저하 시 전환을 즉시 중단한다.

## 1) Frontend Platform
- [ ] Streamlit 외 고급 인터랙션 런타임(부분 분리 또는 전면 분리) 결정을 확정한다.
- [ ] Drag Time-Box/고밀도 타임라인을 새 런타임에서 검증한다.
- [ ] 상태 동기화 전략(서버 원본 + 클라이언트 캐시)을 고정한다.

## 2) Backend Platform
- [ ] gateway/worker 분리 배포를 구현한다.
- [ ] idempotency/retry/backoff/queue depth 제어를 운영 기준으로 설정한다.
- [ ] 장애 시 degraded-mode 자동 전환 경로를 검증한다.

## 3) Data/Analytics
- [ ] 3-tier 위계 지표를 장기 분석 파이프라인에 반영한다.
- [ ] 세션 완료율/재시작 전환율/추천 전환율 대시보드를 고정한다.
- [ ] 레거시 로그와 신규 세션 모델의 조회 호환성을 유지한다.

## 4) Streamlit Limitation Exit Criteria
- [ ] 입력 유실/중복 저장 재현 케이스를 전환 후 0건으로 유지한다.
- [ ] 주요 화면 렌더링 지연 체감이 Phase 2 대비 개선됨을 증명한다.
- [ ] Streamlit은 경량 운영 화면으로 축소하거나 완전 이관 결론을 내린다.

## 5) Demo Runbook
- [ ] 시연 시나리오 1: 고급 Time-Box 편집 + 코어 루프 완료.
- [ ] 시연 시나리오 2: worker 장애 유발 -> degraded-mode에서 루프 유지.
- [ ] 시연 시나리오 3: 3D 고급 필터/클러스터 동작 확인.

## 6) Go/No-Go
- [ ] 코어 루프 p95 성능이 Phase 2 대비 악화되지 않음.
- [ ] 운영팀이 롤백 절차를 5분 내 수행 가능.
- [ ] 발표/데모에서 "아키텍처 전환 완료"를 근거와 함께 설명 가능.

