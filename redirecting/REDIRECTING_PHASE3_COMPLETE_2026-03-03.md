---
doc_type: redirecting_phase_plan
phase: 3
owner: architecture_product
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_PHASE2_MID_MVP_2026-03-03.md
  - redirecting/REDIRECTING_ROLLOUT_MIGRATION_PLAN_2026-03-03.md
  - redirecting/REDIRECTING_CT_MASTER_PROMPT_2026-03-03.md
change_triggers:
  - frontend_architecture_changed
  - worker_topology_changed
sunset_condition: Replace when complete architecture is implemented and stabilized.
---
# Redirecting Phase 3: Complete (2026-03-03)

체크리스트 문서:
- `redirecting/REDIRECTING_PHASE3_DEMO_CHECKLIST_2026-03-03.md`

## 0) 목표
- Streamlit 한계를 구조적으로 우회/해결하는 완성형 아키텍처로 전환한다.
- 코어 루프 안정성은 유지하면서 고급 상호작용과 운영 신뢰성을 확보한다.

## 1) 핵심 원칙
1. 코어 루프를 절대 깨지 않으면서 고급 기능을 증설한다.
2. 프론트엔드 인터랙션 고도화는 전용 런타임으로 분리한다.
3. 비동기/큐/관측은 운영 SLO 기반으로 관리한다.

## 2) 포함 범위 (Must)
1. 프론트엔드 고도화:
   - Streamlit + 전용 UI(또는 전면 분리) 결정 후 실행.
   - 드래그형 Time-Box, 고밀도 타임라인 인터랙션.
2. 비동기 아키텍처 완성:
   - gateway / worker 분리 배포.
   - 멱등성/재시도/백오프/적체 제어.
3. 3D Universe 고도화:
   - 3-tier 위계 유지 하에 고급 필터/클러스터/리플레이 확장.
4. 운영 체계:
   - 대시보드/알림/롤백 자동화.
   - feature flag 기반 단계적 노출.

## 3) 제외 범위 (Not in Phase 3 baseline)
1. 제품 정체성과 무관한 과도한 시각 효과.
2. 코어 루프를 차단하는 강제 UX.

## 4) 구현 단위
1. Frontend Platform:
   - 고급 인터랙션 전용 화면 분리.
   - 상태 동기화/캐시/낙관적 업데이트 전략 정착.
2. Backend Platform:
   - worker 토폴로지와 큐 관측 지표 정식 운영.
   - 장애 전파 차단과 degraded-mode 자동화.
3. Product Analytics:
   - 루프 완료, 전환, 재시작 패턴의 장기 추적.

## 5) 수용 기준
1. 고급 UI에서도 입력 유실/중복 저장이 발생하지 않는다.
2. worker 장애 시 코어 루프는 자동으로 degraded-mode 유지.
3. 3D/추천 고급 기능이 코어 성능(p95)에 악영향을 주지 않는다.

## 6) 리스크/대응
1. 아키텍처 전환 리스크:
   - 대응: Phase 2 경로를 fallback으로 유지한 점진 전환.
2. 운영 복잡도 상승:
   - 대응: 관측 지표와 경보 기준을 코드/문서로 표준화.

## 7) 완료 상태 정의
1. 코어 루프 + 추천 + OCR + 3D 고급 기능이 동시 안정 운영.
2. Streamlit 한계가 사용자 체감 병목으로 나타나지 않음.
3. 운영팀이 장애/롤백 절차를 자동화된 기준으로 수행 가능.
