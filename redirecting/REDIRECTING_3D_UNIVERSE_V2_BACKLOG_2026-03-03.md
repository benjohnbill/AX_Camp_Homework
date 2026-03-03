---
doc_type: redirecting_v2_backlog
owner: product
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/REDIRECTING_KEEP_KILL_DECISION_2026-03-03.md
  - redirecting/REDIRECTING_COMPONENT_PLAN_2026-03-03.md
change_triggers:
  - v2_idea_added
  - v2_priority_changed
sunset_condition: Replace when v2 kickoff plan is approved.
---
# 3D Universe v2 Backlog (2026-03-03)

## 0) 목적
- v1 구현 속도를 해치지 않도록 3D 확장 아이디어를 별도 공간에서 관리한다.
- 아래 항목은 모두 **v1 비범위**이며, 우선순위 재평가 후 착수한다.

## 1) v2 후보 항목

### 1.1 다중 시간 범위
1. 일/주/월 토글 추가
2. 월간 히트맵과 3D 리플레이 연결
3. 특정 기간 필터 저장/재호출

### 1.2 4레인 고급 UI
1. plan(Time-Box)
2. focus(Focus start/end)
3. reflection(회고 이벤트)
4. ocr(이미지 증거)

### 1.3 세션 완결 위계 고도화
1. v1의 3-tier(완결/중단/보조증거)를 다단계 점수 규칙으로 확장
2. 태그 유사도 기반 재시작-완결 전환 판정
3. 완결 전환 성공 패턴 추천 자동화

### 1.4 태그 클러스터 고도화
1. Top10 제한 해제 옵션
2. 연결 강도 가중치 모델(빈도 + 최근성)
3. 클러스터별 리플레이 진입

### 1.5 OCR 시각 고도화
1. OCR 이미지 그룹핑(계획/진행/결과/감정)
2. 썸네일 스트립/필름 모드
3. 이미지 이벤트에서 즉시 회고/블록 생성

## 2) 선행 조건
1. v1 안정화 지표 충족:
   - 3D 종료 CTA 실행률 목표 달성
   - 리플레이 오류율 허용치 이내
2. 데이터 품질:
   - OCR 이벤트 라벨링 누락률 감소
   - Reflection 작성률 기준 충족
3. 운영 여건:
   - 비동기 워커/큐 안정화

## 3) 우선순위 정책
1. v2 착수 순서:
   - 세션 완결 위계 고도화 -> 4레인 UI -> 태그 클러스터 고도화
2. 다중 시간 범위(일/월)는 실제 사용 로그 기반으로 필요성이 확인될 때 착수
3. 모든 v2 항목은 A/B 또는 제한 공개로 검증한다

## 4) 비고
- 이 문서는 아이디어 저장소가 아니라 “v2 검토 대기열”이다.
- 항목 추가 시 반드시 기대효과 KPI를 함께 기록한다.

