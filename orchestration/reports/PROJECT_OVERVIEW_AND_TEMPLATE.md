---
doc_type: report_template
owner: control_tower
authority_level: L2
last_updated: 2026-02-25
sync_with:
  - agent.md
  - integration_status.md
  - orchestration/task.json
change_triggers:
  - cycle_close
  - showcase_video_prep
  - planning_doc_refresh
sunset_condition: Replace when a formal organization template is provided.
review_by: 2026-03-10
---

# 프로젝트 전체 기획서 (정리본 + 재사용 템플릿)

## A. 제출용 정리본 (현재 프로젝트 기준)

### 1) 기획 의도

본 프로젝트는 단순한 생산성 도구가 아니라, 사용자가 자신의 기록을 통해 삶의 맥락을 연결하고 자기결정을 강화하도록 돕는 "자기서사 시스템" 구축을 목표로 한다. 여기서 서사는 소비되는 외부 이야기(story)가 아니라, 개인의 과거-현재-미래를 잇는 내적 방향성과 의미 생성의 구조를 뜻한다.

정보 과잉과 빠른 전환이 일상화된 환경에서 많은 사용자는 자신의 경험을 깊게 이해하기보다 흘려보내기 쉽다. Narrative Loop는 이 단절을 줄이기 위해, 모바일 OCR 기반의 짧은 단상부터 긴 호흡의 에세이까지 글쓰기 행위를 하나의 연속된 루프로 연결한다.

핵심 사용자 루프는 다음과 같다.

1. 기록 작성(Write)
2. 저장 확인(Save)
3. 재오픈/재조회(Re-open/Re-query)
4. 회고 경험(Universe)

MVP의 기준은 기능 개수의 확장이 아니라, 위 루프가 인증/세션 변동 상황에서도 끊기지 않고 사용자가 다음 행동을 선택할 수 있는 신뢰 가능한 경험으로 유지되는지에 있다. 운영 관점에서는 CT(Control Tower)의 계약/증적 기반 의사결정을 채택해 "잘 동작해 보임"이 아니라 "검증 가능한 상태"를 목표로 한다. 이 기획 의도는 이후 구현/분석 방법, 기능 및 유저 시나리오, 한계 및 고도화 방안으로 구체화된다.
### 2) 구현/분석 방법

구현은 역할 분리 아키텍처를 기준으로 진행했다.

1. Android: 사용자 입력 및 런타임 여정
2. Gateway/Auth: 인증 경계, 상태코드 계약, 세션 전이
3. Streamlit: 사용자 경험/서사형 카피
4. Core Logic/DB: 도메인 처리 및 저장/조회

인증은 Contract-first로 관리했다. `missing_token`, `token_expired`, `forbidden_*` 등 코드 중심 분기와 Bearer->Cookie 전이 규칙을 명시했고, 프론트는 Narrative-first 원칙에 따라 기본 UI와 디버그 payload를 분리했다.

검증은 시나리오 기반으로 진행했다.  
에뮬레이터/실기기 공통 기준을 정의했고, backend/frontend는 cycle3 기준 PASS 증적이 확보되었으며, Android 제품 여정은 환경 이슈 복구와 최종 런타임 증적 확보를 병행 중이다.

### 3) 기능 및 유저 시나리오 / 분석 결과 및 인사이트

상태 표기 규칙:

- `PASS`: 증적 기반 재현 확인 완료
- `PARTIAL`: 일부 경로만 확인
- `BLOCKED`: 환경/런타임 이슈로 검증 중단

#### 시나리오 1. 기록 작성/저장

- 가치: 사용자의 첫 의사결정을 시스템에 남기는 시작점
- 인사이트: 저장 성공 피드백의 명확성이 재사용률과 신뢰도에 직접 영향
- 현재 상태: `PARTIAL` (Android 최종 런타임 증적 추가 필요)

#### 시나리오 2. 재오픈/재조회

- 가치: "정말 저장되었는가"에 대한 신뢰 형성
- 인사이트: 재조회 성공은 기능 신뢰뿐 아니라 심리적 안정감에 기여
- 현재 상태: `PARTIAL`

#### 시나리오 3. Universe 진입/렌더링

- 가치: 기록이 회고 경험으로 이어지는 루프 완성
- 인사이트: 렌더링 화려함보다 인증 경로 일관성이 이탈 방지에 더 중요
- 현재 상태: `PARTIAL`

#### 시나리오 4. 401/403 UX

- 가치: 실패 상황에서 복구 행동을 유도
- 인사이트: 기술 용어를 감추고 경계/재연결 중심 카피를 제공하면 실패 체감이 낮아짐
- 현재 상태: `PASS` (frontend 반영 및 backend 계약 증적 확인)

#### 시나리오 5. Lifecycle 안정성

- 가치: 모바일 실사용성 확보(백그라운드/복귀/탭 전환)
- 인사이트: 기능 정확도와 별개로 세션/뷰 상태 복원성이 체감 품질을 좌우
- 현재 상태: `PARTIAL`

스크린샷은 반드시 실제 파일 존재 여부를 확인해 삽입한다.  
예시 경로:

- `android/NarrativeLoopMobile/evidence/screenshots/01_write_save_emulator.png`
- `android/NarrativeLoopMobile/evidence/screenshots/01_write_save_device.png`
- `android/NarrativeLoopMobile/evidence/screenshots/04_auth_401.png`
- `android/NarrativeLoopMobile/evidence/screenshots/05_auth_403.png`

### 4) 한계 및 고도화 방안

현재 한계:

1. Android 빌드/실행 환경 의존성으로 검증 루프가 중단될 수 있음
2. 일부 증적 수집이 수동이라 반복 비용과 편차가 큼
3. 인증 UX 매핑은 핵심 경로 중심으로 닫혔고, OCR/Retrieval 확장 경로는 추가 정합 필요
4. OneDrive/비ASCII 경로와 로컬 Python 실행 경로 변동이 재현성에 영향

고도화 방안:

1. Android CI에 Gradle health check + 핵심 여정 계측 테스트 추가
2. `status+code -> UX copy` 매핑 표준을 공용 모듈 + 테스트로 고정
3. CT 자동판정 파이프라인(Probe + Gate + Scenario Matrix + Schema Validation) 강화
4. Retrieval(RRF) 지표 수집을 정기 리포트로 전환하여 운영 KPI와 연결

---

## B. 재사용 템플릿 (복사해서 다음 프로젝트에 사용)

아래 블록만 복사해서 새 문서로 사용:

```md
# [프로젝트명] 기획서

## 1) 기획 의도
- 본 프로젝트의 목적:
- 해결하려는 사용자 문제:
- MVP 핵심 가치(기능 수가 아닌 루프/경험 관점):
- 운영 철학(검증/의사결정 원칙):

## 2) 구현/분석 방법
- 아키텍처 분리 원칙:
- 핵심 계약(인증/세션/상태코드/데이터 계약):
- UX 설계 원칙:
- 검증 전략(시나리오/디바이스/환경):
- 판정 방식(어떤 증적을 근거로 accept/blocked를 결정하는지):

## 3) 기능 및 유저 시나리오 / 분석 결과 및 인사이트
- 상태 표기 규칙: PASS / PARTIAL / BLOCKED

### 시나리오 1: [이름]
- 가치:
- 인사이트:
- 현재 상태:
- 스크린샷:

### 시나리오 2: [이름]
- 가치:
- 인사이트:
- 현재 상태:
- 스크린샷:

### 시나리오 3: [이름]
- 가치:
- 인사이트:
- 현재 상태:
- 스크린샷:

### 시나리오 4: [이름]
- 가치:
- 인사이트:
- 현재 상태:
- 스크린샷:

### 시나리오 5: [이름]
- 가치:
- 인사이트:
- 현재 상태:
- 스크린샷:

## 4) 한계 및 고도화 방안
- 한계 1:
- 한계 2:
- 한계 3:
- 고도화 1:
- 고도화 2:
- 고도화 3:

## 부록 A) 증적 목록
- result/handoff 파일:
- 테스트 로그:
- 런타임 스크린샷:

## 부록 B) 발표/촬영용 한 줄 요약
- 문제:
- 접근:
- 결과:
- 다음 단계:
```


