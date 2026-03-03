---
doc_type: redirecting_decision
owner: product
authority_level: planning
last_updated: 2026-03-03
sync_with:
  - redirecting/MVP_TIMEBOX_EXECUTION_REFLECTION_DESIGN_2026-03-03.md
  - redirecting/REDIRECTING_INDEX_2026-03-03.md
  - app.py
  - android/NarrativeLoopMobile
change_triggers:
  - feature_priority_changed
  - scope_cut_changed
sunset_condition: Replace when v1 scope is frozen and implementation starts.
---
# Redirecting Keep/Kill Decision (2026-03-03)

## 0) 목적
- 기존 자산(3D Universe, Android OCR, Universe 2D, Desk, Control, Stream)을
  `행동 설계 -> 실행 -> 성찰` 중심 전략에 맞게 보존/축소/통합/제거 판단한다.

## 1) 최종 결론 요약
1. **Android OCR: Keep + Expand**
2. **Universe 2D: Keep + Rescope (행동 추천형으로 전환)**
3. **3D Universe: Keep + Demote (주간 회고 전용)**
4. **Control: Merge into Time-Box**
5. **Desk: Shrink to Archive/Review**
6. **Stream: Keep as Assist (기본 진입 제외)**
7. **Notion/Obsidian MCP: Later (v1 범위 제외)**

## 2) 모듈별 판단

| 모듈 | 결정 | 이유 | v1 조건 |
|---|---|---|---|
| Android OCR | Keep+Expand | 즉시 캡처/입력 진입점, 멀티모달 데이터 확보 | 이미지 이벤트 -> 세션 연결 + AI 보조해석 |
| Universe 2D | Keep+Rescope | 현재 통계판 성격은 약함, 행동 추천형으로 바꿔 가치 가능 | 대시보드에서 바로 실행 CTA |
| 3D Universe | Keep+Demote | 메인 생산성 화면으론 약함, 회고 몰입 모드 가치는 있음 | 주간 회고 전용 진입 |
| Control | Merge | Time-Box와 목적 중복 | 칸반 기능을 블록 플래너로 흡수 |
| Desk | Shrink | 작성 기능 중복, 회고 자산으로는 유효 | Archive/Review 읽기 중심 |
| Stream | Assist-only | 자유채팅은 라이트유저 시작 마찰 큼 | 기본 홈에서 비노출 |
| MCP 연동 | Later | 기술 과도입/운영비 증가 | 단방향 export부터 검증 |

## 3) 왜 이 결정이 사업적으로 유리한가
1. 기존 자산을 버리지 않고 포지셔닝 리스크를 줄인다.
2. "채팅앱" 오해를 줄이고 "실행 시스템" 메시지를 강화한다.
3. Android/OCR라는 차별 입력 채널을 유지해 경쟁우위를 만든다.
4. 3D를 메인에서 분리해 개발비 대비 효용을 맞춘다.

## 4) Universe 재정의 (핵심)

### 4.1 제거할 것
- 의미 없는 장식형 수치/그래프
- 행동과 연결되지 않는 시각화

### 4.2 남길 것
1. 반복 방해요인 Top3
2. 성공 패턴 Top3
3. 다음 실행 1개 추천
4. 추천 실행 버튼

### 4.3 3D 용도 제한
- 기본 홈에서 노출 금지
- `주간 회고 리추얼` 화면으로만 노출
- 종료 액션: `다음 블록 1개 생성` (추천 CTA 우선, Skip 허용)

### 4.4 3D v1 표시 전략 (확정)
1. **v1 UI는 2레인으로 단순화**:
   - 레인 A: 계획(Time-Box)
   - 레인 B: 결과(Focus + Reflection + OCR 이벤트)
2. **내부 데이터 모델은 4레인 호환으로 유지**:
   - Time-Box
   - Focus
   - Reflection
   - OCR
3. 의도:
   - 라이트 유저에게는 이해 가능한 최소 시각화 제공
   - 이후 고급 사용자용 4레인 UI 확장 여지 확보

### 4.5 3D 주간 리플레이 구현 의도/규칙 (확정)
1. **범위 고정**:
   - 기본 조회 범위는 최근 7일(주간 고정)로 시작한다.
   - v1에서는 일/월 토글을 넣지 않는다.
2. **리플레이 종료 액션 고정**:
   - 3D 회고 종료 시 `다음 블록 1개 생성` CTA를 우선 제공한다.
   - 사용자는 `건너뛰기(Skip)`로 종료할 수 있다.
3. **OCR 이벤트 표시 방식 고정**:
   - OCR/이미지 이벤트는 결과 레인에서 `마커 + 썸네일`로 표시한다.
   - 클릭 시 원본/AI 요약/사용자 1줄 의미를 동시에 보여준다.
4. **완결 우선 규칙(v1 라이트 규칙)**:
   - Plan-first와 Focus-first 모두 Reflection까지 도달한 세션을 최우선으로 강조한다.
   - 동일 기간 내에서는 세션 완결 이벤트가 미완결 이벤트보다 상위에 렌더링된다.
   - 미완결/중단 이벤트는 보조 경고로 표시하되, 주 시각 중심은 완결 세션에 둔다.
5. **태그 클러스터 범위 제한**:
   - v1은 Top 10 태그만 노출한다.
   - 동시출현(co-occurrence) 강도에 따라 연결선 두께/투명도를 조절한다.
6. **데이터 우선순위**:
   - 1순위: `세션 완결 이벤트` (Focus 완료 + Reflection 제출).
   - 2순위: Time-Box/Focus/OCR 시간축.
   - 3순위: 태그 클러스터.
   - 즉, 3D의 주 기능은 `완결된 실행 세션 리플레이`이고 나머지는 보조 분석이다.

### 4.6 구현 참고용 이벤트 스키마 (v1)
1. 공통 필드:
   - `event_id`, `session_id`, `event_type`, `timestamp`, `lane`, `title`, `summary`
2. lane 매핑:
   - `plan` (Time-Box)
   - `result` (Focus/Reflection/OCR)
3. event_type:
   - `timebox_block`
   - `focus_start`
   - `focus_end`
   - `reflection_submitted`
   - `image_evidence_added`
   - `session_completed_marker`
   - `session_interrupted_marker`
4. OCR 전용 필드:
   - `image_uri`, `thumbnail_uri`, `intent_label`, `ai_summary`, `user_meaning`

### 4.7 2레인 이벤트 우선순위 (v1 최소 구현 고정)
1. 목적:
   - 구현 난이도를 낮추고, 짧은 기간 내 동작 가능한 3D 회고 리플레이를 확보한다.
2. 우선순위 레벨:
   - **Level 1**: 완료 + Reflection 제출 이벤트
   - **Level 2**: 미완료/중단 이벤트
   - **Level 3**: OCR/기타 보조 이벤트
3. 배치 규칙:
   - 레인 A(계획): Time-Box 블록을 시간순으로 배치
   - 레인 B(결과): 동일 시간 충돌 시 `L1 > L2 > L3` 순서로 상단 배치
4. 시각 규칙:
   - L1 = 초록, L2 = 주황, L3 = 회색
   - v1에서는 가중치/점수 계산을 사용하지 않는다.
5. v1 제외:
   - 동적 랭킹 점수
   - 이벤트 중요도 자동 재정렬 로직
   - 태그 연결선 두께 자동 스코어링

### 4.8 v2 확장 관리 원칙
1. 3D 확장 아이디어는 본 문서에 누적하지 않는다.
2. v2 확장 계획은 별도 문서로 분리해 관리한다.
3. 참조: `REDIRECTING_3D_UNIVERSE_V2_BACKLOG_2026-03-03.md`

## 5) Android OCR 재정의 (핵심)
### 5.1 1차 정책 (확정)
1. OCR의 핵심 목적은 `글쓰기 대체`가 아니라 `행동/맥락 증거 수집`이다.
2. 이미지는 먼저 원본으로 저장하고, 텍스트화(OCR/요약)는 보조 처리로 수행한다.
3. 의미 결정의 최종 권한은 사용자에게 둔다.
4. 업로드 시 의도 라벨 강제 대신, 세션 자동 연결 + Reflection 시점 의미 확정을 기본으로 한다.
5. 운영 원칙:
   - 이미지는 사실(Fact)
   - 사용자 1줄은 의미(Meaning)
   - AI는 연결 보조(Assist)

### 5.2 처리 파이프라인
1. Android에서 이미지 업로드
2. 시스템이 자동 연결 시도:
   - 우선순위: `active focus session -> today latest session -> inbox`
3. 백그라운드 AI가 보조 해석:
   - 텍스트가 있으면 OCR 추출
   - 텍스트가 약하면 장면 요약/키워드 생성
4. Reflection에서 세션 관련 이미지 1~2장만 큐레이션 제시
5. 사용자가 "이 이미지가 오늘 목표와 어떤 관련인지" 1줄 기록 또는 Skip

### 5.3 루프 편입 규칙
1. 작업 전 이미지:
   - Frog 후보 보조 생성에 사용 가능 (선택적)
2. 작업 중 이미지:
   - Time-Box block note 증거로 연결
3. 작업 후 이미지:
   - Reflection 근거 데이터로 연결

### 5.4 v1 범위 제한
1. 이미지 자체 저장 + 세션 연결 + 1줄 의미까지를 v1에 포함
2. 고급 비전 추론(복잡한 장면 이해, 자동 의도 판정)은 v1 제외
3. OCR이 실패해도 이미지 이벤트 저장/연결은 반드시 성공해야 함
4. 대량 이미지 탐색 UI는 v1 제외, Reflection 큐레이션 우선

## 5.5 자유 감상 정책 (확정)
1. 타이머 없이 자유 감상 작성을 허용한다.
2. 자유 감상은 우선 journal로 저장하고 코어 루프 지표에서 분리 집계한다.
3. 사용자가 원할 때 execution session으로 승격할 수 있다.
4. 승격 시 `next_action` 필드를 기준으로 추천/분석 루프에 포함한다.

## 5.6 Core 승격 정책 (확정)
1. v1에서 Core 승격은 사용자 수동 확정만 허용한다.
2. AI는 Core 자동 승격/자동 확정을 수행하지 않는다.
3. AI는 요약/태그 정리와 같은 보조 역할만 수행한다.
4. 승격 출처는 `execution_session` 또는 `journal`로 제한한다.

## 6) v1 Scope 컷

### 6.1 포함
1. Plan-first + Focus-first + Reflection 합류
2. Android OCR -> Frog 후보 연결
3. 자유 감상(Journal) + 세션 승격
4. Core 수동 승격(사용자 직접 확정)
5. Universe 2D 행동 추천 카드
6. 3D Universe 주간 리플레이(7일 고정, 2레인 UI)
7. Stream Assist 보조 모드

### 6.2 제외
1. MBTI류 성격 추론
2. Notion/Obsidian 실시간 양방향 동기화
3. 3D 메인화
4. AI 자동 Core 승격/자동 후보 큐

## 7) KPI (결정 검증용)
1. OCR 캡처 -> 세션 시작 전환율
2. 세션 시작률, Focus 완료율, Reflection 작성률
3. 대시보드 진입 후 추천 CTA 실행률
4. 주간 회고(3D) 진입률과 회고 후 계획 생성률
5. 3D 종료 CTA(다음 블록 생성) 실행률
6. Reflection까지 도달한 세션 비율(주간)

## 8) 리스크와 대응
1. 3D 집착 리스크:
   - 대응: 회고 전용으로 용도 고정, 메인 화면 제외.
2. 기능 과다 리스크:
   - 대응: v1 범위 엄격 제한.
3. MCP 과도입 리스크:
   - 대응: export/deep-link부터 단계 도입.

## 9) 실행 우선순위
1. 메인 루프 전환(Plan-first + Focus-first -> Reflection 합류)
2. OCR 자동 연결 + reflection 큐레이션
3. Journal 허용 + 승격 파이프라인
4. Universe 2D 행동 추천화
5. 3D 회고 전용화
6. MCP 검토는 v1 이후

## 10) 리뷰 방식
- 이 문서를 기준으로 사용자 피드백을 반영해 keep/kill 결정을 반복 업데이트한다.
- 변경 시 반드시 `v1 포함/제외`와 KPI 영향도를 함께 수정한다.
