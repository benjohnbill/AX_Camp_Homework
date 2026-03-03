# agent.md — NarrativeLoopMobile Redirecting 운영 기준 (Standalone Mirror)

이 문서는 Android lane 작업 기준 문서다.
Android가 독립 레포로 동작하는 전제를 기본값으로 하며, 로컬 미러 SSOT를 우선 사용한다.

---

## 0) Source-of-Truth 우선순위

1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. 최신 `orchestration/results/*.result.json`
4. `orchestration/docs/*.md`
5. `integration_status.md`
6. 본 `agent.md`

충돌 시 상위 우선순위를 따른다.

참고:

- 루트 레포의 canonical `orchestration/*`는 CT 집계용이다.
- Android 워커는 로컬 `orchestration/*`만으로 작업 가능해야 한다.

---

## 1) Redirecting 목적 (Android 관점)

Android의 v1 목적은 다음 3가지다.

1. OCR 캡처를 안정적으로 업로드한다.
2. 업로드된 증거가 세션 흐름을 막지 않도록 한다.
3. Phase 1에서는 Android를 `보조 입력 채널`로 명확히 포지셔닝한다.

금지:

- Phase 1에서 Android 단독으로 전체 Focus/Reflection 네이티브 완결을 약속하지 않는다.
- Phase 2 진입 전, CT의 Phase 1 PASS 선언 없이 범위를 확장하지 않는다.

---

## 2) Phase 게이트 규칙

### Phase 1 (선행 필수)

필수 조건:

1. `/v1/ocr/ingest` 계약 정렬
2. 인증 헤더 경로 확인 (`Authorization: Bearer <token>`)
3. 성공/실패 증거를 재현 가능한 경로로 제출

### Phase 2 (Phase 1 PASS 후)

필수 조건:

1. replay/auth 연속성 점검
2. 미구현은 `blocked + root cause + mitigation`로 제출

---

## 3) Android 역할 분담

1. CameraX/이미지 선택 -> 업로드 UX 안정화
2. 네트워크 실패/재시도/타임아웃 처리
3. 인증 토큰 전달 경로 검증
4. 데모에서 Android 역할을 보조 입력 채널로 명확히 고지

---

## 4) OCR 워크플로우 (Redirecting 기준)

1. Android에서 이미지 캡처 또는 선택
2. 인제스트 API로 업로드
3. 서버는 이미지 이벤트를 우선 수락/저장
4. OCR/요약은 비동기로 진행 가능
5. OCR 지연/실패가 코어 루프를 막지 않아야 함

핵심 원칙:

- Evidence saved first, narrative linking later.

---

## 5) API 계약 (v1 Demo 우선)

### 5.1 `POST /v1/ocr/ingest`

우선 계약:

- `multipart/form-data` 업로드 지원
- form-data key: `image` (또는 서버 호환 key `file`)
- 선택 필드: `session_id`

응답 최소 필드(현재 서버 기준):

- `status` (`accepted`)
- `image_event_id`
- `ocr_status`
- `refined_text` (optional/지연 가능)

호환 경로:

- `POST /v1/narrative/vision` alias 지원 가능

주의:

- 과거 `image_base64` JSON 계약은 레거시 호환 대상이며, v1 demo 기준 우선 경로가 아니다.

### 5.2 기타 연계 API (참조)

- `POST /v1/execution/session/start`
- `POST /v1/execution/session/{session_id}/focus/start`
- `POST /v1/execution/session/{session_id}/focus/end`
- `POST /v1/execution/session/{session_id}/reflect`
- `GET /v1/execution/session/today`

---

## 6) 데이터/표현 원칙

1. Android OCR 데이터는 evidence 성격을 우선한다.
2. OCR 실패 시에도 업로드 acceptance 자체는 유지되어야 한다.
3. 사용자 의미 부여는 Reflection 단계에서 확정된다.
4. Core 승격은 사용자 수동 확정만 허용한다(자동 승격 금지).

---

## 7) 보안 원칙

1. API 키/DB 키를 앱 내에 저장하지 않는다.
2. 토큰 기반 인증만 사용한다.
3. 로그/증빙에 민감값을 남기지 않는다.
4. 이미지 원본 보관은 최소화 원칙을 따른다.

---

## 8) 검증/품질 게이트

최소 검증:

```powershell
Get-Content orchestration/results/<android_result>.json | ConvertFrom-Json | Out-Null
Get-Content orchestration/handoff/<android_handoff>.json | ConvertFrom-Json | Out-Null
```

canonical validator 사용 가능 환경(선택):

```powershell
.\tools\project_python.ps1 tools/validate_contracts.py --file orchestration/results/<android_result>.json
.\tools\project_python.ps1 tools/validate_contracts.py --file orchestration/handoff/<android_handoff>.json
```

추가 권장:

- 실기기/에뮬레이터 각각 1회 이상 OCR 업로드 성공/실패 시나리오 증빙
- auth header 전달 검증 증빙

---

## 9) 인계 포맷 (필수)

Fast lane:

- `orchestration/results/<timestamp>.L1-android-redirecting-phase12-update.md`

Slow lane (필수):

1. `schema-valid result.json`
2. `schema-valid handoff.json`

스키마:

- `orchestration/contracts/result.schema.json`
- `orchestration/contracts/handoff.schema.json`

문장 보고만으로는 완료 판정되지 않는다.

---

## 10) 독립 레포 작업 시 규칙

Android가 별도 레포에서 작업되더라도, CT 집계는 아래 2단계 흐름을 따른다.

1. Step A (Android local generation):
   - `orchestration/results/*.json`
   - `orchestration/handoff/*.json`
2. Step B (CT mirror to canonical):
   - 루트 `Narrative_Loop/orchestration/results/*.json`
   - 루트 `Narrative_Loop/orchestration/handoff/*.json`

외부 레포에서 작업한 경우:

- 산출물/증빙 파일을 canonical 경로로 미러링하거나,
- 미러링 불가 시 `blocked + root cause + mitigation`을 명시한다.

CT 판정 규칙:

- Step B 완료 전에는 Android lane PASS 판정이 불가하다.
- narrative 보고만으로는 완료 판정되지 않는다.

---

## 11) 참조 문서

1. `orchestration/docs/REDIRECTING_PHASE1_DEMO_CHECKLIST_2026-03-03.md`
2. `orchestration/docs/REDIRECTING_PHASE2_DEMO_CHECKLIST_2026-03-03.md`
3. `orchestration/docs/REDIRECTING_DATA_API_CONTRACT_2026-03-03.md`
4. `orchestration/docs/REDIRECTING_PHASE2_EXECUTION_PLAN_2026-03-04.md`
5. `orchestration/ANDROID_EXTERNAL_REPO_ARTIFACT_BRIDGE_2026-03-04.md`
6. `orchestration/README_ANDROID_LOCAL_MIRROR.md`
