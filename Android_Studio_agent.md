# Android_Studio_agent.md

Android Studio는 이 프로젝트에서 모바일 클라이언트 구현 책임을 가진다.
작업 전 `agent.md`를 반드시 확인한다.

---

## 1) Mission

- 손글씨 입력을 가장 낮은 마찰로 캡처하고 서버로 전달한다.
- OCR 결과를 사용자가 확인/수정한 뒤 확정 저장하게 만든다.
- 불안정 네트워크 환경에서도 입력 손실을 줄인다.

---

## 2) Scope In

- CameraX 촬영/프리뷰
- 이미지 전처리(회전, 크롭, 압축)
- OCR 업로드 API 호출
- OCR 결과 확인/수정 UX
- AI 응답 렌더링(WebView 또는 Native)
- 재시도/오프라인 큐

## 3) Scope Out

- OpenAI/Gemini 호출 로직
- Supabase 직접 쓰기
- 서버 키 관리

---

## 4) UX Flow

1. Capture
2. Preview
3. OCR 요청
4. OCR 텍스트 확인/수정
5. Confirm 저장
6. AI 응답 확인

원칙:

- 빈 텍스트/오인식은 사용자 수정 후 확정 가능해야 한다.
- "다시 촬영"과 "수정 후 저장"을 항상 제공한다.

---

## 5) API 계약 (고정)

### `POST /v1/ocr/ingest`

Request:

- `user_id`
- `image_base64`
- `client_ts`
- `session_id`
- `mode_hint`
- `manual_override_text` (optional)

Response:

- `request_id`
- `ocr_text_raw`
- `ocr_text_normalized`
- `confidence`
- `saved_log_id`
- `ai_response`
- `related_log_ids`
- `warnings`

Error:

- `400`, `401`, `422`, `429`, `500`

---

## 6) Reliability Rule

- 타임아웃/네트워크 실패 시 재시도
- 중복 전송 방지를 위한 `session_id + local_request_id` 유지
- 실패 요청은 로컬 큐에 저장 후 재전송
- 사용자에게 현재 상태(대기/실패/재시도/완료)를 명확히 표시

---

## 7) Security Rule

1. OpenAI/Gemini 키를 앱에 넣지 않는다.
2. 서버 토큰 기반 인증만 사용한다.
3. 로그/크래시 리포트에 민감 텍스트를 무단 노출하지 않는다.

---

## 8) QA Checklist

- 저조도/흔들림/긴 문장/짧은 문장 케이스 테스트
- 네트워크 on/off 반복 테스트
- OCR 결과 수정 후 저장 테스트
- 동일 이미지 중복 전송 테스트
- API 401/422/429/500 에러 메시지 매핑 테스트

---

## 9) Done Definition

- 촬영 -> OCR -> 확인/수정 -> 저장 -> 응답 흐름이 실기기에서 작동
- 네트워크 실패 후 재전송 성공
- 입력 손실 없이 최소 20건 연속 처리

---

## 10) Handoff Format (to Antigravity)

- request_id
- endpoint + status code
- payload 샘플(민감값 마스킹)
- 재현 절차
- expected vs actual
