# Antigravity_agent.md

Antigravity는 이 프로젝트에서 Backend + Streamlit 통합 구현 책임을 가진다.
작업 전 반드시 `agent.md`를 먼저 읽는다.

---

## 1) Mission

- Android OCR 입력을 기존 내러티브 파이프라인에 안전하게 연결한다.
- Streamlit의 UI 경험을 유지하면서 서버/API 계층을 분리한다.
- Supabase 데이터 무결성을 깨지 않고 기능을 확장한다.

---

## 2) Scope In

- 인제스트 API 계층 구현(FastAPI 기준)
- OCR 텍스트 정규화 및 중복 제어
- `narrative_logic.py`의 검색/응답/저장 흐름 재사용
- `db_manager_postgres.py` 기반 저장 규약 반영
- Streamlit에서 OCR 소스 로그 식별/회고 반영

## 3) Scope Out

- Android UI/Camera 구현
- 대규모 프런트 재디자인
- 운영 인프라 전면 교체

---

## 4) System Boundary

- Streamlit은 UI다. 외부 앱의 공용 API 수신 서버로 쓰지 않는다.
- 모바일 요청은 인제스트 API를 통해서만 진입한다.
- 키 보관 위치:
  - 서버: OpenAI/Gemini/DB 키
  - 클라이언트: 사용자 토큰만

---

## 5) API 계약 (고정)

### `POST /v1/ocr/ingest`

Request:

- `user_id` string
- `image_base64` string
- `client_ts` string
- `session_id` string
- `mode_hint` string
- `manual_override_text` optional string

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

## 6) Data Rule

- `meta_type="Log"`
- `content=정규화된 최종 텍스트`
- `tags`에 `source:android_ocr` 계열 반영
- `dimension`은 필요 시 `handwriting`
- 외부 입력도 기존 회고/검색에서 동일 1급 로그로 취급

---

## 7) Implementation Rule

1. `EvidenceGateway`와 `PolicyEngine` 책임을 섞지 않는다.
2. `narrative_logic.py`에서 UI 세션 상태를 직접 제어하지 않는다.
3. 실패 시에도 가능한 범위에서 로그 저장과 에러 응답을 분리한다.
4. OCR 품질 이슈를 위해 confirm 단계(수정 후 확정)를 지원한다.

---

## 8) Verification Checklist

```powershell
python tools/preflight_postgres_auth.py
python tools/check_supabase_phase1.py
python tools/check_postdeploy_smoke.py --strict-postgres
python tools/check_data_integrity.py --expect-postgres --max-dup-chat 0 --report-json data/integrity_latest.json
python -m pytest -q tests/
```

---

## 9) Done Definition

- Android OCR 텍스트가 Supabase `logs`에 저장된다.
- 저장 직후 관련 로그 검색과 AI 응답이 정상 반환된다.
- Streamlit에서 해당 로그를 source 구분해 확인 가능하다.
- 무결성/스모크/테스트 체크를 통과한다.

---

## 10) Handoff Format

- What changed
- Validation
- Risks
- Next 3 actions
