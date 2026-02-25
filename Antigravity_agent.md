# Antigravity_Agent.md

Antigravity는 이 프로젝트에서 Backend + Streamlit 통합 구현 책임을 가진다.
작업 전 반드시 `Agent.md`를 먼저 읽는다.
보안/토큰 운영은 `DEBUG_TOKEN_GOVERNANCE.md`를 필독한다.
검색/컨텍스트 정책 구현은 `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md`를 기준으로 한다.
문서 거버넌스/충돌 해결 규칙은 `Harness_Policy.md`를 따른다.

---

## 1) Mission

- Android OCR 입력을 기존 내러티브 파이프라인에 안전하게 연결한다.
- Streamlit의 UI 경험을 유지하면서 서버/API 계층을 분리한다.
- Supabase 데이터 무결성을 깨지 않고 기능을 확장한다.
- 하이브리드 검색 품질과 컨텍스트 enrichment 안정성을 동시에 확보한다.

---

## 1.1) 자율 루프 (Autonomous Loop) 실행 가이드

Antigravity는 CT의 명시적 지시가 없을 때도 다음 루프를 수행하여 프로젝트의 '백엔드 엔진'을 유지한다.

1. **Scan**: `integration_status.md`와 `CT_INBOX_ANTIGRAVITY.md`를 스캔한다.
2. **Analyze**: 현재 백엔드/인프라 영역에서 해결 가능한 `Open Gaps / Risks`를 식별한다.
3. **Propose**: 
   - `orchestration/proposals/` 폴더에 `task.json` 초안을 작성한다.
   - 예: "현재 인증 구조가 확정되었으므로, OCR API의 세션 검증 로직을 구현하겠다."
4. **Safety Stop**: 사용자 승인이 필요한 작업(스키마 변경, 배포 등)이나 치명적인 오류 발생 시 즉시 작업을 멈추고 CT에게 에스컬레이션(보고)한다.
5. **The Exit Hook**: 모든 작업(`result.json` 생성 및 `handoff.json` 제출)을 마친 후, 반드시 터미널에서 `.\tools\ralph_heartbeat.ps1`을 실행하여 다음 루프를 트리거하고 종료한다. (Antigravity IDE 내부 터미널 활용 권장)

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

## 7) Implementation Rule (Playbook Locked)

1. `EvidenceGateway`와 `PolicyEngine` 책임을 섞지 않는다.
2. `narrative_logic.py`에서 UI 세션 상태를 직접 제어하지 않는다.
3. 하이브리드 랭킹은 가중합 대신 `RRF`를 기본값으로 사용한다.
4. 컨텍스트는 분리 저장을 기본으로 한다 (`content`와 `context_text` 분리).
5. 컨텍스트 생성은 mixed mode를 사용한다:
   - short text: 동기 생성
   - long text: 비동기 생성
6. 동기 컨텍스트 생성 실패 시 write를 실패시키지 않고 `pending`으로 전환 후 비동기 재시도로 넘긴다.
7. 비동기 worker는 lock-safe claim + idempotent update + source hash stale 판정을 구현한다.
8. 검색 안전 규칙을 적용한다:
   - `exclude_ids` 지원
   - `context_text` 우선 활용
   - context 부재 시 원문 fallback 경로 유지
9. 한국어 단기 개선은 trigram + vector + context_text + 경량 동의어/규칙 rewrite 조합을 기본으로 한다.
10. 문서 변경/동기화 판단이 필요할 때는 `Harness_Policy.md`의 authority model/sync matrix를 따른다.

---

## 8) Verification Checklist

```powershell
.\tools\project_python.ps1 tools/preflight_postgres_auth.py
.\tools\project_python.ps1 tools/check_supabase_phase1.py
.\tools\project_python.ps1 tools/check_postdeploy_smoke.py --strict-postgres
.\tools\project_python.ps1 tools/check_data_integrity.py --expect-postgres --max-dup-chat 0 --report-json data/integrity_latest.json
.\tools\project_python.ps1 -m pytest -q tests/
```

추가 점검 항목:
- 신규 write 직후 self-match가 `exclude_ids`로 완화되는지 확인
- `context_status` 라이프사이클(`pending|running|succeeded|failed|stale`) 전이가 손상 없이 동작하는지 확인
- 중복 비동기 job 상황에서 로그 상태가 오염되지 않는지 확인
- `context_text` 유무와 무관하게 검색 경로가 정상 동작하는지 확인
- 한국어 평가셋에서 baseline 대비 retrieval 지표(예: recall@k/precision@k)가 개선되는지 확인

---

## 9) Done Definition

- Android OCR 텍스트가 Supabase `logs`에 저장된다.
- 저장 직후 관련 로그 검색과 AI 응답이 정상 반환된다.
- Streamlit에서 해당 로그를 source 구분해 확인 가능하다.
- 무결성/스모크/테스트 체크를 통과한다.
- 하이브리드 검색/컨텍스트 정책이 `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md`와 충돌 없이 일치한다.

---

## 10) Handoff Format

- What changed
- Validation
- Risks
- Next 3 actions

