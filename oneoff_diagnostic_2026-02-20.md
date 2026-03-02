---
doc_type: temporary
owner: Codex
authority_level: L4
last_updated: 2026-02-20
sync_with:
  - Agent.md
change_triggers:
  - oneoff_diagnostic_refresh
sunset_condition: delete after gaps in section 3 are resolved or by 2026-02-28
review_by: 2026-12-31
---

# Narrative_Loop 일회성 진단 점검표 (2026-02-20)

이 문서는 **일회성 진단 체크리스트**다.  
요구사항이 충분히 충족되면 삭제하는 것을 전제로 한다.

## 1) 사용 방식

- 목적: 문서-코드 불일치, 잠재 런타임 리스크, 운영 전환 리스크를 단기적으로 정리
- 범위: `Narrative_Loop` 루트의 `.md`, Python 코드, 도구 스크립트, 테스트 결과
- 운영 원칙:
  - 항목 단위로 체크 후 진행
  - 모든 항목이 종료되면 본 파일 폐기

## 2) 진단 요약

- 현재 프로젝트는 Streamlit + Postgres 기반 핵심 기능은 상당 부분 구현되어 있다.
- 다만 **fallback(SQLite) 경로와 문서 정합성**, **일부 런타임 잠재 오류**, **OCR ingest 분리 구현 공백**이 남아 있다.
- 테스트/검증도구는 잘 갖춰져 있지만, 현재 로컬 `venv` 문제로 실행 신뢰도가 떨어진다.

## 3) 점검 항목

### [ ] A-1. fallback DB 문서-코드 불일치 정리

- 문제:
  - SSOT 문서는 fallback을 `db_manager_sqlite.py`로 명시
  - 실제 라우팅은 `db_manager.py` 사용
- 근거:
  - `Agent.md:36`
  - `db_backend.py:32`
- 영향:
  - 유지보수 혼선
  - fallback 경로 기능 편차 확대
- 조치:
  - `db_backend.py` 라우팅과 SSOT를 동일 기준으로 통일

### [ ] A-2. SQLite 인터페이스 parity 실패 해소

- 문제:
  - SQLite 쪽에서 필수 함수 누락으로 인터페이스 테스트 실패
- 근거:
  - `test_result.txt:15`
  - `test_result.txt:16`
  - `test_result.txt:17`
  - `test_result.txt:18`
- 영향:
  - fallback/테스트 신뢰도 하락
- 조치:
  - `db_manager.py`에 누락 함수 추가 또는 `db_manager_sqlite.py`를 fallback 표준으로 전환

### [ ] A-3. Chronos 타이머 함수의 fallback 안전성 확보

- 문제:
  - 앱은 공통 DB 객체에서 Chronos 함수를 호출
  - 해당 함수는 Postgres 구현에만 존재
- 근거:
  - `app.py:78`
  - `app.py:368`
  - `app.py:383`
  - `db_manager_postgres.py:806`
  - `db_manager_postgres.py:817`
  - `db_manager_postgres.py:837`
- 영향:
  - `DATASTORE=sqlite`에서 런타임 예외 가능
- 조치:
  - fallback 구현 추가(실구현 또는 no-op + 명시적 경고)

### [ ] A-4. `evaluate_silence` timezone 참조 오류 수정

- 문제:
  - `datetime.timezone.utc` 참조 코드가 현재 import 방식과 충돌 가능
- 근거:
  - `narrative_logic.py:264`
  - `narrative_logic.py:266`
- 영향:
  - naive datetime 경로에서 예외 발생 가능
- 조치:
  - `from datetime import timezone`로 정리하거나 참조 방식 일관화

### [ ] A-5. 그래프 빈 상태 경로의 `icons` import 보강

- 문제:
  - `generate_graph_html()`에서 `icons.get_icon_svg` 호출
  - `narrative_logic.py` 상단에 `icons` import 없음
- 근거:
  - `narrative_logic.py:1205`
  - `narrative_logic.py:1211`
- 영향:
  - 특정 경로(빈 상태/해당 함수 실행 시) NameError 가능
- 조치:
  - `import icons` 추가 또는 아이콘 의존 제거

### [ ] A-6. 용어 리팩터(`Core/Gap/Log`) 잔여 legacy 제거

- 문제:
  - 일부 SQLite 코드에 `Fragment` 기록 경로가 남아 있음
- 근거:
  - `db_manager.py:976`
- 영향:
  - 분석/집계/검색 시 타입 혼합
- 조치:
  - 저장 meta_type 기준을 `Log`로 통일하고 마이그레이션/호환 로직 정리

### [ ] A-7. OCR ingest 분리 구현 상태 명확화

- 문제:
  - 문서는 FastAPI ingest를 기준으로 하나, 코드에는 구현/의존성 부재
- 근거:
  - `Agent.md:121`
  - `Antigravity_Agent.md:18`
  - `requirements.txt:1`
  - `requirements.txt:2`
  - `requirements.txt:10`
  - `requirements.txt:11`
- 영향:
  - Android OCR 통합 경로가 문서 대비 미완료
- 조치:
  - `POST /v1/ocr/ingest` 최소 구현 스켈레톤 생성 또는 문서 상태를 "planned"로 명시

### [ ] A-8. 로컬 venv 재생성 및 테스트 실행 경로 복구

- 문제:
  - 현재 `venv`가 Windows Store Python 경로를 참조해 pytest 실행 실패
- 근거:
  - `C:\Users\benjohnbill\OneDrive\Desktop\Life_System\01_Active_Projects\08_AX 코딩 아카데미\venv\pyvenv.cfg:1`
  - `C:\Users\benjohnbill\OneDrive\Desktop\Life_System\01_Active_Projects\08_AX 코딩 아카데미\venv\pyvenv.cfg:4`
- 영향:
  - 테스트 자동화/검증 게이트 실행 신뢰도 저하
- 조치:
  - venv 재생성 후 `python -m pytest -q tests/` 재검증

## 4) 종료 기준 (모두 체크되면 문서 삭제)

- [ ] 위 8개 항목 완료
- [ ] 핵심 게이트 통과 (`tools/run_agent_a_gate.py`)
- [ ] `tests/` 전체 실행 결과 확보
- [ ] 문서-코드 기준 충돌 없음 확인

## 5) 폐기 메모

- 폐기 시점:
- 폐기 사유:
- 마지막 검증 로그 위치:
