# DOMAIN_MAP.md

Last updated: 2026-02-23  
Scope: `Narrative_Loop`

## Purpose

도메인 경계를 문서화해 AI/사람 모두가 동일한 용어와 책임 단위로
작업을 분할하도록 한다.

## Domains

1. Narrative Core
- 책임: 기록/회고/결정 선언 로직
- 주요 코드: `narrative_logic.py`

2. Ingest and Auth Gateway
- 책임: Android OCR 입력 수신, 토큰 검증, 세션 전환
- 주요 코드: `universe_auth.py`, `debug_token_server.py`, `gateway_fastapi.py`

3. Retrieval and Context Enrichment
- 책임: hybrid search, context lifecycle, async enrichment
- 주요 코드: `BACKEND_HYBRID_CONTEXT_PLAYBOOK.md`, 관련 tools/tests

4. Runtime UI and Experience
- 책임: Streamlit 화면/상태/모드 흐름, UX 카피
- 주요 코드: `app.py`, `Antigravity_agent.md`, `orchestration/tasks/*frontend*.json`

5. Governance and Orchestration
- 책임: 계약 schema, 상태 동기화, 문서 권한/충돌 해소
- 주요 코드: `Agent.md`, `Harness_Policy.md`, `orchestration/`, `skills/`

## Boundary Rules

1. Gateway/Auth 도메인 변경은 보안 정책 문서와 함께 갱신한다.
2. Retrieval 정책 변경은 playbook + role guide 동기화를 필수로 한다.
3. UI/UX 변경은 API 계약을 임의로 확장하지 않는다.
4. Governance 도메인은 기능 구현 대신 규칙/검증/증적 관리에 집중한다.

