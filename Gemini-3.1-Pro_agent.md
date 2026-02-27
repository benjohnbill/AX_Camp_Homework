# Gemini-3.1-Pro_Agent.md (Legacy Alias)

이 문서는 기존 Gemini UI 역할 문서의 레거시 별칭이다.
현재 프로젝트 운영 기준에서 UI/UX 설계와 구현은 `frontend_ide` 단일 트랙으로 통합되었다.

## 1) Current Role Mapping

- 기존 역할: Gemini UI (설계 중심)
- 현재 역할: `frontend_ide` (설계 + 구현 + 검증)
- 운영 주체: Antigravity (frontend_ide worker)

## 2) Execution Rule

1. UI/UX 관련 실행은 `orchestration/tasks/*frontend*.json`을 기준으로 수행한다.
2. `app.py` 변경, 테스트, 증적(result/handoff)은 frontend worker 계약으로 제출한다.
3. 본 문서는 링크 호환성과 과거 아티팩트 참조를 위해 유지한다.
4. 실행 시작 시 `integration_status.md`, `orchestration/antigravity.current.json`, `CT_INBOX_GEMINI_UI.md`를 함께 확인한다.

## 3) References

- `Agent.md`
- `Antigravity_agent.md`
- `integration_status.md`
- `orchestration/dispatch/20260224-cycle03.worker-prompts.json`
- `skills/integration-status-sync/SKILL.md`
- `skills/cycle-close-packager/SKILL.md`
