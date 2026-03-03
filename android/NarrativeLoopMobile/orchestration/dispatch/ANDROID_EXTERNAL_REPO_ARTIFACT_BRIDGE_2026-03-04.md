---
doc_type: protocol
owner: control_tower
authority_level: operational
last_updated: 2026-03-04
sync_with:
  - orchestration/contracts/result.schema.json
  - orchestration/contracts/handoff.schema.json
  - android/NarrativeLoopMobile/agent.md
  - docs/CT_BASELINE_2026-03-03_REDIRECTING_DEMO.md
change_triggers:
  - external_repo_mode_changed
  - artifact_contract_changed
sunset_condition: Replace when android lane is fully converged into single repo orchestration paths.
---
# Android External Repo Artifact Bridge Protocol (2026-03-04)

## 0) Purpose
- Android worker가 독립 레포(`NarrativeLoopMobile`)만 접근 가능한 상황에서도,
  CT가 `Narrative_Loop`의 canonical orchestration 계약으로 판정할 수 있게 한다.

## 1) Two-Step Artifact Flow
1. Step A (Android local generation):
  - Android worker는 독립 레포 내부 경로에 산출물을 생성한다.
  - 예:
    - `android/NarrativeLoopMobile/orchestration/results/<timestamp>...android...result.json`
    - `android/NarrativeLoopMobile/orchestration/handoff/<timestamp>...android...handoff.json`
2. Step B (CT mirror + normalize):
  - CT 또는 통합 담당자가 Step A 산출물을 메인 레포 canonical 경로로 미러링한다.
  - canonical 경로:
    - `orchestration/results/<timestamp>...android...result.json`
    - `orchestration/handoff/<timestamp>...android...handoff.json`
  - 이후 schema validation을 수행한다.

## 2) Gate Rule
- CT는 Step B 완료 전까지 Android lane을 PASS로 판정하지 않는다.
- narrative 텍스트 보고만으로는 lane 완료로 판정하지 않는다.

## 3) Required Commands (CT side)
```powershell
.\tools\project_python.ps1 tools/validate_contracts.py --file orchestration/results/<android_result>.json
.\tools\project_python.ps1 tools/validate_contracts.py --file orchestration/handoff/<android_handoff>.json
```

## 4) Minimum Artifact Quality
- `result.json`: `orchestration/contracts/result.schema.json` 완전 준수
- `handoff.json`: `orchestration/contracts/handoff.schema.json` 완전 준수
- status 값은 스키마 허용값만 사용:
  - result.status: `success|partial|blocked|failed`
- `PASS/HANDOFF` 같은 임의 상태값 사용 금지

## 5) Evidence Requirements (Android Phase 1)
1. OCR ingest endpoint 정합 증빙 (`POST /v1/ocr/ingest`)
2. auth header 증빙 (`Authorization: Bearer <token>`)
3. 성공 또는 실패 시나리오 로그 경로
4. separate repo 작업인 경우 미러링 여부/불가 사유 명시

## 6) Blocked Handling
- 미러링 권한/경로 제약으로 Step B가 불가능하면:
  1. Android lane status를 `blocked`로 제출
  2. root cause와 mitigation을 `handoff.json`에 명시
  3. CT는 lane unblock 조치를 먼저 수행

## 7) Checklist (CT quick use)
- [ ] Android local result/handoff 존재 확인
- [ ] canonical orchestration 경로로 미러링 완료
- [ ] result schema PASS
- [ ] handoff schema PASS
- [ ] evidence path 재현 가능 확인
- [ ] latest.handoff 집계 반영
