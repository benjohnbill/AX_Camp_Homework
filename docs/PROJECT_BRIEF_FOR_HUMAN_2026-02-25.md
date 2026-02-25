---
doc_type: project_brief
owner: control_tower
authority_level: reference
last_updated: 2026-02-25
sync_with:
  - CT_BASELINE_2026-02-25.md
  - MASTER_PLAN_CYCLE04_06.md
  - integration_status.md
change_triggers:
  - cycle_close
  - major_scope_change
  - risk_profile_change
sunset_condition: Replace at next cycle closure with a newer human-readable brief.
---
# Narrative_Loop 프로젝트 브리프 (사람용, 2026-02-25 기준)

## Quick Links
- [CT Baseline](./CT_BASELINE_2026-02-25.md)
- [Session Bootstrap Protocol](./SESSION_BOOTSTRAP_PROTOCOL.md)
- [Master Plan (Cycle 04-06)](./MASTER_PLAN_CYCLE04_06.md)
- [Pre-Cycle4 Feature Lock And Audit](./PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md)
- [Docs Index](./README.md)

## 한 줄 요약
지금 프로젝트는 "서사 기록 루프(Write/Save/Re-open/Re-query/Universe)"의 기본 작동을 Cycle 03에서 잠그고, Cycle 04~06에서 안정화/품질/운영체계 고정을 진행하려는 단계입니다.

## 1) 지금 실제로 구현되어 있는 기능
- 기록 작성과 저장:
  - Stream, Desk 경로에서 기록 생성/저장 흐름이 존재합니다.
- 재조회와 연결:
  - 저장된 기록을 다시 조회하고 연관 문맥을 찾는 하이브리드 검색 흐름이 있습니다.
- 타이머 기반 몰입:
  - Chronos 경로의 타이머 및 상태 복원 로직이 연결되어 있습니다.
- Universe 회고/탐색:
  - 인증 게이트를 거쳐 3D/대시보드 형태로 회고 화면을 여는 경로가 있습니다.
- Android 연동:
  - 에뮬레이터/실기기 기준 제품 여정 증적이 남아 있고, 네이티브 화면 이동 구조가 구성되어 있습니다.

## 2) 현재 운영 체계 (어떻게 굴러가고 있나)
- CT(Control Tower)가 사이클 목표와 판정 기준을 정의합니다.
- Worker(backend/frontend/android)가 각자 실행하고 증적 JSON을 제출합니다.
- 최종 상태는 `task/result/handoff` JSON으로 판정하며, 설명 문서는 이를 보조합니다.
- 사이클 종료 때는 5종 산출물을 반드시 갱신하도록 운영 규격을 고정했습니다.
- MCP/Skill 운영 원칙:
  - MCP는 현재 read-only 3종만 유지하고, 추가 확장은 보류합니다.
  - 외부 skill은 candidate 상태로만 두고, checksum+검증 통과 전에는 승격하지 않습니다.
  - 내부 운영 스킬 후보(기능잠금/사이클종료/민감정보검증)만 계획에 반영합니다.

## 3) 전체 계획에서 지금 위치
- 완료 구간:
  - Cycle 03 종료 상태를 `2026-02-25T23:59:59Z` 기준으로 PASS baseline 처리.
- 다음 진입 전 필수:
  - 바로 확장 개발로 가지 않고 Pre-Cycle4 기능 잠금/오류 점검을 먼저 수행.
- 확정 계획 범위:
  - Cycle 04~06까지는 구체 실행 계획으로 고정.
- 유보 범위:
  - 그 이후(Cycle 7+)는 아이디어 백로그로만 관리.

## 4) 왜 이렇게 진행하나
- 중기/장기 목표는 복잡도가 매우 높아 지금 확정하면 실패 확률이 커집니다.
- 따라서 먼저 "현재 코드베이스가 실제로 안정 작동하는가"를 잠그고,
- 그 위에서 OCR 안정화, 검색 품질 고도화, 운영 표준화를 단계적으로 진행합니다.

## 5) Cycle 04~06 핵심 체크리스트
- [ ] Cycle 04 시작 전 기능 잠금 게이트 통과 (전 자원 커버리지 감사 포함)
- [ ] Cycle 04: OCR/입력 안정성과 모바일 UX 안정화
- [ ] Cycle 05: 재조회 품질과 회고 UX 품질 고도화
- [ ] Cycle 06: 무맥락 CT 재시작 가능한 운영 체계 고정
- [ ] 각 사이클 종료 시 산출물 5종 동시 갱신

## 6) 실행/탐색/아이디어 구분 (고정 원칙)
- 실행(Committed) 70%: 당장 만들고 검증할 것
- 탐색(Exploratory) 20%: 기술 타당성 확인 후 다음 사이클에 승격할 것
- 아이디어(Speculative) 10%: Cycle 7+ 후보로만 보관할 것

## 7) 지금 당장 팀이 해야 할 일
1. `PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md` 체크리스트를 실제 증적으로 채웁니다.
2. 실패 항목은 Cycle 4로 넘기지 말고, hardening 결과로 먼저 닫습니다.
3. 통과 후 Cycle 4 kickoff JSON 패키지를 발행하고, CT baseline을 갱신합니다.

## 관련 문서
- [CT_BASELINE_2026-02-25.md](./CT_BASELINE_2026-02-25.md)
- [PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md](./PRE_CYCLE4_FEATURE_LOCK_AND_AUDIT.md)
- [MASTER_PLAN_CYCLE04_06.md](./MASTER_PLAN_CYCLE04_06.md)
