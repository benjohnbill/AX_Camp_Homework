---
doc_type: implementation_directive
owner: control_tower
authority_level: L3
last_updated: 2026-02-25
sync_with:
  - agent.md
  - integration_status.md
  - android/NarrativeLoopMobile/CT_INBOX_ANDROID.md
context_ref: "D:/OneDrive/Desktop/Life_System/02_Core_Resources/01_Agent_Orchastration_System"
---

# Cycle 03 Implementation Directive: Android Recovery & Product MVP

## 1. Governance & Context
This directive is issued under the authority of `Agent.md` (SSOT) and `Harness_Policy.md`. It aligns with the System Orchestration standards referenced in the external system resources.

- **Objective**: Finalize Cycle 03 by verifying the Android Product MVP on a recovered environment.
- **Current State**: `IN_PROGRESS (Recovery Phase)`
- **Blocker Resolution**: CT has forcefully resolved the Android emulator 'offline' and 'deployment failed' issues.

## 2. Implementation Scope

### A. Backend & Frontend (Completed)
- **Status**: PASS
- **Verified**:
  - HTTPS Auth Contract (200/401/403/307)
  - Korean Narrative Copy for 401/403
- **Action**: No further changes required. Maintenance mode only.

### B. Android (Active Implementation)
- **Status**: **ENVIRONMENT READY (Worker Action Required)**
- **Target**: `android/NarrativeLoopMobile`
- **Mandatory Scenarios**:
  1. **Write Narrative**: Create -> Save -> Verify Persistence.
  2. **Universe**: Open 3D View -> Verify Render.
  3. **Auth**: Verify Bearer -> Cookie Transition.
  4. **Physical Device**: Replicate above on real hardware.

## 3. Worker Execution Prompt (For Android Studio)

**Paste the following directly to the Android Worker:**

```text
You are android_ide worker for Narrative_Loop Cycle 03.
Date: 2026-02-25
Objective: Complete Product Journey MVP (Write/Save/Query/Universe).

[STATUS UPDATE]
- CT has FIXED the emulator environment (Zombie process killed, ADB online).
- CT has FORCE INSTALLED the latest app via `gradlew installDebug`.
- `CreateNarrativeFragment` CONFIRMED to exist.

[EXECUTION STEPS]
1. Do not try to build/deploy again if not needed. App is installed.
2. Launch app: `adb shell am start -n com.example.narrativeloopmobile/.MainActivity`
3. Navigate to "Write Narrative" (Bottom Menu).
4. Execute: Write "Cycle3 Test" -> Save -> Restart App -> Check History.
5. Execute: Open Universe Tab -> Verify Load.
6. Repeat on Physical Device.

[REPORTING]
- Output: `android/NarrativeLoopMobile/ANDROID_REPORT.md`
- Format: 4-block standard (What changed, Validation, Risks, Next 3 actions).
- Evidence: Screenshots and Logcat are MANDATORY.
```

## 4. Validation Criteria (Completion Definition)
Cycle 03 will be marked **ACCEPTED** only when:
1. `ANDROID_REPORT.md` contains positive runtime evidence (PASS) for all mandatory scenarios.
2. `latest.handoff.json` is updated with the final Android validation links.
3. RRF (Retrieval) benchmark script has been executed (`tools/eval_korean_retrieval.py`).

## 5. Rollback Plan
If Android verification fails again due to *new* blockers:
1. Mark Cycle 03 as `CONDITIONAL_PASS` (Backend/Frontend only).
2. Move Android Product Journey to Cycle 04 as a high-priority debt.
