# USER_INTERVENTION_REQUIRED (Emergency Report)

## 1) Incident Summary
- **Status**: LOOP_HALTED
- **Trigger**: [e.g., Fail count exceeded / Boundary violation / User approval needed]
- **Last Active Agent**: [Agent Name]
- **Timestamp**: [ISO-8601]

## 2) Root Cause Analysis
- **Problem**: [Detailed description of the error or block]
- **Evidence**: [Reference to result.json or log files]

## 3) Decision Required
- [ ] Option 1: [Manual fix and restart loop]
- [ ] Option 2: [Modify task strategy and retry]
- [ ] Option 3: [Abort current cycle and redesign]

## 4) How to Restart
1. Resolve the issue above.
2. Manually set `"status": "running"` and `"fail_count": 0` in `orchestration/loop_state.json`.
3. Run `.	oolsalph_heartbeat.ps1` to resume.
