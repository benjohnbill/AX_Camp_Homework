---
doc_type: playbook
owner: control_tower
authority_level: operational
last_updated: 2026-02-25
sync_with:
  - docs/CHAT_CLI_MESSAGE_PROTOCOL_2026-02-25.md
change_triggers:
  - worker_reporting_format_change
sunset_condition: Replace when L1 format is versioned by external schema.
---
# L1 Worker Update Template (Fast Lane)

```text
[L1_WORKER_UPDATE]
worker: <frontend_ide|backend_cli|android_ide>
task_id: <task_id>
status: <running|partial|blocked|success>
blocker_class: <none|auth|storage|universe|android_runtime|other>
summary: <one line>
evidence_top3:
1) <path1>
2) <path2>
3) <path3>
next_3:
1) <action1>
2) <action2>
3) <action3>
```
