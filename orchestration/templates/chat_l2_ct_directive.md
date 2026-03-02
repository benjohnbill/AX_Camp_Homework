---
doc_type: playbook
owner: control_tower
authority_level: operational
last_updated: 2026-03-02
sync_with:
  - docs/CHAT_CLI_MESSAGE_PROTOCOL_2026-02-25.md
change_triggers:
  - ct_directive_format_change
sunset_condition: Replace when CT directive format is schema-enforced.
---
# L2 CT Directive Template

```text
[L2_CT_DIRECTIVE]
target_worker: <frontend_ide|backend_cli|android_ide>
trace_id: <trace_id>
task_file: <orchestration/tasks/...task.json>
priority: <P0|P1|P2>
common_guard: Canonical JSON verdict > markdown narrative. No deploy/schema migration/permission elevation/destructive command without explicit approval.
scope:
- <in-scope bullet 1>
- <in-scope bullet 2>
exit_criteria:
1) <criterion1>
2) <criterion2>
required_output:
1) L1 update (fast lane)
2) schema-valid result.json (slow lane)
```
