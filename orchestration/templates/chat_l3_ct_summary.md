---
doc_type: playbook
owner: control_tower
authority_level: operational
last_updated: 2026-02-25
sync_with:
  - docs/CHAT_CLI_MESSAGE_PROTOCOL_2026-02-25.md
  - docs/SESSION_BOOTSTRAP_PROTOCOL.md
change_triggers:
  - ct_summary_format_change
sunset_condition: Replace when cycle/session summary format is standardized elsewhere.
---
# L3 CT Summary Template

```text
[L3_CT_SUMMARY]
timestamp_basis: <ISO-8601>
files_read:
1) <file1>
2) <file2>
3) <file3>
current_state: <one paragraph>
next_3_actions:
1) <action1>
2) <action2>
3) <action3>
conflicts:
1) <conflict or none>
2) <conflict or none>
```
