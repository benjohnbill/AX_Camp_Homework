# Orchestration Workspace

This folder contains shared contracts and example artifacts for Control Tower orchestration.

## Structure

- `contracts/`: JSON schemas for task/result/handoff
- `examples/`: sample payloads that must pass schema validation
- `dispatch/`: worker prompt packages
- `tasks/`: worker-specific task contracts
- `templates/`: chat fast-lane (`L1`) and CT directive/summary (`L2/L3`) templates
- `antigravity.current.json`: CT->frontend_ide(Antigravity) single-pointer channel file
- `backend.current.json`: CT->backend_cli single-pointer channel file
- `android.current.json`: CT->android_ide single-pointer channel file

## Validate

```powershell
.\tools\project_python.ps1 tools/validate_contracts.py
```

Optional single-file validation:

```powershell
.\tools\project_python.ps1 tools/validate_contracts.py --file orchestration/examples/sample.task.json
```

## Chat Mode

When operating in chat-triggered CLI mode:
- Use `orchestration/templates/chat_l1_worker_update.md` for quick status reports.
- Keep final decisions based on schema-valid `result/handoff` artifacts.
