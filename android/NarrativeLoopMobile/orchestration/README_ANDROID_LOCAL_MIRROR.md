# Android Local Mirror (Standalone Repo)

This folder makes Android lane runnable without access to the root Narrative_Loop repo.

## Local SSOT (Android only)
1. `orchestration/handoff/latest.handoff.json`
2. `orchestration/task.json`
3. Latest `orchestration/results/*.result.json`
4. `agent.md`

## Bridge Model
- Step A: Android worker writes artifacts in this repo under `orchestration/results` and `orchestration/handoff`.
- Step B: CT mirrors validated artifacts into canonical root repo `orchestration/`.

## Required Files
- Schemas: `orchestration/contracts/*.schema.json`
- Templates: `orchestration/templates/result.template.json`, `orchestration/templates/handoff.template.json`
- Active task pointer: `orchestration/task.json`
