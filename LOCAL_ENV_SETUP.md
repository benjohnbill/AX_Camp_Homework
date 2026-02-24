# LOCAL_ENV_SETUP.md

## Purpose

OneDrive 동기화 환경에서 노트북/데스크톱 모두 동일하게 동작하는 로컬 Python 런타임 복구 절차.

## Rule

1. 코드/문서만 OneDrive로 공유한다.
2. 가상환경은 기기 로컬 경로로 분리한다.
3. venv root는 동적으로 해석한다:
- 1순위: `$env:LIFE_VENV_ROOT`
- 2순위(default): `$env:USERPROFILE\.venvs_hub`
- non-ASCII profile fallback: `C:\venvs_hub`
4. canonical venv 이름:
- `Narrative_Loop.venv`
5. 호환 별칭 이름:
- `narrative_loop` (junction으로 canonical venv를 가리켜도 됨)
6. 문서/로그 표기 규칙:
- 절대경로 하드코딩 대신 `LIFE_VENV_ROOT` + `Narrative_Loop.venv` 형태로 기록한다.

## Desktop / Laptop Setup (same steps)

```powershell
# Optional explicit override:
# $env:LIFE_VENV_ROOT = (Join-Path $env:USERPROFILE ".venvs_hub")
.\tools\bootstrap_env.ps1 -Recreate -InstallPreCommit
.\tools\project_python.ps1 --version
.\tools\project_python.ps1 tools/check_docs_contract.py --mode warn
.\tools\project_python.ps1 tools/check_skill_registry.py --mode warn
.\tools\project_python.ps1 tools/run_agent_a_gate.py --policy-mode warn
```

Optional lock refresh:

```powershell
.\tools\bootstrap_env.ps1 -WriteLock
```

## Do Not

- 프로젝트 내부 `venv/.venv`를 canonical runtime으로 사용하지 않는다.
- `pyvenv.cfg`에 타 기기 절대경로가 들어간 환경을 재사용하지 않는다.
- 특정 드라이브/사용자명 절대경로(`D:\...`, `C:\Users\...`)를 기준 경로로 문서에 고정하지 않는다.
