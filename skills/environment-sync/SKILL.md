# SKILL: environment-sync (Project Instance)

> **Status**: ACTIVE ENGINE (Model B)
> **Reference**: [Master Template](D:\OneDrive\바탕 화면\Life_System\02_Core_Resources\01_Agent_Orchastration_System\SKILL_ENVIRONMENT_SYNC.md)

## 1. Local Goal
Ensure the 
arrative-loop environment matches the latest requirements on this device.

## 2. Dynamic Path Resolution (Local)
- Current Project: 
arrative-loop
- Local Venv Hub: \.venvs_hub\narrative-loop

## 3. Operational Protocol (Model B: Advisory Assistant)

### Step 1: Silent Scan (Post-Pull/Clone)
Whenever a session starts or git pull is detected:
1. **Check venv**: Verify \.venvs_hub\narrative-loop\Scripts\Activate.ps1.
2. **Check requirements**: Scan equirements.txt for updates since last install.
3. **Check .env**: Compare local .env with mandatory keys in MULTI_DEVICE_MIGRATION_GUIDE.md.

### Step 2: Advisory Report
If drift is detected, report to the user:
- [Missing] Virtual Environment
- [Out-of-Sync] Packages (e.g., pandas added)
- [Missing] Secret Keys (e.g., GEMINI_API_KEY)

### Step 3: Ask & Execute (Y/n)
Ask: *'Environment drift detected. Should I automatically sync and fix these issues? (Y/n)'*

Upon 'Y':
1. Create venv if missing: python -m venv \.venvs_hub\narrative-loop
2. Run: pip install -r requirements.txt
3. Prompt for any missing mandatory .env keys.

## 4. Maintenance Rule
When adding a new library or environment variable, the Agent MUST:
1. Update equirements.txt.
2. Update MULTI_DEVICE_MIGRATION_GUIDE.md (Mandatory list).
3. Push changes so other devices can detect the drift.
