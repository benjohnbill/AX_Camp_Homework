# ralph_heartbeat.ps1 - The Universal Heartbeat Engine for Ralph Loop (Heterogeneous Mode)
# Objective: Scan Board -> Update Inboxes -> Physically Wake up the next Agent (Codex/Antigravity/Android Studio)

$LoopStatePath = "orchestration/loop_state.json"
$IntegrationStatusPath = "integration_status.md"

# --- Helper: Find and Run Agent Command ---
function Start-Agent([string]$AgentName, [string]$Reason) {
    Write-Host ">>> WAKING UP AGENT: [$AgentName] (Reason: $Reason) <<<" -ForegroundColor Green
    
    # 1. Update State
    $State = Get-Content $LoopStatePath | ConvertFrom-Json
    $State.last_agent = $AgentName
    $State.last_update = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $State | ConvertTo-Json | Set-Content $LoopStatePath

    # 2. Execution Switch (Heterogeneous Agents)
    switch ($AgentName) {
        "Antigravity" {
            # IDE Based (Frontend/Backend Hybrid)
            $Path = "$env:LOCALAPPDATA\Programs\Antigravity\Antigravity.exe"
            if (Test-Path $Path) { 
                Write-Host "Launching Antigravity IDE..." -ForegroundColor Cyan
                Start-Process $Path -ArgumentList "." # Open current project
            } else {
                Write-Host "Warning: Antigravity.exe not found at $Path. Please focus on your IDE." -ForegroundColor Yellow
            }
            [System.Media.SystemSounds]::Beep.Play() # Audio Trigger
        }
        "Android" {
            # IDE Based (Mobile)
            $Path = "$env:ProgramFiles\Android\Android Studio\bin\studio64.exe"
            if (Test-Path $Path) {
                Write-Host "Launching Android Studio IDE..." -ForegroundColor Cyan
                Start-Process $Path
            } else {
                Write-Host "Warning: studio64.exe not found at $Path. Please focus on Android Studio." -ForegroundColor Yellow
            }
            [System.Media.SystemSounds]::Hand.Play() # Special Audio Trigger
        }
        "Backend" {
            # CLI Based (Codex)
            if (Get-Command "codex" -ErrorAction SilentlyContinue) {
                Write-Host "Starting Codex (5.1-mini) for Backend..." -ForegroundColor Green
                Start-Process "powershell.exe" -ArgumentList "-NoExit", "-Command", "codex -m 'codex-5.1-mini' 'Scan INBOX and integration_status.md. Work on current backend gaps.'"
            } else {
                Write-Host "Error: 'codex' command not found in PATH." -ForegroundColor Red
            }
        }
        "CT" {
            # CLI Based (Control Tower)
            if (Get-Command "codex" -ErrorAction SilentlyContinue) {
                Write-Host "Starting CT CLI..." -ForegroundColor Cyan
                Start-Process "powershell.exe" -ArgumentList "-NoExit", "-Command", "codex 'Review all proposals in orchestration/proposals/ and update status board.'"
            } else {
                Write-Host "Error: 'codex' command not found in PATH." -ForegroundColor Red
            }
        }
    }
}

# --- 1. Loop Health Check ---
$State = Get-Content $LoopStatePath | ConvertFrom-Json
if ($State.status -eq "halted") {
    Write-Host "Loop is currently HALTED. Check USER_INTERVENTION_REQUIRED.md." -ForegroundColor Red
    exit 0
}

if ($State.fail_count -ge 3) {
    Write-Host "Fail count exceeded (3). HALTING loop and escalating to User." -ForegroundColor Red
    $State.status = "halted"
    $State | ConvertTo-Json | Set-Content $LoopStatePath
    # Create or update emergency report
    # (Existing USER_INTERVENTION_REQUIRED.md template will be used)
    exit 1
}

# --- 2. Sync Board & Dispatch Inboxes ---
Write-Host "Pulse: Syncing Board and Inboxes..." -ForegroundColor Gray
# [NOTE] Manual sync required if autonomous skill is not available. 
# Codex_CT will handle the sync logic upon awakening.

# --- 3. Routing Logic (Ralph Loop Decision) ---
$StatusContent = Get-Content $IntegrationStatusPath -Raw

# A. If there is a pending proposal, wake up CT
$Proposals = Get-ChildItem "orchestration/proposals/*.json"
if ($Proposals.Count -gt 0) {
    Start-Agent -AgentName "CT" -Reason "New Proposals to Review"
} 
# B. If a task is active in task.json, wake up the assigned worker
elseif (Test-Path "orchestration/task.json") {
    $Task = Get-Content "orchestration/task.json" | ConvertFrom-Json
    Start-Agent -AgentName $Task.worker -Reason "Assigned Task in task.json"
}
# C. If idling, nudge a proactive agent
else {
    Write-Host "System is idling. Nudging Antigravity for proactive improvements..." -ForegroundColor Gray
    Start-Agent -AgentName "Antigravity" -Reason "Proactive Nudge"
}

Write-Host "Heartbeat Complete. Listening for next signal..." -ForegroundColor Gray
