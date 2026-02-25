# approve_proposal.ps1 - Move a proposal to formal task.json
# Usage: .	ools\approve_proposal.ps1 <proposal_filename>

param (
    [Parameter(Mandatory=$true)]
    [string]$ProposalFile
)

$TaskPath = "orchestration/task.json"
$ProposalDir = "orchestration/proposals"
$ProposalFullPath = Join-Path $ProposalDir $ProposalFile

if (-Not (Test-Path $ProposalFullPath)) {
    Write-Host "Error: Proposal $ProposalFile not found in $ProposalDir." -ForegroundColor Red
    exit 1
}

# 1. Archive current task.json if exists
if (Test-Path $TaskPath) {
    $ArchiveName = "orchestration/task_archive_$(Get-Date -Format 'yyyyMMddHHmmss').json"
    Move-Item $TaskPath $ArchiveName -Force
    Write-Host "Archived current task.json to $ArchiveName" -ForegroundColor Gray
}

# 2. Convert proposal to formal task.json
Move-Item $ProposalFullPath $TaskPath -Force
Write-Host "Proposal $ProposalFile approved and promoted to $TaskPath" -ForegroundColor Green

# 3. Resume the loop if halted
$LoopStatePath = "orchestration/loop_state.json"
if (Test-Path $LoopStatePath) {
    $State = Get-Content $LoopStatePath | ConvertFrom-Json
    $State.status = "running"
    $State.fail_count = 0
    $State.last_update = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $State | ConvertTo-Json | Set-Content $LoopStatePath
    Write-Host "Ralph Loop RESUMED." -ForegroundColor Cyan
}

# 4. Wake up the relevant agent based on the task
# This is a simple nudge. For Antigravity, gemini can be called.
# For Android, it remains a manual trigger or a nudge if the agent is already running.
Write-Host "Triggering the worker loop..." -ForegroundColor Gray
.	oolsalph_heartbeat.ps1
