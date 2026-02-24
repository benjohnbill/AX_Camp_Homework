[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$VenvRoot = "",
    [string]$ProjectKey = "narrative_loop",
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs
)

$ErrorActionPreference = "Stop"

$defaultVenvRoot = Join-Path $env:USERPROFILE ".venvs_hub"
if ($env:USERPROFILE -match "[^\u0000-\u007F]") {
    $defaultVenvRoot = "C:\venvs_hub"
}
$resolvedVenvRoot = if ($VenvRoot) { $VenvRoot } elseif ($env:LIFE_VENV_ROOT) { $env:LIFE_VENV_ROOT } else { $defaultVenvRoot }
$projectCandidates = @()
if ($ProjectKey -eq "narrative_loop") {
    # Canonical venv path first, then legacy compatibility alias.
    $projectCandidates += "Narrative_Loop.venv"
    $projectCandidates += "narrative_loop"
} else {
    $projectCandidates += $ProjectKey
}

$pythonPath = $null
foreach ($candidate in $projectCandidates) {
    $candidatePython = Join-Path (Join-Path $resolvedVenvRoot $candidate) "Scripts\python.exe"
    if (Test-Path -LiteralPath $candidatePython) {
        $pythonPath = $candidatePython
        break
    }
}

if (-not $pythonPath) {
    $searched = $projectCandidates | ForEach-Object { Join-Path (Join-Path $resolvedVenvRoot $_) "Scripts\python.exe" }
    throw "Project python not found. Searched:`n$($searched -join "`n")`nRun .\\tools\\bootstrap_env.ps1 first."
}

if ($PyArgs.Count -eq 0) {
    & $pythonPath --version
    Write-Output $pythonPath
    exit $LASTEXITCODE
}

& $pythonPath @PyArgs
exit $LASTEXITCODE
