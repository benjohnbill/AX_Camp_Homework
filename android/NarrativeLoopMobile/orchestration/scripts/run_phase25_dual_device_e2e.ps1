param(
    [string]$Timestamp = (Get-Date -AsUTC -Format "yyyyMMddTHHmmssZ"),
    [string]$RepoRoot = "D:\dev\Narrative_Loop",
    [string]$EmulatorSerial = "emulator-5554",
    [string]$PhysicalSerial = "R3CR80HR90W"
)

$ErrorActionPreference = "Stop"

$adb = "C:\Users\benjohnbill\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$appId = "com.example.narrativeloopmobile"
$package = "com.example.narrativeloopmobile"
$activity = ".MainActivity"

$evidenceDir = Join-Path $RepoRoot "android\NarrativeLoopMobile\evidence"
New-Item -ItemType Directory -Force -Path $evidenceDir | Out-Null
$walkthroughPath = Join-Path $evidenceDir "${Timestamp}_android_phase25_e2e_walkthrough.md"
$logcatPath = Join-Path $evidenceDir "${Timestamp}_android_phase25_e2e_logcat.log"

function Invoke-Adb {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    (& $adb -s $Serial @Args 2>&1 | Out-String)
}

function Dump-UiXml {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [Parameter(Mandatory = $true)][string]$Name
    )
    Invoke-Adb -Serial $Serial -Args @("shell", "uiautomator", "dump", "/sdcard/$Name") | Out-Null
    Invoke-Adb -Serial $Serial -Args @("shell", "cat", "/sdcard/$Name")
}

function Get-BoundsForId {
    param(
        [Parameter(Mandatory = $true)][string]$Xml,
        [Parameter(Mandatory = $true)][string]$Id
    )
    $pattern = "resource-id=`"$($appId):id/$Id`"[^>]*bounds=`"\[(\d+),(\d+)\]\[(\d+),(\d+)\]`""
    $match = [regex]::Match($Xml, $pattern)
    if (-not $match.Success) {
        return $null
    }
    [pscustomobject]@{
        x1 = [int]$match.Groups[1].Value
        y1 = [int]$match.Groups[2].Value
        x2 = [int]$match.Groups[3].Value
        y2 = [int]$match.Groups[4].Value
    }
}

function Get-TextForId {
    param(
        [Parameter(Mandatory = $true)][string]$Xml,
        [Parameter(Mandatory = $true)][string]$Id
    )
    $pattern = "text=`"([^`"]*)`"\s+resource-id=`"$($appId):id/$Id`""
    $match = [regex]::Match($Xml, $pattern)
    if ($match.Success) {
        return $match.Groups[1].Value
    }
    return ""
}

function Tap-ById {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [Parameter(Mandatory = $true)][string]$Id,
        [int]$MaxAttempts = 6
    )
    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        $xml = Dump-UiXml -Serial $Serial -Name "nl_tap_$Id.xml"
        $bounds = Get-BoundsForId -Xml $xml -Id $Id
        if ($null -ne $bounds) {
            $x = [int](($bounds.x1 + $bounds.x2) / 2)
            $y = [int](($bounds.y1 + $bounds.y2) / 2)
            Invoke-Adb -Serial $Serial -Args @("shell", "input", "tap", "$x", "$y") | Out-Null
            Start-Sleep -Milliseconds 900
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Wait-ForId {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [Parameter(Mandatory = $true)][string]$Id,
        [int]$TimeoutSec = 30
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $xml = Dump-UiXml -Serial $Serial -Name "nl_wait_$Id.xml"
        if ($xml -match [regex]::Escape("$($appId):id/$Id")) {
            return $true
        }
        Start-Sleep -Milliseconds 700
    }
    return $false
}

function Ensure-CreateScreen {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [int]$TimeoutSec = 40
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $xml = Dump-UiXml -Serial $Serial -Name "nl_ensure_create.xml"
        if ($xml -match [regex]::Escape("$($appId):id/edit_text_narrative")) {
            return $true
        }

        if ($xml -match [regex]::Escape("$($appId):id/button_write_narrative")) {
            [void](Tap-ById -Serial $Serial -Id "button_write_narrative" -MaxAttempts 1)
            Start-Sleep -Milliseconds 900
            continue
        }

        if ($xml -match [regex]::Escape("$($appId):id/nav_create_narrative")) {
            [void](Tap-ById -Serial $Serial -Id "nav_create_narrative" -MaxAttempts 1)
            Start-Sleep -Milliseconds 900
            continue
        }

        Start-Sleep -Milliseconds 700
    }
    return $false
}

function Input-TextById {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$Text
    )
    if (-not (Tap-ById -Serial $Serial -Id $Id)) {
        throw "Cannot focus id=$Id on serial=$Serial"
    }
    $sanitized = ($Text -replace "\s+", "%s") -replace "[^A-Za-z0-9%._-]", ""
    Invoke-Adb -Serial $Serial -Args @("shell", "input", "text", $sanitized) | Out-Null
    Start-Sleep -Milliseconds 700
    Invoke-Adb -Serial $Serial -Args @("shell", "input", "keyevent", "4") | Out-Null
    Start-Sleep -Milliseconds 400
}

function Read-CreateState {
    param(
        [Parameter(Mandatory = $true)][string]$Serial
    )
    $xml = Dump-UiXml -Serial $Serial -Name "nl_state.xml"
    [pscustomobject]@{
        stage = Get-TextForId -Xml $xml -Id "text_stage_state"
        status = Get-TextForId -Xml $xml -Id "text_action_status"
        evidence = Get-TextForId -Xml $xml -Id "text_evidence_state"
        xml = $xml
    }
}

function Wait-ForCompletion {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [int]$TimeoutSec = 220
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $state = Read-CreateState -Serial $Serial
    while ((Get-Date) -lt $deadline) {
        $state = Read-CreateState -Serial $Serial
        if ($state.status -like "Status: E2E save completed*") {
            return $state
        }
        if ($state.status -like "Status: E2E save flow failed*") {
            return $state
        }
        Start-Sleep -Seconds 2
    }
    return $state
}

function Run-Scenario {
    param(
        [Parameter(Mandatory = $true)][string]$Serial,
        [Parameter(Mandatory = $true)][ValidateSet("plan", "focus")][string]$Mode,
        [Parameter(Mandatory = $true)][string]$Narrative,
        [bool]$RunRefine = $false
    )
    Invoke-Adb -Serial $Serial -Args @("logcat", "-c") | Out-Null
    Invoke-Adb -Serial $Serial -Args @("shell", "am", "force-stop", $package) | Out-Null
    Invoke-Adb -Serial $Serial -Args @("shell", "am", "start", "-n", "$package/$activity") | Out-Null
    Start-Sleep -Seconds 2

    if (-not (Ensure-CreateScreen -Serial $Serial -TimeoutSec 45)) {
        throw "Create screen not reached on serial=$Serial"
    }

    Input-TextById -Serial $Serial -Id "edit_text_narrative" -Text $Narrative

    if ($Mode -eq "focus") {
        [void](Tap-ById -Serial $Serial -Id "radio_mode_focus")
    } else {
        [void](Tap-ById -Serial $Serial -Id "radio_mode_plan")
    }

    if ($RunRefine) {
        if (Tap-ById -Serial $Serial -Id "button_ai_refine") {
            Start-Sleep -Seconds 5
        }
    }

    if (-not (Tap-ById -Serial $Serial -Id "button_save_narrative")) {
        throw "Save button not available on serial=$Serial"
    }

    $final = Wait-ForCompletion -Serial $Serial -TimeoutSec 220
    $modeLabel = if ($Mode -eq "focus") { "Focus-first" } else { "Plan-first" }
    $statusOk = $final.status -like "Status: E2E save completed ($modeLabel)*"
    $imageOk = $false
    $imageEvent = ""
    $linkedCount = -1
    if ($final.evidence -match "image_event_id=([^ /]+)") {
        $imageEvent = $matches[1]
        $imageOk = $imageEvent -ne "n/a"
    }
    if ($final.evidence -match "linked_count=(\d+)") {
        $linkedCount = [int]$matches[1]
    }
    $pass = $statusOk -and $imageOk -and ($linkedCount -gt 0)

    $logLines = Invoke-Adb -Serial $Serial -Args @("logcat", "-d", "-v", "time", "CreateNarrativeFragment:I", "*:S")

    [pscustomobject]@{
        serial = $Serial
        mode = $Mode
        stage = $final.stage
        status = $final.status
        evidence = $final.evidence
        image_event_id = $imageEvent
        linked_count = $linkedCount
        status_ok = $statusOk
        image_ok = $imageOk
        pass = $pass
        logcat = $logLines
    }
}

function Get-TopActivitySnippet {
    param(
        [Parameter(Mandatory = $true)][string]$Serial
    )
    $raw = Invoke-Adb -Serial $Serial -Args @("shell", "dumpsys", "activity", "activities")
    $lines = $raw -split "`r?`n" | Where-Object {
        $_ -match "com.example.narrativeloopmobile/.MainActivity" -or $_ -match "topResumedActivity" -or $_ -match "Task\{"
    }
    (($lines | Select-Object -First 14) -join "`n")
}

$devicesOutput = (& $adb devices -l | Out-String)
if ($devicesOutput -notmatch [regex]::Escape($EmulatorSerial)) {
    throw "Emulator not connected: $EmulatorSerial"
}
if ($devicesOutput -notmatch [regex]::Escape($PhysicalSerial)) {
    throw "Physical device not connected: $PhysicalSerial"
}

$scA = Run-Scenario -Serial $EmulatorSerial -Mode "plan" -Narrative "phase25 plan emulator e2e proof" -RunRefine $false
$scB = Run-Scenario -Serial $EmulatorSerial -Mode "focus" -Narrative "phase25 focus emulator e2e proof" -RunRefine $false

$scC = $scA.image_ok -and $scB.image_ok -and ($scA.linked_count -gt 0) -and ($scB.linked_count -gt 0)

Invoke-Adb -Serial $EmulatorSerial -Args @("shell", "am", "start", "-n", "$package/$activity") | Out-Null
Invoke-Adb -Serial $PhysicalSerial -Args @("shell", "am", "start", "-n", "$package/$activity") | Out-Null
Start-Sleep -Seconds 2
$topPhysical = Get-TopActivitySnippet -Serial $PhysicalSerial
$topEmulator = Get-TopActivitySnippet -Serial $EmulatorSerial
$scD = ($topPhysical -match "MainActivity") -and ($topEmulator -match "MainActivity")

$logcatContent = @"
# Android Phase2.5 E2E Logcat ($Timestamp)

## Devices
$devicesOutput

## SC-A / $($scA.serial) / mode=$($scA.mode)
$($scA.logcat)

## SC-B / $($scB.serial) / mode=$($scB.mode)
$($scB.logcat)
"@
Set-Content -Path $logcatPath -Value $logcatContent -Encoding utf8

$overall = $scA.pass -and $scB.pass -and $scC -and $scD

$walkthrough = @"
# Android Phase2.5 E2E Walkthrough ($Timestamp)

- trace_id: trace-narrative_loop-20260305-rp25-it2
- task_id: T-nl-20260305-rp25-it2-android

## Device Window
```
$devicesOutput
```

## SC-A (Plan-first complete on emulator)
- serial: $($scA.serial)
- mode: $($scA.mode)
- stage: $($scA.stage)
- status: $($scA.status)
- evidence: $($scA.evidence)
- pass: $($scA.pass)

## SC-B (Focus-first + retro complete)
- serial: $($scB.serial)
- mode: $($scB.mode)
- stage: $($scB.stage)
- status: $($scB.status)
- evidence: $($scB.evidence)
- pass: $($scB.pass)

## SC-C (OCR image_event -> reflect evidence_links)
- sc_a image_event_id: $($scA.image_event_id)
- sc_b image_event_id: $($scB.image_event_id)
- sc_a linked_count: $($scA.linked_count)
- sc_b linked_count: $($scB.linked_count)
- decision: $scC

## SC-D (physical/emulator same-window core flow)
- physical top activity snippet:
```
$topPhysical
```
- emulator top activity snippet:
```
$topEmulator
```
- decision: $scD

## Verdict
- success_gate: $overall
- logcat_path: $logcatPath
"@
Set-Content -Path $walkthroughPath -Value $walkthrough -Encoding utf8

$summary = [ordered]@{
    timestamp = $Timestamp
    walkthrough_path = $walkthroughPath
    logcat_path = $logcatPath
    sc_a = [bool]$scA.pass
    sc_b = [bool]$scB.pass
    sc_c = [bool]$scC
    sc_d = [bool]$scD
    success_gate = [bool]$overall
}
$summary | ConvertTo-Json -Depth 6
