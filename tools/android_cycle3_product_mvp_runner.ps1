[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("all", "auth", "device", "template")]
    [string]$Mode = "all",
    [string]$ExecutionUnit = "cycle03-product-mvp-validation",
    [string]$TraceId = "trace-narrative_loop-20260225-cycle03",
    [string]$GatewayUrl = "https://ax-camp-universe-gateway-staging.onrender.com/gateway/universe_3d",
    [string]$DebugTokenUrl = "https://ax-camp-debug-token-staging.onrender.com/debug/token",
    [string]$ValidUserId = "android-cycle3-valid",
    [string]$ForbiddenUserId = "android-cycle3-forbidden",
    [ValidateRange(1, 120)]
    [int]$TtlMinutes = 10,
    [string]$AdbPath = "",
    [string]$PackageName = "com.example.narrativeloopmobile",
    [string]$ActivityName = "com.example.narrativeloopmobile/.MainActivity",
    [switch]$LaunchApp,
    [switch]$CopyValidToken,
    [switch]$CopyForbiddenToken,
    [string]$JsonOut = "data/evidence/android_cycle3_product_mvp_latest.json",
    [string]$MarkdownOut = "android/NarrativeLoopMobile/evidence/android_cycle3_product_mvp_latest.md"
)

$ErrorActionPreference = "Stop"

function Ensure-ParentDirectory {
    param([string]$PathValue)
    $dir = Split-Path -Parent $PathValue
    if (-not [string]::IsNullOrWhiteSpace($dir) -and -not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

function Invoke-CurlStatus {
    param(
        [string]$Url,
        [string]$Token = "",
        [string]$CookieRead = "",
        [string]$CookieWrite = ""
    )

    $args = @("-s", "-o", "NUL", "-w", "%{http_code}", "--max-redirs", "0")
    if (-not [string]::IsNullOrWhiteSpace($Token)) {
        $args += @("-H", "Authorization: Bearer $Token")
    }
    if (-not [string]::IsNullOrWhiteSpace($CookieRead)) {
        $args += @("-b", $CookieRead)
    }
    if (-not [string]::IsNullOrWhiteSpace($CookieWrite)) {
        $args += @("-c", $CookieWrite)
    }
    $args += $Url

    $statusRaw = & curl.exe @args
    if ($LASTEXITCODE -ne 0) {
        return -1
    }
    $text = ($statusRaw | Out-String).Trim()
    if ($text -notmatch "^\d{3}$") {
        return -1
    }
    return [int]$text
}

function Select-AdbPath {
    param([string]$RequestedPath)

    if (-not [string]::IsNullOrWhiteSpace($RequestedPath) -and (Test-Path -LiteralPath $RequestedPath)) {
        return $RequestedPath
    }
    $candidate = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }
    return "adb"
}

function Parse-AdbDeviceRows {
    param([string[]]$RawLines)
    $rows = @()
    foreach ($line in $RawLines) {
        $trimmed = ($line | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }
        if ($trimmed -like "List of devices attached*") {
            continue
        }
        if ($trimmed -match "^(\S+)\s+(\S+)$") {
            $rows += [ordered]@{
                serial = $matches[1]
                state  = $matches[2]
            }
        }
    }
    return $rows
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$runAuth = $Mode -in @("all", "auth")
$runDevice = $Mode -in @("all", "device")
$runTemplate = $Mode -in @("all", "template")

$utcNow = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
$authResult = $null
$deviceResult = $null
$validToken = ""
$forbiddenToken = ""

if ($runAuth) {
    if ([string]::IsNullOrWhiteSpace($env:DEBUG_TOKEN_ADMIN_KEY)) {
        throw "DEBUG_TOKEN_ADMIN_KEY is missing in current shell environment."
    }

    $headers = @{ "X-Debug-Admin-Key" = $env:DEBUG_TOKEN_ADMIN_KEY }
    $bodyValid = @{ user_id = $ValidUserId; ttl_minutes = $TtlMinutes } | ConvertTo-Json -Compress
    $respValid = Invoke-WebRequest -Method Post `
        -Uri $DebugTokenUrl `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $bodyValid `
        -UseBasicParsing `
        -ErrorAction Stop

    $validPayload = $respValid.Content | ConvertFrom-Json
    $validToken = ("" + $validPayload.token).Trim()
    if ([string]::IsNullOrWhiteSpace($validToken)) {
        throw "Valid debug token issuance succeeded but token field is empty."
    }

    $forbiddenRaw = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot "tools/project_python.ps1") `
        (Join-Path $repoRoot "tools/issue_local_debug_token.py") `
        "--user-id" $ForbiddenUserId `
        "--aud" "forbidden-audience" `
        "--ttl-minutes" ([string]$TtlMinutes)
    if ($LASTEXITCODE -ne 0) {
        throw "Local forbidden token issuance failed."
    }
    $forbiddenToken = (($forbiddenRaw | Out-String).Trim())
    if ([string]::IsNullOrWhiteSpace($forbiddenToken)) {
        throw "Forbidden token output is empty."
    }

    $cookieJar = Join-Path ([System.IO.Path]::GetTempPath()) ("nl_cycle3_cookie_" + [Guid]::NewGuid().ToString("N") + ".txt")
    $validFirstStatus = Invoke-CurlStatus -Url $GatewayUrl -Token $validToken -CookieWrite $cookieJar
    $cookieFollowStatus = Invoke-CurlStatus -Url $GatewayUrl -CookieRead $cookieJar
    $forbiddenStatus = Invoke-CurlStatus -Url $GatewayUrl -Token $forbiddenToken
    $noAuthStatus = Invoke-CurlStatus -Url $GatewayUrl
    if (Test-Path -LiteralPath $cookieJar) {
        Remove-Item -LiteralPath $cookieJar -Force -ErrorAction SilentlyContinue
    }

    if ($CopyValidToken) {
        $validToken | Set-Clipboard
    }
    if ($CopyForbiddenToken) {
        $forbiddenToken | Set-Clipboard
    }

    $authResult = [ordered]@{
        debug_token_issue_status = [int]$respValid.StatusCode
        debug_token_issue_code = ("" + $validPayload.code)
        token_presence = [ordered]@{
            valid_set = (-not [string]::IsNullOrWhiteSpace($validToken))
            forbidden_set = (-not [string]::IsNullOrWhiteSpace($forbiddenToken))
            valid_length = $validToken.Length
            forbidden_length = $forbiddenToken.Length
        }
        gateway_status = [ordered]@{
            valid_first = $validFirstStatus
            cookie_follow_up = $cookieFollowStatus
            forbidden = $forbiddenStatus
            no_auth = $noAuthStatus
        }
        expected = [ordered]@{
            valid_first = 307
            cookie_follow_up = 307
            forbidden = 403
            no_auth = 401
        }
        pass = (
            $validFirstStatus -eq 307 -and
            $cookieFollowStatus -eq 307 -and
            $forbiddenStatus -eq 403 -and
            $noAuthStatus -eq 401
        )
        clipboard = [ordered]@{
            valid_copied = [bool]$CopyValidToken
            forbidden_copied = [bool]$CopyForbiddenToken
        }
    }
}

if ($runDevice) {
    $adbExe = Select-AdbPath -RequestedPath $AdbPath
    $adbRaw = & $adbExe devices 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "adb devices failed. adb_path=$adbExe"
    }

    $rows = Parse-AdbDeviceRows -RawLines $adbRaw
    $onlineRows = @($rows | Where-Object { $_.state -eq "device" })
    $deviceList = @()

    foreach ($row in $rows) {
        $serial = "" + $row.serial
        $state = "" + $row.state
        $model = ""
        $sdk = ""
        $packageInstalled = $false

        if ($state -eq "device") {
            $model = ((& $adbExe -s $serial shell getprop ro.product.model | Out-String).Trim())
            $sdk = ((& $adbExe -s $serial shell getprop ro.build.version.sdk | Out-String).Trim())
            $pkgLines = & $adbExe -s $serial shell pm list packages $PackageName
            $pkgText = ($pkgLines | Out-String)
            $packageInstalled = $pkgText -match ("package:" + [Regex]::Escape($PackageName))
        }

        $deviceList += [ordered]@{
            serial = $serial
            state = $state
            model = $model
            sdk = $sdk
            package_installed = $packageInstalled
        }
    }

    $launchOutcome = [ordered]@{
        attempted = [bool]$LaunchApp
        serial = ""
        ok = $false
        details = "not_requested"
    }

    if ($LaunchApp -and $onlineRows.Count -gt 0) {
        $serial = "" + $onlineRows[0].serial
        $launchText = (& $adbExe -s $serial shell am start -n $ActivityName 2>&1 | Out-String).Trim()
        $launchOutcome = [ordered]@{
            attempted = $true
            serial = $serial
            ok = ($LASTEXITCODE -eq 0)
            details = $launchText
        }
    }

    $deviceResult = [ordered]@{
        adb_path = $adbExe
        connected_rows = $rows.Count
        online_device_count = $onlineRows.Count
        devices = $deviceList
        launch = $launchOutcome
        pass = ($onlineRows.Count -gt 0)
    }
}

$result = [ordered]@{
    generated_at_utc = $utcNow
    trace_id = $TraceId
    execution_unit = $ExecutionUnit
    mode = $Mode
    gateway_url = $GatewayUrl
    debug_token_url = $DebugTokenUrl
    auth = $authResult
    device = $deviceResult
}

if ($runTemplate) {
    $authPass = if ($authResult) { [bool]$authResult.pass } else { $false }
    $devicePass = if ($deviceResult) { [bool]$deviceResult.pass } else { $false }
    $validFirst = if ($authResult) { $authResult.gateway_status.valid_first } else { "n/a" }
    $cookieFollow = if ($authResult) { $authResult.gateway_status.cookie_follow_up } else { "n/a" }
    $forbidden = if ($authResult) { $authResult.gateway_status.forbidden } else { "n/a" }
    $noAuth = if ($authResult) { $authResult.gateway_status.no_auth } else { "n/a" }
    $onlineCount = if ($deviceResult) { $deviceResult.online_device_count } else { "n/a" }
    $lastUpdatedDate = [DateTime]::UtcNow.ToString("yyyy-MM-dd")
    $reviewByDate = [DateTime]::UtcNow.AddDays(2).ToString("yyyy-MM-dd")

    $md = @(
        "---",
        "doc_type: runtime_evidence",
        "owner: android_ide",
        "authority_level: L2",
        "last_updated: $lastUpdatedDate",
        "sync_with:",
        "  - android/NarrativeLoopMobile/ANDROID_REPORT.md",
        "  - data/evidence/android_cycle3_product_mvp_latest.json",
        "change_triggers:",
        "  - cycle3 runtime rerun",
        "  - product mvp checklist update",
        "sunset_condition: Replace on next cycle3 evidence rerun.",
        "review_by: $reviewByDate",
        "---",
        "",
        "# Cycle3 Product MVP Evidence (Sanitized)",
        "",
        "- Generated (UTC): $utcNow",
        "- Trace ID: $TraceId",
        "- Execution unit: $ExecutionUnit",
        "- Mode: $Mode",
        "",
        "## 1) Automation Snapshot",
        "- Auth prep pass: $authPass",
        "- Device probe pass: $devicePass",
        "- Online device count: $onlineCount",
        "- Status pre-check: valid_first=$validFirst, cookie_follow_up=$cookieFollow, forbidden=$forbidden, no_auth=$noAuth",
        "",
        "## 2) Product MVP User Journey Checklist",
        "1. Write new narrative entry in mobile runtime path: [ ] pass / [ ] fail",
        "2. Confirm save success response and UI confirmation: [ ] pass / [ ] fail",
        "3. Restart app and re-open entry list/history: [ ] pass / [ ] fail",
        "4. Re-query and confirm written entry is present: [ ] pass / [ ] fail",
        "5. Open Universe flow and confirm render/redirect path: [ ] pass / [ ] fail",
        "6. Verify auth UX fallback: empty token 401 + forbidden token 403: [ ] pass / [ ] fail",
        "7. Lifecycle smoke (tab switch/background/foreground/resume): [ ] pass / [ ] fail",
        "",
        "## 3) Runtime Evidence Pointers",
        "- App build/install command and output:",
        "- adb/logcat snippets:",
        "- Screenshots path:",
        "- Notes on any manual intervention:",
        "",
        "## 4) Security Notes",
        "- Do not include raw token, admin key, or secret values.",
        "- Include only status codes and sanitized logs."
    ) -join "`n"

    Ensure-ParentDirectory -PathValue $MarkdownOut
    Set-Content -LiteralPath $MarkdownOut -Value $md -Encoding UTF8
    $result["markdown_path"] = $MarkdownOut
}

Ensure-ParentDirectory -PathValue $JsonOut
$json = $result | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $JsonOut -Value $json -Encoding UTF8

Write-Output ("[INFO] json report saved: {0}" -f (Resolve-Path -LiteralPath $JsonOut))
if ($runTemplate) {
    Write-Output ("[INFO] markdown template saved: {0}" -f (Resolve-Path -LiteralPath $MarkdownOut))
}
if ($authResult) {
    Write-Output ("[INFO] auth precheck: valid_first={0}, cookie_follow_up={1}, forbidden={2}, no_auth={3}" -f $authResult.gateway_status.valid_first, $authResult.gateway_status.cookie_follow_up, $authResult.gateway_status.forbidden, $authResult.gateway_status.no_auth)
}
if ($deviceResult) {
    Write-Output ("[INFO] device probe: online_device_count={0}" -f $deviceResult.online_device_count)
}
Write-Output "[PASS] android cycle3 runner completed."
