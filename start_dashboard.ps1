param(
    [int]$Port = 8501,
    [switch]$NoBrowser,
    [switch]$DashboardOnly,
    [switch]$AllowUnapprovedRuntime,
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dashboardUrl = "http://localhost:$Port"
$liveLockPath = Join-Path $projectRoot ".live_bot.lock"
$logRoot = Join-Path $projectRoot "logs"

function Test-TcpPortOpen {
    param([int]$TargetPort)

    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $asyncResult = $client.BeginConnect("127.0.0.1", $TargetPort, $null, $null)
        if (-not $asyncResult.AsyncWaitHandle.WaitOne(500)) {
            $client.Close()
            return $false
        }
        $client.EndConnect($asyncResult)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-TcpPortReady {
    param(
        [int]$TargetPort,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-TcpPortOpen -TargetPort $TargetPort) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Resolve-PythonExecutable {
    param([string]$RequestedPython)

    if ($RequestedPython) {
        return $RequestedPython
    }

    $candidates = @()
    if ($env:VIRTUAL_ENV) {
        $candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
    }
    $candidates += @(
        (Join-Path $projectRoot ".venv\Scripts\python.exe"),
        (Join-Path $projectRoot "venv\Scripts\python.exe"),
        "python"
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }
        if (Test-Path $candidate) {
            return $candidate
        }
        try {
            $null = Get-Command $candidate -ErrorAction Stop
            return $candidate
        } catch {
        }
    }

    throw "Could not find a Python executable. Set -PythonExe explicitly or activate the project environment first."
}

function Test-PastSessionFlattenDeadline {
    try {
        $tz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
    } catch {
        return $false
    }

    $nowEt = [System.TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $tz)
    $deadline = Get-Date -Date $nowEt -Hour 15 -Minute 55 -Second 0
    return $nowEt -ge $deadline
}

function Get-LiveLockMetadata {
    if (-not (Test-Path $liveLockPath)) {
        return $null
    }
    try {
        $raw = Get-Content -LiteralPath $liveLockPath -Raw -Encoding UTF8
        return $raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-LatestStartupConfigWriteTime {
    if (-not (Test-Path $logRoot)) {
        return $null
    }

    $latest = Get-ChildItem -Path $logRoot -Recurse -Filter "startup_config.json" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        return $null
    }
    return $latest.LastWriteTimeUtc
}

function Get-LatestStartupSessionInfo {
    if (-not (Test-Path $logRoot)) {
        return $null
    }

    $latest = Get-ChildItem -Path $logRoot -Recurse -Filter "startup_config.json" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        return $null
    }

    $sessionId = $null
    try {
        $payload = Get-Content -LiteralPath $latest.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        $sessionId = "$($payload.session_id)"
    } catch {
        $sessionId = $null
    }

    return [pscustomobject]@{
        session_id = $sessionId
        write_time_utc = $latest.LastWriteTimeUtc
        path = $latest.FullName
    }
}

function Get-LatestLifecycleEventForSession {
    param(
        [string]$SessionId
    )

    if ([string]::IsNullOrWhiteSpace($SessionId) -or -not (Test-Path $logRoot)) {
        return $null
    }

    $lifecyclePath = Get-ChildItem -Path $logRoot -Recurse -Filter "lifecycle.jsonl" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $lifecyclePath) {
        return $null
    }

    $matched = $null
    foreach ($line in Get-Content -LiteralPath $lifecyclePath.FullName -Tail 400 -ErrorAction SilentlyContinue) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $event = $line | ConvertFrom-Json
        } catch {
            continue
        }
        if ("$($event.session_id)" -ne $SessionId) {
            continue
        }
        if ("$($event.event)" -ne "process.lifecycle") {
            continue
        }
        $matched = $event
    }

    return $matched
}

function Test-LiveRuntimeApproval {
    $runtimePath = Join-Path $projectRoot "config\live_config.json"
    if (-not (Test-Path $runtimePath)) {
        return
    }

    try {
        $payload = Get-Content -LiteralPath $runtimePath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return
    }

    if ($AllowUnapprovedRuntime) {
        return
    }

    $approved = $payload.source.approved
    $approvedText = if ($null -eq $approved) { "" } else { "$approved" }
    if ($approvedText.ToLowerInvariant() -ne "false") {
        return
    }

    $reasons = @()
    if ($null -ne $payload.source.rejection_reasons) {
        $reasons = @($payload.source.rejection_reasons | ForEach-Object { "$_" })
    }
    $reasonText = if ($reasons.Count -gt 0) { $reasons -join "; " } else { "no approval reasons recorded" }
    $relativePath = Resolve-Path -LiteralPath $runtimePath | ForEach-Object {
        $_.Path.Replace($projectRoot + "\", "")
    }
    throw "Refusing to start live trading because $relativePath is marked approved=false ($reasonText). Re-run with -AllowUnapprovedRuntime only if you intend to override the approval gate."
}

function Test-ProcessRunning {
    param([int]$TargetProcessId)

    if ($TargetProcessId -le 0) {
        return $false
    }

    try {
        $null = Get-Process -Id $TargetProcessId -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Get-ProcessIdentity {
    param([int]$TargetProcessId)

    if ($TargetProcessId -le 0) {
        return $null
    }

    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $TargetProcessId" -ErrorAction Stop
    } catch {
        try {
            $proc = Get-Process -Id $TargetProcessId -ErrorAction Stop
        } catch {
            return $null
        }

        $startedAtUtc = $null
        try {
            if ($proc.StartTime) {
                $startedAtUtc = $proc.StartTime.ToUniversalTime().ToString("o")
            }
        } catch {
            $startedAtUtc = $null
        }

        return [pscustomobject]@{
            pid = [int]$proc.Id
            started_at_utc = $startedAtUtc
            command_line = "python -m tradeos live"
        }
    }

    if ($null -eq $proc) {
        return $null
    }

    $startedAtUtc = $null
    if ($proc.CreationDate) {
        try {
            $startedAtUtc = ([System.Management.ManagementDateTimeConverter]::ToDateTime($proc.CreationDate)).ToUniversalTime().ToString("o")
        } catch {
            $startedAtUtc = $null
        }
    }

    [pscustomobject]@{
        pid = [int]$proc.ProcessId
        started_at_utc = $startedAtUtc
        command_line = [string]$proc.CommandLine
    }
}

function Get-RunningLiveBotProcesses {
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction Stop
    } catch {
        return @()
    }

    $liveBotProcesses = @()
    foreach ($proc in @($procs)) {
        $commandLine = [string]$proc.CommandLine
        if ([string]::IsNullOrWhiteSpace($commandLine)) {
            continue
        }

        $normalized = $commandLine.ToLowerInvariant()
        if ($normalized -notmatch "tradeos" -or $normalized -notmatch "(^|\\s)live(\\s|$)") {
            continue
        }

        $startedAtUtc = $null
        if ($proc.CreationDate) {
            try {
                $startedAtUtc = ([System.Management.ManagementDateTimeConverter]::ToDateTime($proc.CreationDate)).ToUniversalTime().ToString("o")
            } catch {
                $startedAtUtc = $null
            }
        }

        $liveBotProcesses += [pscustomobject]@{
            pid = [int]$proc.ProcessId
            started_at_utc = $startedAtUtc
            command_line = $commandLine
        }
    }

    return @($liveBotProcesses)
}

function Write-LiveLockMetadata {
    param(
        [Parameter(Mandatory = $true)]
        $Identity
    )

    $payload = [ordered]@{
        pid = [int]$Identity.pid
        command = "tradeos live"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        workspace = $projectRoot
        process_started_at_utc = $Identity.started_at_utc
    }

    $json = $payload | ConvertTo-Json -Compress
    Set-Content -LiteralPath $liveLockPath -Value $json -Encoding UTF8
}

function Test-LiveBotIdentity {
    param(
        [Parameter(Mandatory = $true)]
        $Metadata,
        [int]$ExpectedPid = 0
    )

    if ($null -eq $Metadata) {
        return $false
    }

    $pidValue = 0
    if (-not [int]::TryParse("$($Metadata.pid)", [ref]$pidValue)) {
        return $false
    }
    if ($ExpectedPid -gt 0 -and $pidValue -ne $ExpectedPid) {
        return $false
    }
    if (-not (Test-ProcessRunning -TargetProcessId $pidValue)) {
        return $false
    }

    $identity = Get-ProcessIdentity -TargetProcessId $pidValue
    if ($null -eq $identity) {
        return $false
    }

    $commandLine = "$($identity.command_line)".ToLowerInvariant()
    if ($commandLine -notmatch "tradeos" -or $commandLine -notmatch "(^|\\s)live(\\s|$)") {
        return $false
    }

    $lockedStartedAt = "$($Metadata.process_started_at_utc)"
    if (-not [string]::IsNullOrWhiteSpace($lockedStartedAt)) {
        return $lockedStartedAt -eq "$($identity.started_at_utc)"
    }

    return "$($Metadata.command)".ToLowerInvariant() -eq "tradeos live"
}

function Test-LaunchedLiveBotReady {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        $Metadata
    )

    if ($null -eq $Metadata) {
        return $false
    }

    $pidValue = 0
    if (-not [int]::TryParse("$($Metadata.pid)", [ref]$pidValue)) {
        return $false
    }
    if ($pidValue -ne $Process.Id) {
        return $false
    }

    if (-not (Test-ProcessRunning -TargetProcessId $pidValue)) {
        return $false
    }

    return $true
}

function Wait-LiveBotReady {
    param(
        [System.Diagnostics.Process]$Process,
        $PreviousStartupConfigWriteTime = $null,
        [int]$TimeoutSeconds = 15
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            return $false
        }
        $metadata = Get-LiveLockMetadata
        if ($null -eq $metadata -or -not (Test-LaunchedLiveBotReady -Process $Process -Metadata $metadata)) {
            Start-Sleep -Milliseconds 250
            continue
        }
        $sessionInfo = Get-LatestStartupSessionInfo
        if ($null -eq $sessionInfo) {
            Start-Sleep -Milliseconds 250
            continue
        }
        if ($null -ne $PreviousStartupConfigWriteTime -and $sessionInfo.write_time_utc -le $PreviousStartupConfigWriteTime) {
            Start-Sleep -Milliseconds 250
            continue
        }
        $lifecycleEvent = Get-LatestLifecycleEventForSession -SessionId $sessionInfo.session_id
        if ($null -eq $lifecycleEvent) {
            Start-Sleep -Milliseconds 250
            continue
        }
        if ("$($lifecycleEvent.stage)" -eq "startup.ready") {
            return $true
        }
        if ("$($lifecycleEvent.stage)" -eq "startup.failed") {
            return $false
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Start-LiveBot {
    $existing = Get-LiveLockMetadata
    if ($null -ne $existing -and (Test-LiveBotIdentity -Metadata $existing)) {
        Write-Host "Reusing existing live bot process (pid=$($existing.pid))"
        return
    }

    $runningLiveBots = @(Get-RunningLiveBotProcesses)
    if ($runningLiveBots.Count -gt 1) {
        $pidList = ($runningLiveBots | ForEach-Object { "$($_.pid)" }) -join ", "
        throw "Refusing to start another live bot because multiple 'python -m tradeos live' processes are already running without a recoverable single lock owner. Running PIDs: $pidList"
    }
    if ($runningLiveBots.Count -eq 1) {
        $orphan = $runningLiveBots[0]
        try {
            Write-LiveLockMetadata -Identity $orphan
            Write-Warning "Recovered missing live bot lock for existing process pid=$($orphan.pid)."
            Write-Host "Reusing existing live bot process (pid=$($orphan.pid))"
            return
        } catch {
            throw "Found an existing live bot process (pid=$($orphan.pid)) but could not recover ${liveLockPath}: $($_.Exception.Message)"
        }
    }

    if (Test-PastSessionFlattenDeadline) {
        Write-Host "Skipping live bot start because the session is past the 3:55 PM ET flatten deadline."
        return
    }

    Write-Host "Starting live bot via 'python -m tradeos live'"
    if ($AllowUnapprovedRuntime) {
        Write-Host "WARNING: allowing unapproved runtime config for this launch only."
    }
    Test-LiveRuntimeApproval
    $previousStartupConfigWriteTime = Get-LatestStartupConfigWriteTime
    $previousOverride = $env:TRADEOS_ALLOW_UNAPPROVED_RUNTIME
    $stdoutPath = Join-Path $logRoot "live_bot_stdout.log"
    $stderrPath = Join-Path $logRoot "live_bot_stderr.log"
    $launchedWithoutRedirection = $false
    try {
        if ($AllowUnapprovedRuntime) {
            $env:TRADEOS_ALLOW_UNAPPROVED_RUNTIME = "true"
        } else {
            Remove-Item Env:TRADEOS_ALLOW_UNAPPROVED_RUNTIME -ErrorAction SilentlyContinue
        }
        try {
            $process = Start-Process `
                -FilePath $python `
                -ArgumentList @("-m", "tradeos", "live") `
                -WorkingDirectory $projectRoot `
                -RedirectStandardOutput $stdoutPath `
                -RedirectStandardError $stderrPath `
                -PassThru `
                -ErrorAction Stop
        } catch [System.ArgumentException] {
            $message = $_.Exception.Message
            if ($message -notmatch "Key in dictionary: 'Path'.*Key being added: 'PATH'") {
                throw
            }
            Write-Warning (
                "PowerShell could not launch the live bot with redirected stdout/stderr because this session " +
                "contains duplicate Path/PATH environment keys. Retrying without redirection."
            )
            $process = Start-Process `
                -FilePath $python `
                -ArgumentList @("-m", "tradeos", "live") `
                -WorkingDirectory $projectRoot `
                -PassThru `
                -ErrorAction Stop
            $launchedWithoutRedirection = $true
        }
    } finally {
        if ($null -eq $previousOverride) {
            Remove-Item Env:TRADEOS_ALLOW_UNAPPROVED_RUNTIME -ErrorAction SilentlyContinue
        } else {
            $env:TRADEOS_ALLOW_UNAPPROVED_RUNTIME = $previousOverride
        }
    }

    if (-not (Wait-LiveBotReady -Process $process -PreviousStartupConfigWriteTime $previousStartupConfigWriteTime)) {
        $latestStartupConfigWriteTime = Get-LatestStartupConfigWriteTime
        if ($null -ne $latestStartupConfigWriteTime -and ($null -eq $previousStartupConfigWriteTime -or $latestStartupConfigWriteTime -gt $previousStartupConfigWriteTime)) {
            $latestMetadata = Get-LiveLockMetadata
            if ($null -ne $latestMetadata -and -not (Test-LaunchedLiveBotReady -Process $process -Metadata $latestMetadata)) {
                try {
                    Remove-Item -LiteralPath $liveLockPath -Force -ErrorAction Stop
                    Write-Warning "Recovered a stale live bot lock after startup metadata was written."
                } catch {
                }
            }
            throw (
                "Live bot wrote startup metadata, but the launcher could not confirm startup readiness in time. " +
                "Failing closed to avoid overlapping live sessions. " +
                "Inspect logs at $stdoutPath and $stderrPath before retrying."
            )
        }
        throw "Live bot did not reach startup readiness within the timeout window."
    }

    $metadata = Get-LiveLockMetadata
    if ($null -ne $metadata -and $metadata.pid) {
        Write-Host "Live bot ready (pid=$($metadata.pid))"
    }
    if ($launchedWithoutRedirection) {
        Write-Warning "Live bot launched without stdout/stderr redirection; inspect JSONL logs and dashboard artifacts for runtime activity."
    } else {
        Write-Host "Live bot stdout: $stdoutPath"
        Write-Host "Live bot stderr: $stderrPath"
    }
}

function Start-Dashboard {
    if (Test-TcpPortOpen -TargetPort $Port) {
        Write-Host "Reusing existing dashboard at $dashboardUrl"
        return
    }

    Write-Host "Starting dashboard via 'python -m tradeos dashboard' on $dashboardUrl"
    Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "tradeos", "dashboard", "--port", "$Port") `
        -WorkingDirectory $projectRoot

    if (-not (Wait-TcpPortReady -TargetPort $Port)) {
        throw "Dashboard did not become ready within the timeout window."
    }
}

$python = Resolve-PythonExecutable -RequestedPython $PythonExe

if (-not $DashboardOnly) {
    Start-LiveBot
}

Start-Dashboard

if (-not $NoBrowser) {
    Start-Process $dashboardUrl
}

if ($DashboardOnly) {
    Write-Host "Dashboard ready at $dashboardUrl"
} else {
    Write-Host "TradeOS live session and dashboard ready."
    Write-Host "Dashboard: $dashboardUrl"
    if ($AllowUnapprovedRuntime) {
        Write-Host "Runtime approval override was enabled for this launch."
    }
}
