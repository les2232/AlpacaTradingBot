param(
    [int]$Port = 8501,
    [switch]$NoBrowser,
    [switch]$DashboardOnly,
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
        [int]$TimeoutSeconds = 20
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

function Wait-LiveBotReady {
    param(
        [System.Diagnostics.Process]$Process,
        [datetime]$PreviousStartupConfigWriteTime,
        [int]$TimeoutSeconds = 15
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $metadata = Get-LiveLockMetadata
        if ($null -ne $metadata) {
            $pidText = "$($metadata.pid)"
            $pidValue = 0
            if ([int]::TryParse($pidText, [ref]$pidValue) -and (Test-ProcessRunning -TargetProcessId $pidValue)) {
                return $true
            }
        }
        $latestStartupWriteTime = Get-LatestStartupConfigWriteTime
        if ($null -ne $latestStartupWriteTime) {
            if ($null -eq $PreviousStartupConfigWriteTime -or $latestStartupWriteTime -gt $PreviousStartupConfigWriteTime) {
                return $true
            }
        }
        if ($Process.HasExited) {
            return $false
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Start-LiveBot {
    $existing = Get-LiveLockMetadata
    if ($null -ne $existing) {
        $pidText = "$($existing.pid)"
        $pidValue = 0
        if ([int]::TryParse($pidText, [ref]$pidValue) -and (Test-ProcessRunning -TargetProcessId $pidValue)) {
            Write-Host "Reusing existing live bot process (pid=$pidValue)"
            return
        }
    }

    Write-Host "Starting live bot via 'python -m tradeos live'"
    $previousStartupConfigWriteTime = Get-LatestStartupConfigWriteTime
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "tradeos", "live") `
        -WorkingDirectory $projectRoot `
        -PassThru

    if (-not (Wait-LiveBotReady -Process $process -PreviousStartupConfigWriteTime $previousStartupConfigWriteTime)) {
        throw "Live bot did not create a healthy lock file within the timeout window."
    }

    $metadata = Get-LiveLockMetadata
    if ($null -ne $metadata -and $metadata.pid) {
        Write-Host "Live bot ready (pid=$($metadata.pid))"
    } else {
        if ($process.HasExited) {
            Write-Host "Live bot started and exited quickly, likely because the session is already outside trading hours."
        } else {
            Write-Host "Live bot started."
        }
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
}
