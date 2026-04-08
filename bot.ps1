# bot.ps1
# -----------------------------------------------------------------------
# Unified launcher for the most common AlpacaTradingBot workflows.
#
# Examples:
#   .\bot.ps1 preview
#   .\bot.ps1 live
#   .\bot.ps1 setup
#   .\bot.ps1 dataset-short
#   .\bot.ps1 train-model
#   .\bot.ps1 backtest-short
#   .\bot.ps1 compare
#   .\bot.ps1 dashboard
#   .\bot.ps1 research
#   .\bot.ps1 help
# -----------------------------------------------------------------------

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script,

        [string[]]$Arguments = @()
    )

    Write-Host ("python " + $Script + " " + ($Arguments -join " ")).Trim()
    & python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

function Show-Help {
    Write-Host "AlpacaTradingBot launcher"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\bot.ps1 <command>"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup           Verify Python, .env, model, and key datasets"
    Write-Host "  preview         Run trading_bot.py with EXECUTE_ORDERS=false"
    Write-Host "  live            Run trading_bot.py with normal execution behavior"
    Write-Host "  dataset-short   Build the short validation dataset"
    Write-Host "  train-model     Run python -m ml.train"
    Write-Host "  backtest-short  Run the short SMA vs ML vs HYBRID comparison"
    Write-Host "  compare         Run run_compare_suite.ps1"
    Write-Host "  dashboard       Run the Streamlit dashboard"
    Write-Host "  research        Run run_research.py"
    Write-Host "  help            Show this help"
    Write-Host ""
}

function Write-Check {
    param(
        [string]$Label,
        [bool]$Passed,
        [string]$Detail
    )

    $status = if ($Passed) { "OK" } else { "MISSING" }
    $color = if ($Passed) { "Green" } else { "Yellow" }
    Write-Host ("[{0}] {1} - {2}" -f $status, $Label, $Detail) -ForegroundColor $color
}

function Get-NewestDatasetName {
    $datasetsRoot = Join-Path $projectRoot "datasets"
    if (-not (Test-Path $datasetsRoot)) {
        return $null
    }

    $datasetDir = Get-ChildItem -LiteralPath $datasetsRoot -Directory |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1

    if ($null -eq $datasetDir) {
        return $null
    }

    return $datasetDir.Name
}

switch ($Command.ToLowerInvariant()) {
    "setup" {
        Write-Section "Setup Check"

        $checkResults = @()
        function Add-CheckResult {
            param(
                [string]$Label,
                [bool]$Passed,
                [string]$Detail
            )
            $script:checkResults += [pscustomobject]@{
                Label = $Label
                Passed = $Passed
                Detail = $Detail
            }
            Write-Check -Label $Label -Passed $Passed -Detail $Detail
        }

        $pythonOk = $true
        try {
            $pythonVersion = (& python --version) 2>&1
        } catch {
            $pythonOk = $false
            $pythonVersion = "python not found on PATH"
        }
        Add-CheckResult -Label "Python" -Passed $pythonOk -Detail $pythonVersion

        $requirementsPath = Join-Path $projectRoot "requirements.txt"
        Add-CheckResult -Label "requirements.txt" -Passed (Test-Path $requirementsPath) -Detail $requirementsPath

        $importsOk = $false
        $importsDetail = "import check skipped because python is unavailable"
        if ($pythonOk) {
            & python -c "import alpaca, dotenv, pytz, streamlit, pandas, pyarrow, sklearn; print('core imports OK')" 2>$null
            $importsOk = ($LASTEXITCODE -eq 0)
            $importsDetail = if ($importsOk) { "alpaca-py, python-dotenv, pytz, streamlit, pandas, pyarrow, scikit-learn" } else { "one or more required imports failed" }
        }
        Add-CheckResult -Label "Core imports" -Passed $importsOk -Detail $importsDetail

        $envPath = Join-Path $projectRoot ".env"
        $envExists = Test-Path $envPath
        Add-CheckResult -Label ".env" -Passed $envExists -Detail $envPath

        $envKeys = @("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_PAPER", "BOT_SYMBOLS")
        foreach ($key in $envKeys) {
            $present = $false
            if ($envExists) {
                $present = Select-String -Path $envPath -Pattern "^$key=" -Quiet
            }
            Add-CheckResult -Label "env:$key" -Passed $present -Detail (if ($present) { "set" } else { "missing" })
        }

        $modelPath = Join-Path $projectRoot "ml\models\logistic_latest.pkl"
        Add-CheckResult -Label "Offline model" -Passed (Test-Path $modelPath) -Detail $modelPath

        $datasetsRoot = Join-Path $projectRoot "datasets"
        $resultsRoot = Join-Path $projectRoot "results"
        Add-CheckResult -Label "datasets path" -Passed (Test-Path $datasetsRoot) -Detail $datasetsRoot
        Add-CheckResult -Label "results path" -Passed (Test-Path $resultsRoot) -Detail $resultsRoot

        $shortDataset = Join-Path $projectRoot "datasets\AAPL-MSFT-NVDA__15Min__20260101T000000Z__20260201T000000Z__iex__74ddb7ff4b45"
        $baselineDataset = Join-Path $projectRoot "datasets\AAPL-GOOGL-JPM-KO-META-MSFT__15Min__20251004T000000Z__20260404T000000Z__sip__7bdcdff15c5f"
        Add-CheckResult -Label "Short dataset" -Passed (Test-Path $shortDataset) -Detail $shortDataset
        Add-CheckResult -Label "Baseline dataset" -Passed (Test-Path $baselineDataset) -Detail $baselineDataset

        $passedCount = ($checkResults | Where-Object { $_.Passed }).Count
        $failedCount = ($checkResults | Where-Object { -not $_.Passed }).Count
        $summaryColor = if ($failedCount -eq 0) { "Green" } else { "Yellow" }

        Write-Host ""
        Write-Host ("Setup summary: {0} passed, {1} flagged" -f $passedCount, $failedCount) -ForegroundColor $summaryColor
        if ($failedCount -eq 0) {
            Write-Host "Setup looks ready for testing." -ForegroundColor Green
        } else {
            Write-Host "Setup is usable, but review the flagged items above before testing." -ForegroundColor Yellow
        }
        Write-Host ""
        Write-Host "Suggested next commands:" -ForegroundColor Cyan
        Write-Host "  .\bot.ps1 preview"
        Write-Host "  .\bot.ps1 backtest-short"
        Write-Host "  .\bot.ps1 compare"
        break
    }
    "preview" {
        Write-Section "Live Preview"
        $env:EXECUTE_ORDERS = "false"
        Invoke-PythonScript -Script "trading_bot.py"
        break
    }
    "live" {
        Write-Section "Live Bot"
        if (Test-Path Env:EXECUTE_ORDERS) {
            Remove-Item Env:EXECUTE_ORDERS
        }
        Invoke-PythonScript -Script "trading_bot.py"
        break
    }
    "dataset-short" {
        Write-Section "Short Dataset Snapshot"
        $beforeDataset = Get-NewestDatasetName
        Invoke-PythonScript -Script "dataset_snapshotter.py" -Arguments @(
            "--symbols", "AAPL", "MSFT", "NVDA",
            "--start", "2026-01-01T00:00:00Z",
            "--end", "2026-02-01T00:00:00Z",
            "--timeframe", "15Min",
            "--feed", "iex"
        )
        $afterDataset = Get-NewestDatasetName
        Write-Host ""
        if ($null -ne $afterDataset) {
            $datasetPath = Join-Path $projectRoot ("datasets\" + $afterDataset)
            if ($beforeDataset -ne $afterDataset) {
                Write-Host ("Dataset saved to: {0}" -f $datasetPath) -ForegroundColor Green
            } else {
                Write-Host ("Dataset available at: {0}" -f $datasetPath) -ForegroundColor Green
            }
        }
        break
    }
    "train-model" {
        Write-Section "Train Offline Model"
        Write-Host "python -m ml.train"
        & python -m ml.train
        if ($LASTEXITCODE -ne 0) {
            throw "Model training failed with exit code $LASTEXITCODE."
        }
        $modelPath = Join-Path $projectRoot "ml\models\logistic_latest.pkl"
        Write-Host ""
        if (Test-Path $modelPath) {
            Write-Host ("Model training succeeded. Artifact: {0}" -f $modelPath) -ForegroundColor Green
        } else {
            Write-Host "Model training finished, but the expected artifact was not found." -ForegroundColor Yellow
        }
        break
    }
    "backtest-short" {
        Write-Section "Short Backtest"
        Invoke-PythonScript -Script "backtest_runner.py" -Arguments @(
            "--dataset", "datasets/AAPL-MSFT-NVDA__15Min__20260101T000000Z__20260201T000000Z__iex__74ddb7ff4b45",
            "--strategy-mode-list", "sma,ml,hybrid"
        )
        break
    }
    "compare" {
        Write-Section "Compare Suite"
        Write-Host "powershell -ExecutionPolicy Bypass -File run_compare_suite.ps1"
        & powershell -ExecutionPolicy Bypass -File ".\run_compare_suite.ps1"
        if ($LASTEXITCODE -ne 0) {
            throw "Compare suite failed with exit code $LASTEXITCODE."
        }
        break
    }
    "dashboard" {
        Write-Section "Dashboard"
        Write-Host "python -m streamlit run dashboard.py"
        & python -m streamlit run "dashboard.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Dashboard failed with exit code $LASTEXITCODE."
        }
        break
    }
    "research" {
        Write-Section "Research Pipeline"
        Invoke-PythonScript -Script "run_research.py"
        break
    }
    "help" {
        Show-Help
        break
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
