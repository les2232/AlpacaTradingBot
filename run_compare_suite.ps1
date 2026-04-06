$ErrorActionPreference = "Stop"

Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsRoot = Join-Path $projectRoot "results"
$outputDir = Join-Path $resultsRoot "compare_suite_$timestamp"

if (-not (Test-Path -LiteralPath $resultsRoot)) {
    New-Item -ItemType Directory -Path $resultsRoot | Out-Null
}

New-Item -ItemType Directory -Path $outputDir | Out-Null

function Invoke-BacktestCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Write-Host ("python " + ($Arguments -join " "))

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Backtest command failed during '$Label' with exit code $LASTEXITCODE."
    }
}

$baselineDataset = "datasets/AAPL-GOOGL-JPM-KO-META-MSFT__15Min__20251004T000000Z__20260404T000000Z__sip__7bdcdff15c5f"
$largeDataset = "datasets/AAPL-AMD-AMZN-GOOGL-JPM-KO-META-MSFT-NVDA-TSLA-XOM__15Min__20251004T000000Z__20260404T000000Z__iex__2b7b3366b676"
$shortDataset = "datasets/AAPL-MSFT-NVDA__15Min__20260101T000000Z__20260201T000000Z__iex__74ddb7ff4b45"

$baselineOutput = Join-Path $outputDir "baseline_compare.csv"
$largeOutput = Join-Path $outputDir "large_compare.csv"
$shortOutput = Join-Path $outputDir "short_compare.csv"
$mlSweepOutput = Join-Path $outputDir "baseline_ml_thresholds.csv"

Invoke-BacktestCommand -Label "Job 1 - Baseline comparison" -Arguments @(
    "backtest_runner.py",
    "--dataset", $baselineDataset,
    "--strategy-mode-list", "sma,ml,hybrid",
    "--output-csv", $baselineOutput
)

Invoke-BacktestCommand -Label "Job 2 - Large-universe comparison" -Arguments @(
    "backtest_runner.py",
    "--dataset", $largeDataset,
    "--strategy-mode-list", "sma,ml,hybrid",
    "--output-csv", $largeOutput
)

Invoke-BacktestCommand -Label "Job 3 - Short-window comparison" -Arguments @(
    "backtest_runner.py",
    "--dataset", $shortDataset,
    "--strategy-mode-list", "sma,ml,hybrid",
    "--output-csv", $shortOutput
)

Write-Host ""
Write-Host "=== Job 4 - Baseline ML threshold sweep ===" -ForegroundColor Cyan

$buyThresholds = @(0.52, 0.55, 0.58)
$sellThresholds = @(0.48, 0.45, 0.42)
$mlSweepRows = @()

foreach ($buyThreshold in $buyThresholds) {
    foreach ($sellThreshold in $sellThresholds) {
        $tempOutput = Join-Path $outputDir ("baseline_ml_thresholds_buy_{0}_sell_{1}.csv" -f $buyThreshold.ToString("0.00"), $sellThreshold.ToString("0.00"))

        Invoke-BacktestCommand -Label ("Job 4 - ML threshold sweep buy={0:0.00} sell={1:0.00}" -f $buyThreshold, $sellThreshold) -Arguments @(
            "backtest_runner.py",
            "--dataset", $baselineDataset,
            "--strategy-mode", "ml",
            "--ml-probability-buy", $buyThreshold.ToString("0.00"),
            "--ml-probability-sell", $sellThreshold.ToString("0.00"),
            "--output-csv", $tempOutput
        )

        $rows = Import-Csv -LiteralPath $tempOutput
        foreach ($row in $rows) {
            $row.ml_probability_buy = $buyThreshold.ToString("0.00")
            $row.ml_probability_sell = $sellThreshold.ToString("0.00")
            $mlSweepRows += $row
        }
    }
}

$mlSweepRows |
    Export-Csv -LiteralPath $mlSweepOutput -NoTypeInformation

Write-Host ""
Write-Host "Compare suite complete." -ForegroundColor Green
Write-Host "Results saved to: $outputDir"
