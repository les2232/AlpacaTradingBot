param(
    [string]$Dataset = ".\datasets\39syms_10535fab__15Min__20251004T000000Z__20260404T000000Z__iex__18c5b2aebc6b",
    [string]$RunName = "",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$argsList = @(
    ".\run_backtest_experiments.py",
    "--dataset", $Dataset
)

if ($RunName) {
    $argsList += @("--run-name", $RunName)
}

if ($SkipExisting) {
    $argsList += "--skip-existing"
}

Write-Host "Running breakout experiment batch..." -ForegroundColor Cyan
Write-Host ("python " + ($argsList -join " "))

& python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Experiment batch failed with exit code $LASTEXITCODE."
}

Write-Host "Done." -ForegroundColor Green
