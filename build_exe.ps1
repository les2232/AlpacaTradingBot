param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$distRoot = Join-Path $projectRoot "dist\AlpacaTradingBot"
$buildRoot = Join-Path $projectRoot "build\AlpacaTradingBot"
$iconScript = Join-Path $projectRoot "tools\generate_app_icon.py"

Write-Host "Generating app icon..." -ForegroundColor Cyan
python $iconScript

if (Test-Path $distRoot) {
    Write-Host "Removing previous dist output..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $distRoot -Recurse -Force
}

if (Test-Path $buildRoot) {
    Write-Host "Removing previous build output..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}

Write-Host "Building AlpacaTradingBot.exe..." -ForegroundColor Cyan
python -m PyInstaller --noconfirm --clean .\AlpacaTradingBot.spec

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "App folder: $projectRoot\dist\AlpacaTradingBot"
Write-Host "Executable: $projectRoot\dist\AlpacaTradingBot\AlpacaTradingBot.exe"
