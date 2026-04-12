param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktopPath = [Environment]::GetFolderPath("Desktop")
$targetPath = Join-Path $projectRoot "dist\AlpacaTradingBot\AlpacaTradingBot.exe"
$shortcutPath = Join-Path $desktopPath "AlpacaTradingBot.lnk"
$iconPath = $targetPath

if (-not (Test-Path $targetPath)) {
    throw "Built executable not found: $targetPath"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = Split-Path -Parent $targetPath
$shortcut.IconLocation = "$iconPath,0"
$shortcut.WindowStyle = 1
$shortcut.Description = "Launch AlpacaTradingBot"
$shortcut.Save()

Write-Host "Desktop shortcut created: $shortcutPath" -ForegroundColor Green
