# run_research.ps1
# -----------------------------------------------------------------------
# Runs the automated research pipeline for the Alpaca Trading Bot.
# Suitable for manual use or Windows Task Scheduler.
#
# What it does:
#   1. Changes to the project directory
#   2. Activates the virtual environment if one is found
#   3. Runs python run_research.py
#   4. Writes all console output to logs\research_YYYYMMDD_HHMMSS.log
#
# No admin privileges required.
# -----------------------------------------------------------------------

# --- Configuration ------------------------------------------------------
# Absolute path to the project root. Edit this if you move the folder.
$ProjectDir = "C:\Users\lesco\Desktop\AlpacaTradingBot-1"

# Subdirectory names to search for a virtual environment (in order).
$VenvCandidates = @(".venv", "venv", "env")

# Python executable to fall back to if no venv is found.
$FallbackPython = "python"
# -----------------------------------------------------------------------


# ---- Resolve log path --------------------------------------------------
$LogDir = Join-Path $ProjectDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile   = Join-Path $LogDir "research_$Timestamp.log"


# ---- Helper: write a line to both the console and the log file ---------
function Write-Log {
    param([string]$Message)
    $Line = "[$(Get-Date -Format 'HH:mm:ss')] $Message"
    Write-Host $Line
    Add-Content -Path $LogFile -Value $Line -Encoding UTF8
}


# ---- Move to project directory -----------------------------------------
if (-not (Test-Path $ProjectDir)) {
    Write-Error "Project directory not found: $ProjectDir"
    exit 1
}
Set-Location $ProjectDir
Write-Log "Working directory: $ProjectDir"
Write-Log "Log file: $LogFile"


# ---- Activate virtual environment if present ---------------------------
$Python = $FallbackPython
foreach ($Candidate in $VenvCandidates) {
    $ActivateScript = Join-Path $ProjectDir "$Candidate\Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        Write-Log "Activating virtual environment: $Candidate"
        & $ActivateScript
        # After activation, 'python' on PATH resolves to the venv interpreter.
        $Python = "python"
        break
    }
}

if ($Python -eq $FallbackPython) {
    Write-Log "No virtual environment found — using system Python"
}

# Log which interpreter will be used (helps debug environment issues).
$PythonPath = & $Python -c "import sys; print(sys.executable)" 2>&1
Write-Log "Python interpreter: $PythonPath"


# ---- Run the research pipeline -----------------------------------------
Write-Log "Starting run_research.py ..."
Write-Log "---"

# Redirect both stdout and stderr into the log file while also displaying
# them in the console (Tee-Object). PowerShell's 2>&1 merges stderr into
# the success stream so Tee-Object captures both.
& $Python run_research.py 2>&1 | Tee-Object -FilePath $LogFile -Append

$ExitCode = $LASTEXITCODE


# ---- Report result -----------------------------------------------------
Write-Log "---"
if ($ExitCode -eq 0) {
    Write-Log "SUCCESS — pipeline completed (exit code 0)"
} else {
    Write-Log "FAILED  — pipeline exited with code $ExitCode"
}
Write-Log "Full log: $LogFile"

# Exit with the same code as the Python process so Task Scheduler can
# detect failures (Last Run Result will be non-zero on failure).
exit $ExitCode
