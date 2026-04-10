# bot.ps1
# -----------------------------------------------------------------------
# Unified launcher for the most common AlpacaTradingBot workflows.
#
# Examples:
#   .\bot.ps1 preview
#   .\bot.ps1 live
#   .\bot.ps1 setup
#   .\bot.ps1 dataset-short
#   .\bot.ps1 dataset-liquid
#   .\bot.ps1 research-liquid
#   .\bot.ps1 show-config
#   .\bot.ps1 preflight
#   .\bot.ps1 paper-run
#   .\bot.ps1 live-run --confirm-live
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
    [string]$Command = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
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
    Write-Host "  .\bot.ps1 <command> [--raw] [--json]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup           Verify Python, .env, model, and key datasets"
    Write-Host "  preview         Run trading_bot.py with EXECUTE_ORDERS=false"
    Write-Host "  live            Run trading_bot.py with normal execution behavior"
    Write-Host "  dataset-short   Build the short validation dataset"
    Write-Host "  dataset-liquid  Build a broader liquid-universe validation dataset"
    Write-Host "  research-liquid Build a liquid-universe dataset, then run dataset-driven research"
    Write-Host "  show-config     Show the promoted live runtime config summary"
    Write-Host "  preflight       Check trading readiness before launching the bot"
    Write-Host "  paper-run       Run preflight, then launch trading_bot.py with EXECUTE_ORDERS=false"
    Write-Host "  live-run        Run preflight, require --confirm-live, then launch with EXECUTE_ORDERS=true"
    Write-Host "  train-model     Run python -m ml.train"
    Write-Host "  backtest-short  Run the short SMA vs ML vs HYBRID comparison"
    Write-Host "  compare         Run run_compare_suite.ps1"
    Write-Host "  dashboard       Run the Streamlit dashboard"
    Write-Host "  research        Run run_research.py"
    Write-Host "  help            Show this help"
    Write-Host ""
}

function Get-FlagPresent {
    param([string]$Flag)
    return ($script:ExtraArgs -contains $Flag)
}

function Get-DotEnvValue {
    param([string]$Key)

    $envPath = Join-Path $projectRoot ".env"
    if (-not (Test-Path $envPath)) {
        return $null
    }

    $line = Select-String -Path $envPath -Pattern ("^{0}=(.*)$" -f [regex]::Escape($Key)) | Select-Object -First 1
    if ($null -eq $line) {
        return $null
    }
    return $line.Matches[0].Groups[1].Value
}

function Get-SettingPresence {
    param([string]$Key)

    $processValue = [Environment]::GetEnvironmentVariable($Key)
    if (-not [string]::IsNullOrWhiteSpace($processValue)) {
        return [pscustomobject]@{
            Present = $true
            Source = "process"
        }
    }

    $dotEnvValue = Get-DotEnvValue -Key $Key
    if (-not [string]::IsNullOrWhiteSpace($dotEnvValue)) {
        return [pscustomobject]@{
            Present = $true
            Source = ".env"
        }
    }

    return [pscustomobject]@{
        Present = $false
        Source = "missing"
    }
}

function Get-StatusStyle {
    param([string]$Status)

    switch ($Status) {
        "PASS" { return "Green" }
        "WARN" { return "Yellow" }
        "FAIL" { return "Red" }
        default { return "White" }
    }
}

function Write-StatusHeader {
    param(
        [string]$Label,
        [string]$Status
    )

    Write-Host ("{0}: {1}" -f $Label, $Status) -ForegroundColor (Get-StatusStyle -Status $Status)
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

function Get-LiquidUniverseSelection {
    $selectionJsonPath = Join-Path ([System.IO.Path]::GetTempPath()) ("alpaca_liquid_universe_" + [System.Guid]::NewGuid().ToString("N") + ".json")
    $selectionScript = @'
from __future__ import annotations

import json
import os
from pathlib import Path

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass
from dotenv import load_dotenv

from universe import UniverseAsset, UniverseConfig, build_universe, UNIVERSE_MODE_FILTERED

project_root = Path.cwd()
load_dotenv(project_root / ".env")

api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")
if not api_key or not api_secret:
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in .env.")

trading = TradingClient(api_key, api_secret, paper=os.getenv("ALPACA_PAPER", "true").lower() != "false")
data = StockHistoricalDataClient(api_key, api_secret)

config = UniverseConfig(
    mode=UNIVERSE_MODE_FILTERED,
    include_otc=False,
    exchanges=["NYSE", "NASDAQ"],
    min_price=5.0,
    max_price=200.0,
    min_avg_volume=500_000.0,
    min_dollar_volume=5_000_000.0,
    max_symbols=25,
)

all_assets = trading.get_all_assets()
candidate_assets = []
for asset in all_assets:
    if getattr(asset, "asset_class", None) != AssetClass.US_EQUITY:
        continue
    exchange = str(getattr(asset, "exchange", "") or "")
    if exchange.upper() not in {"NYSE", "NASDAQ"}:
        continue
    candidate_assets.append(asset)

symbols = [str(getattr(asset, "symbol", "") or "").upper() for asset in candidate_assets]
symbols = [symbol for symbol in symbols if symbol]

assets_for_selection: list[UniverseAsset] = []
batch_size = 200
for offset in range(0, len(symbols), batch_size):
    batch_symbols = symbols[offset: offset + batch_size]
    snapshots = data.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=batch_symbols, feed=DataFeed.IEX))
    for asset in candidate_assets[offset: offset + batch_size]:
        symbol = str(getattr(asset, "symbol", "") or "").upper()
        snapshot = snapshots.get(symbol)
        previous_daily_bar = getattr(snapshot, "previous_daily_bar", None) if snapshot is not None else None
        if previous_daily_bar is None:
            avg_price = None
            avg_volume = None
            avg_dollar_volume = None
        else:
            close_price = getattr(previous_daily_bar, "close", None)
            volume = getattr(previous_daily_bar, "volume", None)
            avg_price = float(close_price) if close_price is not None else None
            avg_volume = float(volume) if volume is not None else None
            avg_dollar_volume = (
                avg_price * avg_volume
                if avg_price is not None and avg_volume is not None
                else None
            )

        assets_for_selection.append(
            UniverseAsset(
                symbol=symbol,
                exchange=str(getattr(asset, "exchange", "") or ""),
                tradable=bool(getattr(asset, "tradable", False)),
                status=str(getattr(asset, "status", "") or ""),
                asset_class=str(getattr(asset, "asset_class", "") or ""),
                avg_price=avg_price,
                avg_volume=avg_volume,
                avg_dollar_volume=avg_dollar_volume,
                is_otc=str(getattr(asset, "exchange", "") or "").upper() == "OTC",
            )
        )

assets_for_selection.sort(
    key=lambda asset: (
        asset.avg_dollar_volume if asset.avg_dollar_volume is not None else -1.0,
        asset.avg_volume if asset.avg_volume is not None else -1.0,
        asset.symbol,
    ),
    reverse=True,
)

selected_symbols = build_universe(config, assets_for_selection)

payload = {
    "symbols": selected_symbols,
    "filters": {
        "exchanges": config.exchanges,
        "include_otc": config.include_otc,
        "min_price": config.min_price,
        "max_price": config.max_price,
        "min_avg_volume": config.min_avg_volume,
        "min_dollar_volume": config.min_dollar_volume,
        "max_symbols": config.max_symbols,
    },
}

Path(os.environ["BOT_SELECTION_JSON_PATH"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
'@

    try {
        $env:BOT_SELECTION_JSON_PATH = $selectionJsonPath
        & python -c $selectionScript
        if ($LASTEXITCODE -ne 0) {
            throw "Liquid universe selection failed with exit code $LASTEXITCODE."
        }
        if (-not (Test-Path $selectionJsonPath)) {
            throw "Liquid universe selection did not produce a result file."
        }
        return Get-Content $selectionJsonPath -Raw | ConvertFrom-Json
    } finally {
        if (Test-Path Env:BOT_SELECTION_JSON_PATH) {
            Remove-Item Env:BOT_SELECTION_JSON_PATH
        }
        if (Test-Path $selectionJsonPath) {
            Remove-Item -LiteralPath $selectionJsonPath -Force
        }
    }
}

function Invoke-LiquidDatasetSnapshot {
    Write-Section "Liquid Universe Dataset Snapshot"
    $selection = Get-LiquidUniverseSelection
    $selectedSymbols = @($selection.symbols)
    if ($selectedSymbols.Count -eq 0) {
        throw "Universe selection returned no symbols. Review Alpaca credentials, data access, or filter thresholds."
    }

    Write-Host "Selected symbols:" -ForegroundColor Cyan
    Write-Host ("  " + ($selectedSymbols -join ", "))
    Write-Host ""
    Write-Host "Universe filters:" -ForegroundColor Cyan
    Write-Host ("  exchanges={0} exclude_otc={1} min_price={2} max_price={3} min_avg_volume={4} min_dollar_volume={5} max_symbols={6}" -f `
        (($selection.filters.exchanges) -join ","), `
        (-not [bool]$selection.filters.include_otc), `
        $selection.filters.min_price, `
        $selection.filters.max_price, `
        $selection.filters.min_avg_volume, `
        $selection.filters.min_dollar_volume, `
        $selection.filters.max_symbols)

    $beforeDataset = Get-NewestDatasetName
    $snapshotArguments = @(
        "--symbols"
    ) + $selectedSymbols + @(
        "--start", "2026-01-01T00:00:00Z",
        "--end", "2026-02-01T00:00:00Z",
        "--timeframe", "15Min",
        "--feed", "iex"
    )
    Invoke-PythonScript -Script "dataset_snapshotter.py" -Arguments $snapshotArguments

    $afterDataset = Get-NewestDatasetName
    if ($null -eq $afterDataset) {
        throw "Dataset snapshot completed, but no dataset folder was found under datasets."
    }

    $datasetPath = Join-Path $projectRoot ("datasets\" + $afterDataset)
    Write-Host ""
    if ($beforeDataset -ne $afterDataset) {
        Write-Host ("Dataset saved to: {0}" -f $datasetPath) -ForegroundColor Green
    } else {
        Write-Host ("Dataset available at: {0}" -f $datasetPath) -ForegroundColor Green
    }

    return $datasetPath
}

function Show-LiveConfig {
    param(
        [switch]$Raw
    )

    $configPath = Join-Path $projectRoot "config\live_config.json"
    if (-not (Test-Path $configPath)) {
        Write-Host "No promoted config found. Run research-liquid first." -ForegroundColor Yellow
        return
    }

    try {
        $payload = Get-Content $configPath -Raw | ConvertFrom-Json
    } catch {
        Write-Host ("Unable to read promoted config: {0}" -f $configPath) -ForegroundColor Red
        Write-Host ("Reason: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        return
    }

    if ($Raw) {
        Write-Section "LIVE CONFIG RAW JSON"
        Get-Content $configPath
        return
    }

    $runtime = $payload.runtime
    $source = $payload.source
    if ($null -eq $runtime) {
        Write-Host ("Malformed live config: missing 'runtime' object in {0}" -f $configPath) -ForegroundColor Red
        return
    }

    $symbols = @($runtime.symbols)
    $symbolCount = $symbols.Count
    $symbolText = if ($symbolCount -le 12) {
        $symbols -join ", "
    } else {
        (@($symbols | Select-Object -First 12) -join ", ") + ", ..."
    }

    $timeframeMinutes = $runtime.bar_timeframe_minutes
    $timeframeText = if ($null -ne $timeframeMinutes) { "{0}m" -f $timeframeMinutes } else { "n/a" }

    Write-Section "LIVE TRADING CONFIG"
    $strategyMode = if ($null -ne $runtime.strategy_mode -and "$($runtime.strategy_mode)".Trim()) {
        $runtime.strategy_mode
    } else {
        "n/a"
    }
    Write-Host ("Strategy: {0}" -f $strategyMode)
    Write-Host ("Timeframe: {0}" -f $timeframeText)

    Write-Host ""
    Write-Host ("Symbols ({0}):" -f $symbolCount) -ForegroundColor Cyan
    Write-Host ("  {0}" -f $(if ($symbolText) { $symbolText } else { "n/a" }))

    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Cyan
    if ($null -ne $runtime.sma_bars) {
        Write-Host ("  - SMA Bars: {0}" -f $runtime.sma_bars)
    }
    if ($null -ne $runtime.entry_threshold_pct) {
        Write-Host ("  - Entry Threshold: {0}" -f $runtime.entry_threshold_pct)
    }
    if ($null -ne $runtime.ml_probability_buy) {
        Write-Host ("  - ML Buy: {0}" -f $runtime.ml_probability_buy)
    }
    if ($null -ne $runtime.ml_probability_sell) {
        Write-Host ("  - ML Sell: {0}" -f $runtime.ml_probability_sell)
    }
    if ($null -ne $runtime.threshold_mode) {
        Write-Host ("  - Threshold Mode: {0}" -f $runtime.threshold_mode)
    }
    if ($null -ne $runtime.atr_multiple) {
        Write-Host ("  - ATR Multiple: {0}" -f $runtime.atr_multiple)
    }
    if ($null -ne $runtime.atr_percentile_threshold) {
        Write-Host ("  - ATR Percentile Threshold: {0}" -f $runtime.atr_percentile_threshold)
    }
    if ($null -ne $runtime.time_window_mode) {
        Write-Host ("  - Time Window: {0}" -f $runtime.time_window_mode)
    }
    if ($null -ne $runtime.regime_filter_enabled) {
        Write-Host ("  - Regime Filter: {0}" -f $runtime.regime_filter_enabled)
    }
    if ($null -ne $runtime.orb_filter_mode) {
        Write-Host ("  - ORB Filter: {0}" -f $runtime.orb_filter_mode)
    }
    if ($null -ne $runtime.breakout_exit_style) {
        Write-Host ("  - Breakout Exit: {0}" -f $runtime.breakout_exit_style)
    }
    if ($null -ne $runtime.breakout_tight_stop_fraction) {
        Write-Host ("  - Breakout Tight Stop Fraction: {0}" -f $runtime.breakout_tight_stop_fraction)
    }
    if ($null -ne $runtime.mean_reversion_exit_style) {
        Write-Host ("  - Mean Reversion Exit: {0}" -f $runtime.mean_reversion_exit_style)
    }
    if ($null -ne $runtime.mean_reversion_max_atr_percentile) {
        Write-Host ("  - Mean Reversion Max ATR Percentile: {0}" -f $runtime.mean_reversion_max_atr_percentile)
    }

    Write-Host ""
    Write-Host "Source:" -ForegroundColor Cyan
    if ($null -ne $source.dataset) {
        Write-Host ("  - Dataset: {0}" -f $source.dataset)
    }
    if ($null -ne $source.dataset_symbol_source) {
        Write-Host ("  - Symbol Source: {0}" -f $source.dataset_symbol_source)
    }
    if ($null -ne $payload.saved_at) {
        Write-Host ("  - Saved At: {0}" -f $payload.saved_at)
    }
    Write-Host ("  - Config Path: {0}" -f $configPath)
}

function Invoke-Preflight {
    param(
        [switch]$Json
    )

    $configPath = Join-Path $projectRoot "config\live_config.json"
    $configExists = Test-Path $configPath
    $configPayload = $null
    $configParseError = $null
    if ($configExists) {
        try {
            $configPayload = Get-Content $configPath -Raw | ConvertFrom-Json
        } catch {
            $configParseError = $_.Exception.Message
        }
    }

    $runtime = if ($null -ne $configPayload) { $configPayload.runtime } else { $null }
    $source = if ($null -ne $configPayload) { $configPayload.source } else { $null }
    $symbols = if ($null -ne $runtime -and $null -ne $runtime.symbols) { @($runtime.symbols) } else { @() }
    $symbolCount = @($symbols).Count
    $strategyMode = if ($null -ne $runtime) { $runtime.strategy_mode } else { $null }
    $timeframeMinutes = if ($null -ne $runtime) { $runtime.bar_timeframe_minutes } else { $null }
    $datasetPath = if ($null -ne $source) { $source.dataset } else { $null }
    $symbolSource = if ($null -ne $source) { $source.dataset_symbol_source } else { $null }
    $savedAt = if ($null -ne $configPayload) { $configPayload.saved_at } else { $null }

    $configStatus = "PASS"
    $configNotes = @()
    if (-not $configExists) {
        $configStatus = "FAIL"
        $configNotes += "No promoted config found. Run research-liquid first."
    } elseif ($null -ne $configParseError) {
        $configStatus = "FAIL"
        $configNotes += ("Malformed live config: {0}" -f $configParseError)
    } else {
        $configNotes += ("Found config: {0}" -f $configPath)
        $configNotes += ("Strategy: {0}" -f $(if ($strategyMode) { $strategyMode } else { "missing" }))
        $configNotes += ("Symbols: {0}" -f $symbolCount)
        $configNotes += ("Timeframe: {0}" -f $(if ($null -ne $timeframeMinutes) { "{0}m" -f $timeframeMinutes } else { "missing" }))
        if ($datasetPath) {
            $configNotes += ("Dataset: {0}" -f $datasetPath)
        }
        if ($symbolSource) {
            $configNotes += ("Symbol Source: {0}" -f $symbolSource)
        }
        if ($savedAt) {
            $configNotes += ("Saved At: {0}" -f $savedAt)
        }
        if ($symbolCount -eq 0) {
            $configStatus = "FAIL"
            $configNotes += "Configured symbol list is empty."
        }
        if ([string]::IsNullOrWhiteSpace("$strategyMode")) {
            $configStatus = "FAIL"
            $configNotes += "Strategy mode is missing."
        }
        if ($null -eq $timeframeMinutes) {
            if ($configStatus -ne "FAIL") {
                $configStatus = "WARN"
            }
            $configNotes += "Bar timeframe is missing."
        }
    }

    $executeOrdersSet = Test-Path Env:EXECUTE_ORDERS
    $executeOrdersRaw = if ($executeOrdersSet) { [Environment]::GetEnvironmentVariable("EXECUTE_ORDERS") } else { $null }
    $executeOrdersEnabled = $true
    if ($executeOrdersSet -and $executeOrdersRaw -and $executeOrdersRaw.ToLowerInvariant() -eq "false") {
        $executeOrdersEnabled = $false
    }

    $alpacaPaperPresence = Get-SettingPresence -Key "ALPACA_PAPER"
    $alpacaPaperRaw = [Environment]::GetEnvironmentVariable("ALPACA_PAPER")
    if ([string]::IsNullOrWhiteSpace($alpacaPaperRaw)) {
        $alpacaPaperRaw = Get-DotEnvValue -Key "ALPACA_PAPER"
    }
    $paperMode = $true
    if (-not [string]::IsNullOrWhiteSpace($alpacaPaperRaw) -and $alpacaPaperRaw.Trim().ToLowerInvariant() -eq "false") {
        $paperMode = $false
    }

    $environmentStatus = if ($executeOrdersEnabled) { "WARN" } else { "PASS" }
    $environmentNotes = @()
    $environmentNotes += ("EXECUTE_ORDERS set: {0}" -f $(if ($executeOrdersSet) { "yes" } else { "no (defaults to true)" }))
    $environmentNotes += ("EXECUTE_ORDERS effective value: {0}" -f $(if ($executeOrdersEnabled) { "true" } else { "false" }))
    $environmentNotes += ("Execution mode: {0}" -f $(if ($executeOrdersEnabled) { "live-order capable" } else { "dry-run / paper-safe" }))
    $environmentNotes += ("ALPACA_PAPER: {0}" -f $(if ($alpacaPaperPresence.Present) { "$alpacaPaperRaw ($($alpacaPaperPresence.Source))" } else { "missing (defaults to true in trading_bot.py)" }))
    $environmentNotes += ("Account mode: {0}" -f $(if ($paperMode) { "paper" } else { "live" }))

    $apiKeyPresence = Get-SettingPresence -Key "ALPACA_API_KEY"
    $apiSecretPresence = Get-SettingPresence -Key "ALPACA_API_SECRET"
    $credentialsStatus = if ($apiKeyPresence.Present -and $apiSecretPresence.Present) { "PASS" } else { "FAIL" }
    $credentialsNotes = @(
        ("Alpaca API key: {0}" -f $(if ($apiKeyPresence.Present) { "present via $($apiKeyPresence.Source)" } else { "missing" })),
        ("Alpaca API secret: {0}" -f $(if ($apiSecretPresence.Present) { "present via $($apiSecretPresence.Source)" } else { "missing" })),
        "Base URL: not used by this project"
    )

    $easternZone = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
    $nowLocal = Get-Date
    $nowEastern = [System.TimeZoneInfo]::ConvertTime($nowLocal, $easternZone)
    $weekday = $nowEastern.DayOfWeek
    $isWeekday = $weekday -ne "Saturday" -and $weekday -ne "Sunday"
    $entryStart = [datetime]::Today.AddHours(9).AddMinutes(45)
    $entryEnd = [datetime]::Today.AddHours(15).AddMinutes(45)
    $entryTime = [datetime]::Today.AddHours($nowEastern.Hour).AddMinutes($nowEastern.Minute)
    $withinEntryWindow = $isWeekday -and $entryTime -ge $entryStart -and $entryTime -le $entryEnd
    $marketStatus = if ($withinEntryWindow) { "PASS" } else { "WARN" }
    $marketNotes = @(
        ("Current local time: {0}" -f $nowLocal.ToString("yyyy-MM-dd HH:mm:ss zzz")),
        ("Current ET time: {0}" -f $nowEastern.ToString("yyyy-MM-dd HH:mm:ss")),
        ("Entry window check: {0}" -f $(if ($withinEntryWindow) { "inside 09:45-15:45 ET" } else { "outside 09:45-15:45 ET" }))
    )

    $readyForPaper = ($configStatus -ne "FAIL") -and ($credentialsStatus -eq "PASS")
    $readyForLive = $readyForPaper -and $executeOrdersEnabled
    $overallStatus = if ($readyForLive) { "PASS" } elseif ($readyForPaper) { "WARN" } else { "FAIL" }
    $overallSummary = if ($readyForLive) {
        "READY FOR LIVE-ORDER CAPABLE RUN"
    } elseif ($readyForPaper) {
        "READY FOR PAPER / DRY-RUN ONLY"
    } else {
        "NOT READY FOR TRADING"
    }

    $report = [pscustomobject]@{
        config = [pscustomobject]@{
            status = $configStatus
            exists = $configExists
            strategy_mode = $strategyMode
            symbol_count = $symbolCount
            timeframe_minutes = $timeframeMinutes
            dataset = $datasetPath
            symbol_source = $symbolSource
            saved_at = $savedAt
            notes = $configNotes
        }
        environment = [pscustomobject]@{
            status = $environmentStatus
            execute_orders_set = $executeOrdersSet
            execute_orders_effective = $executeOrdersEnabled
            alpaca_paper = $alpacaPaperRaw
            paper_mode_enabled = $paperMode
            notes = $environmentNotes
        }
        credentials = [pscustomobject]@{
            status = $credentialsStatus
            api_key_present = $apiKeyPresence.Present
            api_secret_present = $apiSecretPresence.Present
            notes = $credentialsNotes
        }
        market_time = [pscustomobject]@{
            status = $marketStatus
            local_time = $nowLocal.ToString("o")
            eastern_time = $nowEastern.ToString("o")
            within_entry_window = $withinEntryWindow
            notes = $marketNotes
        }
        overall = [pscustomobject]@{
            status = $overallStatus
            ready_for_paper = $readyForPaper
            ready_for_live = $readyForLive
            summary = $overallSummary
        }
    }

    if ($Json) {
        $report | ConvertTo-Json -Depth 6
        return $report
    }

    Write-Section "TRADING PREFLIGHT"
    Write-StatusHeader -Label "Config" -Status $configStatus
    foreach ($note in $configNotes) {
        Write-Host ("  - {0}" -f $note)
    }

    Write-Host ""
    Write-StatusHeader -Label "Environment" -Status $environmentStatus
    foreach ($note in $environmentNotes) {
        Write-Host ("  - {0}" -f $note)
    }

    Write-Host ""
    Write-StatusHeader -Label "Credentials" -Status $credentialsStatus
    foreach ($note in $credentialsNotes) {
        Write-Host ("  - {0}" -f $note)
    }

    Write-Host ""
    Write-StatusHeader -Label "Market Time" -Status $marketStatus
    foreach ($note in $marketNotes) {
        Write-Host ("  - {0}" -f $note)
    }

    Write-Host ""
    Write-StatusHeader -Label "Overall" -Status $overallStatus
    Write-Host ("  - {0}" -f $overallSummary)
    return $report
}

function Write-LaunchSummary {
    param(
        [string]$LaunchMode,
        [psobject]$PreflightReport,
        [string]$CommandText
    )

    try {
        $launchDir = Join-Path $projectRoot "logs\launches"
        New-Item -ItemType Directory -Force -Path $launchDir | Out-Null

        $timestamp = Get-Date
        $fileTimestamp = $timestamp.ToString("yyyy-MM-dd_HH-mm-ss")
        $launchPath = Join-Path $launchDir ("launch_{0}_{1}.json" -f $fileTimestamp, $LaunchMode)

        $configPath = Join-Path $projectRoot "config\live_config.json"
        $gitCommit = $null
        $executeOrdersEffective = $PreflightReport.environment.execute_orders_effective
        $executeOrdersRaw = [System.Environment]::GetEnvironmentVariable("EXECUTE_ORDERS", "Process")
        try {
            $gitCommit = ((& git rev-parse HEAD 2>$null) | Select-Object -First 1)
            if ([string]::IsNullOrWhiteSpace($gitCommit)) {
                $gitCommit = $null
            }
        } catch {
            $gitCommit = $null
        }

        if (-not [string]::IsNullOrWhiteSpace($executeOrdersRaw)) {
            $executeOrdersEffective = @("1", "true", "yes", "on") -contains $executeOrdersRaw.Trim().ToLowerInvariant()
        }

        $payload = [ordered]@{
            timestamp = $timestamp.ToString("o")
            launch_mode = $LaunchMode
            command = $CommandText
            config_path = $configPath
            strategy_mode = $PreflightReport.config.strategy_mode
            symbols = @()
            symbol_count = $PreflightReport.config.symbol_count
            timeframe_minutes = $PreflightReport.config.timeframe_minutes
            dataset = $PreflightReport.config.dataset
            symbol_source = $PreflightReport.config.symbol_source
            saved_at = $PreflightReport.config.saved_at
            execute_orders_effective = $executeOrdersEffective
            alpaca_paper = $PreflightReport.environment.alpaca_paper
            paper_mode_enabled = $PreflightReport.environment.paper_mode_enabled
            ready_for_paper = $PreflightReport.overall.ready_for_paper
            ready_for_live = $PreflightReport.overall.ready_for_live
            git_commit = $gitCommit
        }

        if (Test-Path $configPath) {
            try {
                $configPayload = Get-Content $configPath -Raw | ConvertFrom-Json
                if ($null -ne $configPayload.runtime -and $null -ne $configPayload.runtime.symbols) {
                    $payload.symbols = @($configPayload.runtime.symbols)
                    $payload.symbol_count = @($payload.symbols).Count
                }
            } catch {
                # Keep audit logging best-effort; preflight already validated what it could.
            }
        }

        $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchPath -Encoding utf8
        return $launchPath
    } catch {
        Write-Host ("Warning: unable to write launch summary. {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        return $null
    }
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
    "dataset-liquid" {
        [void](Invoke-LiquidDatasetSnapshot)
        break
    }
    "research-liquid" {
        Write-Section "Liquid Research Pipeline"
        Write-Host "Stage 1/3: building liquid dataset" -ForegroundColor Cyan
        $datasetPath = Invoke-LiquidDatasetSnapshot

        Write-Host ""
        Write-Host "Stage 2/3: dataset path selected" -ForegroundColor Cyan
        Write-Host ("  {0}" -f $datasetPath) -ForegroundColor Green

        Write-Host ""
        Write-Host "Stage 3/3: running research on dataset" -ForegroundColor Cyan
        Invoke-PythonScript -Script "run_research.py" -Arguments @(
            "--dataset", $datasetPath
        )

        $promotedConfigPath = Join-Path $projectRoot "config\live_config.json"
        Write-Host ""
        Write-Host ("Promoted config: {0}" -f $promotedConfigPath) -ForegroundColor Green
        Write-Host "Next paper-trading command:" -ForegroundColor Cyan
        Write-Host '  $env:EXECUTE_ORDERS="false"'
        Write-Host '  python trading_bot.py'
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
    "show-config" {
        Show-LiveConfig -Raw:(Get-FlagPresent "--raw")
        break
    }
    "preflight" {
        Invoke-Preflight -Json:(Get-FlagPresent "--json")
        break
    }
    "paper-run" {
        $preflight = Invoke-Preflight
        $criticalFailure = (
            $preflight.config.status -eq "FAIL" -or
            $preflight.credentials.status -eq "FAIL"
        )

        Write-Host ""
        if ($criticalFailure) {
            Write-Host "Launch aborted due to failed preflight checks." -ForegroundColor Red
            exit 1
        }

        Write-Host "Launching trading_bot.py in paper-safe mode..." -ForegroundColor Green
        $env:EXECUTE_ORDERS = "false"
        $launchSummaryPath = Write-LaunchSummary -LaunchMode "paper" -PreflightReport $preflight -CommandText "python trading_bot.py"
        if ($null -ne $launchSummaryPath) {
            Write-Host ("Launch summary: {0}" -f $launchSummaryPath) -ForegroundColor Green
        }
        Invoke-PythonScript -Script "trading_bot.py"
        break
    }
    "live-run" {
        $preflight = Invoke-Preflight
        $criticalFailure = (
            $preflight.config.status -eq "FAIL" -or
            $preflight.credentials.status -eq "FAIL" -or
            $preflight.config.symbol_count -le 0
        )

        Write-Host ""
        if ($criticalFailure) {
            Write-Host "Launch aborted due to failed preflight checks." -ForegroundColor Red
            exit 1
        }

        if ($preflight.environment.paper_mode_enabled) {
            Write-Host "Launch aborted: environment still points to paper mode." -ForegroundColor Red
            exit 1
        }

        if (-not (Get-FlagPresent "--confirm-live")) {
            Write-Host "Launch aborted: missing --confirm-live" -ForegroundColor Red
            exit 1
        }

        Write-Host "WARNING: Launching trading_bot.py in LIVE-ORDER mode..." -ForegroundColor Red
        $env:EXECUTE_ORDERS = "true"
        $launchSummaryPath = Write-LaunchSummary -LaunchMode "live" -PreflightReport $preflight -CommandText "python trading_bot.py"
        if ($null -ne $launchSummaryPath) {
            Write-Host ("Launch summary: {0}" -f $launchSummaryPath) -ForegroundColor Green
        }
        Invoke-PythonScript -Script "trading_bot.py"
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
