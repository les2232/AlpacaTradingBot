Legacy desktop operator surface quarantined on 2026-04-12.

Why it was moved:
- It created a second operator workflow beside `alpaca-bot live` and `alpaca-bot dashboard`.
- The Streamlit dashboard is now monitoring-only, so the control-panel launch path was conflicting and misleading.
- Packaging metadata, README guidance, and CLI commands were cleaned to remove this path from normal use.

Contents:
- `desktop_app/`: retired CustomTkinter control panel
- `bot.ps1`: legacy PowerShell launcher with overlapping live/paper commands
- `launch_control_panel.py`: wrapper that opened the retired desktop app
- `launch_dashboard.bat` / `launch_dashboard.vbs`: browser-wrapper launchers superseded by `alpaca-bot dashboard`
