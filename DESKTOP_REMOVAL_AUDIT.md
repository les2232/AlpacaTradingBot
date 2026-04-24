# Desktop Removal Audit

## Scope

This audit covers all repository references to:

- `customtkinter`
- `tkinter`
- quarantined desktop UI modules and launchers

The goal was to confirm whether the old desktop UI still had any dependency edges into the active TradeOS runtime before removing it.

## Findings

### 1. `customtkinter`

- File/path: `requirements.txt`
- Reference: listed as a dependency
- Classification: active dependency declaration, not active runtime usage
- Status: safe to remove if no non-desktop code imports it

- File/path: `repo_quarantine/legacy_desktop_runtime/desktop_app/app.py`
- Reference: `import customtkinter as ctk`
- Classification: quarantined/legacy desktop UI
- Status: safe to remove with the rest of the legacy desktop runtime

### 2. `tkinter`

- File/path: `repo_quarantine/legacy_desktop_runtime/desktop_app/app.py`
- Reference: found indirectly via `customtkinter` search context only; no standalone active repo usage outside the quarantined desktop app
- Classification: quarantined/legacy desktop UI
- Status: safe to remove with the rest of the legacy desktop runtime

Repo-wide result:

- No active runtime, broker, strategy, CLI, or Streamlit files import `tkinter` or `customtkinter`.

### 3. Desktop UI modules and launchers

- File/path: `repo_quarantine/legacy_desktop_runtime/launch_control_panel.py`
- Reference: imports `desktop_app.app`
- Classification: quarantined/legacy
- Status: only referenced inside the quarantined desktop runtime

- File/path: `repo_quarantine/legacy_desktop_runtime/desktop_app/app.py`
- Reference: imports `desktop_app.log_formatter`, `desktop_app.paths`, `desktop_app.repo_status`, `desktop_app.runner`
- Classification: quarantined/legacy
- Status: only referenced inside the quarantined desktop runtime

- File/path: `repo_quarantine/legacy_desktop_runtime/desktop_app/log_formatter.py`
- Classification: quarantined/legacy
- Status: only referenced by `desktop_app.app`

- File/path: `repo_quarantine/legacy_desktop_runtime/desktop_app/paths.py`
- Classification: quarantined/legacy
- Status: only referenced by `desktop_app.app`

- File/path: `repo_quarantine/legacy_desktop_runtime/desktop_app/repo_status.py`
- Classification: quarantined/legacy
- Status: only referenced by `desktop_app.app`

- File/path: `repo_quarantine/legacy_desktop_runtime/desktop_app/runner.py`
- Classification: quarantined/legacy
- Status: only referenced by `desktop_app.app`

- File/path: `repo_quarantine/legacy_desktop_runtime/bot.ps1`
- Classification: quarantined/legacy launcher
- Status: only referenced by the quarantined desktop app

- File/path: `repo_quarantine/legacy_desktop_runtime/launch_dashboard.bat`
- Classification: quarantined/legacy launcher
- Status: only referenced by quarantine docs; not used by active code

- File/path: `repo_quarantine/legacy_desktop_runtime/launch_dashboard.vbs`
- Classification: quarantined/legacy launcher
- Status: only referenced by quarantine docs; not used by active code

- File/path: `repo_quarantine/legacy_desktop_runtime/README.md`
- Classification: quarantined/legacy documentation
- Status: safe to remove with the desktop runtime folder

## Active-Code Reference Check

I searched the full repository for:

- `customtkinter`
- `tkinter`
- `launch_control_panel`
- `desktop_app`
- `launch_dashboard.bat`
- `launch_dashboard.vbs`
- `bot.ps1`
- `legacy_desktop_runtime`

Result:

- No active runtime file under the current operator path references the desktop UI.
- No active Streamlit dashboard code references the desktop UI.
- No active CLI path references the desktop UI.
- No strategy, trading, or broker code references the desktop UI.

Only these categories referenced it:

- the quarantined desktop runtime itself
- quarantine documentation
- repository cleanup/project-map documentation
- `requirements.txt`

## Removal Decision

The legacy desktop UI is isolated from the active TradeOS runtime and can be removed safely, provided these changes are included together:

1. Delete `repo_quarantine/legacy_desktop_runtime/`
2. Remove `customtkinter` from `requirements.txt`
3. Update documentation to state that the supported interfaces are:
   - `tradeos` CLI
   - Streamlit dashboard

## Guardrails Observed

The following files were intentionally not modified during this removal:

- `trading_bot.py`
- `strategy.py`
- `tradeos/brokers/*`
- runtime config behavior
- active Streamlit dashboard code
