This folder holds legacy files removed from the active operator path during the pre-market cleanup on 2026-04-12.

Quarantine policy:
- Files were moved here when they looked stale or conflicting, but deleting them outright felt less safe than preserving them one layer away.
- Nothing in `alpaca-bot live`, `alpaca-bot preview`, or `alpaca-bot dashboard` should depend on these files anymore.
- If a quarantined file is needed again, move it back deliberately and re-wire the active entrypoints/docs at the same time.

Current quarantine groups:
- `legacy_desktop_runtime/`: retired desktop-control-panel and launcher surfaces that conflicted with the single-process Streamlit workflow.
