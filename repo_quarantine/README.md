This folder holds legacy files removed from the active operator path during the pre-market cleanup on 2026-04-12.

Quarantine policy:
- Files were moved here when they looked stale or conflicting, but deleting them outright felt less safe than preserving them one layer away.
- Nothing in `tradeos live`, `tradeos preview`, or `tradeos dashboard` should depend on these files anymore.
- Older `alpaca-bot` naming may still appear inside compatibility shims or quarantined files, but it is not the preferred current operator surface.
- If a quarantined file is needed again, move it back deliberately and re-wire the active entrypoints/docs at the same time.

Current quarantine groups:
- None currently checked in. The retired desktop-control-panel runtime was removed after confirming it was isolated from the active `tradeos` + Streamlit workflow.
