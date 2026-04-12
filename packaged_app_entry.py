from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent.parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = _resolve_project_root()
os.chdir(PROJECT_ROOT)
os.environ["ALPACA_BOT_PROJECT_ROOT"] = str(PROJECT_ROOT)


def _launch_dashboard() -> None:
    from desktop_app.dashboard_home import main as native_dashboard_main

    native_dashboard_main()


def _launch_control_panel() -> None:
    from desktop_app.app import main as control_panel_main

    control_panel_main()


def main() -> None:
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "dashboard"
    if mode in {"control-panel", "--control-panel"}:
        _launch_control_panel()
        return
    _launch_dashboard()


if __name__ == "__main__":
    main()
