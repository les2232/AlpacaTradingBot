from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path
from tkinter import Tk, messagebox


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGED_EXE = PROJECT_ROOT / "dist" / "AlpacaTradingBot" / "AlpacaTradingBot.exe"


def main() -> None:
    os.chdir(PROJECT_ROOT)

    # Prefer the packaged Windows app when it exists so double-clicking this
    # launcher behaves like a normal program instead of depending on Python deps.
    if PACKAGED_EXE.exists():
        subprocess.Popen(
            [str(PACKAGED_EXE)],
            cwd=str(PACKAGED_EXE.parent),
        )
        return

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from desktop_app.dashboard_home import main as dashboard_main

    dashboard_main()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        root = Tk()
        root.withdraw()
        messagebox.showerror(
            "AlpacaTradingBot Launch Error",
            (
                f"{exc}\n\n"
                "If you want the clickable app experience, use:\n"
                f"{PACKAGED_EXE}\n\n"
                f"{traceback.format_exc()}"
            ),
        )
        root.destroy()
