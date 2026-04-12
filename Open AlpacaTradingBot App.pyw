from __future__ import annotations

import subprocess
from pathlib import Path
from tkinter import Tk, messagebox


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGED_EXE = PROJECT_ROOT / "dist" / "AlpacaTradingBot" / "AlpacaTradingBot.exe"


def main() -> None:
    if not PACKAGED_EXE.exists():
        raise FileNotFoundError(
            f"Packaged app not found.\n\nExpected at:\n{PACKAGED_EXE}"
        )

    subprocess.Popen(
        [str(PACKAGED_EXE)],
        cwd=str(PACKAGED_EXE.parent),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        root = Tk()
        root.withdraw()
        messagebox.showerror(
            "AlpacaTradingBot Launch Error",
            str(exc),
        )
        root.destroy()
