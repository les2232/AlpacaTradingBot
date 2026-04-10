from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import which


OutputCallback = Callable[[str, str], None]
UpdateCallback = Callable[["ProcessUpdate"], None]


@dataclass(frozen=True)
class ProcessUpdate:
    status: str
    command: list[str]
    exit_code: int | None = None
    detail: str = ""


class ProcessRunner:
    """Central process runner for GUI-launched commands."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._command: list[str] = []
        self._status = "idle"
        self._stop_requested = False
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def snapshot(self) -> ProcessUpdate:
        with self._lock:
            exit_code = None if self._process is None else self._process.poll()
            return ProcessUpdate(
                status=self._status,
                command=list(self._command),
                exit_code=exit_code,
            )

    def start(
        self,
        command: list[str],
        *,
        cwd: Path,
        on_output: OutputCallback,
        on_update: UpdateCallback,
    ) -> None:
        self._validate_launch(command=command, cwd=cwd)

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A command is already running.")

            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
            self._process = process
            self._command = list(command)
            self._status = "running"
            self._stop_requested = False
            update = ProcessUpdate(status="running", command=list(self._command))

        on_update(update)

        threading.Thread(
            target=self._read_stream,
            args=(process, "stdout", on_output),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stream,
            args=(process, "stderr", on_output),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._wait_for_exit,
            args=(process, on_update),
            daemon=True,
        ).start()

    def stop(self) -> ProcessUpdate | None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return None

            command = list(self._command)
            self._status = "stopping"
            self._stop_requested = True

            if os.name == "nt":
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    return ProcessUpdate(
                        status="stopping",
                        command=command,
                        detail="Sent CTRL+BREAK to the process group.",
                    )
                except Exception:
                    process.terminate()
                    return ProcessUpdate(
                        status="stopping",
                        command=command,
                        detail="CTRL+BREAK unavailable; sent terminate() instead.",
                    )

            process.terminate()
            return ProcessUpdate(
                status="stopping",
                command=command,
                detail="Sent terminate() to the running process.",
            )

    def _validate_launch(self, *, command: list[str], cwd: Path) -> None:
        if not command:
            raise ValueError("No command was provided.")
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory was not found: {cwd}")

        executable = command[0]
        if which(executable) is None and not Path(executable).exists():
            raise FileNotFoundError(f"Command was not found on PATH: {executable}")

    def _read_stream(
        self,
        process: subprocess.Popen[str],
        stream_name: str,
        on_output: OutputCallback,
    ) -> None:
        stream = process.stdout if stream_name == "stdout" else process.stderr
        if stream is None:
            return

        for line in stream:
            on_output(stream_name, line.rstrip("\r\n"))

    def _wait_for_exit(
        self,
        process: subprocess.Popen[str],
        on_update: UpdateCallback,
    ) -> None:
        exit_code = process.wait()
        with self._lock:
            if self._process is process:
                command = list(self._command)
                stop_requested = self._stop_requested
                self._process = None
                self._command = []
                self._status = "idle"
                self._stop_requested = False
            else:
                command = []
                stop_requested = False

        if stop_requested:
            status = "stopped"
        elif exit_code == 0:
            status = "completed"
        else:
            status = "failed"
        on_update(
            ProcessUpdate(
                status=status,
                command=command,
                exit_code=exit_code,
            )
        )
