"""Headless replacement for the pygame pad: keys come from the terminal.

Same interface the control loop uses (``update(dashboard_data) -> "start" |
"save" | "discard" | "normal"`` and ``banner(...)``), no window, no rendering.
Type in the terminal the launcher runs in:

    Enter        start recording (after the 3-2-1 countdown)
    s + Enter    end and save
    d + Enter    end and discard

Selected with ``--no_dashboard``. Camera-stale warnings are logged to the
terminal in both modes.
"""

import select
import sys
import time
from typing import Any, Dict, Optional


class TerminalKeys:
    def __init__(self, stream=None) -> None:
        self._in = stream or sys.stdin
        self._last_status = ""
        self._last_status_t = 0.0
        print("Headless mode: [Enter] start   s+Enter save   d+Enter discard", flush=True)

    def _read_line(self) -> Optional[str]:
        try:
            ready, _, _ = select.select([self._in], [], [], 0.0)
        except (ValueError, OSError):
            return None
        if not ready:
            return None
        line = self._in.readline()
        return line.rstrip("\r\n").strip().lower() if line else None

    def update(self, dashboard_data: Optional[Dict[str, Any]] = None) -> str:
        if dashboard_data is not None:
            status = str(dashboard_data.get("status_text", ""))
            if status != self._last_status and time.time() - self._last_status_t > 0.5:
                print(f"\r{status}                    ", flush=True)
                self._last_status, self._last_status_t = status, time.time()
        line = self._read_line()
        if line is None:
            return "normal"
        if line == "":
            return "start"
        if line in ("s", "save"):
            return "save"
        if line in ("d", "discard"):
            return "discard"
        print(f"(ignored input {line!r}: Enter / s / d)", flush=True)
        return "normal"

    def banner(self, text: str, dashboard_data: Optional[Dict[str, Any]] = None,
               duration_s: float = 1.5, color=None) -> None:
        print(f"\n>>> {text}", flush=True)

    def close(self) -> None:
        pass
