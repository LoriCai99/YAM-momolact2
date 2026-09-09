"""Headless replacement for the pygame pad: keys come from the terminal.

Same interface the control loop uses (``update(dashboard_data) -> "start" |
"save" | "discard" | "normal"`` and ``banner(...)``), no window, no rendering.
On a real terminal the keys work like the pad -- a single keypress, no Enter
needed (the tty is put into cbreak mode lazily on the first ``update()`` and
restored on ``close()`` / exit):

    Enter    start recording (after the 3-2-1 countdown)
    s        end and save
    d        end and discard

When stdin is not a tty (tests, pipes) it falls back to one command per line.
Selected with ``--no_dashboard``.
"""

import atexit
import os
import select
import sys
import time
from typing import Any, Dict, Optional

_START = {"\r", "\n"}
_SAVE = {"s", "S"}
_DISCARD = {"d", "D"}


class TerminalKeys:
    def __init__(self, stream=None) -> None:
        self._in = stream or sys.stdin
        self._last_status = ""
        self._last_status_t = 0.0
        self._saved_termios = None
        self._is_tty = False
        try:
            self._is_tty = os.isatty(self._in.fileno())
        except (ValueError, OSError, AttributeError):
            self._is_tty = False
        mode = "single keypress" if self._is_tty else "one command per line"
        print(f"Headless mode ({mode}): [Enter] start   s save   d discard", flush=True)

    # -- tty handling ---------------------------------------------------------
    def _enter_cbreak(self) -> None:
        if not self._is_tty or self._saved_termios is not None:
            return
        try:
            import termios
            import tty

            fd = self._in.fileno()
            self._saved_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)  # keeps ISIG, so Ctrl-C still works
            atexit.register(self.close)
        except Exception as e:  # pragma: no cover - depends on the terminal
            print(f"(could not switch terminal to single-key mode: {e}; use key+Enter)", flush=True)
            self._is_tty = False

    def close(self) -> None:
        if self._saved_termios is not None:
            try:
                import termios

                termios.tcsetattr(self._in.fileno(), termios.TCSADRAIN, self._saved_termios)
            except Exception:
                pass
            self._saved_termios = None

    # -- input ------------------------------------------------------------------
    def _read_token(self) -> Optional[str]:
        try:
            ready, _, _ = select.select([self._in], [], [], 0.0)
        except (ValueError, OSError):
            return None
        if not ready:
            return None
        if self._is_tty:
            ch = os.read(self._in.fileno(), 1).decode(errors="ignore")
            return ch if ch else None
        line = self._in.readline()
        if not line:
            return None
        line = line.rstrip("\r\n").strip()
        return "\n" if line == "" else line

    def update(self, dashboard_data: Optional[Dict[str, Any]] = None) -> str:
        self._enter_cbreak()
        if dashboard_data is not None:
            status = str(dashboard_data.get("status_text", ""))
            if status != self._last_status and time.time() - self._last_status_t > 0.5:
                print(f"\r{status}                    ", flush=True)
                self._last_status, self._last_status_t = status, time.time()
        tok = self._read_token()
        if tok is None:
            return "normal"
        if tok in _START:
            return "start"
        if tok in _SAVE or tok.lower() == "save":
            return "save"
        if tok in _DISCARD or tok.lower() == "discard":
            return "discard"
        if tok.strip():
            print(f"(ignored key {tok!r}: Enter / s / d)", flush=True)
        return "normal"

    def banner(self, text: str, dashboard_data: Optional[Dict[str, Any]] = None,
               duration_s: float = 1.5, color=None) -> None:
        print(f"\n>>> {text}", flush=True)
