"""Desktop actions (pyautogui wrapper).

All functions are no-ops (with a log message) when ``dry_run`` is True, so the
agent can be exercised end-to-end without moving the mouse.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional

# pyautogui (via mouseinfo) opens an X11 connection at import time, so it is
# imported lazily on first real action. That keeps the agent importable (and
# offline grounding runnable) in headless environments.
pyautogui = None
_CONFIGURED = False


def _pg():
    global pyautogui, _CONFIGURED
    import pyautogui as _m

    if not _CONFIGURED:
        _m.FAILSAFE = True  # slam the mouse into the top-left corner to abort
        _m.PAUSE = 0.01  # per-call settle; the task's own waits handle loads
        _CONFIGURED = True
    pyautogui = _m
    return _m


class DryRunError(RuntimeError):
    """Raised if a real action is attempted while dry_run is active."""


def _log(action: str, dry_run: bool) -> None:
    print(f"    [action{' (dry-run)' if dry_run else ''}] {action}")


def click(x: float, y: float, button: str = "left", clicks: int = 1, dry_run: bool = False) -> None:
    desc = f"click {button} x{clicks} @ ({x:.0f}, {y:.0f})"
    _log(desc, dry_run)
    if dry_run:
        return
    _pg().click(x, y, button=button, clicks=clicks)


def move(x: float, y: float, dry_run: bool = False) -> None:
    desc = f"move mouse -> ({x:.0f}, {y:.0f})"
    _log(desc, dry_run)
    if dry_run:
        return
    _pg().moveTo(x, y)


def type_text(text: str, interval: float = 0.005, dry_run: bool = False) -> None:
    """Type ASCII text. Non-ASCII characters are typed as empty placeholders;
    for unicode input use `key` combos or a helper like `xdotool type`."""
    _log(f"type {text!r}", dry_run)
    if dry_run:
        return
    pg = _pg()
    ascii_part = text.encode("ascii", "ignore").decode()
    if ascii_part:
        pg.typewrite(ascii_part, interval=interval)
    if ascii_part != text:
        # Best-effort: emit <space> for the dropped characters so layout is kept.
        pg.typewrite(" " * (len(text) - len(ascii_part)), interval=interval)


def hotkey(combo: str, dry_run: bool = False) -> None:
    """Press a key combo like 'ctrl+alt+t' or 'enter'."""
    keys = [k.strip() for k in combo.split("+") if k.strip()]
    _log(f"key {combo}", dry_run)
    if dry_run:
        return
    _pg().hotkey(*keys)


def scroll(amount: int, x: float | None = None, y: float | None = None, dry_run: bool = False) -> None:
    """Scroll at a point. Positive = up, negative = down."""
    _log(f"scroll {amount:+d} @ ({x:.0f}, {y:.0f})" if x is not None else f"scroll {amount:+d}", dry_run)
    if dry_run:
        return
    _pg().scroll(amount, x=x, y=y)


def drag_to(x1: float, y1: float, x2: float, y2: float, duration: float = 0.5, dry_run: bool = False) -> None:
    _log(f"drag ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})", dry_run)
    if dry_run:
        return
    pg = _pg()
    pg.moveTo(x1, y1)
    pg.dragTo(x2, y2, duration=duration)


def pause(seconds: float) -> None:
    time.sleep(seconds)


def launch(command: str, dry_run: bool = False) -> Optional[str]:
    """Launch a detached GUI application (e.g. "thunar" or
    "firefox --no-sandbox"), inheriting the agent's X display.

    The child runs in its own session and outlives the agent process; its
    stdout/stderr go to /dev/null so a chatty app (browser console) never
    drowns the agent's log. Returns the resolved binary, or None if no
    candidate is installed.
    """
    binary = command.split()[0]
    if shutil.which(binary) is None:
        return None
    _log(f"launch {command}", dry_run)
    if dry_run:
        return binary
    subprocess.Popen(
        command.split(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return binary