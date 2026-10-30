"""Screen capture helpers (mss-based) with multi-monitor coordinate mapping.

Two capture scopes:
  * "screen"  – a full monitor (default: the one under the mouse)
  * "window"  – the active window's rectangle (via xdotool); gives the model
                near-native resolution for the UI it actually operates on,
                which matters a lot on 8 GB GPUs with a limited vision-token
                budget.

Implementation notes (verified empirically on this box):
  * The mss context and the Xlib display are created once per process and
    MUST be created before torch's CUDA driver is loaded (see ``prime`` in
    ``__init__``); opening a fresh X connection after CUDA init segfaults.
  * Arbitrary-rectangle X grabs fail on this X server (XProtoError), so a
    window crop is produced by grabbing the containing full monitor and
    cropping with PIL (lossless).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import mss
from PIL import Image

# Process-wide handles (created by prime(); see module docstring).
_SCT = None
_XDISPLAY = None


def _sct():
    global _SCT
    if _SCT is None:
        # CUA_MSS_BACKEND selects the mss X11 backend (mss >= 10). The
        # default (XCB + XShmGetImage) returns corrupted frames on Xvfb;
        # the sandbox sets "xlib" (legacy Xlib XGetImage) which is clean.
        backend = os.environ.get("CUA_MSS_BACKEND", "default")
        maker = getattr(mss, "MSS", None) or mss.mss  # MSS in mss >= 10
        try:
            _SCT = maker(backend=backend)
        except TypeError:
            _SCT = maker()  # older mss without backend support
    return _SCT


def prime() -> bool:
    """Open the X11 connection early (call BEFORE importing torch).

    Returns True if a display is available.
    """
    global _XDISPLAY
    try:
        _sct()
    except Exception:
        return False
    try:
        from Xlib.display import Display

        _XDISPLAY = Display()
    except Exception:
        _XDISPLAY = None
    return True


def mouse_position() -> tuple[int, int]:
    """Mouse position in root-window coordinates."""
    if _XDISPLAY is not None:
        try:
            p = _XDISPLAY.query_pointer()
            return p.root_x, p.root_y
        except Exception:
            pass
    # Fallback: xdotool subprocess (no in-process X connection).
    try:
        out = subprocess.check_output(
            ["xdotool", "getmouselocation"], text=True, timeout=3
        )
        vals = dict(line.split("=", 1) for line in out.split())
        return int(vals["X"]), int(vals["Y"])
    except Exception:
        return 0, 0


@dataclass
class Capture:
    """A screenshot plus the rectangle it was grabbed from."""

    image: Image.Image
    monitor: dict  # {"left", "top", "width", "height"}

    def to_screen(self, x_img: float, y_img: float) -> tuple[float, float]:
        """Map image pixels to absolute desktop coordinates."""
        m = self.monitor
        # The capture is normally 1:1 with the grabbed rect; scale anyway in
        # case of HiDPI / scaling mismatches.
        sx = m["width"] / self.image.width
        sy = m["height"] / self.image.height
        return m["left"] + x_img * sx, m["top"] + y_img * sy


def list_monitors() -> list[dict]:
    """Monitors excluding the virtual 'all screens' entry (index 0)."""
    return list(_sct().monitors[1:])


def _monitor_under_mouse(monitors: list[dict], mouse: tuple[int, int]) -> dict:
    mx, my = mouse
    for m in monitors:
        if m["left"] <= mx < m["left"] + m["width"] and m["top"] <= my < m["top"] + m["height"]:
            return m
    return monitors[0]


def _monitor_overlapping(rect: dict, monitors: list[dict]) -> dict:
    """The monitor with the largest overlap with `rect`."""
    best, best_area = monitors[0], -1
    for m in monitors:
        x1 = max(rect["left"], m["left"])
        y1 = max(rect["top"], m["top"])
        x2 = min(rect["left"] + rect["width"], m["left"] + m["width"])
        y2 = min(rect["top"] + rect["height"], m["top"] + m["height"])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best, best_area = m, area
    return best


def active_window_rect() -> Optional[dict]:
    """Geometry of the active window via xdotool, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
            text=True, timeout=3, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    vals = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip()
    try:
        return {
            "left": int(vals["X"]),
            "top": int(vals["Y"]),
            "width": int(vals["WIDTH"]),
            "height": int(vals["HEIGHT"]),
        }
    except (KeyError, ValueError):
        return None


def capture(scope: str = "screen", monitor_index: Optional[int] = None,
            rect: Optional[dict] = None) -> Capture:
    """Grab a screenshot.

    Args:
        scope: "window" (active window) or "screen" (a monitor).
        monitor_index: for scope="screen": 0-based monitor index; None = the
            monitor the mouse is on.
        rect: explicit grab rectangle, overrides everything.

    A window rect is served by grabbing the containing full monitor and
    cropping with PIL (arbitrary-rect X grabs fail on this X server).
    """
    sct = _sct()
    if rect is None:
        if scope == "window":
            rect = active_window_rect()
        if rect is None:
            monitors = sct.monitors[1:]
            if monitor_index is None:
                rect = _monitor_under_mouse(monitors, mouse_position())
            else:
                rect = monitors[monitor_index]

    monitors = sct.monitors[1:]
    mon = _monitor_overlapping(rect, monitors)
    shot = sct.grab(mon)
    full = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    is_monitor = (
        (rect["left"], rect["top"], rect["width"], rect["height"])
        == (mon["left"], mon["top"], mon["width"], mon["height"])
    )
    if is_monitor:
        image = full
    else:
        rx1 = max(0, rect["left"] - mon["left"])
        ry1 = max(0, rect["top"] - mon["top"])
        rx2 = min(mon["width"], rect["left"] + rect["width"] - mon["left"])
        ry2 = min(mon["height"], rect["top"] + rect["height"] - mon["top"])
        image = full.crop((rx1, ry1, rx2, ry2))

    return Capture(image=image, monitor=dict(rect))