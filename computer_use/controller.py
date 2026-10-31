"""The computer-use loop: screenshot -> ground with LocateAnything -> act.

The perception model only *localizes*; planning is a small, deterministic task
language so the agent is transparent and easy to test:

    click <phrase>            left-click the grounded element   (default verb)
    double-click <phrase>     double-click
    right-click <phrase>      right-click
    type <text>               type text (ASCII)
    key <combo>               press a key combo, e.g. ctrl+alt+t
    scroll <phrase> <n>       scroll n notches at the element (positive = down)
    wait <seconds>            idle
    open <app>                tool step: launch/focus an app directly —
                              firefox | file manager | terminal | applications
    search <query>            tool step: open Firefox on a DuckDuckGo
                              search for <query> (alias: duckduckgo <query>)
                              — the results page is shown; the agent
                              itself grounds and clicks the top result
    wait-load                 wait until the active window's title
                              changes (a navigation has produced a new
                              page); a measured load wait, capped at 8 s
    done                      stop

Steps are separated by ';' or newlines. A bare phrase is treated as `click`.
"""

from __future__ import annotations

import re
import subprocess
import time
from urllib.parse import quote_plus
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from . import actions, screen
from .grounding import best_target
from .worker import GroundingResult, LocateAnythingWorker

ImageLike = Union[str, None]

_VERB_RE = re.compile(
    r"^\s*(?P<verb>"
    r"click|left[- ]?click|double[- ]?click|right[- ]?click|"
    r"type|key|scroll|wait-load|wait|open|search|duckduckgo|done"
    r")\s*(?P<arg>.*?)\s*$",
    re.IGNORECASE,
)
_SCROLL_RE = re.compile(r"^(?P<phrase>.+?)\s+(?P<amount>[+-]?\d+)\s*$")

# Scroll targets are large scrollable regions: the grounded point only has
# to land inside one, so scroll grounding runs a coarse single pass (vision
# patches bounded to this) instead of a full-resolution one — ~320 patches
# takes ~2-3 s on the 1070 vs ~34 s at full window resolution (measured).
# The full-resolution capture is still saved as step evidence, and
# Capture.to_screen maps the coarse point back to desktop coordinates.
_SCROLL_COARSE_PATCHES = 320

# Two-stage zoom for "<thing> in the <container>" phrasings (e.g. "the
# terminal icon in the panel"): the container is large and easy to box, so
# locate it first, then ground the phrase in a margin-expanded crop of the
# full capture. Containers are whitelisted so odd phrasings ("in the middle
# of the page") never trigger a zoom.
_CONTAINER_RE = re.compile(
    r"\b(?:in|on|under|above|within)\s+(?:the|a|an)?\s*"
    r"([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,2})\s*$",
    re.IGNORECASE,
)
_CONTAINERS = frozenset({
    "panel", "panels", "tab bar", "tabs", "title bar", "toolbar", "menu bar",
    "address bar", "status bar", "sidebar", "side bar", "taskbar", "task bar",
    "dock", "tray", "clock", "task switcher", "launcher", "window", "desktop",
    "screen", "browser", "menu", "bar", "ribbon", "footer", "header",
    "dialog", "modal", "popup", "dialog box", "modal dialog",
    "popup window", "cookie dialog",
    "search results", "results", "results list", "search results list",
})

# Containers that by definition float near the center of their window
# (modal dialogs: cookie walls, confirmation popups). Their exact box is
# unknown, but the buttons of a centered modal sit in a small zone
# slightly below the window center, so a fixed box around that zone
# contains the target without the model's box pass — one fewer model
# pass per step, and the crop is small enough for a fast pass. Measured
# on the cookie wall of the NVIDIA support page: the "Reject Optional"
# button center is at window (739, 384) of a 1280x620 window (center
# 310), the button row spans y≈370-398; the box below
# (0.35w..0.7w, center-70..center+100) contains it with ~25 px margin
# while the crop is ~300 patches (~1.7 s prefill on the 1070 vs ~8.7 s
# for a 280 px full-width band).
# The dialog is centered in x on every browser, so the x extent is a
# fraction of the width; the y extent is a pixel offset from the center
# because the wall is a fixed-pixel modal. If the target isn't in the
# box, the deterministic strip fails and the normal two-pass zoom runs
# instead (same as the edge strips).
_CENTER_CONTAINERS = frozenset({
    "dialog", "modal", "popup", "dialog box", "modal dialog",
    "popup window", "cookie dialog",
})
_CENTER_BOX_X = (0.35, 0.70)      # fractions of the capture width
_CENTER_BOX_Y = (-70.0, 100.0)    # pixel offsets from the capture center

# Containers that are a vertical list of results (a search results page):
# the FIRST organic result sits in the left column, below the engine's
# search header. Measured on DuckDuckGo at 720p: the first result title
# row spans window y≈426-440 of a 620 px window, the title text spans
# x≈108..728 of a 1280 px window. A deterministic band around that zone
# (x 0..0.6w, y 0.62h..0.8h) contains it with ~30 px margin while the
# crop is ~430 patches (~2 s prefill on the 1070 vs ~16.5 s for the
# full window). Same fallback as the other deterministic boxes.
_RESULTS_CONTAINERS = frozenset({
    "search results", "results", "results list", "search results list",
})
_RESULTS_BOX = (0.0, 0.62, 0.6, 0.8)   # fractions of the capture (x1, y1, x2, y2)

# Containers that live on a screen edge (XFCE: top menu panel, bottom
# launcher panel, tray, clock). For these, a box touching the edge is
# snapped to the edge strip even when the box is loose (the model
# sometimes boxes a whole screen quadrant for "the panel").
_EDGE_CONTAINERS = frozenset({
    "panel", "panels", "taskbar", "task bar", "dock", "tray",
    "footer", "status bar", "menu bar", "clock",
})

# Edge containers that by definition span the whole edge (the panel, the
# taskbar, the menu bar). For these the strip crop uses the full screen
# length: a box covering only part of the edge is the model's guess and
# must not clip the target out of the crop.
_FULL_EDGE = frozenset({
    "panel", "panels", "taskbar", "task bar", "menu bar", "dock",
    "footer", "status bar", "sidebar", "side bar",
})

# Edge containers whose position is fixed by the desktop (XFCE defaults:
# the launcher panel/taskbar on the bottom edge, the menu bar on top).
# For these the strip crop can be built without the model's box pass —
# one fewer model pass per step. If the deterministic strip fails, the
# normal two-pass zoom still runs (the box pass then verifies the edge).
_EDGE_DEFAULT = {
    "panel": "bottom", "panels": "bottom", "taskbar": "bottom",
    "task bar": "bottom", "footer": "bottom", "status bar": "bottom",
    "tray": "bottom",
    "menu bar": "top",
}

# Height of the deterministic edge strip (px). A taskbar/menu bar is ~30-60 px
# tall, so 64 px captures the whole bar with a small margin while excluding
# the windows/wallpaper just above (or below) the edge. Fewer pixels -> fewer
# vision patches -> a quadratically cheaper prefill (measured on the 1070:
# 120 px -> 1096 patches -> 5.7 s prefill; 64 px -> 548 patches -> 2.9 s).
_EDGE_STRIP_H = 64
# A menu bar is a thin bar (~20-30 px on every desktop), so the top-edge
# strip is shorter. A 64 px top strip drags in the window content just below
# the bar, and the model then lands its vertical estimate in that content
# (measured: "Applications" at y=43 of a 64 px strip — below the 27 px bar —
# vs y=2-6 in a 48 px strip, on the button).
_EDGE_STRIP_H_TOP = 48

# `open <app>` tools (Agent._open): launch/focus an app directly instead of
# grounding its launcher icon. They cover the common app-launch actions —
# which are exactly where grounding is weakest (measured: "file manager
# icon" deterministically lands on the directory-menu icon, a semantic bias
# that upscaling does not fix) — and each saves one model pass. The step
# returns only when the app's window holds the focus, so a following
# type/key step always lands in the right place. Value: (launch command,
# WM_CLASS match tokens). Firefox needs --no-sandbox: the sandbox runs as
# root, same as the entrypoint's own launch.
_OPEN_APPS = {
    "firefox": ("firefox --no-sandbox", ("firefox", "navigator")),
    "firefox browser": ("firefox --no-sandbox", ("firefox", "navigator")),
    "web browser": ("firefox --no-sandbox", ("firefox", "navigator")),
    "browser": ("firefox --no-sandbox", ("firefox", "navigator")),
    "file manager": ("thunar", ("thunar",)),
    "files": ("thunar", ("thunar",)),
    "thunar": ("thunar", ("thunar",)),
    "terminal": ("xfce4-terminal", ("xfce4-terminal",)),
    "console": ("xfce4-terminal", ("xfce4-terminal",)),
}
_OPEN_APPS_MENU = frozenset({
    "applications", "applications menu", "application menu", "apps", "apps menu",
})


def _xdotool(*args: str) -> str:
    """Run xdotool, returning stdout ('' on any failure)."""
    try:
        r = subprocess.run(["xdotool", *args], capture_output=True,
                           text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def _visible_window_ids() -> set[str]:
    """Top-level visible window ids."""
    return {t for t in _xdotool("search", "--onlyvisible", "--name", ".").split()
            if t.isdigit()}


def _window_class(wid: str) -> str:
    """WM_CLASS of a window as 'instance,Class' (xprop output format)."""
    out = subprocess.run(["xprop", "-id", str(wid), "WM_CLASS"],
                         capture_output=True, text=True, timeout=5).stdout
    m = re.search(r'"([^"]*)",\s*"([^"]*)"', out)
    return f"{m.group(1)},{m.group(2)}" if m else out


def _container_of(phrase: str) -> Optional[str]:
    """Return the container noun if the phrase is '<thing> in the <container>'."""
    m = _CONTAINER_RE.search(phrase)
    if not m:
        return None
    noun = m.group(1).lower().strip()
    return noun if noun in _CONTAINERS else None


@dataclass
class Step:
    verb: str
    arg: str
    raw: str
    amount: int = 0  # scroll amount


def parse_task(task: str) -> list[Step]:
    """Split a task string into steps."""
    steps: list[Step] = []
    for line in re.split(r";|\n", task):
        line = line.strip()
        if not line:
            continue
        m = _VERB_RE.match(line)
        if m:
            verb = m.group("verb").lower().replace(" ", "-")
            if verb == "left-click":
                verb = "click"
            arg = m.group("arg").strip()
            amount = 0
            if verb == "done":
                steps.append(Step(verb="done", arg=arg, raw=line))
                continue
            if verb == "wait-load":
                steps.append(Step(verb="wait-load", arg="", raw=line))
                continue
            if not arg:
                raise ValueError(f"step needs an argument: {line!r}")
            if verb == "wait" and not arg.replace(".", "", 1).isdigit():
                raise ValueError(f"wait needs seconds: {line!r}")
            if verb == "scroll":
                sm = _SCROLL_RE.match(arg)
                if sm:
                    arg, amount = sm.group("phrase").strip(), int(sm.group("amount"))
                else:
                    raise ValueError(f"scroll step needs '<phrase> <n>': {line!r}")
            steps.append(Step(verb=verb, arg=arg, raw=line, amount=amount))
        else:
            # Bare phrase -> click.
            steps.append(Step(verb="click", arg=line, raw=line))
    return steps


def _frame_diff(a, b) -> float:
    """Mean absolute grayscale difference (0-255) between two captures,
    compared at 320x180 so even a 1080p grab stays cheap."""
    from PIL import ImageChops
    from PIL import ImageStat as _IS  # local import

    box = (320, 180)
    d = ImageChops.difference(a.convert("L").resize(box),
                              b.convert("L").resize(box))
    # PIL 13 renamed ImageStat.ImageStat -> ImageStat.Stat.
    _cls = getattr(_IS, "Stat", None) or _IS.ImageStat
    return _cls(d).mean[0]


# Mean grayscale diff (0-255) below which two frames count as "the same
# page" — a stable render. A real repaint (navigation, dialog) moves the
# mean by far more; a cursor or antialiasing flicker stays below this.
_SETTLE_DIFF = 8.0


class Agent:
    """Perceive -> ground -> act loop on top of a LocateAnythingWorker."""

    def __init__(
        self,
        worker: LocateAnythingWorker,
        dry_run: bool = False,
        delay: float = 0.2,
        max_steps: int = 32,
        scope: str = "window",  # "window" (active window) or "screen" (full monitor)
        monitor: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        box_fallback: bool = True,
    ):
        self.worker = worker
        self.dry_run = dry_run
        self.delay = delay
        self.max_steps = max_steps
        self.scope = scope
        self.monitor = monitor
        self.save_dir = Path(save_dir) if save_dir else None
        self.box_fallback = box_fallback
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ run

    def run(self, task: str) -> bool:
        steps = parse_task(task)
        print(f"[agent] {len(steps)} step(s), dry_run={self.dry_run}")
        t0 = time.time()
        for i, step in enumerate(steps):
            if i >= self.max_steps:
                print(f"[agent] max_steps={self.max_steps} reached, stopping")
                return False
            if step.verb == "done":
                print(f"[agent] done ({time.time() - t0:.1f} s total)")
                return True
            t = time.time()
            if not self.execute(i, step):
                print(f"[agent] step {i} failed, aborting")
                return False
            print(f"    step time: {time.time() - t:.2f} s")
        print(f"[agent] all steps finished ({time.time() - t0:.1f} s total)")
        return True

    # ------------------------------------------------------------- execute

    def execute(self, i: int, step: Step) -> bool:
        print(f"[agent] step {i}: {step.raw}")

        if step.verb == "wait":
            actions.pause(float(step.arg))
            return True
        if step.verb == "wait-load":
            return self._wait_load()
        if step.verb == "type":
            actions.type_text(step.arg, dry_run=self.dry_run)
            return True
        if step.verb == "key":
            actions.hotkey(step.arg, dry_run=self.dry_run)
            return True
        if step.verb == "open":
            return self._open(step.arg, i)
        if step.verb in ("search", "duckduckgo"):
            return self._search(step.arg, i)

        # All remaining verbs need a grounded target.
        if step.verb == "scroll":
            if step.amount == 0:
                print("[agent] scroll amount missing, treating as no-op")
                return True
            # Coarse grounding: region-level accuracy is all a scroll point
            # needs, so bound the vision patch count (~6x faster on a
            # full-window capture).
            target = self._ground(step.arg, i,
                                  coarse_patches=_SCROLL_COARSE_PATCHES)
        else:
            target = self._ground(step.arg, i)
        if target is None:
            return False
        ax, ay = target

        if step.verb == "click":
            actions.click(ax, ay, dry_run=self.dry_run)
        elif step.verb == "double-click":
            actions.click(ax, ay, clicks=2, dry_run=self.dry_run)
        elif step.verb == "right-click":
            actions.click(ax, ay, button="right", dry_run=self.dry_run)
        elif step.verb == "scroll":
            # Task language: positive scrolls DOWN (the reading direction);
            # pyautogui's wheel clicks count positive = up, so negate.
            actions.scroll(-step.amount, ax, ay, dry_run=self.dry_run)
        else:  # pragma: no cover
            return False

        if self.delay > 0:
            actions.pause(self.delay)
        return True

    # ------------------------------------------------------------ open tools

    def _open(self, arg: str, i: int) -> bool:
        """`open <app>`: a tool step — launch/focus the app, no grounding."""
        name = arg.lower().strip()
        if name.startswith("the "):
            name = name[4:].strip()
        if name in _OPEN_APPS_MENU:
            return self._open_applications_menu(i)
        entry = _OPEN_APPS.get(name)
        if entry is None:
            print(f"    [open] no tool for {arg!r}; known apps: "
                  f"{', '.join(sorted(_OPEN_APPS))}, applications")
            return False
        command, class_tokens = entry
        if self.dry_run:
            print(f"    [action (dry-run)] open {arg!r} ({command})")
            return True
        if self._focus_existing(class_tokens) is not None:
            print(f"    [action] focus existing {command.split()[0]} window "
                  f"(class {class_tokens})")
            actions.pause(0.8)  # let the app settle before the next step types
            return True
        before = _visible_window_ids()
        if actions.launch(command) is None:
            print(f"    [open] binary not found: {command.split()[0]!r}")
            return False
        print(f"    [action] launch {command}; waiting for focus (class {class_tokens})")
        if self._wait_app_focus(class_tokens, before, timeout=15.0):
            actions.pause(0.8)  # let the app settle before the next step types
            return True
        print("    [open] the app window did not take focus within 15 s")
        return False

    def _wait_app_focus(self, class_tokens: tuple[str, ...], exclude: set[str],
                        timeout: float, tick: float = 0.3,
                        new_only: bool = False) -> bool:
        """Activate the first new window whose WM_CLASS matches
        ``class_tokens``; by default also accept the app already holding
        the focus (singleton launchers can race the window-id poll).
        ``new_only`` disables that fallback: the caller only wants a
        window that `launch` actually spawned (search opens a new window
        in the running session; typing before it exists loses keystrokes
        into the previous window — measured).
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            for wid in sorted(_visible_window_ids() - exclude):
                cls = _window_class(wid).lower()
                if any(t in cls for t in class_tokens):
                    _xdotool("windowactivate", "--sync", wid)
                    return True
            if not new_only:
                active = _xdotool("getactivewindow")
                if active and any(t in _window_class(active).lower()
                                  for t in class_tokens):
                    return True
            time.sleep(tick)
        return False

    def _focus_existing(self, class_tokens: tuple[str, ...]) -> Optional[str]:
        """Activate a visible window matching ``class_tokens`` and return
        its id, or None if the app has no window yet.

        Attaching instead of launching matters where a second launch
        opens a NEW window: a new Firefox window in a running session
        shows the "Open previous tabs?" notification bar, which pushes
        the page layout down ~35 px and breaks the results-band prior
        (measured: the first result fell out of the band and the click
        hit dead space). A fresh first window shows no such bar
        (measured), so launching is still right when the app is not
        running. With several windows, the active one wins, else the
        largest (stray first-run windows are smaller than the
        established geometry).
        """
        ids = [w for w in _xdotool("search", "--onlyvisible", "--name", ".").split()
               if w.isdigit()
               and any(t in _window_class(w).lower() for t in class_tokens)]
        if not ids:
            return None
        active = _xdotool("getactivewindow")
        if active in ids:
            return active

        def _area(w: str) -> int:
            m = re.search(r"Geometry:\s*(\d+)x(\d+)",
                          _xdotool("getwindowgeometry", w))
            return int(m.group(1)) * int(m.group(2)) if m else 0

        wid = max(ids, key=_area)
        _xdotool("windowactivate", "--sync", wid)
        return wid

    def _open_applications_menu(self, i: int) -> bool:
        """`open applications`: click the Applications button in the top
        menu bar. Reuses the deterministic top-strip path — the strip is
        full-width, so it tracks the resolution (verified hit: (81,2) on
        1080p; the button is the leftmost menu-bar item on every XFCE).
        """
        if self.dry_run:
            print("    [action (dry-run)] open applications menu")
            return True
        cap = screen.capture(scope="screen", monitor_index=self.monitor)
        target = self._ground_capture(cap, "Applications in the menu bar")
        if target is None:
            print("    [open] applications menu button not grounded")
            return False
        ix, iy = target
        ax, ay = cap.to_screen(ix, iy)
        print(f"    target     : image=({ix:.0f}, {iy:.0f})  desktop=({ax:.0f}, {ay:.0f})  "
              f"[{cap.image.width}x{cap.image.height}]")
        actions.click(ax, ay, dry_run=False)
        actions.pause(0.5)  # let the menu popup open before the next step
        return True

    def _wait_load(self) -> bool:
        """`wait-load`: wait until a navigation's new page is painted.

        Two gates, both measured (no fixed guess):
          1. title change — a navigation leaves the old title up until the
             new document sets its own (0.05 s poll, 8 s cap). A slow
             page can set its title after the cap (measured: a heavy docs
             page, ~10 s), so the gate also accepts a SETTLED frame
             change: two consecutive captures that differ from the
             pre-navigation frame (0.3 s poll) — the new page is up and
             stable even if the title never arrives in time.
          2. render settle — the title is set BEFORE the new page paints
             (measured paint lag 0.44-1.1 s on DDG results), so gate 1
             alone would let the next step act on the old page. After
             gate 1, poll captures (0.1 s poll, 3 s cap) until the frame
             differs from the pre-navigation frame AND two consecutive
             frames agree. Skipped when nothing changed at all
             (nothing navigated).
        """
        if self.dry_run:
            print("    [action (dry-run)] wait-load")
            return True
        wid = _xdotool("getactivewindow")
        if not wid:
            return True
        pre = _xdotool("getwindowname", wid).strip()
        pre_cap = screen.capture(scope=self.scope,
                                 monitor_index=self.monitor).image
        t0 = time.time()
        outcome = "unchanged (cap)"
        prev = None
        next_frame = t0
        while time.time() - t0 < 8.0:
            if _xdotool("getwindowname", wid).strip() != pre:
                outcome = "title changed"
                break
            if time.time() >= next_frame:  # a slow page sets its title late
                frame = screen.capture(scope=self.scope,
                                       monitor_index=self.monitor).image
                if (prev is not None
                        and _frame_diff(prev, frame) <= _SETTLE_DIFF
                        and _frame_diff(pre_cap, frame) > _SETTLE_DIFF):
                    outcome = "page changed (title late)"
                    break
                prev = frame
                next_frame = time.time() + 0.3
            time.sleep(0.05)
        if outcome != "unchanged (cap)":
            t1 = time.time()
            prev = None
            while time.time() - t1 < 3.0:
                frame = screen.capture(scope=self.scope,
                                       monitor_index=self.monitor).image
                if (prev is not None
                        and _frame_diff(prev, frame) <= _SETTLE_DIFF
                        and _frame_diff(pre_cap, frame) > _SETTLE_DIFF):
                    break  # new content, settled
                prev = frame
                time.sleep(0.1)
        print(f"    [action] wait-load: {outcome} after {time.time() - t0:.2f} s")
        return True

    def _search(self, query: str, i: int) -> bool:
        """`search <query>`: open Firefox on a DuckDuckGo search for it.

        The results page is opened for the agent to see — the top result
        is NOT resolved server-side (that would pick the answer without
        the agent ever looking at the results); a following click step
        grounds and opens it. Flow: attach to the running Firefox
        window (launch only if none is running — a second launch opens
        a NEW window, and a new window in a running session shows the
        "Open previous tabs?" bar that shifts the results layout down
        ~35 px and breaks the results-band prior, measured; a fresh
        first window shows no such bar, measured) -> click the address
        bar (a mouse click, not Ctrl+L: a page that auto-focuses its
        own search box — DDG does — swallows the synthetic Ctrl+L and
        the URL lands in the page's search box instead, measured
        twice; the toolbar row sits at y≈33-68 of the window on every
        browser at the established geometry) -> type the URL -> Return
        -> poll the window title until the results page loads (the
        title changes from the previous page to '<query> at DuckDuckGo'
        ; if the tab already showed this exact search the title never
        changes and the 8 s cap bounds the cost) -> poll until the
        results region actually renders (measured: on DDG the title is
        set BEFORE the results paint, and the render lag varied 0.44 s
        to >1.1 s across runs — a fixed settle is a guess, so wait for
        real content instead of a fixed pause) -> Escape + Home,
        because a tab that already held a scroll position (a restored
        session, a page scrolled by a previous run) carries that offset
        into the new document (measured: a results page opened ~320 px
        down, the first result cut out of the results band and the
        click opening the wrong result); the results are read from the
        top, so the page is returned to its top and re-polled. The
        address-bar click works in any tab and any session state — the
        URL argument to the browser binary was measured unreliable (a
        fresh session taking over a stale profile drops it and shows
        New Tab). The URL is quote_plus-encoded, so every character
        xdotool types is plain ASCII.
        """
        url = "https://duckduckgo.com/?q=" + quote_plus(query)
        if self.dry_run:
            print(f"    [action (dry-run)] search {query!r} -> {url}")
            return True
        wid = self._focus_existing(("firefox", "navigator"))
        if wid is not None:
            print(f"    [action] search {query!r}; using the running firefox window {wid}")
        else:
            before = _visible_window_ids()
            if actions.launch("firefox --no-sandbox") is None:
                print("    [search] firefox binary not found")
                return False
            print(f"    [action] search {query!r}; waiting for firefox to take focus")
            if not self._wait_app_focus(("firefox", "navigator"), before,
                                        timeout=15.0, tick=0.1, new_only=True):
                print("    [search] firefox window did not take focus within 15 s")
                return False
            wid = _xdotool("getactivewindow")
        pre = _xdotool("getwindowname", wid).strip()
        actions.pause(0.2)  # let focus settle before typing
        # A browser launched over a killed session opens the
        # "Restore Session" interstitial (measured); its restore
        # machinery races early navigations (the tab bounced back to
        # the interstitial right after the results loaded — measured).
        # "Start New Session" (fixed position on this fixed chrome page
        # at the established geometry) clears it deterministically.
        if "Restore Session" in pre:
            print("    [search] Restore Session interstitial; "
                  "starting a new session")
            _xdotool("mousemove", "--window", wid, "737", "537")
            _xdotool("click", "--window", wid, "1")
            t0 = time.time()
            while time.time() - t0 < 5.0:
                if "Restore Session" not in _xdotool("getwindowname", wid).strip():
                    break
                time.sleep(0.1)
            pre = _xdotool("getwindowname", wid).strip()
            actions.pause(0.2)
        # The address-bar sequence, verified: the title must come to
        # contain "DuckDuckGo". A bare title change is not proof of a
        # navigation (a transient restore attempt changes the title
        # without loading anything — measured), and a focus race can
        # eat the whole sequence, so verify and retry.
        ok = False
        for attempt in range(3):
            # Focus the address bar with a mouse click, not Ctrl+L: a
            # page that auto-focuses its own search box (DDG does)
            # swallows the synthetic Ctrl+L and the URL lands in the
            # page's search box instead (measured twice). The toolbar
            # row sits at y≈33-68 of the window on every browser at the
            # established geometry, so the window-center of that row is
            # the address bar. A click in it selects the old URL, so
            # the typed URL replaces it.
            _xdotool("mousemove", "--window", wid, "640", "50")
            _xdotool("click", "--window", wid, "1")
            time.sleep(0.2)
            _xdotool("type", "--window", wid, "--clearmodifiers", "--delay", "3", url)
            _xdotool("key", "--window", wid, "Return")
            t0 = time.time()
            while time.time() - t0 < 8.0:
                if "DuckDuckGo" in _xdotool("getwindowname", wid).strip():
                    ok = True
                    break
                time.sleep(0.05)
            if ok:
                # catch a fast bounce (the restore machinery
                # reasserting itself — measured once)
                time.sleep(0.5)
                if "DuckDuckGo" not in _xdotool("getwindowname", wid).strip():
                    ok = False
            if not ok:
                print(f"    [search] results not up after attempt "
                      f"{attempt + 1}; retrying the address bar")
        if not ok:
            print(f"    [search] the DuckDuckGo results page did not load "
                  f"(title: {_xdotool('getwindowname', wid).strip()!r})")
            return False
        # Measured: on DDG the title is set BEFORE the results render
        # (the render lag varied 0.44 s to >1.1 s across runs), so a
        # fixed settle is a guess. Poll until the first-result band
        # shows real content (dark text pixels), capped at 3 s.
        t1 = time.time()
        while time.time() - t1 < 3.0:
            cap = screen.capture(scope="window")
            w, h = cap.image.size
            # Clamp the band to the live capture: a freshly mapped window
            # is briefly captured smaller than the band while xfwm is
            # still sizing it, and an unclamped px[x, y] read raises
            # IndexError("image index out of range"). A too-small crop has
            # few/no dark pixels, so the poll just keeps going until the
            # window renders at full size.
            x1, y1, x2, y2 = 120, 400, min(700, w), min(470, h)
            px = cap.image.convert("L").load()
            dark = sum(1 for y in range(y1, y2, 4)
                       for x in range(x1, x2, 8) if px[x, y] < 150)
            if dark > 20:
                break
            time.sleep(0.1)
        # A tab that already held a scroll position (a restored session,
        # a page scrolled by a previous run) carries that offset into
        # the new document (measured: a results page opened ~320 px
        # down, the first result cut out of the results band and the
        # click opening the wrong result). The address bar still holds
        # the focus after the Return: Escape drops it, Home scrolls the
        # document to its top — the results are read from the top, so
        # the results-band prior sees the first result again.
        _xdotool("key", "--window", wid, "Escape")
        _xdotool("key", "--window", wid, "Home")
        time.sleep(0.3)
        # the scroll repaints the band: confirm the first result is in
        # it (same dark-pixel poll, short cap)
        t2 = time.time()
        while time.time() - t2 < 2.0:
            cap = screen.capture(scope="window")
            w, h = cap.image.size
            x1, y1, x2, y2 = 120, 400, min(700, w), min(470, h)
            px = cap.image.convert("L").load()
            dark = sum(1 for y in range(y1, y2, 4)
                       for x in range(x1, x2, 8) if px[x, y] < 150)
            if dark > 20:
                break
            time.sleep(0.1)
        return True

    # -------------------------------------------------------------- ground

    def _ground(self, phrase: str, step: int,
                coarse_patches: Optional[int] = None) -> Optional[tuple[float, float]]:
        """Screenshot, ground `phrase`, map to desktop coords, save evidence.

        `coarse_patches` bounds the vision patch count: if the capture is
        larger, it is shrunk to fit before grounding (the full-resolution
        capture is still saved as evidence, and Capture.to_screen maps the
        coarse point back to desktop coordinates). Step types that only
        need a point *inside* a large region (scroll) pass this — region
        accuracy at a fraction of the prefill time.
        """
        cap = screen.capture(scope=self.scope, monitor_index=self.monitor)
        w, h = cap.image.size

        view = cap
        if coarse_patches is not None:
            patches = (w // 14) * (h // 14)
            if patches > coarse_patches:
                import math
                from PIL import Image  # local import

                scale = math.sqrt(coarse_patches / patches)
                vw, vh = max(14, int(w * scale)), max(14, int(h * scale))
                while (vw // 14) * (vh // 14) > coarse_patches:
                    vw, vh = int(vw * 0.98), int(vh * 0.98)
                resample = getattr(Image, "Resampling", Image)
                view = screen.Capture(
                    image=cap.image.resize((vw, vh), resample.BILINEAR),
                    monitor=cap.monitor,
                )
                print(f"    coarse view: {w}x{h} ({patches} patches) -> {vw}x{vh} "
                      f"({(vw // 14) * (vh // 14)} patches)")

        target = self._ground_capture(view, phrase)
        if target is None:
            print("    grounding failed: no point/box in the answer")
            return None
        ix, iy = target
        ax, ay = view.to_screen(ix, iy)
        if view is cap:
            print(f"    target     : image=({ix:.0f}, {iy:.0f})  desktop=({ax:.0f}, {ay:.0f})  "
                  f"[{w}x{h} @ left={cap.monitor['left']}, top={cap.monitor['top']}]")
        else:
            print(f"    target     : image=({ix:.0f}, {iy:.0f})  desktop=({ax:.0f}, {ay:.0f})  "
                  f"[coarse {view.image.width}x{view.image.height} of {w}x{h} "
                  f"@ left={cap.monitor['left']}, top={cap.monitor['top']}]")

        if self.save_dir:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            png = self.save_dir / f"step{step:02d}-{stamp}.png"
            # compress_level=1: lossless, ~3x faster encode on a 1080p grab.
            cap.image.save(png, compress_level=1)
            print(f"    screenshot : {png}")
        return ax, ay

    def _ground_capture(self, cap, phrase: str) -> Optional[tuple[float, float]]:
        """Ground `phrase` inside a capture. Returns cap-image pixels.

        Strategies, in order of preference:
          0. two-stage zoom – "<thing> in the <container>": box the (large,
             easy) container on the full capture, then ground the phrase in
             a margin-expanded crop. Small targets inside panels/tab bars are
             exactly where the strategies below fail
          1. native single pass – the capture fits the vision-token budget
          2. shrunk single pass – a modest overrun (<= 1.5x budget): shrink
             the whole capture to fit. One pass keeps the full context in
             view (tab bars, sidebars, the whole window), which beats tiling
             for typical windows and is also faster (one pass, not four)
          3. tiling – a large overrun: split into 2-4 tiles at near-native
             resolution and pick the candidate deepest inside its tile
        """
        img = cap.image
        w, h = img.size
        budget = getattr(self.worker, "in_token_limit", None) or 2048
        patches = (w // 14) * (h // 14)

        container = _container_of(phrase)
        if container:
            zt = self._zoom_ground(img, phrase, container)
            if zt is not None:
                return zt

        view, scale = None, 1.0
        if patches <= budget:
            view = img
        elif patches <= budget * 1.5:
            import math
            from PIL import Image  # local import

            scale = math.sqrt(budget / patches)
            vw, vh = max(14, int(w * scale)), max(14, int(h * scale))
            while (vw // 14) * (vh // 14) > budget:
                vw, vh = int(vw * 0.98), int(vh * 0.98)
            resample = getattr(Image, "Resampling", Image)
            view = img.resize((vw, vh), resample.BILINEAR)
            print(f"    capture {w}x{h} slightly over budget ({patches} patches) -> "
                  f"single pass at {vw / w:.2f}x ({vw}x{vh})")
        if view is not None:
            result = self.worker.point(view, phrase, verbose=False)
            print(f"    raw answer : {result.answer.strip()!r}")
            why = self._degenerate(result, view.size[0], view.size[1], "point")
            if why is None and result.center is None:
                why = "no point/box in the answer"
            if why is not None and self.box_fallback:
                print(f"    point answer rejected ({why}); retrying with box prompt")
                result = self.worker.ground(view, phrase, output_type="box", verbose=False)
                print(f"    box answer : {result.answer.strip()!r}")
                why = self._degenerate(result, view.size[0], view.size[1], "box")
                if why is None and result.center is None:
                    why = "no point/box in the answer"
            if why is not None:
                print(f"    grounding failed: {why}")
                return None
            x, y = result.center
            return (x / scale, y / scale)

        # Tiling pass: split along the dominant axis (only for very elongated
        # captures; a 16:9 desktop stays 2x2 so each tile is near-native),
        # or 2x2 when roughly square.
        tx, ty = (w + 1) // 2, (h + 1) // 2
        if w > h * 2:
            tiles = [(0, 0, img.crop((0, 0, tx, h))), (tx, 0, img.crop((tx, 0, w, h)))]
        elif h > w * 2:
            tiles = [(0, 0, img.crop((0, 0, w, ty))), (0, ty, img.crop((0, ty, w, h)))]
        else:
            tiles = [(0, 0, img.crop((0, 0, tx, ty))), (tx, 0, img.crop((tx, 0, w, ty))),
                     (0, ty, img.crop((0, ty, tx, h))), (tx, ty, img.crop((tx, ty, w, h)))]
        print(f"    capture {w}x{h} exceeds token budget ({patches} patches) -> "
              f"{len(tiles)} tile(s)")

        cands = []  # (margin_px, cap_x, cap_y, tile, x, y)
        for i, (ox, oy, tile) in enumerate(tiles):
            tw, th = tile.size
            result = self.worker.point(tile, phrase, verbose=False)
            if result.center is None and self.box_fallback:
                result = self.worker.ground(tile, phrase, output_type="box", verbose=False)
                print(f"    tile {i} answer   : (box) {result.answer.strip()!r}")
            else:
                print(f"    tile {i} answer   : {result.answer.strip()!r}")
            why = self._degenerate(result, tw, th, "point")
            if why is not None and self.box_fallback:
                print(f"    tile {i} point spam ({why}); retrying with box prompt")
                result = self.worker.ground(tile, phrase, output_type="box", verbose=False)
                print(f"    tile {i} answer   : (box) {result.answer.strip()!r}")
                why = self._degenerate(result, tw, th, "box")
                if why is None and result.center is None:
                    why = "no point/box in the answer"
            if why is not None:
                print(f"    tile {i} rejected : {why}")
                continue
            if result.center is None:
                continue
            x, y = result.center
            margin = min(x, tw - x, y, th - y)
            print(f"    tile {i} point    : ({x:.0f}, {y:.0f})  edge-margin={margin:.0f}px")
            cands.append((margin, ox + x, oy + y, tile, x, y))

        # Content gate: prefer candidates sitting on real UI content.
        # A hallucinated point on flat regions (wallpaper, blank pages)
        # has a low local grayscale std; a real icon/button does not.
        # When the candidates clearly differ in content, keep only the
        # high-content ones; otherwise fall back to the margin heuristic.
        if len(cands) > 1:
            from PIL import ImageStat  # local import

            stds = []
            for _, _, _, tile, x, y in cands:
                g = tile.convert("L")
                r = 32
                box = g.crop((max(0, int(x) - r), max(0, int(y) - r),
                              min(g.width, int(x) + r), min(g.height, int(y) + r)))
                stds.append(ImageStat.Stat(box).stddev[0])
            hi = [c for c, s in zip(cands, stds) if s >= 30 and s >= 1.8 * min(stds)]
            if 0 < len(hi) < len(cands):
                print(f"    content gate : {len(hi)}/{len(cands)} candidates on real content "
                      f"(local std {min(stds):.0f}..{max(stds):.0f})")
                cands = hi

        best = max(cands, key=lambda c: c[0]) if cands else None
        if best is not None:
            print(f"    chosen tile  : edge-margin={best[0]:.0f}px")
        return None if best is None else (best[1], best[2])

    def _zoom_ground(self, img, phrase: str, container: str) -> Optional[tuple[float, float]]:
        """Two-stage grounding for '<thing> in the <container>'.

        Pass 1 boxes the container on the full capture (the processor
        shrinks it to the token budget; containers are large, so this is
        reliable). Pass 2 grounds the original phrase in a margin-expanded
        crop of the capture, shrunk to fit the budget or upscaled when
        small. Returns capture pixels, or None to fall back to the normal
        path.

        The decode is stochastic on legacy hardware (the same prompt can
        box a 120 px corner for a full-window container in one run and the
        window itself in the next), so the whole zoom is retried once when
        it fails.

        Edge containers with a fixed desktop position (_EDGE_DEFAULT) skip
        the box pass entirely: the strip is the known edge of the screen,
        and only pass 2 (the target in the strip) costs a model pass.
        Center containers (_CENTER_CONTAINERS: modal dialogs, cookie
        walls) do the same with a fixed box around the button zone
        slightly below the capture center, and result lists
        (_RESULTS_CONTAINERS) with a fixed band where the first organic
        result of a search results page sits.
        """
        if container in _EDGE_DEFAULT:
            w, h = img.width, img.height
            edge = _EDGE_DEFAULT[container]
            strip_h = _EDGE_STRIP_H_TOP if edge == "top" else _EDGE_STRIP_H
            box = ((0.0, float(h) - strip_h, float(w), float(h))
                   if edge == "bottom" else (0.0, 0.0, float(w), strip_h))
            zt = self._zoom_attempt(
                img, phrase, container, box=box,
                box_note=f"known {edge} edge", det_edge=edge)
            if zt is not None:
                return zt
            print("    zoom: deterministic strip failed; falling back "
                  "to the box pass")
        if container in _CENTER_CONTAINERS:
            w, h = img.width, img.height
            box = ((float(w) * _CENTER_BOX_X[0],
                    max(0.0, h / 2 + _CENTER_BOX_Y[0]),
                    float(w) * _CENTER_BOX_X[1],
                    min(float(h), h / 2 + _CENTER_BOX_Y[1])))
            zt = self._zoom_attempt(
                img, phrase, container, box=box,
                box_note="known center region", det_edge="center")
            if zt is not None:
                return zt
            print("    zoom: deterministic center box failed; falling back "
                  "to the box pass")
        if container in _RESULTS_CONTAINERS:
            w, h = img.width, img.height
            box = ((float(w) * _RESULTS_BOX[0], float(h) * _RESULTS_BOX[1],
                    float(w) * _RESULTS_BOX[2], float(h) * _RESULTS_BOX[3]))
            zt = self._zoom_attempt(
                img, phrase, container, box=box,
                box_note="known results band", det_edge="results",
                pass2_box_first=True, det_trust=True)
            if zt is not None:
                return zt
            print("    zoom: deterministic results band failed; falling back "
                  "to the box pass")
        for attempt in (1, 2):
            zt = self._zoom_attempt(img, phrase, container)
            if zt is not None:
                return zt
            if attempt == 1:
                print("    zoom: attempt 1 failed; retrying once "
                      "(decode varies between attempts)")
        return None

    def _zoom_attempt(self, img, phrase: str, container: str,
                      box=None, box_note: Optional[str] = None,
                      det_edge: Optional[str] = None,
                      pass2_box_first: bool = False,
                      det_trust: bool = False) -> Optional[tuple[float, float]]:
        """One pass-1 (+ pass-2) attempt. Returns capture pixels or None.

        ``box``: a pre-supplied container box (the deterministic edge
        strip) — pass 1 is skipped and the box flows through the snap/
        crop logic unchanged. ``pass2_box_first`` swaps pass 2's order:
        box grounding before point grounding (for targets where the box
        is the reliable answer — e.g. a search result block, where a
        point can wander into the whitespace around the text; measured
        on the DDG results band: point std 22 < floor 43, box std 57).
        ``det_trust``: skip the content gate for a deterministic box —
        only for boxes whose content is known to BE the target region
        (the results band: every pixel in it is part of the first
        result, so the model's box center is clickable even when the
        local-std floor is high on a text-dense crop; measured: a
        correct box on the title row, std 56 < floor 70, was rejected
        by the gate).
        """
        import math
        from PIL import Image  # local import

        w, h = img.width, img.height

        def _point_box() -> Optional[tuple[float, float, float, float]]:
            # Point prompts are more robust than box prompts at low
            # resolution (an OOM retry can degrade the view); a point on the
            # container plus a margin is a usable crop.
            result = self.worker.point(img, f"the {container}", verbose=False)
            print(f"    zoom point   : {result.answer.strip()!r}")
            why = self._degenerate(result, w, h, "point")
            if why is not None or result.center is None:
                return None
            x, y = result.center
            m = 320.0
            return (max(0.0, x - m), max(0.0, y - m),
                    min(float(w), x + m), min(float(h), y + m))

        def _usable(b) -> bool:
            # Unusable: absent, microscopic, whole-frame, or a tiny box in
            # a screen corner (the model's failure mode: e.g.
            # (0,0)-(120,120) when asked to box a full-window container).
            # Thin-but-real strips (tab bars) are kept.
            if b is None:
                return False
            x1, y1, x2, y2 = b
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 8 or bw * bh > 0.9 * w * h:
                return False
            if bw * bh < 0.04 * w * h and (
                    (x1 <= 4 or x2 >= w - 4) and (y1 <= 4 or y2 >= h - 4)):
                return False
            return True

        if box is None:
            box = self._ground_boxes(img, f"the {container}")
            whole_frame = (box is not None and
                           (box[2] - box[0]) * (box[3] - box[1])
                           > 0.9 * w * h)
            if whole_frame:
                # The container fills the capture (e.g. the active window
                # is the browser and the phrase is "… in the browser"): a
                # whole-frame box is the correct answer, not garbage — the
                # crop is the whole capture and pass 2 grounds the target
                # in it (one less retry).
                box = (0.0, 0.0, float(w), float(h))
                box_note = "whole-frame box"
            elif not _usable(box):
                print(f"    zoom: box for 'the {container}' unusable ({box}); "
                      "retrying box prompt")
                box = self._ground_boxes(img, f"the {container}")
                if not _usable(box):
                    box = _point_box()
                    if box is None:
                        print(f"    zoom: no usable box/point for 'the {container}'")
                        return None
        box_note = box_note or "box"
        x1, y1, x2, y2 = box
        if det_edge is not None:
            # A deterministic edge strip: the supplied box is already the
            # exact crop (it hugs a known screen edge, so no snap, no margin,
            # no upscale). Keep the content gate (trust_box False) because
            # the strip still includes wallpaper beside the panel; det_trust
            # containers (the results band) hold the target itself, so their
            # box center is trusted as-is.
            cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)
            trust_box = det_trust
        else:
            # Models box containers loosely (the box spills into the
            # surrounding area). Panels/bars hug a screen edge, so snap a box
            # that touches an edge to a thin strip along that edge: the crop
            # gets smaller and the target bigger in pass 2.
            # Only thin boxes (real strips: panels, bars) are snapped: a big
            # window that merely touches a screen edge must keep its full
            # box, so the thinness test is 25% of the capture dimension. For
            # known edge containers (panel, taskbar, ...) the model sometimes
            # boxes a whole screen quadrant; the edge touch is then the
            # reliable signal, so the threshold relaxes to 60%.
            edge = container in _EDGE_CONTAINERS

            def _thin(span: float, size: float) -> bool:
                return span < size * (0.6 if edge else 0.25)

            snap_x = snap_y = 0  # -1: snapped to bottom/right edge, 1: top/left
            if y2 >= h - 4 and _thin(y2 - y1, h):
                y1, snap_y = h - 120, -1
            elif y1 <= 4 and _thin(y2 - y1, h):
                y2, snap_y = 120, 1
            if x2 >= w - 4 and _thin(x2 - x1, w):
                x1, snap_x = w - 120, -1
            elif x1 <= 4 and _thin(x2 - x1, w):
                x2, snap_x = 120, 1
            # A box that needed no snapping is the model's own tight box
            # (e.g. a browser window): a point inside it is inside the
            # container, so it is trusted even on flat surfaces (an address
            # bar is plain white). Snapped strips keep the content gate,
            # because a loose box spills onto the wallpaper.
            trust_box = (snap_x == 0 and snap_y == 0)

            # Expand in FULL-capture coordinates: a container (e.g. a panel)
            # can straddle tile boundaries, so the crop must come from the
            # capture, not from whatever view produced the box. A snapped edge
            # already hugs the screen, so only a small margin is needed on its
            # inside.
            m, m_edge = 160, 40
            # A full-edge container (panel, taskbar, menu bar) spans the whole
            # edge: when it snaps to a horizontal edge the strip runs the full
            # screen width, and to a vertical edge the full screen height. The
            # box's extent along the edge is the model's guess, not a boundary.
            if container in _FULL_EDGE and snap_y and not snap_x:
                cx1, cx2 = 0, int(w)
            else:
                cx1 = max(0, int(x1) - (m_edge if snap_x == -1 else m))
                cx2 = min(w, int(x2) + (m_edge if snap_x == 1 else m))
            if container in _FULL_EDGE and snap_x and not snap_y:
                cy1, cy2 = 0, int(h)
            else:
                m_y = m_edge if snap_y != 0 else m
                cy1 = max(0, int(y1) - m_y)
                cy2 = min(h, int(y2) + m_y)

        if cx2 - cx1 < 28 or cy2 - cy1 < 28:
            print("    zoom: container box too small after expansion")
            return None
        crop = img.crop((cx1, cy1, cx2, cy2))

        # Keep the pass-2 prefill small while giving the target as many
        # patches as possible: upscale small/thin crops (a 32 px icon in a
        # 160 px strip becomes 64 px in 320 px -> twice the patch rows),
        # and shrink to the token budget if that overruns it. A
        # full-screen-length edge strip (and the deterministic strip) stays
        # at 1.0x: it is already tall enough that a 32 px icon spans ~2 patch
        # rows, and the 2x upscale doubles the CPU-ViT cost (measured
        # 3.7 s -> 1.4 s for the bottom-panel strip on the 1070).
        if det_edge is not None:
            factor = 1.0
        else:
            full_strip = container in _FULL_EDGE and (snap_x or snap_y)
            budget = getattr(self.worker, "in_token_limit", None) or 2048
            cpatches = (crop.width // 14) * (crop.height // 14)
            factor = 1.0
            if not full_strip and (max(crop.size) < 400 or min(crop.size) <= 160):
                factor = 2.0
            if (int(crop.width * factor) // 14) * (int(crop.height * factor) // 14) > budget:
                factor = math.sqrt(budget / cpatches)
                while factor > 0.5 and (int(crop.width * factor) // 14) * (
                        int(crop.height * factor) // 14) > budget:
                    factor *= 0.98
        if factor != 1.0:
            resample = getattr(Image, "Resampling", Image)
            crop = crop.resize((max(14, int(crop.width * factor)),
                                 max(14, int(crop.height * factor))), resample.BILINEAR)
        print(f"    zoom: 'the {container}' -> {box_note} ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
              f"-> crop ({cx1},{cy1})-({cx2},{cy2}) at {factor:.2f}x")

        # Pass 2. The model can land on flat content (wallpaper) even when
        # the target is in the crop, so candidates are scored by local
        # edge density and must clearly stand out from the crop's own
        # background; if the point pass is weak, a box pass is added.
        import statistics
        from PIL import ImageStat  # local imports
        gray = crop.convert("L")

        def _std_at(x: float, y: float) -> float:
            r = 32
            b = gray.crop((max(0, int(x) - r), max(0, int(y) - r),
                           min(gray.width, int(x) + r), min(gray.height, int(y) + r)))
            return ImageStat.Stat(b).stddev[0]

        # Background level: a low quantile of a coarse grid over the crop.
        # The median is right when the crop is mostly empty background, but
        # a crop can contain large non-background regions (a window
        # straddling the strip), and those push the median up until the
        # real target is rejected (measured: floor 83 vs icon std 85 on
        # the bottom strip with a terminal window open). The lower quartile
        # stays at the empty-background level while such regions are
        # present, and plays the median's role on uniform crops.
        bg = []
        for gx in range(1, 12):
            for gy in range(1, 7):
                x, y = gray.width * gx / 12, gray.height * gy / 7
                bg.append(_std_at(x, y))
        floor = max(30.0, 1.5 * sorted(bg)[len(bg) // 4])

        def _score(result, kind: str) -> Optional[tuple[float, float, float]]:
            why = self._degenerate(result, crop.size[0], crop.size[1], kind)
            if why is not None or result.center is None:
                return None
            # A single full-crop box is the model's "target not found in
            # this crop" answer: its center is a blind guess (measured:
            # on a layout-shifted results page the box pass returned the
            # full crop and the trusted-band click hit dead space).
            # Reject it so the caller tries the next pass.
            if kind == "box" and len(result.boxes) == 1:
                bx1, by1, bx2, by2 = result.boxes[0]
                if (bx1 <= 2 and by1 <= 2
                        and bx2 >= crop.size[0] - 2
                        and by2 >= crop.size[1] - 2):
                    print("    zoom answer: full-frame box = no target in "
                          "the crop; trying the next pass")
                    return None
            x, y = result.center
            s = _std_at(x, y)
            fx, fy = cx1 + x / factor, cy1 + y / factor
            if trust_box and (x1 - 4 <= fx <= x2 + 4) and (y1 - 4 <= fy <= y2 + 4):
                print(f"    zoom content : local std {s:.0f} at ({x:.0f}, {y:.0f}) "
                      f"inside container box -> accepted")
                return (s, x, y)
            print(f"    zoom content : local std {s:.0f} at ({x:.0f}, {y:.0f}) "
                  f"(background floor {floor:.0f})")
            return (s, x, y) if s >= floor else None

        # The crop IS the container, so ground the bare target phrase in it
        # ("the address bar", not "the address bar in the browser" - the
        # container clause is noise when the container fills the view).
        bare = _CONTAINER_RE.sub("", phrase).strip() or phrase

        if pass2_box_first:
            result = self.worker.ground(crop, bare, output_type="box",
                                        verbose=False)
            print(f"    zoom answer: (box) {result.answer.strip()!r}")
            best = _score(result, "box")
            if best is None and self.box_fallback:
                best = _score(self.worker.point(crop, bare, verbose=False),
                              "point")
        else:
            best = _score(self.worker.point(crop, bare, verbose=False), "point")
            if best is None and self.box_fallback:
                result = self.worker.ground(crop, bare, output_type="box",
                                            verbose=False)
                print(f"    zoom answer: (box) {result.answer.strip()!r}")
                best = _score(result, "box")
        if best is None:
            print("    zoom failed: no candidate on real content; "
                  "falling back to normal grounding")
            return None
        _, x, y = best
        return (cx1 + x / factor, cy1 + y / factor)

    def _ground_boxes(self, img, phrase: str) -> Optional[tuple[float, float, float, float]]:
        """Box-ground `phrase` on the full image in a single pass.

        The processor shrinks the image to the token budget internally,
        which is fine for containers (large targets). Returns the first
        box in img pixel coords, or None.
        """
        result = self.worker.ground(img, phrase, output_type="box", verbose=False)
        print(f"    zoom box     : {result.answer.strip()!r}")
        why = self._degenerate(result, img.width, img.height, "box")
        if why is not None:
            print(f"    zoom box rejected: {why}")
            return None
        if result.boxes:
            return result.boxes[0]
        if result.points:  # a point answer is still usable as a rough box
            x, y = result.points[0]
            b = 60.0
            return (max(0.0, x - b), max(0.0, y - b),
                    min(float(img.width), x + b), min(float(img.height), y + b))
        return None

    @staticmethod
    def _degenerate(result: GroundingResult, w: int, h: int,
                    kind: str = "both") -> Optional[str]:
        """Why a model answer is unusable, or None if it looks sane.

        Catches the failure modes seen on small GPUs: decoding spam (dozens
        of boxes) and degenerate repeats (identical or full-frame boxes)
        that would otherwise make us click a random spot. ``kind`` scopes
        the checks to the output actually used: a point answer commonly
        trails decode-spam boxes that must not disqualify a good point.
        """
        from collections import Counter

        if kind in ("both", "box"):
            boxes = result.boxes
            if boxes:
                if len(boxes) > 20:
                    return f"{len(boxes)} boxes (decoding spam)"
                full = sum(
                    1 for b in boxes
                    if b[0] <= 2 and b[1] <= 2 and b[2] >= w - 2 and b[3] >= h - 2
                )
                if full > 1:
                    return f"{full} full-frame boxes"
                common, n = Counter(boxes).most_common(1)[0]
                if n >= 3 and n * 2 > len(boxes):
                    return f"{n} identical boxes"
                if n >= 2 and common[0] <= 2 and common[1] <= 2:
                    return f"{n} identical boxes at the image corner"
        if kind in ("both", "point"):
            pts = result.points
            if pts:
                _, n = Counter(pts).most_common(1)[0]
                if n >= 3 and n * 2 > len(pts):
                    return f"{n} identical points"
        return None


def offline_ground(
    worker: LocateAnythingWorker,
    image: ImageLike,
    phrase: str,
    output_type: str = "point",
    save_dir: Optional[Union[str, Path]] = None,
    draw: bool = True,
) -> GroundingResult:
    """Ground a phrase in an image file, without touching the screen.

    If ``draw`` is True and an output image exists, saves an annotated copy
    next to the original as `<name>.grounded.png`.
    """
    if not image:
        raise ValueError("image path is required")
    from PIL import Image, ImageDraw  # local import

    result = worker.ground(image, phrase, output_type=output_type)
    if draw:
        img = Image.open(image).convert("RGB")
        draw = ImageDraw.Draw(img)
        for x, y in result.points:
            draw.ellipse([x - 12, y - 12, x + 12, y + 12], outline="red", width=3)
            draw.line([x - 20, y, x + 20, y, x, y - 20, x, y + 20], fill="red", width=2)
        for x1, y1, x2, y2 in result.boxes:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        out = Path(image).with_suffix("").with_name(Path(image).stem + ".grounded" + Path(image).suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"annotated copy: {out}")
    print(f"answer   : {result.answer!r}")
    print(f"points   : {[(round(x), round(y)) for x, y in result.points]}")
    print(f"boxes    : {[(round(a), round(b), round(c), round(d)) for a, b, c, d in result.boxes]}")
    print(f"target   : {result.center}")
    return result