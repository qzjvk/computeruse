"""The computer-use loop: screenshot -> ground with LocateAnything -> act.

The perception model only *localizes*; planning is a small, deterministic task
language so the agent is transparent and easy to test:

    click <phrase>            left-click the grounded element   (default verb)
    double-click <phrase>     double-click
    right-click <phrase>      right-click
    type <text>               type text (ASCII)
    key <combo>               press a key combo, e.g. ctrl+alt+t
    scroll <phrase> <n>       scroll n notches at the element (+up / -down)
    wait <seconds>            idle
    done                      stop

Steps are separated by ';' or newlines. A bare phrase is treated as `click`.
"""

from __future__ import annotations

import re
import time
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
    r"type|key|scroll|wait|done"
    r")\s*(?P<arg>.*?)\s*$",
    re.IGNORECASE,
)
_SCROLL_RE = re.compile(r"^(?P<phrase>.+?)\s+(?P<amount>[+-]?\d+)\s*$")


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


class Agent:
    """Perceive -> ground -> act loop on top of a LocateAnythingWorker."""

    def __init__(
        self,
        worker: LocateAnythingWorker,
        dry_run: bool = False,
        delay: float = 0.8,
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
        for i, step in enumerate(steps):
            if i >= self.max_steps:
                print(f"[agent] max_steps={self.max_steps} reached, stopping")
                return False
            if step.verb == "done":
                print("[agent] done")
                return True
            if not self.execute(i, step):
                print(f"[agent] step {i} failed, aborting")
                return False
        print("[agent] all steps finished")
        return True

    # ------------------------------------------------------------- execute

    def execute(self, i: int, step: Step) -> bool:
        print(f"[agent] step {i}: {step.raw}")

        if step.verb == "wait":
            actions.pause(float(step.arg))
            return True
        if step.verb == "type":
            actions.type_text(step.arg, dry_run=self.dry_run)
            return True
        if step.verb == "key":
            actions.hotkey(step.arg, dry_run=self.dry_run)
            return True

        # All remaining verbs need a grounded target.
        if step.verb == "scroll":
            if step.amount == 0:
                print("[agent] scroll amount missing, treating as no-op")
                return True
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
            actions.scroll(step.amount, ax, ay, dry_run=self.dry_run)
        else:  # pragma: no cover
            return False

        if self.delay > 0:
            actions.pause(self.delay)
        return True

    # -------------------------------------------------------------- ground

    def _ground(self, phrase: str, step: int) -> Optional[tuple[float, float]]:
        """Screenshot, ground `phrase`, map to desktop coords, save evidence."""
        cap = screen.capture(scope=self.scope, monitor_index=self.monitor)
        w, h = cap.image.size

        target = self._ground_capture(cap, phrase)
        if target is None:
            print("    grounding failed: no point/box in the answer")
            return None
        ix, iy = target
        ax, ay = cap.to_screen(ix, iy)
        print(f"    target     : image=({ix:.0f}, {iy:.0f})  desktop=({ax:.0f}, {ay:.0f})  "
              f"[{w}x{h} @ left={cap.monitor['left']}, top={cap.monitor['top']}]")

        if self.save_dir:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            png = self.save_dir / f"step{step:02d}-{stamp}.png"
            cap.image.save(png)
            print(f"    screenshot : {png}")
        return ax, ay

    def _ground_capture(self, cap, phrase: str) -> Optional[tuple[float, float]]:
        """Ground `phrase` inside a capture. Returns cap-image pixels.

        Single pass when the capture fits the worker's vision-token budget;
        otherwise split it into 2-4 tiles and ground each at near-native
        resolution (far more accurate for a 3B model on a small-VRAM GPU),
        choosing the candidate point that lies deepest inside its tile.
        """
        img = cap.image
        w, h = img.size
        budget = getattr(self.worker, "in_token_limit", None) or 2048
        patches = (w // 14) * (h // 14)

        if patches <= budget:
            result = self.worker.point(img, phrase, verbose=False)
            print(f"    raw answer : {result.answer.strip()!r}")
            if result.center is None and self.box_fallback:
                result = self.worker.ground(img, phrase, output_type="box", verbose=False)
                print(f"    box answer : {result.answer.strip()!r}")
            return result.center

        # Tiling pass: split along the dominant axis, or 2x2 when square-ish.
        tx, ty = (w + 1) // 2, (h + 1) // 2
        if w > h * 1.5:
            tiles = [(0, 0, img.crop((0, 0, tx, h))), (tx, 0, img.crop((tx, 0, w, h)))]
        elif h > w * 1.5:
            tiles = [(0, 0, img.crop((0, 0, w, ty))), (0, ty, img.crop((0, ty, w, h)))]
        else:
            tiles = [(0, 0, img.crop((0, 0, tx, ty))), (tx, 0, img.crop((tx, 0, w, ty))),
                     (0, ty, img.crop((0, ty, tx, h))), (tx, ty, img.crop((tx, ty, w, h)))]
        print(f"    capture {w}x{h} exceeds token budget ({patches} patches) -> "
              f"{len(tiles)} tile(s)")

        best = None  # (margin_px, cap_x, cap_y)
        for i, (ox, oy, tile) in enumerate(tiles):
            tw, th = tile.size
            result = self.worker.point(tile, phrase, verbose=False)
            if result.center is None and self.box_fallback:
                result = self.worker.ground(tile, phrase, output_type="box", verbose=False)
                print(f"    tile {i} answer   : (box) {result.answer.strip()!r}")
            else:
                print(f"    tile {i} answer   : {result.answer.strip()!r}")
            if result.center is None:
                continue
            x, y = result.center
            margin = min(x, tw - x, y, th - y)
            print(f"    tile {i} point    : ({x:.0f}, {y:.0f})  edge-margin={margin:.0f}px")
            if best is None or margin > best[0]:
                best = (margin, ox + x, oy + y)
        if best is not None:
            print(f"    chosen tile  : edge-margin={best[0]:.0f}px")
        return None if best is None else (best[1], best[2])


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