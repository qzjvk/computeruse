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
})


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
            why = self._degenerate(result, view.size[0], view.size[1])
            if why is None and result.center is None:
                why = "no point/box in the answer"
            if why is not None and self.box_fallback:
                print(f"    point answer rejected ({why}); retrying with box prompt")
                result = self.worker.ground(view, phrase, output_type="box", verbose=False)
                print(f"    box answer : {result.answer.strip()!r}")
                why = self._degenerate(result, view.size[0], view.size[1])
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
            why = self._degenerate(result, tw, th)
            if why is not None and self.box_fallback:
                print(f"    tile {i} point spam ({why}); retrying with box prompt")
                result = self.worker.ground(tile, phrase, output_type="box", verbose=False)
                print(f"    tile {i} answer   : (box) {result.answer.strip()!r}")
                why = self._degenerate(result, tw, th)
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
            why = self._degenerate(result, w, h)
            if why is not None or result.center is None:
                return None
            x, y = result.center
            m = 160.0
            return (max(0.0, x - m), max(0.0, y - m),
                    min(float(w), x + m), min(float(h), y + m))

        box = self._ground_boxes(img, f"the {container}")
        if (box is None
                or box[2] - box[0] < 8
                or box[3] - box[1] < 8
                or (box[2] - box[0]) * (box[3] - box[1]) > 0.9 * w * h):
            box = _point_box()
            if box is None:
                print(f"    zoom: no usable box/point for 'the {container}'")
                return None
        x1, y1, x2, y2 = box
        # Models box containers loosely (the box spills into the surrounding
        # area). Panels/bars hug a screen edge, so snap a box that touches
        # an edge to a thin strip along that edge: the crop gets smaller and
        # the target bigger in pass 2.
        if y2 >= h - 4 and (y2 - y1) < h * 0.75:
            y1 = h - 120
        elif y1 <= 4 and (y2 - y1) < h * 0.75:
            y2 = 120
        if x2 >= w - 4 and (x2 - x1) < w * 0.75:
            x1 = w - 120
        elif x1 <= 4 and (x2 - x1) < w * 0.75:
            x2 = 120

        # Expand in FULL-capture coordinates: a container (e.g. a panel) can
        # straddle tile boundaries, so the crop must come from the capture,
        # not from whatever view produced the box.
        m = 160
        cx1 = max(0, int(x1) - m)
        cy1 = max(0, int(y1) - m)
        cx2 = min(w, int(x2) + m)
        cy2 = min(h, int(y2) + m)
        if cx2 - cx1 < 28 or cy2 - cy1 < 28:
            print("    zoom: container box too small after expansion")
            return None
        crop = img.crop((cx1, cy1, cx2, cy2))

        # Keep the pass-2 prefill small: shrink the crop to the token budget
        # when it overruns, upscale small crops so the target is large.
        budget = getattr(self.worker, "in_token_limit", None) or 2048
        cpatches = (crop.width // 14) * (crop.height // 14)
        factor = 1.0
        if cpatches > budget:
            factor = math.sqrt(budget / cpatches)
            while factor > 0.5 and (int(crop.width * factor) // 14) * (
                    int(crop.height * factor) // 14) > budget:
                factor *= 0.98
        elif max(crop.size) < 400:
            factor = 2.0
        if factor != 1.0:
            resample = getattr(Image, "Resampling", Image)
            crop = crop.resize((max(14, int(crop.width * factor)),
                                max(14, int(crop.height * factor))), resample.BILINEAR)
        print(f"    zoom: 'the {container}' -> box ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
              f"-> crop ({cx1},{cy1})-({cx2},{cy2}) at {factor:.2f}x")

        result = self.worker.point(crop, phrase, verbose=False)
        print(f"    zoom answer: {result.answer.strip()!r}")
        why = self._degenerate(result, crop.size[0], crop.size[1])
        if why is None and result.center is None:
            why = "no point/box in the answer"
        if why is not None and self.box_fallback:
            result = self.worker.ground(crop, phrase, output_type="box", verbose=False)
            print(f"    zoom answer: (box) {result.answer.strip()!r}")
            why = self._degenerate(result, crop.size[0], crop.size[1])
            if why is None and result.center is None:
                why = "no point/box in the answer"
        if why is not None:
            print(f"    zoom failed: {why}; falling back to normal grounding")
            return None
        x, y = result.center
        return (cx1 + x / factor, cy1 + y / factor)

    def _ground_boxes(self, img, phrase: str) -> Optional[tuple[float, float, float, float]]:
        """Box-ground `phrase` on the full image in a single pass.

        The processor shrinks the image to the token budget internally,
        which is fine for containers (large targets). Returns the first
        box in img pixel coords, or None.
        """
        result = self.worker.ground(img, phrase, output_type="box", verbose=False)
        print(f"    zoom box     : {result.answer.strip()!r}")
        why = self._degenerate(result, img.width, img.height)
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
    def _degenerate(result: GroundingResult, w: int, h: int) -> Optional[str]:
        """Why a model answer is unusable, or None if it looks sane.

        Catches the failure modes seen on small GPUs: decoding spam (dozens
        of boxes) and degenerate repeats (identical or full-frame boxes)
        that would otherwise make us click a random spot.
        """
        from collections import Counter

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