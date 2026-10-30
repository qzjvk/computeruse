"""Parse LocateAnything outputs into pixel coordinates.

The model emits coordinates as normalized integers in [0, 1000] wrapped in
special tokens:

    point:  <box><x><y></box>
    box:    <box><x1><x2><y1><y2></box>   (4 coordinates; the on-disk order
                                           can be (x1,x2,y1,y2) or (x1,y1,x2,y2)
                                           depending on the decoding path, so
                                           we resolve it by axis min/max)
    empty:  <box>none</box>
"""

from __future__ import annotations

import re

# <box> followed by 2 or 4 integer coordinates
_BOX_RE = re.compile(r"<box><(\d+)>(?:<(\d+)>(?:<(\d+)>(?:<(\d+)>)?)?)?</box>")


def _scale(norm: int, size: int) -> float:
    return norm / 1000.0 * size


def parse_points(answer: str, image_width: int, image_height: int) -> list[tuple[float, float]]:
    """Extract 2-coordinate points from the model answer."""
    points: list[tuple[float, float]] = []
    for m in _BOX_RE.finditer(answer):
        g = m.groups()
        if g[2] is None:  # exactly two coords -> point
            points.append((_scale(int(g[0]), image_width), _scale(int(g[1]), image_height)))
    return points


def parse_boxes(answer: str, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    """Extract 4-coordinate boxes as (x1, y1, x2, y2) pixel rects.

    The model's coordinate order varies by decode path ((x1,x2,y1,y2) or
    (x1,y1,x2,y2)), so both are tried and the larger-area interpretation is
    kept: the cross-interpretation scrambles a real box into a smaller (or
    degenerate) one.
    """
    boxes: list[tuple[float, float, float, float]] = []
    for m in _BOX_RE.finditer(answer):
        g = m.groups()
        if g[3] is None:  # not a 4-coord box
            continue
        c1, c2, c3, c4 = (int(v) for v in g)
        # A: first two are X, last two are Y.  B: (x1, y1, x2, y2).
        axa, axb = min(c1, c2), max(c1, c2)
        aya, ayb = min(c3, c4), max(c3, c4)
        bxa, bxb = min(c1, c3), max(c1, c3)
        bya, byb = min(c2, c4), max(c2, c4)
        if (bxb - bxa) * (byb - bya) > (axb - axa) * (ayb - aya):
            xa, xb, ya, yb = bxa, bxb, bya, byb
        else:
            xa, xb, ya, yb = axa, axb, aya, ayb
        boxes.append((_scale(xa, image_width), _scale(ya, image_height),
                      _scale(xb, image_width), _scale(yb, image_height)))
    return boxes


def best_target(
    answer: str, image_width: int, image_height: int
) -> tuple[float, float] | None:
    """Best click target: first point, else center of first box, else None."""
    points = parse_points(answer, image_width, image_height)
    if points:
        return points[0]
    boxes = parse_boxes(answer, image_width, image_height)
    if boxes:
        x1, y1, x2, y2 = boxes[0]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return None