"""computer_use: a computer-use agent powered by NVIDIA LocateAnything-3B.

LocateAnything is a vision-language grounding model (Qwen2.5-3B + MoonViT) that
localizes GUI elements / objects from natural language. This package turns it
into a computer-use agent: screenshot -> ground -> click/type/key.

The agent is tuned to run on legacy NVIDIA GPUs (e.g. a GTX 1070, sm_61) using
float16 weights, the SDPA math attention backend, and a CPU-offloaded vision
encoder.
"""

import os
import sys

# On Linux the X11 connection must be opened BEFORE torch's CUDA driver is
# loaded: creating a fresh Xlib connection after CUDA init segfaults in the
# mss/Xlib native path (verified empirically in this environment). Import the
# libraries up front and hold one mss context for the whole process.
if sys.platform.startswith("linux"):
    for _mod in ("Xlib.display", "mss"):
        try:
            __import__(_mod)
        except ImportError:
            pass
    if os.environ.get("DISPLAY"):
        from . import screen as _screen

        if not _screen.prime():
            print(
                "warning: computer_use could not open the X11 display before "
                "loading torch; screen capture may fail",
                file=sys.stderr,
            )

from .worker import LocateAnythingWorker
from .controller import Agent, Step
from .grounding import parse_points, parse_boxes

__all__ = [
    "LocateAnythingWorker",
    "Agent",
    "Step",
    "parse_points",
    "parse_boxes",
]

__version__ = "0.1.0"