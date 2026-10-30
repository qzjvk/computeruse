"""computer_use: a computer-use agent powered by NVIDIA LocateAnything-3B.

LocateAnything is a vision-language grounding model (Qwen2.5-3B + MoonViT) that
localizes GUI elements / objects from natural language. This package turns it
into a computer-use agent: screenshot -> ground -> click/type/key.

The agent is tuned to run on legacy NVIDIA GPUs (e.g. a GTX 1070, sm_61) using
float16 weights, the SDPA math attention backend, and a CPU-offloaded vision
encoder.
"""

# On Linux the X11 connection must be opened BEFORE torch's CUDA driver is
# loaded: creating a fresh Xlib connection after CUDA init segfaults in the
# mss/Xlib native path (verified empirically in this environment). Only
# processes that load CUDA are concerned: the worker primes the mss/Xlib
# handles at the start of its init, before its first CUDA call. The task
# client talks to the worker over a socket and never touches X or CUDA, so
# importing the package must stay cheap (~10 ms, not ~2.5 s of torch +
# X-connection).

# Heavy re-exports (worker -> torch, controller -> worker) are resolved
# lazily (PEP 562) so a socket client does not import torch at all.
_LAZY = {
    "LocateAnythingWorker": ".worker",
    "Agent": ".controller",
    "Step": ".controller",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    if name in ("parse_points", "parse_boxes"):
        from . import grounding

        return getattr(grounding, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LocateAnythingWorker",
    "Agent",
    "Step",
    "parse_points",
    "parse_boxes",
]

__version__ = "0.1.0"