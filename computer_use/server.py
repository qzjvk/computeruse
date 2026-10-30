"""Resident worker server.

Keeps the LocateAnything model loaded in VRAM and executes tasks over a Unix
socket, so each agent run skips the ~25 s model load (which dominates on a
Pascal GPU):

    python -m computer_use serve --scope window     # keep this running once
    python -m computer_use "click the search bar"   # auto-connects when up

Model-level options (--model/--gpu/--dtype/--attn/--in-token-limit/--scope/
--vit-cpu) are fixed by the ``serve`` invocation; per-run options (--dry-run,
--delay, --max-steps, --monitor, --save-dir, --no-box-fallback) are sent with
each task.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import socketserver

SOCKET_PATH = os.environ.get("CUA_SOCKET", "/tmp/computeruse.sock")


def build_worker(args):
    """Build the worker from CLI args (shared by local runs and the server).

    Returns (worker, scope).
    """
    import torch

    from .worker import LocateAnythingWorker

    dtype = None
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    vit_cpu = {"auto": "auto", "on": True, "off": False}[args.vit_cpu]

    # Window grounding wants the highest token budget the card can afford
    # (near-native resolution for typical windows); full-monitor grounding
    # gets the conservative budget.
    if args.image:
        scope, limit = "screen", args.in_token_limit
    elif args.scope == "window":
        scope, limit = "window", args.in_token_limit if args.in_token_limit else 4096
    else:
        scope, limit = "screen", args.in_token_limit

    worker = LocateAnythingWorker(
        model_path=args.model,
        device=args.gpu,
        dtype=dtype,
        attn=args.attn,
        in_token_limit=limit,
        vit_cpu=vit_cpu,
        verbose=not args.quiet,
    )
    return worker, scope


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        buf = io.StringIO()
        resp: dict
        try:
            req = json.loads(self.rfile.readline().decode() or "{}")
            if "task" not in req and "kind" not in req:
                return  # stray/empty connection (e.g. a readiness probe)
            with contextlib.redirect_stdout(buf):
                if req.get("kind") == "offline":
                    from .controller import offline_ground

                    res = offline_ground(
                        self.server.worker, req["image"], req["query"],
                        output_type=req.get("output_type", "point"),
                    )
                    ok = res.center is not None
                else:
                    from .controller import Agent

                    agent = Agent(
                        self.server.worker,
                        dry_run=bool(req.get("dry_run", False)),
                        delay=float(req.get("delay", 0.8)),
                        max_steps=int(req.get("max_steps", 32)),
                        scope=self.server.scope,
                        monitor=req.get("monitor"),
                        save_dir=req.get("save_dir") or None,
                        box_fallback=bool(req.get("box_fallback", True)),
                    )
                    ok = agent.run(req["task"])
            resp = {"ok": ok, "log": buf.getvalue()}
        except Exception as e:  # report back, keep the server alive
            resp = {"ok": False, "error": repr(e), "log": buf.getvalue()}
        self.wfile.write(json.dumps(resp).encode() + b"\n")


class WorkerServer:
    def __init__(self, args):
        self.worker, self.scope = build_worker(args)
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

    def serve_forever(self) -> None:
        if not hasattr(socketserver, "UnixStreamServer"):
            raise RuntimeError("Unix sockets are not supported on this platform")
        srv = socketserver.UnixStreamServer(SOCKET_PATH, _Handler)
        srv.worker = self.worker
        srv.scope = self.scope
        print(
            f"[serve] worker ready on {SOCKET_PATH} "
            f"(scope={self.scope}, in_token_limit={self.worker.in_token_limit}, "
            f"device={self.worker.device}); send tasks with: "
            f'python -m computer_use "task"'
        )
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                os.unlink(SOCKET_PATH)
            except OSError:
                pass


def send_task(req: dict, timeout: float = 1800.0) -> dict | None:
    """Send a task to a running server. Returns None if no server is up."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(SOCKET_PATH)
    except OSError:
        return None
    try:
        s.settimeout(timeout)
        s.sendall(json.dumps(req).encode() + b"\n")
        line = b""
        while not line.endswith(b"\n"):
            chunk = s.recv(1 << 16)
            if not chunk:
                return {"ok": False, "error": "server closed the connection", "log": ""}
            line += chunk
        return json.loads(line.decode())
    except OSError as e:
        return {"ok": False, "error": f"socket error: {e}", "log": ""}
    finally:
        s.close()