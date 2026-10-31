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
import re
import socket
import socketserver
import time
import traceback

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
        quant=args.quant,
        verbose=not args.quiet,
    )
    # Fit the vision budget to the VRAM that is actually free after load.
    # The limit only bounds the prefill (read at ground time), and the
    # prefill's transient logits copy is ~160 MiB per 1024 vision tokens
    # (measured on the 1070). Raising it buys near-native resolution;
    # stepping it down keeps the agent running (instead of OOMing) while
    # another process shares the card.
    if worker.device.type == "cuda" and worker.device.index is not None:
        free, _ = torch.cuda.mem_get_info(worker.device.index)
        need = lambda tok: int(tok / 1024 * 160) * 1024 * 1024
        margin = 300 * 1024 * 1024  # allocator slack + sharing pressure

        limit = worker.in_token_limit
        if scope == "screen":
            # Screen scope: raise as far as the free VRAM allows.
            while limit < 4096 and free >= need(limit + 1024) + margin:
                limit += 1024
        while limit > 1024 and free < need(limit) + margin:
            limit -= 1024
        if limit != worker.in_token_limit:
            worker.in_token_limit = limit
            try:
                worker.processor.image_processor.in_token_limit = limit
            except AttributeError:
                pass
            print(f"[serve] in_token_limit={limit} "
                  f"({free / 1e9:.2f}G free after load, scope={scope})", flush=True)
    return worker, scope


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        buf = io.StringIO()
        req: dict = {}
        resp: dict
        t0 = time.time()
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
                elif req.get("kind") == "orchestrate":
                    # LLM planning over the same worker: the planner
                    # (LFM3, its own GPU share) picks the next step;
                    # this worker does all the pixel-level work.
                    from .orchestrator import OrchestratedAgent, Planner

                    planner = Planner()
                    planner.ensure_server()
                    ok = OrchestratedAgent(
                        self.server.worker, planner,
                        dry_run=bool(req.get("dry_run", False)),
                        delay=float(req.get("delay", 0.8)),
                        scope=req.get("scope") or self.server.scope,
                        monitor=req.get("monitor"),
                        save_dir=req.get("save_dir") or None,
                        max_steps=int(req.get("max_steps", 32)),
                        max_seconds=float(req.get("max_seconds", 180.0)),
                        box_fallback=bool(req.get("box_fallback", True)),
                    ).run(req["goal"])
                else:
                    from .controller import Agent

                    agent = Agent(
                        self.server.worker,
                        dry_run=bool(req.get("dry_run", False)),
                        delay=float(req.get("delay", 0.8)),
                        max_steps=int(req.get("max_steps", 32)),
                        # Per-run scope override; the serve invocation sets
                        # the default. (The token budget is fixed by the
                        # worker, but both scopes share one budget here.)
                        scope=req.get("scope") or self.server.scope,
                        monitor=req.get("monitor"),
                        save_dir=req.get("save_dir") or None,
                        box_fallback=bool(req.get("box_fallback", True)),
                    )
                    ok = agent.run(req["task"])
            resp = {"ok": ok, "log": buf.getvalue()}
        except Exception as e:  # report back, keep the server alive
            tb = traceback.format_exc()
            # Full traceback on the serve stdout (audit log) and in the
            # client response — repr alone lost the failing line.
            print(f"[serve] task failed:\n{tb}", flush=True)
            resp = {"ok": False, "error": repr(e),
                    "traceback": tb, "log": buf.getvalue()}
        # Task history on the serve process's stdout, so
        # `docker logs cuasandbox` doubles as a task audit log.
        try:
            if "task" in req:
                label = f"task={req['task']!r}"
            elif req.get("kind") == "orchestrate":
                label = f"orchestrate goal={req.get('goal', '')!r}"
            elif "kind" in req:
                label = f"offline query={req.get('query', '')!r}"
            else:
                label = "stray connection"
            m = re.search(r"target\s*:\s*(.+)", resp.get("log", ""))
            target = m.group(1).strip() if m else "-"
            print(f"[serve] {time.strftime('%Y-%m-%d %H:%M:%S')} {label} "
                  f"ok={resp.get('ok')} elapsed={time.time() - t0:.1f}s target={target}",
                  flush=True)
        except Exception:
            pass
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


def send_task_wait(req: dict, wait: float = 0.0,
                   timeout: float = 1800.0) -> dict | None:
    """send_task, but while the worker is mid-restart the socket is
    briefly missing: wait for it instead of giving up (or, without the
    wait budget, loading a second model copy on the same GPU and OOMing
    the card)."""
    deadline = time.time() + max(0.0, wait)
    while True:
        resp = send_task(req, timeout=timeout)
        if resp is not None or time.time() >= deadline:
            return resp
        print("[client] worker not ready yet (restarting?); waiting...",
              flush=True)
        time.sleep(2)