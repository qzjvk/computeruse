"""CLI entry point: ``python -m computer_use ...``

Examples:
    # Offline grounding on an image (no screen control)
    python -m computer_use --image assets/demo.png --query "the search box"

    # Dry run of a task (grounds + prints targets, no mouse movement)
    python -m computer_use --dry-run "click the start menu; type firefox; key enter"

    # Real run
    python -m computer_use "click File; click New; type hello"

    # Resident worker: load the model once, then every run skips the ~25 s load
    python -m computer_use serve --scope window
    python -m computer_use "click File; click New"   # auto-connects when up

    # LLM-orchestrated: a goal instead of a step script (LFM3 planner,
    # see computer_use/orchestrator.py); the resident worker must be up
    python -m computer_use orchestrate "search nvidia container cuda and open the first result"
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="computer_use",
        description="Computer-use agent powered by NVIDIA LocateAnything-3B.",
    )
    p.add_argument("task", nargs="*", help="Task to execute (see README for the step language)")
    p.add_argument("--task-file", help="Read the task from a file (overrides positional task)")
    p.add_argument("--image", help="Offline grounding mode: path to an image")
    p.add_argument("--query", help="Phrase to ground (offline mode)")
    p.add_argument("--output-type", choices=["point", "box"], default="point",
                   help="Offline mode: ground as point (default) or box")

    p.add_argument("--model", default=os.environ.get("LOCATEANYTHING_MODEL", "locatenv"),
                   help="Path to the model directory (default: $LOCATEANYTHING_MODEL or ./locatenv)")
    p.add_argument("--gpu", help="CUDA device, e.g. '0' or 'cuda:2'. Default: most free VRAM")
    p.add_argument("--dtype", choices=["fp16", "bf16"], help="Weight dtype (default: bf16 on sm_80+, fp16 elsewhere)")
    p.add_argument("--attn", choices=["sdpa", "eager", "flash_attention_2", "magi"],
                   help="Attention implementation (default: sdpa)")
    p.add_argument("--in-token-limit", type=int, default=None,
                   help="Max vision patch tokens (default auto: 1024 on legacy GPUs, 2048 elsewhere; raise on big-VRAM GPUs, e.g. 4096/25600)")
    p.add_argument("--quant", choices=["off", "nf4"], default="off",
               help="Weight-only quantization of the LLM (nf4 = 4-bit; "
                    "~4x less VRAM and ~4x less weight traffic per pass)")
    p.add_argument("--vit-cpu", choices=["auto", "on", "off"], default="auto",
                   help="Run the MoonViT vision encoder on the CPU (default auto: on for sm<8 / 8 GB cards)")
    p.add_argument("--generation-mode", choices=["hybrid", "fast", "slow"], default="hybrid")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy (recommended for the agent)")
    p.add_argument("--verbose", action="store_true", help="Print per-step model statistics")

    p.add_argument("--dry-run", action="store_true", help="Ground and log actions but do not move the mouse")
    p.add_argument("--scope", choices=["window", "screen"], default="window",
                   help="Capture scope: the active window (default, near-native resolution) or the full monitor")
    p.add_argument("--delay", type=float, default=0.2,
                   help="Pause after each grounded action (seconds); "
                        "0.2 lets a click settle. A task's own 'wait N' "
                        "(or 'wait-load') covers longer loads.")
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--max-seconds", type=float, default=180.0,
                   help="Wall-clock budget for an orchestrate run "
                        "(default 180 s; research goals need more, e.g. "
                        "900)")
    p.add_argument("--planner-ctx", type=int, default=None,
                   help="LFM3 planner context window for this run (e.g. "
                        "8192 for research goals whose history carries "
                        "read dumps; default 4096)")
    p.add_argument("--monitor", type=int, help="Monitor index for screenshots (default: mouse position)")
    p.add_argument("--save-dir", default="runs", help="Where to save step screenshots ('' to disable)")
    p.add_argument("--no-box-fallback", action="store_true",
                   help="Do not retry with box grounding when point grounding returns nothing")
    p.add_argument("--no-server", action="store_true",
                   help="Load the model locally even if a resident 'serve' worker is running")
    p.add_argument("--quiet", action="store_true", help="Suppress worker banners/progress")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Resident worker: keep the model loaded and serve tasks over a socket.
    # Model-level options (--model/--gpu/--dtype/--attn/--scope/...) are fixed
    # by this invocation; per-run options (--dry-run/--delay/...) come with
    # each task.
    if args.task == ["serve"]:
        from . import server

        server.WorkerServer(args).serve_forever()
        return 0

    # LLM-orchestrated mode: a goal instead of a step script. The planner
    # (LFM3, LFM3_* env) picks the next step in the controller vocabulary;
    # the resident worker does the pixel-level work (see orchestrator.py).
    if args.task and args.task[0] == "orchestrate":
        if args.task_file:
            goal = open(args.task_file).read().strip()
        elif len(args.task) > 1:
            goal = " ".join(args.task[1:])
        else:
            print('orchestrate needs a goal, e.g.:\n'
                  '  python -m computer_use orchestrate "open a terminal"')
            return 2
        req = {
            "kind": "orchestrate", "goal": goal,
            "scope": args.scope,
            "dry_run": args.dry_run, "delay": args.delay,
            "max_steps": args.max_steps, "monitor": args.monitor,
            "max_seconds": args.max_seconds,
            "planner_ctx": args.planner_ctx,
            "save_dir": args.save_dir or None,
            "box_fallback": not args.no_box_fallback,
        }
        if not args.no_server:
            from . import server

            wait = max(0, int(os.environ.get("CUA_WAIT_SERVER", "0") or 0))
            resp = server.send_task_wait(req, wait=wait)
            if resp is None:
                print("[client] no resident worker on the socket and "
                      "--no-server not set; start one with:\n"
                      '  python -m computer_use serve --scope window')
                return 1
            print(resp.get("log", ""), end="")
            if "error" in resp:
                print(resp["error"])
                if resp.get("traceback"):
                    print(resp["traceback"])
                return 1
            return 0 if resp["ok"] else 1

        from . import server
        from .orchestrator import OrchestratedAgent, Planner, set_planner_ctx

        worker, scope = server.build_worker(args)
        planner = Planner()
        if args.planner_ctx:
            set_planner_ctx(args.planner_ctx)
        planner.ensure_server()
        ok = OrchestratedAgent(
            worker, planner, dry_run=args.dry_run, delay=args.delay,
            scope=scope, monitor=args.monitor,
            save_dir=args.save_dir or None,
            max_steps=args.max_steps,
            max_seconds=args.max_seconds,
            box_fallback=not args.no_box_fallback,
        ).run(goal)
        return 0 if ok else 1

    task = None
    if args.task_file:
        task = open(args.task_file).read()
    elif args.task:
        task = " ".join(args.task)
    elif args.image and args.query:
        task = None  # offline grounding mode
    if not (task or (args.image and args.query)):
        print(__doc__)
        return 2

    from . import server

    # Offline grounding mode
    if args.image and args.query:
        if not args.no_server:
            resp = server.send_task({
                "kind": "offline", "image": args.image,
                "query": args.query, "output_type": args.output_type,
            })
            if resp is not None:
                print(resp.get("log", ""), end="")
                if "error" in resp:
                    print(resp["error"])
                    if resp.get("traceback"):
                        print(resp["traceback"])
                    return 1
                return 0 if resp["ok"] else 1

        worker, _scope = server.build_worker(args)
        from .controller import offline_ground

        res = offline_ground(
            worker, args.image, args.query,
            output_type=args.output_type,
            save_dir=args.save_dir or None,
        )
        return 0 if res.center else 1

    # Agent mode: use the resident worker when one is up (skips the ~25 s load)
    if not args.no_server:
        # CUA_WAIT_SERVER (seconds): while the worker is mid-restart the
        # socket is briefly missing; wait for it instead of loading a second
        # model copy on the same GPU (which would OOM the card).
        wait = max(0, int(os.environ.get("CUA_WAIT_SERVER", "0") or 0))
        req = {
            "kind": "task", "task": task,
            "scope": args.scope,
            "dry_run": args.dry_run, "delay": args.delay,
            "max_steps": args.max_steps, "monitor": args.monitor,
            "save_dir": args.save_dir or None,
            "box_fallback": not args.no_box_fallback,
        }
        resp = server.send_task_wait(req, wait=wait)
        if resp is not None:
            print(resp.get("log", ""), end="")
            if "error" in resp:
                print(resp["error"])
                if resp.get("traceback"):
                    print(resp["traceback"])
                return 1
            return 0 if resp["ok"] else 1

    worker, scope = server.build_worker(args)
    from .controller import Agent

    agent = Agent(
        worker,
        dry_run=args.dry_run,
        delay=args.delay,
        max_steps=args.max_steps,
        scope=scope,
        monitor=args.monitor,
        save_dir=args.save_dir or None,
        box_fallback=not args.no_box_fallback,
    )
    ok = agent.run(task)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())