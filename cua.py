#!/usr/bin/env python3
"""cua — take over the computer with one command, get a written report back.

Usage:
  cua "goal to accomplish" [options]    run the agent on the HOST desktop
  cua --sandbox "goal" [options]        run it in the cuasandbox container
  cua show [run-id]                     re-print a saved report
  cua list                              list past runs

Every run writes into cua_runs/<run-id>/:
  report.md     the result: status, step count, the done summary (the
                answer, if any), the step table
  report.json   the same, machine-readable
  run.log       the full transcript
  stepNN-*.png  per-step screenshots (evidence)

Safety: host mode on a live desktop asks for an explicit ack before the
first move (skip with --yes; --dry-run never touches the screen). Abort
at any time: Ctrl-C, or slam the mouse into the top-left corner of the
screen (pyautogui failsafe — the agent dies, the report is still written
from whatever ran).

Stdlib only; the agent itself runs through .venv.
"""

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
VENV_PY = REPO / ".venv" / "bin" / "python"
RUNS = REPO / "cua_runs"
LFM3_NAME = "LFM3-1.7B-Q4_K_M.gguf"
# Free VRAM the 1070 must have before a host run: the NF4 LocateAnything
# worker (~3.3 GB) plus the planner's own spawn check (>= 2.5 GB free,
# orchestrator.ensure_server). Both load on the same shared card.
MIN_FREE_MIB = 6144

USAGE = __doc__


def die(msg: str, code: int = 2) -> None:
    print(f"cua: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------- parsing

def _free_mib_1070() -> int | None:
    """Free VRAM on the GTX 1070, or None when unreadable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            name, _, free = line.partition(",")
            if "1070" in name:
                return int(free.strip().split()[0])
    except Exception:
        pass
    return None


def _lfm3_bin() -> Path | None:
    env = os.environ.get("LFM3_BIN")
    if env:
        return Path(env)
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    sibling = REPO.parent / "llama.cpp" / "build" / "bin" / "llama-server"
    return sibling if sibling.exists() else None


def _lfm3_model() -> Path | None:
    env = os.environ.get("LFM3_MODEL")
    if env and os.path.exists(env):
        return Path(env)
    home = Path.home() / "models" / LFM3_NAME
    return home if home.exists() else None


def _sandbox_up() -> bool:
    try:
        out = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=cuasandbox"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return bool(out)
    except Exception:
        return False


def _provision_host_lfm3(bin_path: Path | None, model: Path) -> None:
    """First-run setup on the host: the planner's llama-server binary and
    GGUF must exist. The GGUF is the same file the sandbox already runs
    with — copy it out of the container (1.05 GB) instead of downloading."""
    if bin_path is None or not bin_path.exists():
        die("llama-server binary not found "
            f"({bin_path or 'LFM3_BIN / PATH'}) — set LFM3_BIN or put "
            f"llama-server on the PATH")
    if model.exists():
        return
    if not _sandbox_up():
        die(f"planner model missing: {model} — start the cuasandbox "
            f"container (cua can copy the GGUF out of it) or download it: "
            f"huggingface-cli download LiquidAI/LFM3-1.7B "
            f"--local-dir {Path.home() / 'models'} "
            f"--include {LFM3_NAME}")
    print(f"cua: first run — copying the LFM3 planner weights out of the "
          f"cuasandbox container (1.05 GB) to {model} …", flush=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        ["docker", "cp", f"cuasandbox:/root/models/{LFM3_NAME}",
         str(model)]).returncode
    if rc != 0 or not model.exists():
        die(f"could not copy the planner model out of the container "
            f"(docker cp rc={rc})")


# ---------------------------------------------------------------- report

_STEP_RE = re.compile(
    r"^\s+step (\d+): ([a-z-]+)\((.*)\) -> (ok|FAILED) in ([\d.]+)s;"
    r" frame: (.*?)% changed; title: '((?:[^'\\]|\\.)*)'"
    r"(?: \(([^)]*)\))?(?: \| (.+))?$")
_DONE_RE = re.compile(
    r"\[orch\] done: (.*) \(([\d.]+) s total, (\d+) steps\)\s*$")
_GOAL_RE = re.compile(
    r"\[orch\] goal: (.+?) \((\d+) step cap, (\d+) s cap\)$", re.M)


def build_report(log_text: str, save_dir: Path, goal: str, rc: int,
                 mode: str, opts: dict) -> dict:
    """Parse the orchestrator's transcript into the run report.

    The transcript grammar is owned by orchestrator.py (all flush=True):
      [orch] goal: {goal!r} ({n} step cap, {s} s cap)
      [orch] subgoals:\\n  1. …
      [orch] step N: {name} {json}            (planner turn)
      [orch] step N: protocol: …              (fast path)
          step N: {name}({json}) -> ok|FAILED in Xs; …; title: '…'
      [orch] done: {summary} ({t} s total, {n} steps)
      [orch] budget of N s exceeded; stopping at step N
      [orch] N consecutive failures; stopping
      [orch] max_steps=N reached without done
      [orch] planner error at step N: {e}
    """
    lines = log_text.splitlines()
    rep = {
        "run_id": str(save_dir.name),
        "started": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "goal": goal,
        "caps": {"max_steps": opts.get("max_steps"),
                 "max_seconds": opts.get("max_seconds"),
                 "planner_ctx": opts.get("planner_ctx")},
    }

    m = _GOAL_RE.search(log_text)
    if m:
        rep["caps"]["step_cap"] = int(m.group(2))
        rep["caps"]["second_cap"] = int(m.group(3))

    subs: list[str] = []
    in_subs = False
    for line in lines:
        if line.startswith("[orch] subgoals:"):
            in_subs = True
            continue
        if in_subs:
            mm = re.match(r"\s{2}(\d+)\. (.+)$", line)
            if mm:
                subs.append(mm.group(2))
            elif line.strip():
                in_subs = False
    rep["subgoals"] = subs

    steps = []
    for line in lines:
        mm = _STEP_RE.match(line)
        if not mm:
            continue
        n, name, args, ok, dt, frame, title, _unch, note = mm.groups()
        try:
            parsed = json.loads(args)
        except Exception:
            parsed = args
        steps.append({
            "n": int(n), "step": name, "args": parsed,
            "ok": ok == "ok", "seconds": float(dt),
            "title": title,
            "note": note if (note and ok == "FAILED") else None,
        })
    rep["steps"] = steps
    rep["step_count"] = len(steps)
    rep["failed"] = [s for s in steps if not s["ok"]]

    done = None
    for line in lines:
        mm = _DONE_RE.search(line)
        if mm:
            done = mm.group(1)
            rep["duration_s"] = float(mm.group(2))
    rep["done"] = done
    rep["mechanical_done"] = (
        "planner ignored the done hint twice" in log_text)

    stop = None
    for line in lines:
        if "budget of" in line and "exceeded" in line:
            stop = "time budget exceeded"
        elif "consecutive failures" in line and "stopping" in line:
            stop = "3 consecutive step failures"
        elif "reached without done" in line:
            cap = (rep["caps"].get("step_cap")
                   or rep["caps"].get("max_steps") or "?")
            stop = f"step cap ({cap}) reached"
        elif "planner error" in line:
            stop = f"planner error: {line.split(':', 2)[-1].strip()}"
    rep["stop_reason"] = stop

    if opts.get("dry_run"):
        rep["status"] = "dry-run: first move shown, nothing touched"
    elif done and rc == 0:
        rep["status"] = "done"
    elif rc == 130:
        rep["status"] = "interrupted (Ctrl-C)"
    elif stop:
        rep["status"] = f"incomplete: {stop}"
    elif rc == 0:
        rep["status"] = "done (client reported ok)"
    else:
        rep["status"] = f"incomplete: exit {rc}"

    rep["last_title"] = steps[-1]["title"] if steps else None

    pngs = sorted(save_dir.glob("step*.png"))
    rep["evidence"] = [p.name for p in pngs]
    return rep


def render_markdown(rep: dict) -> str:
    out = [f"# cua report — {rep['run_id']}", ""]
    out.append(f"- **goal:** {rep['goal']!r}")
    out.append(f"- **status:** {rep['status']}")
    out.append(f"- **mode:** {rep['mode']}")
    caps = rep.get("caps", {})
    step_cap = caps.get("step_cap") or caps.get("max_steps") or "?"
    second_cap = caps.get("second_cap") or caps.get("max_seconds") or "?"
    cap_bits = [f"{step_cap}-step cap", f"{second_cap} s cap"]
    if caps.get("planner_ctx"):
        cap_bits.append(f"planner ctx {caps['planner_ctx']}")
    out.append(f"- **caps:** {', '.join(cap_bits)}")
    if rep.get("done"):
        how = ("mechanical (the planner ignored the done hint twice — "
               "the protocol executed it)"
               if rep.get("mechanical_done") else "planner's")
        out.append(f"- **result ({how} done summary):**")
        out.append("")
        for line in rep["done"].splitlines() or [rep["done"]]:
            out.append(f"  > {line}" if line.strip() else "  >")
        out.append("")
    elif rep.get("stop_reason"):
        out.append(f"- **result:** none — {rep['stop_reason']}")
    if rep.get("step_count"):
        ok = rep["step_count"] - len(rep.get("failed", []))
        dur = rep.get("duration_s")
        dur_s = f"{dur:.1f} s" if dur is not None else "?"
        out.append(f"- **steps:** {rep['step_count']} executed "
                   f"({ok} ok, {len(rep.get('failed', []))} failed), "
                   f"{dur_s}")
    if rep.get("last_title"):
        out.append(f"- **last window title:** {rep['last_title']!r}")
    if rep.get("evidence"):
        out.append(f"- **evidence:** {len(rep['evidence'])} screenshots "
                   f"in this directory")
    out.append("- **log:** run.log")
    if rep.get("subgoals"):
        out += ["", "## Subgoals (decomposed)"]
        for i, s in enumerate(rep["subgoals"], 1):
            out.append(f"{i}. {s}")
    if rep.get("steps"):
        out += ["", "## Steps", "",
                "| # | step | result | s | title |",
                "|---|------|--------|---|-------|"]
        for s in rep["steps"]:
            args = json.dumps(s["args"], ensure_ascii=False)
            if len(args) > 60:
                args = args[:57] + "…"
            title = s["title"][:70]
            if s.get("note"):
                title += f" — {s['note'][:60]}"
            out.append(f"| {s['n']} | `{s['step']}` {args} "
                       f"| {'ok' if s['ok'] else 'FAILED'} "
                       f"| {s['seconds']:.1f} | {title} |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- runs

def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _check_host(dry_run: bool, force: bool) -> None:
    if not VENV_PY.exists():
        die(f"{VENV_PY} not found — set up the venv first "
            f"(README, Setup)")
    if not dry_run and not force:
        free = _free_mib_1070()
        if free is not None and free < MIN_FREE_MIB:
            hint = ""
            if _sandbox_up():
                hint = (" The cuasandbox worker is resident on the same "
                        "1070 — run `cua --sandbox \"…\"` instead, or free "
                        "VRAM first.")
            die(f"the 1070 has {free} MiB free; a host run needs ~"
                f"{MIN_FREE_MIB // 1024} GiB (NF4 worker + planner).{hint} "
                f"Use --force to override.")


def _banner(goal: str, mode: str, run_id: str, extra: str) -> None:
    w = 60
    print("═" * w)
    print(f" CUA TAKEOVER — {mode}")
    print("═" * w)
    print(f" goal:    {goal!r}")
    if mode == "host":
        print(f" display: {os.environ.get('DISPLAY', '?')} "
              f"(your live desktop)")
    else:
        print(f" target:  cuasandbox container (your desktop is untouched)")
    print(" worker:  " + extra)
    print(" abort:   Ctrl-C, or slam the mouse into the TOP-LEFT CORNER")
    print("           of the screen (pyautogui failsafe)")
    print(f" report:  cua_runs/{run_id}/ "
          f"(report.md, report.json, run.log, evidence)")
    print("─" * w)


def _ack(goal: str, dry_run: bool, yes: bool, mode: str) -> None:
    if dry_run or yes:
        return
    if not sys.stdin.isatty():
        die("no TTY for the takeover ack — re-run with --yes "
            "(explicit) or --dry-run (no screen changes)")
    print("This will move your mouse and type in "
          + ("the cuasandbox" if mode == "sandbox" else "your session")
          + ".")
    input("Press Enter to start (Ctrl-C to cancel): ")


def _tee(proc, log_path: Path) -> int:
    """Stream the child's output to the terminal AND the run log.

    Returns the child's exit code; a Ctrl-C here (SIGINT to the process
    group already hit the child) still yields the partial log."""
    rc = 0
    try:
        with log_path.open("w") as f:
            for raw in proc.stdout:
                sys.stdout.write(raw)
                sys.stdout.flush()
                f.write(raw)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            rc = proc.wait(timeout=10)
        except Exception:
            proc.kill()
            rc = 130
    return rc if rc is not None else 130


def _run_host(goal: str, run_id: str, opts: dict) -> int:
    _check_host(opts["dry_run"], opts["force"])
    bin_path = _lfm3_bin()
    model = _lfm3_model()
    if model is None:
        # First run: copies the GGUF out of the sandbox; dies with a
        # clear message when the binary is missing or the sandbox is
        # down.
        _provision_host_lfm3(bin_path, model)
        model = _lfm3_model()

    use_socket = os.path.exists(
        os.environ.get("CUA_SOCKET", "/tmp/computeruse.sock"))
    cmd = [str(VENV_PY), "-m", "computer_use"]
    if use_socket:
        extra = "resident host worker on the socket (transcript prints " \
                "when the run completes)"
        # The serve worker's model flags are fixed at its start; only the
        # per-run options travel with the task.
    else:
        extra = "fresh load in this process (NF4, ~25 s, live transcript)"
        cmd += ["--no-server", "--quant", "nf4", "--scope", "window"]
    cmd += ["--save-dir", str(RUNS / run_id)]
    if opts["max_steps"]:
        cmd += ["--max-steps", str(opts["max_steps"])]
    if opts["max_seconds"]:
        cmd += ["--max-seconds", str(opts["max_seconds"])]
    if opts["planner_ctx"]:
        cmd += ["--planner-ctx", str(opts["planner_ctx"])]
    if opts["dry_run"]:
        cmd.append("--dry-run")
    cmd += ["orchestrate", goal]

    _banner(goal, "host", run_id, extra)
    _ack(goal, opts["dry_run"], opts["yes"], "host")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if bin_path is not None:
        env["LFM3_BIN"] = str(bin_path)
    if model is not None:
        env["LFM3_MODEL"] = str(model)
    if opts["keep_warm"]:
        env["LFM3_PERSIST"] = "1"

    save_dir = RUNS / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            env=env, cwd=str(REPO),
                            text=True, bufsize=1)
    rc = _tee(proc, save_dir / "run.log")
    return _finish(rc, save_dir, goal, "host", opts)


# The sandbox's verified reset (eval/eval_one.sh): close the terminal
# (retry), put Firefox at about:blank (the location bar is FOCUSED from
# the a11y tree — grab_focus — not clicked; its node name varies per
# search engine; fallback ctrl+l; verify+retry). A run expects a clean
# desktop: a leftover unsubmitted line in the terminal prompt appends
# to the goal's first command and corrupts it (measured: a leftover
# line from an earlier run broke T9's first echo — the file came out
# 16/17 lines).
RESET_SCRIPT = r"""
export DISPLAY=:99
rm -f /root/note.txt /root/note2.txt /root/multi.txt /root/pool.conf
ok_reset=0
for i in 1 2 3; do
  xdotool search --name "Terminal" windowclose 2>/dev/null; sleep 0.5
  [ -z "$(xdotool search --name "Terminal" 2>/dev/null)" ] && { ok_reset=1; break; }
done
ok_nav=0
for i in 1 2 3; do
  W=$(xdotool search --name "Mozilla Firefox" | head -1)
  xdotool windowactivate --sync "$W" 2>/dev/null; sleep 0.6
  [ "$(xdotool getactivewindow 2>/dev/null)" != "$W" ] \
    && { xdotool windowactivate --sync "$W" 2>/dev/null; sleep 1; }
  if ! timeout 20 /usr/bin/python3 -c "
import gi; gi.require_version('Atspi', '2.0')
from gi.repository import Atspi
Atspi.init()
d = Atspi.get_desktop(0)
found = []
def walk(a):
    if a is None or found: return
    try:
        name = a.get_name() or ''
        role = a.get_role_name() or ''
    except Exception:
        return
    if role in ('combo box', 'entry', 'text') and (
            'address' in name.lower() or 'enter' in name.lower()
            or 'search with' in name.lower()
            or 'location' in name.lower()):
        found.append(a); return
    try:
        n = a.get_child_count()
    except Exception:
        n = 0
    for i in range(min(n, 400)):
        try:
            walk(a.get_child_at_index(i))
        except Exception:
            pass
for i in range(d.get_child_count()):
    a = d.get_child_at_index(i)
    if a and 'firefox' in (a.get_name() or '').lower():
        walk(a)
if found and found[0].grab_focus():
    print('focus-ok')
" 2>/dev/null | grep -q focus-ok; then
    xdotool key ctrl+l 2>/dev/null; sleep 0.4
  fi
  xdotool key ctrl+a 2>/dev/null; sleep 0.2
  xdotool type --clearmodifiers "about:blank" 2>/dev/null; sleep 0.3
  xdotool key Return 2>/dev/null; sleep 1.2
  T=$(xdotool getwindowname "$W" 2>/dev/null)
  [ "$T" = "Mozilla Firefox" ] && { ok_nav=1; break; }
done
if [ $ok_nav = 1 ] && [ $ok_reset = 1 ]; then echo reset-ok; else echo reset-fail; fi
"""


def _run_sandbox(goal: str, run_id: str, opts: dict) -> int:
    if not _sandbox_up():
        die("the cuasandbox container is not running "
            "(`./sandbox.sh up`) — or drop --sandbox for host mode")
    _banner(goal, "sandbox", run_id,
            "resident sandbox worker (the transcript prints when the run "
            "completes — the sandbox client waits on the socket)")
    _ack(goal, opts["dry_run"], opts["yes"], "sandbox")
    if not opts["dry_run"]:
        print("cua: resetting the sandbox desktop "
              "(close terminal, Firefox → about:blank) …", flush=True)
        out = subprocess.run(
            ["docker", "exec", "-w", "/app", "cuasandbox", "bash",
             "-lc", RESET_SCRIPT],
            capture_output=True, text=True).stdout.strip()
        if "reset-ok" not in out:
            die(f"sandbox reset failed ({out or 'no output'}) — a run "
                f"needs a clean desktop; re-run cua --sandbox")
        print("cua: reset ok", flush=True)
    py = [f"python -m computer_use --save-dir /tmp/cua_runs/{run_id}"]
    if opts["max_steps"]:
        py.append(f"--max-steps {opts['max_steps']}")
    if opts["max_seconds"]:
        py.append(f"--max-seconds {opts['max_seconds']}")
    if opts["planner_ctx"]:
        py.append(f"--planner-ctx {opts['planner_ctx']}")
    if opts["dry_run"]:
        py.append("--dry-run")
    py.append(f"orchestrate {shlex.quote(goal)}")
    script = ("export DISPLAY=:99 CUA_WAIT_SERVER=60 PYTHONUNBUFFERED=1; "
              + " ".join(py))
    cmd = ["docker", "exec", "-w", "/app", "cuasandbox",
           "bash", "-lc", script]

    save_dir = RUNS / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    rc = _tee(proc, save_dir / "run.log")
    # Pull evidence + the authoritative log out of the container.
    subprocess.run(["docker", "cp",
                    f"cuasandbox:/tmp/cua_runs/{run_id}/.",
                    str(save_dir)],
                   capture_output=True)
    return _finish(rc, save_dir, goal, "sandbox", opts)


def _finish(rc: int, save_dir: Path, goal: str, mode: str,
            opts: dict) -> int:
    log_path = save_dir / "run.log"
    log_text = log_path.read_text(errors="replace") \
        if log_path.exists() else ""
    rep = build_report(log_text, save_dir, goal, rc, mode, opts)
    (save_dir / "report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False))
    md = render_markdown(rep)
    (save_dir / "report.md").write_text(md)
    print()
    print(md)
    return 0 if rep["status"].startswith(("done", "dry-run")) else 1


# ---------------------------------------------------------------- commands

def cmd_list() -> int:
    if not RUNS.exists():
        print(f"cua: no runs yet ({RUNS} is empty)")
        return 0
    rows = []
    for d in sorted(RUNS.iterdir()):
        j = d / "report.json"
        if not j.exists():
            continue
        try:
            rep = json.loads(j.read_text())
        except Exception:
            continue
        rows.append((d.name, rep.get("status", "?"),
                     (rep.get("goal") or "?")[:60]))
    if not rows:
        print("cua: no completed runs yet")
        return 0
    print(f"{'run-id':16}  {'status':34}  goal")
    for rid, status, goal in rows:
        print(f"{rid:16}  {status:34}  {goal}")
    print(f"\ncua: re-print one with: cua show <run-id>")
    return 0


def cmd_show(run_id: str | None) -> int:
    if not RUNS.exists():
        die("no runs yet")
    if run_id:
        d = RUNS / run_id
    else:
        ids = [p.name for p in RUNS.iterdir() if (p / "report.md").exists()]
        if not ids:
            die("no completed runs yet")
        d = RUNS / sorted(ids)[-1]
    md = d / "report.md"
    if not md.exists():
        die(f"no report for {d.name} (still running?)")
    print(md.read_text())
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] == "list":
        return cmd_list()
    if argv[0] == "show":
        return cmd_show(argv[1] if len(argv) > 1 else None)

    opts = {
        "sandbox": False, "dry_run": False, "yes": False, "force": False,
        "keep_warm": False,
        "max_steps": None, "max_seconds": None, "planner_ctx": None,
    }
    pos: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--sandbox",):
            opts["sandbox"] = True
        elif a in ("--dry-run",):
            opts["dry_run"] = True
        elif a in ("--yes",):
            opts["yes"] = True
        elif a in ("--force",):
            opts["force"] = True
        elif a in ("--keep-warm",):
            opts["keep_warm"] = True
        elif a == "--max-steps":
            i += 1
            opts["max_steps"] = int(argv[i])
        elif a == "--max-seconds":
            i += 1
            opts["max_seconds"] = float(argv[i])
        elif a == "--planner-ctx":
            i += 1
            opts["planner_ctx"] = int(argv[i])
        elif a.startswith("-"):
            die(f"unknown option {a!r} (see cua --help)")
        else:
            pos.append(a)
        i += 1

    goal = " ".join(pos).strip()
    if not goal:
        die('give a goal: cua "open firefox and search nvidia cuda docs" '
            '(see cua --help)')

    run_id = _run_id()
    if opts["sandbox"]:
        return _run_sandbox(goal, run_id, opts)
    return _run_host(goal, run_id, opts)


if __name__ == "__main__":
    sys.exit(main())