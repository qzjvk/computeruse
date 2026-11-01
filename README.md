# computer-use

A **computer-use agent** built on NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
grounding model, tuned to run on a **GTX 1070** (Pascal, sm_61, 8 GB).

The perception model turns a screenshot + a phrase into exact coordinates
(`Point to: the search box.` → `<box><x><y></box>` in normalized [0,1000] space).
A small deterministic task interpreter converts your task into
screenshot → ground → click/type/key steps:

```
┌────────────┐   ┌──────────────────────┐   ┌────────────┐   ┌─────────────┐
│   task     │ → │  LocateAnything-3B   │ → │  parse pt  │ → │  pyautogui  │
│  (steps)   │   │  GUI grounding (GPU) │   │  (0–1000)  │   │  click/type │
└────────────┘   └──────────────────────┘   └────────────┘   └─────────────┘
        ▲                                                        │
        │                    screenshot (mss) ◄──────────────────┘
```

## What the model does

LocateAnything-3B (Qwen2.5-3B + MoonViT) grounds GUI elements and objects from
natural language, using parallel box decoding. Supported tasks: GUI grounding
(points *and* boxes), open-set detection, phrase grounding, text/layout
grounding, pointing. This agent uses the **point** mode as the primary click
source and falls back to **box** mode (clicking the box center).

## How grounding works: scope and tiling

Accuracy on an 8 GB card comes down to how many pixels the model can actually
see, so the agent manages its capture:

* **`--scope window` (default)** — grabs the *focused* window's rectangle
  (via `xdotool`) instead of the full monitor, so the same token budget buys
  near-native resolution of the UI being operated on.
* **`--scope screen`** — full monitor (the one under the mouse).
* **Capture strategy** — the capture is compared to the token budget and
  one of three passes is used:
  1. *fits* → one pass at native resolution.
  2. *slight overrun* (≤ 1.5× budget) → the **whole** capture is shrunk to
     fit and grounded in a single pass. Full context stays in view (tab
     bars, sidebars), which is what a 3B model needs to pick the right
     small element, and it is also faster than tiling (one pass, not four).
  3. *large overrun* (e.g. a full 1920-wide desktop) → split into 2-4
     tiles, each grounded at native resolution; the point deepest inside
     its tile wins. This costs one model pass per tile (~15 s on the 1070)
     but is what makes large captures clickable.
* **Two-stage zoom for "… in the …" phrasings** — "click the terminal
  icon in the panel" is exactly where the passes above fail (a 32 px
  icon inside a 1920×1080 capture). So when the phrase names a known
  container (panel, tab bar, toolbar, address bar, title bar, …), the
  agent first *boxes the container* (large and easy, one pass over the
  whole capture), then re-grounds the original phrase in a margin-expanded
  crop of it — upscaled when small. The container clause is dropped in
  that second pass: "the address bar", not "the address bar in the
  browser" (the crop already is the container, and the clause only
  confuses the model). Points landing back inside the container box are
  trusted even on flat content (an address bar is plain white).
  Edge-hugging containers (panels,
  bars) are snapped to a thin strip along the screen edge so the crop
  stays small; a loose box for such a container (a whole quadrant, half
  the edge) still snaps, because the edge touch is the reliable signal.
  For full-length containers (panel, taskbar, menu bar) the strip runs
  the whole screen edge, so a box covering only part of it cannot clip
  the target out of the crop.
  Containers with a fixed position skip the box pass entirely: edge
  containers (panel, taskbar, menu bar, …) use a deterministic strip at
  the known screen edge, center containers (modal dialogs, cookie
  walls, popups) use a deterministic box around the button zone slightly
  below the capture center, and result lists (a search results page) use
  a deterministic band where the first organic result sits (left column,
  below the search header). All of them cut the crop to a few hundred
  patches — one fast pass on the 1070 instead of a ~16 s full-window
  pass; if the target isn't in the box the normal two-pass zoom runs
  instead.
  Tiny corner boxes — the model's failure mode for
  full-window containers — are rejected with one box retry, and the
  whole zoom runs again once on failure (decodes vary between runs). If
  the container still can't be boxed (point prompt fallback
  first, then the normal path), zoom is skipped and the normal strategy
  runs. A whole-frame box is the opposite failure: the container *is*
  the whole capture (the browser being the focused window), so it is
  accepted and the crop is the whole capture — one less retry.
  Measured: "address bar in the browser" 1 m 25 s end-to-end (two model
  passes + the 12 s page-load wait), "terminal icon in the panel"
  1 m 15 s, both within a few px of the real target.
* **Decode stops at the first box** — single-object grounding (point and
  box prompts) stops the moment the first complete box is emitted and
  caps the decode at 64 tokens: the model otherwise rams on with
  duplicated boxes up to the cap (measured on the 1070: 512 tokens /
  ~60 s of pure spam where the answer was the first 4 tokens; the
  model-side early-stop flag lives in the git-ignored `locatenv` model
  code and is a no-op for multi-box `detect` prompts). This is why a
  pass now costs its prefill (~35-40 s on the 1070) instead of
  prefill + up to a minute of decode spam.
* **VRAM adapts at load and at run time** — after the model loads, the
  vision-token budget is fit to the free VRAM (prefill needs ~160 MiB per
  1024 vision tokens; the budget steps down/up so a shared card never
  OOMs). If a prefill OOMs anyway (another process grew, fragmentation),
  the worker retries once at half the budget and continues.
* **Degenerate answers are rejected** — box spam (dozens of boxes),
  repeated identical boxes, or full-frame boxes are treated as "not found"
  and the step fails cleanly instead of clicking a random spot.

Measured on the 1070 (dry-run): a 1010×900 browser window grounded both of
two page buttons within ~5 px; a 1005×829 window tiled 2×2 hit the X.com
search bar within ~4 px. Small text on a *full* 1080p desktop at the 1024
token budget is usually out of reach — that is a model/VRAM limit, not an
agent bug.

Internals worth knowing (all verified on this box):

* One `mss.MSS` context and one `Xlib.display.Display` are opened **before**
  torch is imported (`computer_use/__init__.py`). Opening a fresh X
  connection after CUDA driver init segfaults in the Xlib/mss native path,
  so the handles are held for the process lifetime.
* Arbitrary-rectangle X grabs fail on this X server (XProtoError), so a
  window crop = grab the containing full monitor + `PIL.crop` (lossless).
* `pyautogui` is imported lazily on the first real action, keeping the
  package importable in headless environments.

## Setup

Requires Python **3.10–3.12** (not 3.13 — see "GTX 1070 notes").

```bash
# 1. environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. model weights (≈ 7.3 GB) into ./locatenv
#    (git-ignored; the repo only contains the agent code)
huggingface-cli download nvidia/LocateAnything-3B --local-dir locatenv
```

On a headless server: install `python3-xlib` (X11) for pyautogui and make sure
`DISPLAY` points at the session whose screen you want to control.

## Usage

```bash
# Offline grounding: ground a phrase in an image, save an annotated copy.
python -m computer_use --image locatenv/assets/pointing.png --query "the object being pointed at"

# Same, but ask for a box instead of a point
python -m computer_use --image screenshot.png --query "the settings gear" --output-type box

# Dry run of a task: grounds everything, prints click targets, touches nothing.
python -m computer_use --dry-run "click the start menu; type firefox; key enter"

# Real run (moves your mouse — failsafe: slam the mouse into the top-left corner)
python -m computer_use "click File; click New; type hello"

# Useful knobs
python -m computer_use \
  --model ./locatenv --gpu 2 --in-token-limit 4096 \
  --scope window --generation-mode hybrid --max-new-tokens 2048 \
  --delay 1.0 --monitor 0 --save-dir runs \
  --dry-run "click the close button"
```

Programmatic use:

```python
from computer_use import LocateAnythingWorker, Agent

worker = LocateAnythingWorker("locatenv")          # picks the GPU with most free VRAM
agent = Agent(worker, dry_run=True, save_dir="runs")
agent.run("click the start menu; type notepad; key enter")

# or just ground one phrase
res = worker.point("screenshot.png", "the search box")
print(res.center)                                   # (x, y) in image pixels
```

### Resident worker (`serve`)

Every local run pays a ~25 s model load, which on the 1070 is as long as the
grounding itself. A resident worker keeps the model in VRAM and serves tasks
over a Unix socket (`/tmp/computeruse.sock`, override with `CUA_SOCKET`):

```bash
# terminal 1 — load the model once (same model flags as a normal run)
python -m computer_use serve --scope window

# terminal 2 — each run auto-connects and skips the load
python -m computer_use "click the search bar"
python -m computer_use --dry-run "click File; click New"
python -m computer_use --image screenshot.png --query "the gear"
```

Model-level options (`--model/--gpu/--dtype/--attn/--in-token-limit/--scope/
--vit-cpu/--quant`) are fixed by the `serve` invocation; per-run options (`--dry-run`,
`--delay`, `--max-steps`, `--monitor`, `--save-dir`, `--no-box-fallback`) are
sent with each task. `--no-server` forces a local load. Measured on the 1070:
~90 s per run without the worker vs ~35–45 s with it (2 tiled passes at
4096 tokens remain the bottleneck — that is Pascal prefill, not startup).

### Sandbox: let the agent click in a container, not on your desktop

If you want the agent to run while you are still using the machine, give it
its own XFCE desktop in Docker. It is managed with docker compose
(`docker-compose.yml` + the `./sandbox.sh` wrapper): Xvfb + XFCE at
1280×720 (720p is deliberate — see below), a noVNC browser view **plus a raw
VNC port (5900) for native clients**, and a resident `serve` worker
**pinned to the GTX 1070**. The
card is found by name in CUDA's own device enumeration — nvidia-smi
indices do not match torch/cuda indices on this host, and the index can
change across boots; without the card the worker falls back to auto-pick
(most free VRAM). The container restarts automatically (`unless-stopped`)
and shows as `healthy` once the model is loaded:

```bash
./sandbox.sh up                        # build (cached) + start
./sandbox.sh restart                   # reload the worker only (~40 s)
# desktop view, two ways:
#   browser:  http://localhost:6090/vnc.html   (noVNC)
#   native:   VNC :5900  (any VNC client; crisp 1:1, less lag)
./sandbox.sh dry "click the firefox icon"   # grounding only, no clicks
./sandbox.sh task "click the terminal icon in the panel; done"
./sandbox.sh status                    # state + recent logs (healthy = ready)
./sandbox.sh down
```

Every task the agent runs is logged on the serve process's stdout, so
`docker logs cuasandbox` doubles as a task audit log (task text, ok/fail,
elapsed time, and the grounded target for each step).

`./sandbox.sh up` is just `docker compose up -d --build` (plus killing a
host-side `serve` if one is running); plain compose works too:

```bash
docker compose up -d --build
docker compose ps        # STATUS "healthy" = model loaded
docker compose down
```

Everything the agent sees/clicks happens inside the container; your own
session is never touched. Step screenshots land in `runs/` on the host.

Notes:

- The container gets the GPUs via `--gpus all` **plus explicit `--device`
  flags for `/dev/nvidia*`. On this host (runc 1.1.x, cgroup v2) the nvidia
  hook's device injection is not visible to runc, so unprivileged GPU
  access needs the `--device` flags (NVML otherwise fails with
  "Unknown Error"). `sandbox.sh` builds the flag list from whatever
  `/dev/nvidia*` nodes exist.
- The GPU's kernel driver lives on the host; `entrypoint.sh` symlinks the
  host driver libs that the nvidia hook mounts so `libnvidia-ml.so.1`
  matches the kernel module.
- `locatenv/` is bind-mounted read-only into the container, so no copy of
  the weights is needed.
- **720p, not 1080p**: the full 1280×720 screen is 4641 vision patches,
  just over the 4096 token budget, so a full-frame step is ONE ~0.94× pass
  (~29 s on the 1070) instead of the 1080p 4-tile pass (~67 s) — and the
  model sees the UI at near-native scale instead of down-scaled tiles. The
  edge strips (64 px bottom / 48 px top) are full-width either way and are
  unaffected.
- Ports 6090 (noVNC) and 5900 (raw VNC) are used because 6080 is already
  taken on this host; change `docker-compose.yml` and `entrypoint.sh` if
  you move them.

### Orchestrated runs (`orchestrate`): a small LLM plans, the controller executes

For goals the scripted task language cannot express, a small local LLM
plans: **LiquidAI LFM3-1.7B** (Q4_K_M GGUF, served by llama.cpp's
OpenAI-compatible `llama-server`) on the same GTX 1070, ~1.4 GB
footprint. The LLM sees only **text** — the goal, and the reported
result of each executed step (ok/fail, elapsed, frame-change %, active
window title) — and each turn emits exactly ONE tool call from a closed
vocabulary (the controller verbs plus `observe`/`verify`/`done`). The
deterministic executor in `controller.py` is the only actor that touches
the screen; grounding stays inside it. The LLM is never asked "what do
you see" (LocateAnything is a grounding model, not VQA), so perception
is the deterministic window-title + 320×180 frame-diff signal.

The planner is 1.7B, so the architecture compensates for measured
weaknesses:

- **Goal decomposition + an external progress tracker.** The goal is
  split once into 2–6 subgoals; every planning prompt re-lists them with
  `[DONE]`/`[NEXT]`/`[ ]` marks that advance only on *measured* state
  (DDG title, non-DDG title, terminal title, type/key run in the
  terminal). The model loses multi-part goals mid-run, so progress is
  tracked outside it.
- **Suggestion fast-path (the latency design).** After a measured state
  signature (results page up, page settled, terminal focused), the
  deterministic layer proposes a concrete next machine step whose args are
  extracted from the goal's own words (scroll count, "terminal", the
  quoted command): `search → click first result → wait-load → scroll N →
  open terminal → type <command verbatim> → enter → done`. When a pending
  suggestion exists the system **executes it directly — no planning round
  trip**: each suggestion is anchored on a measured signature, so
  following it is always safe, and skipping the ~1–2 s LFM3 call keeps
  canonical transitions at executor speed (measured: consulting the
  planner per transition cost ~15 s of pure latency in the 9-step goal).
  The planner keeps full authority over every turn *without* a pending
  suggestion — goal start, states outside the canonical chain, recovery
  after a failed step — and sees every executed result in the history.
  `done` is suggested by the system once all subgoals are marked on
  measured evidence (the 1.7B never calls done on its own).
- **Decomposer-variance hardening.** The 1.7B decomposer's wording varies
  run-to-run (it merged "type … and press Enter" into one subgoal, lifted
  the goal's terminal clause into a subgoal — run 15 — dropped the
  quoted-command line entirely — battery T1, wrote meta-prose instead of
  actions — battery T3, and repeated lines verbatim — battery T3). So: a
  subgoal asking for typing *and* Enter is marked at the Enter step, not
  the type step; a "call done when …" subgoal is a termination condition,
  not an action — it is dropped at decomposition; a goal that quotes a
  command to type must have a subgoal carrying it (one re-decompose, then
  a deterministic append); duplicate lines are dropped; and lines outside
  the mark surface (no controller step could ever mark them) are dropped,
  since they would block the all-done signal forever. Marks additionally
  require the subgoal's *own* quoted command(s) in the typed-text history
  (a merged "both commands" subgoal is not marked at the first Enter —
  battery T5), and the fast path types the goal's next still-untyped
  command after an Enter, so multi-command goals finish deterministically.
  All of this was measured on the 9-task battery below, not on one ad-hoc
  goal.
- **Caps:** 32 steps (overridable with `--max-steps` — the T9
  17-command goal runs with a 48 cap), 180 s, 3 consecutive failures
  (blocked repeats count).

The server is spawned at orchestrate start and stopped at the end. The
spawn pins the 1070 **alone** via `CUDA_VISIBLE_DEVICES`: llama.cpp's
CUDA init creates a context on every *visible* device, and on this shared
host the other two GPUs (RTX 3060s) can sit at under 1 GB free under the
user's workloads — the `cudaSetDevice` there OOMs and aborts the whole
load even though `--main-gpu` points at the 1070 (measured: worked while
the 3060s had headroom, aborted at 110 MiB free). With the filter the
1070 is device 0 (`--main-gpu 0`), the model loads in ~2.7 s, and the
spawn is deterministic. The spawn also waits for ≥ 2.5 GB free on the
1070 and retries (3×) — the card is shared with the LocateAnything
worker, whose fp16 reload peaks at ~6.8 GB, and with the user's live
desktop. `entrypoint.sh` also kills stray llama-servers before each
worker load, and the planner re-ensures the server if it dies mid-run.
`LFM3_PERSIST=1` keeps it warm between runs. The 1.05 GB GGUF is
expected at `LFM3_MODEL` (downloaded at runtime, never committed).

```bash
# in the sandbox (the worker serves the "orchestrate" kind too):
docker exec cuasandbox bash -lc \
  'CUA_WAIT_SERVER=60 python -m computer_use --save-dir /tmp/orch_runs_real \
   --task-file /tmp/goal.txt orchestrate'
```

Knobs: `LFM3_URL`/`LFM3_PORT` (attach to an existing server),
`LFM3_MODEL`, `LFM3_BIN`, `LFM3_GPU` (default: the device named *1070*),
`LFM3_PERSIST`, `LFM3_CTX` (4096), `LFM3_MAX_TOKENS` (256),
`LFM3_TEMPERATURE` (0.1), `LFM3_TOP_K` (50); `CUA_WAIT_SERVER=<s>` makes
the client wait for the worker socket. Sampling: temp 0.1, top-k 50,
thinking disabled, `tool_choice: required`.

Measured on the 1070: the 9-step goal (search → open first result →
scroll 3 → open terminal → type a verbatim `echo … > /root/note.txt` →
Enter) completes in **8 steps / 17–20 s loop, 1 planner call** (search is
the only planning turn — every canonical transition is fast-path
executed; ~22 s cold including the ~2.7 s LFM3 spawn), with the exact
file content + fresh mtime and the article still in the tab. The
wait-load catches the slow-navigation quirk mid-flight at 1.0–1.2 s
(pre-capture title == post-capture title means the navigation is still
in flight); when the title already moved within the 0.2 s post-capture
window the fast path settles directly instead of burning the wait-load's
8.1 s cap. The scripted path for the same goal is still the fast lane at
~13 s — orchestration is for goals that cannot be scripted.

**Resetting the desktop between runs.** Runs expect a clean desktop: no
`/root/note.txt`, no terminal window, Firefox at about:blank. The reset
is VERIFY-AND-RETRY, not fire-and-forget — `windowclose` was measured to
transiently fail right after a run (the X server is busy), and a
surviving terminal with a pending typed line poisoned the next run's
output (run 15→15b: the leftover line was appended to and executed as
one concatenated command):

```bash
docker exec cuasandbox bash -lc '
  export DISPLAY=:99; rm -f /root/note.txt
  for i in 1 2 3; do
    xdotool search --name "Terminal" windowclose 2>/dev/null; sleep 0.5
    [ -z "$(xdotool search --name "Terminal")" ] && break
  done
  xdotool search --name "Mozilla Firefox" windowactivate --sync
  sleep 0.4; xdotool key ctrl+l; sleep 0.3
  xdotool type "about:blank"; sleep 0.3; xdotool key Return; sleep 1.2
  t=$(xdotool getactivewindow getwindowname)
  [ "$t" = "Mozilla Firefox" ] && echo "desktop clean" || echo "RESET FAILED: $t"'
```

 ### Task accuracy: the 9-task battery

 Every task runs through `orchestrate` (LFM3 planner + fast-path over the
 deterministic controller) — never the scripted path. Each run's driver
 prints `lfm3_ready=1` as proof the orchestration layer was used, plus
 per-run evidence: file content + fresh mtime, the final Firefox tab
 title, and `pgrep -c xfce4-terminal`. All 9 goals share one shape
 (search → open first result → scroll → terminal → type a verbatim
 command → Enter) with per-task twists: a no-terminal/no-file goal (T3),
 a quoted-quotes file (T4), two commands in one goal (T5), a
 pre-opened terminal (T6), a cold Firefox launch (T7), preloaded
 unrelated DDG results (T8), and a 17-command long-horizon goal (T9 —
 40 steps, run with a raised 48-step cap). Goals: `eval/goals/t1..t9.txt`;
 driver: `eval/eval_one.sh` (verify-and-retry reset, per-task setup,
 optional `--max-steps`, evidence summary).

 Final code, one run per task, 2026-11-01, GTX 1070 shared desktop:

 | Task | Setup twist | Steps | Loop | Planner calls | Evidence (file / tab) |
 |------|-------------|-------|------|---------------|------------------------|
 | T1 | — | 8 | 27.4 s | 1 | `T1: k8s gpu plugin docs` / GitHub k8s-device-plugin |
 | T2 | — | 7 | 20.6 s | 1 | `hello world 123` / GeeksforGeeks disk-space |
 | T3 | no terminal, no file | 4 | 18.2 s | 1 | (none) / CUDA Runtime API docs; 0 terminal procs |
 | T4 | single+double quotes in file | 8 | 13.7 s | 1 | `T4: single and "double" quotes ok` / Stack Overflow nvidia-runtime |
 | T5 | two commands, one 2-line file | 10 | 16.4 s | 1 | exactly `first-line`⏎`second-line` / NVIDIA Enterprise Portal |
 | T6 | terminal pre-opened | 8 | 24.2 s | 1 | `T6: attach existing terminal` / GitHub k8s-device-plugin |
 | T7 | Firefox killed (cold launch) | 8 | 36.3 s | 1 | `T7: cold launch ok` / GitHub k8s-device-plugin |
 | T8 | DDG preloaded with unrelated results | 7 | 19.3 s | 1 | `T8: preloaded results replaced` / GeeksforGeeks disk-space |
 | T9 | 17 commands, one 17-line file; 48-step cap | 40 | 29.9 s | 1 | `/root/pool.conf` all 17 lines exact / GitHub k8s-device-plugin |

 9/9 pass; every run `exit=0`, `failed_lines=0`, and **1 planner call**
 (the decompose — the whole loop, including the done, is fast-path).
 Loop time is dominated by the two grounding passes (search 3.7–6.1 s,
 click 2.5–7.9 s); everything else is sub-2 s. T3 is 4 steps because the
 goal has no scroll/terminal half. T5 is 10 steps because the goal has
 two type+Enter pairs. T9 is 40 steps: the 3-step preamble (search,
 click, scroll), 17 type+Enter pairs, and the done — the decompose's
 checklist cap and output budget are sized to the goal's command count
 (a fixed 8-line checklist / 300-token budget would have truncated the
 17-command goal and fired all_done before every command was typed).

 The battery's value is the failure analysis, not the 9/9. Before the
 hardening above, each task exposed a distinct orchestration-layer bug,
 root-caused from logs + evidence screenshots, and fixed generally (no
 case-specific branches):

 - **Stale grounding on a moving page** (T1): the DDG Search Assist
   widget expanded *during* the click's vision pass, so the click hit
   the widget. Fix: re-capture after the vision pass and re-ground once
   if the frame moved > 2% (`_REGROUND_DIFF_PCT`), plus a retry budget
   that lets a same-state step run exactly one more time (fresh
   grounding is the remedy) and blocks the second.
 - **The results-band prior misses the first result when the widget is
   up** (T1/T2/T3): the deterministic band (0.62h–0.8h) is tuned for
   the bare results layout; an expanded widget pushes the result below
   0.8h. Fix: two measured triggers escalate to the band extended to
   the window bottom — the box pass answering "no target in this crop"
   (with `strict_box` skipping the trusted-point fallback that would
   latch onto widget text), and a grounded point in the band's bottom
   30% (a band-edge caption latch). In the extended band a THIN
   full-height box is a "column, not the item" answer (rejected), while
   a FULL-WIDTH one is a "whole region" answer whose 50%-height center
   lands in the widget–result gap — the click uses the 70%-height point
   instead (measured: 50% missed the title row, 70% hit it, first
   click navigated).
 - **Decomposer dropped the typed-command subgoal** (T1): all_done fired
   with the command never typed and the file never written. Fix: the
   coverage check (re-decompose once, then a deterministic append of the
   goal's own words).
 - **The open-terminal mark fired on "In the terminal, type …"** (T2):
   the open mark now excludes type/run/enter phrasing.
 - **A merged multi-command subgoal marked at the first Enter** (T5):
   the second command never ran, yet all_done fired. Fix: marks require
   the subgoal's *own* quoted command(s) in the typed-text history, and
   the fast path types the goal's next still-untyped command after an
   Enter.
 - **Unmarkable decomposer lines block all_done forever** (T3):
   "Verify that the article is opened" ×2, junk alternatives ("Use a
   command-line tool like `curl`…"), exact duplicates, and meta-prose
   ("Formulate query string: …") — the planner then polled the same
   `observe` until the failure budget. Fix: the verify-article mark
   branch, duplicate drop, and the mark-surface filter (lines no
   controller step could mark are dropped at decomposition).
 - **The repeat guard read state as the window title alone** (T9):
   the terminal title is constant (`Terminal -`) across every type and
   Enter, so after the second Enter the third canonical Enter was
   judged "already executed twice in this exact state", the fast path
   dropped the pending suggestion, and the planner restarted the chain
   from the search — re-typing commands with no Enters — until the
   failure budget. Fix: state identity is now (title, frame
   signature) — the md5 of a 32×18 grayscale capture quantized to
   8 levels (`_frame_sig`) — used by both the repeat guard and the
   fast-path stale check, so they stay in sync. A typed character moves
   the frame, so each Enter is a new state; a truly STATIC screen still
   blocks after the second retry (same title, same frame), which is
   the planner fixation the guard exists for. The signature capture
   short-circuits: only a repeated step in the same title with a spent
   retry budget costs one extra grab.
 - **The decomposer dropped the "press Enter" clause from a command
   subgoal** (T2, second run): the subgoal ended at the em-dash, so the
   mark logic — which marks a type+enter subgoal only at the Enter step
   when the subgoal itself says Enter — marked it at the type step, and
   all_done fired with the command sitting on the terminal prompt, file
   never written (the typed-only guard passed, because the command WAS
   typed). Fix: all_done now also requires `_goal_commands_run` — every
   command the goal explicitly quotes must be in the typed history AND,
   when the goal asks for Enter, an ok Enter step in the EXECUTED
   history between each command's ok type step and the next command's
   type step (unbounded for the last). Measured steps only — never the
   planner's claims.

 Re-run the battery from the sandbox (each task ~20–30 s):

 ```bash
 docker exec cuasandbox bash -lc \
   'bash /tmp/eval_one.sh T1 /tmp/goal_t1.txt none /root/note.txt "T1: k8s gpu plugin docs"'
 # …t2–t9 the same way (goal files, setups and expected contents are in
 # eval/ in the repo; see eval/eval_one.sh usage in its header).
 ```

### Task language

Steps are separated by `;` or newlines. A bare phrase is treated as `click`.

| Step                  | Action                                   |
|-----------------------|------------------------------------------|
| `click <phrase>`      | left-click the grounded element          |
| `double-click <phrase>` | double-click                            |
| `right-click <phrase>`| right-click                              |
| `type <text>`         | type ASCII text                          |
| `key <combo>`         | key combo, e.g. `ctrl+alt+t`, `enter`    |
| `scroll <phrase> <n>` | scroll at the element (positive = down, − = up). The target is grounded on a coarse pass (≤ 320 vision patches): a scroll target is a large region — the point only has to land inside it — so region-level accuracy at ~1/10 of a full-window pass |
| `wait <seconds>`      | idle                                     |
| `open <app>`          | tool step: focus an app directly — attach to its running window, launch only if none is running — `firefox`, `file manager`, `terminal`, `applications`. No grounding pass; the step returns only once the app's window has the focus, so a following `type`/`key` lands in it. Unknown apps fail loudly. |
| `search <query>`      | tool step: open Firefox on a DuckDuckGo search for `<query>` (alias `duckduckgo <query>`). The results page is shown — the top result is *not* resolved server-side (that would pick the answer without the agent ever seeing the results); a following `click the first result in the search results` grounds and opens it. Flow: attach to the running Firefox window (launch only if none is running) → click the address bar (a mouse click — pages that auto-focus their own search box swallow a synthetic Ctrl+L, measured) → type → Return, sent by window id → verify the title actually becomes the DuckDuckGo title (a bare title change is not proof of navigation — measured), retrying up to 3× and failing loudly → poll until the results region has rendered → Escape+Home, because a tab that held a scroll position carries it into the new document (measured) (no fixed pauses). |
| `wait-load`           | wait until a navigation's new page is painted. Two measured gates, no fixed guess: (1) the active window's title changes (50 ms poll, 8 s cap — a navigation leaves the old title up until the new document sets its own); a slow page can set its title after the cap (measured: a heavy docs page, ~10 s), so gate 1 also accepts a *settled* frame change — two consecutive captures differing from the pre-navigation frame (0.3 s poll); (2) the screen settles on the new page (100 ms frame compares, 3 s cap) — the title is set *before* the paint (measured), so gate 1 alone would let the next step act on the old page |
| `done`                | stop                                     |

The `open` tools exist because app launching is exactly where icon grounding
is weakest (measured: "file manager icon" in the panel deterministically
lands on the directory-menu icon — a semantic bias that upscaling the strip
does not fix) and because each saves a model pass. Everything else still
goes through the VL grounding above.

**Measured: a real research task in ~9 s.** The task "search a topic,
open the top result, dismiss the cookie wall, read the answer" runs in
**8.2–10.5 s** on the 1070 (fresh sandbox container, wall clock, the task's
own waits included; the spread is DDG/article load variance — e.g. the
article title being set 0.8 s vs 1.2 s after the click. The two-gate
`wait-load` adds a measured 0.02–0.25 s over the title-only wait). The
search is honest: the tool opens the DuckDuckGo **results page** (the top
result is not resolved server-side), and the agent grounds and clicks the
top result itself. The budget, per step:

| Step | Cost | Why |
|---|---|---|
| `search <query>` | ~1.5–2.4 s (attach); ~4.5 s (cold launch) | no model pass — attach to the running Firefox window (launch only if none is running: a second launch opens a *new* window, and a new window in a running session shows the "Open previous tabs?" bar that shifts the results layout down ~35 px and breaks the results-band prior — measured), click the address bar (a mouse click, not Ctrl+L — DDG auto-focuses its search box and swallows the synthetic Ctrl+L, measured twice), type → Return sent by window id, verify the title becomes the DuckDuckGo title (a bare title change is not proof — a transient restore attempt changed it without a navigation, measured), retry up to 3× then fail loudly, then poll until the results region actually renders (measured: on DDG the title is set *before* the paint; the render lag varied 0.44 s to >1.1 s across runs) |
| `wait 0.4` | 0.4 s | the task's own settle margin |
| `click the first result in the search results` | ~2.8 s | the results-band prior crops the fixed band where the first organic result sits (left column below the search header; (0,384)–(768,496) of a 1280×620 window ≈ 432 patches) — one pass, no box pass. The model's box is trusted (everything in the band is part of the first result) and box-grounding is tried before point-grounding (a point wanders into the whitespace around the text; measured) |
| `wait-load` | ~0.6–1.2 s | gate 1: poll the window title every 50 ms until the navigation sets a new one (a slow page that sets its title after the 8 s cap is caught by a settled frame change instead — two consecutive frames differing from the pre-navigation capture); gate 2: wait for the new page's paint to settle (100 ms frame compares) — measured overhead of gate 2: 0.02–0.25 s |
| `wait 0.9` | 0.9 s | the OneTrust wall renders ~0.93 s after the title change (measured on a fresh profile: the dialog appears in a single frame) |
| `click … in the dialog` | ~2.2 s | the center-region prior crops a fixed box around the button zone of a centered modal ((448,240)–(896,410) ≈ 384 patches) — one pass, no box pass |

Before these layout priors the same task took ~81 s: the top result was
opened by grounding a click on the DuckDuckGo results page (~33 s, a full
window pass), and the cookie wall was grounded on the whole browser window
(~39 s, another full pass). An intermediate design got it to ~16.5 s by
resolving the top result server-side from DuckDuckGo's JS-free `/html/`
endpoint — but that step picked the answer without the agent ever looking
at the results, so it was removed.

**Measured: a complex multi-app task in ~13 s.** The 9-step task "search a
topic, open the top result, scroll the article, open a terminal, write a note
file, press enter" spans two applications and runs in **13–15.6 s** on the
1070 across the attach and launch paths (wall clock, the task's own waits
included; earlier fresh-container runs took 11.5–12.1 s). No task-specific
shortcuts: every step is grounded from pixels. The budget of the slowest
verified run (launch path over a killed session whose restored tab carried a
scroll position — the worst realistic state, ~15.6 s with the paced type):

| Step | Cost | Why |
|---|---|---|
| `search <query>` | 4.94 s | no model pass — cold launch of Firefox (restore, window map, focus) + address-bar sequence + DDG load + render poll + Escape+Home normalize |
| `wait 0.4` | 0.40 s | the task's own settle margin |
| `click the first result in the search results` | 3.01 s | the results-band prior — one pass |
| `wait-load` | 1.11 s | the title changed 1.09 s after the click; the settle gate adds 0.02 s |
| `scroll the article 3` | 3.96 s | coarse grounding pass: the capture is downsampled to 300 vision patches (≤ 320) — a scroll target is a large region; the point only has to land inside it. ~30 s cheaper than the 4004-patch full-window pass; the full-res capture is still saved as evidence |
| `open terminal` | 0.95 s | the `open` tool — no grounding pass; attaches to the running terminal window |
| `type echo … > /root/note.txt` | ~1.2 s | typed directly into the focused window, paced at ~100 chars/s (was 0.71 s at 5 ms/char — the faster burst was measured to drop the last characters of a long line on the loaded shared card) |
| `key enter` | 0.01 s | — |

(Attach-path re-runs: 13–13.2 s — search is 1.5–2.4 s instead of 4.9 s.)
Verified on each run: `/root/note.txt` contains the echoed line, the correct
article is open (not the second or third result), and the article actually
scrolled — the article region differs by 37–57 % of its pixels between the
pre-scroll and post-scroll captures, and the "Solution:" section is visible in
the final screenshot. (Before the fixes below the same region differed by
0.02 %: the scroll had silently been a no-op.) The pixel diffs and the
failed re-runs exposed five bugs:

1. **Scroll sign.** pyautogui takes positive amounts as *up*; the task
   language "scroll 3" means down. With the old sign a scroll at the top of a
   freshly loaded article did nothing to the screen. The controller now
   passes the negated amount — `scroll <phrase> <n>` in task language is
   positive = down.
2. **Scrolling the stale page.** The original `wait-load` waited for the
   title change only; on DDG the article title is set 0.44–1.1 s *before* the
   new page paints, so a scroll right after hit the old results page. The
   settle gate (gate 2 of `wait-load`) fixes this at a measured 0.02–0.25 s
   overhead.
3. **A new window on re-run.** `search` always launched a second Firefox
   instance; the new window it opened in the running session showed the
   "Open previous tabs?" bar, which pushed the results layout down ~35 px and
   put the first result out of the results band — the model then answered
   with a full-crop box (its "not found" answer) and the trusted-band logic
   clicked the crop center: dead space, no navigation (measured: `wait-load`
   capped at 8 s and the task ran on the results page). Now `search` and
   `open` attach to the running window (launch only if none is running — a
   fresh first window shows no such bar, measured), and a single full-frame
   box in the zoom pass is rejected as "no target in the crop" instead of
   being clicked blindly.
4. **Inherited scroll position.** A tab that already held a scroll position
   (a restored session, a page scrolled by a previous run) carries that offset
   into the newly loaded document — measured: the results page opened ~320 px
   down (3 wheel notches), the first result cut out of the results band, and
   the click opened result 3 instead of result 1. `search` now ends with
   Escape + Home (Escape drops the address-bar focus, Home scrolls the
   document to its top) plus a short re-poll of the results band; the
   controlled repro (scrolled tab → search → capture) confirms the results
   render at the top again.
5. **The "Restore Session" interstitial.** Launching Firefox over a killed
   session occasionally shows its restore-failure interstitial ("Sorry. We're
   having trouble getting your pages back", with "Start New Session" /
   "Restore Session" buttons) — observed once, with its early restore machinery
   racing the first navigation: the typed results URL bounced back to the
   interstitial right after loading (measured). `search` now proactively
   clicks "Start New Session" (fixed position on this fixed-chrome page at the
   established geometry) and then *verifies* that the title actually becomes
   the DuckDuckGo title — a bare title change is not proof of a navigation
   (the transient restore attempt changed the title without loading anything,
   measured) — retrying the address-bar sequence up to 3× and failing loudly
   if the results page never loads, instead of clicking a dead band. The
   interstitial's trigger could not be reproduced on demand (controlled
   session-file corruption at both the LZ4 and JSON level yields a logged
   "No readable session files found to restore, starting with empty session"
   instead — measured), so the verify+retry path is the general guard for
   whatever startup state appears.

## GTX 1070 notes

The 1070 is a Pascal GPU (compute capability 6.1, 8 GB). Five constraints
shape this repo:

1. **torch ≤ 2.5.1** — PyTorch 2.6+ dropped sm_61 kernels entirely (the build
   will *run* but CUDA ops fail). torch 2.5.1 has no cp313 wheels, hence the
   Python 3.12 requirement.
2. **float16, not bfloat16** — bf16 needs sm_80+. The worker selects fp16
   automatically on sm < 8.
3. **SDPA math attention** — flash-attn and MagiAttention need sm_80+, the
   mem-efficient SDPA kernel needs sm_75+. On Pascal the worker forces
   PyTorch's arch-independent math backend (and disables cuDNN so MoonViT's
   convs use native kernels).
4. **MoonViT on the CPU** — the 3B LLM in fp16 (≈ 6.3 GB) plus CUDA context
   leaves only ~1 GB for activations on an 8 GB card, and the prefill logits
   copy is the biggest one. The 0.4B vision encoder therefore runs on the CPU
   (one extra second per query) via `--vit-cpu auto` (default).
5. **NF4 weights on a shared card** — the sandbox worker defaults to
   `--quant nf4` (≈ 3.3 GB) instead of fp16 (≈ 6.6 GB). In this setup the
   1070 is *also* the host's display GPU (X server, compositor and browser
   run on it): fp16 left only ~1.1 GB for the desktop, and a model-reload
   VRAM spike in that headroom wedged the driver and froze the host until a
   reboot (measured incident, 2026-10-31). NF4 leaves ~4.4 GB for the
   desktop, and reloads are no longer a freeze risk. Speed is unchanged on
   the 1070 (measured: prefill is attention/compute-bound, not
   weight-stream-bound).

VRAM budget (8 GB card, full-HD screenshot):

| Setting                       | Vision tokens | Notes                         |
|-------------------------------|--------------:|-------------------------------|
| `in_token_limit=1024` (1070 default) | ~260 | fits 8 GB with headroom |
| `in_token_limit=2048`         | ~530 | usually fits; more detail    |
| `in_token_limit=4096`         | ~1030 | tight; only if VRAM allows   |
| `in_token_limit=25600` (model default) | ~4000 | needs ≥ 12 GB |

On modern GPUs (RTX 30/40/50, A/H-series) everything works as-is — the worker
auto-selects bf16, keeps the vision encoder on the GPU, and you can opt into
`--attn magi`/`flash_attention_2` if those packages are installed.

## Layout

```
computer_use/
├── worker.py       # model loading + inference (1070-tuned), GroundingResult
├── grounding.py    # parse <box><x>…</box> output into pixel coords
├── screen.py       # mss screenshot capture + monitor coordinate mapping
├── actions.py      # pyautogui click/type/key/scroll + detached launch (dry-run aware)
├── controller.py   # task language + perceive→ground→act loop
├── server.py       # resident worker: model stays in VRAM, tasks over a Unix socket
└── __main__.py     # CLI (agent / offline / serve)
Dockerfile          # XFCE sandbox image (Xvfb + noVNC + agent)
entrypoint.sh       # sandbox entrypoint: desktop + VNC + serve
sandbox.sh          # host wrapper: up/down/task/dry/status
locatenv/           # model weights (git-ignored)
```

## Limitations

- One step = one grounded phrase; there is no LLM planner. Express multi-step
  flows in the task language above (or add a planner that emits it).
- Window scope follows the *focused* window — keep the target app in the
  foreground; if the focus moves between steps, steps operate on the new
  window.
- Pointing mode is the click source; if the model can't see the element, the
  step fails (a box fallback is attempted automatically).
- `type` is ASCII-only (X11 unicode input needs an extra layer, e.g. xdotool).
- The model is a 3B grounding model: expect occasional mis-groundings on
  cluttered or small UIs. Use `--dry-run` to review targets before executing.
- On the 1070, prefill takes ~10-15 s per query (Pascal + math attention);
  plan your `--delay` accordingly.

## License

Agent code: MIT.
Model weights: **NVIDIA non-commercial research license** (see
`locatenv/LICENSE`) — academic/non-profit use only; the `locatenv/` directory
is intentionally git-ignored and excluded from this repository.