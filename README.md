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
* **Auto-tiling** — if the capture has more vision patches than the token
  budget (e.g. a maximized 1920-wide browser at `in_token_limit=4096`), the
  capture is split into 2-4 tiles and each tile is grounded at native
  resolution; the point deepest inside its tile wins. This costs one model
  pass per tile (~15 s on the 1070) but is what makes large windows
  clickable.

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

### Task language

Steps are separated by `;` or newlines. A bare phrase is treated as `click`.

| Step                  | Action                                   |
|-----------------------|------------------------------------------|
| `click <phrase>`      | left-click the grounded element          |
| `double-click <phrase>` | double-click                            |
| `right-click <phrase>`| right-click                              |
| `type <text>`         | type ASCII text                          |
| `key <combo>`         | key combo, e.g. `ctrl+alt+t`, `enter`    |
| `scroll <phrase> <n>` | scroll at the element (+up / −down)      |
| `wait <seconds>`      | idle                                     |
| `done`                | stop                                     |

## GTX 1070 notes

The 1070 is a Pascal GPU (compute capability 6.1, 8 GB). Four constraints
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
├── actions.py      # pyautogui click/type/key/scroll (dry-run aware)
├── controller.py   # task language + perceive→ground→act loop
└── __main__.py     # CLI
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