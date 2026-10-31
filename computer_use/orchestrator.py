"""LLM orchestrator: LFM3-1.7B (Q4) plans over the deterministic controller.

The planner is a small local LLM (LiquidAI LFM3-1.7B, Q4_K_M GGUF, served
by llama.cpp's OpenAI-compatible ``llama-server``) that sees only TEXT:
the goal, the reported result of each executed step, and the active
window title. Each turn it emits exactly ONE tool call from a closed
step vocabulary (the controller verbs plus ``observe``/``verify``/
``done``); the deterministic executor in ``controller.py`` is the only
actor that touches the screen.

Division of labor:
  * LFM3 planner     — high-level state, which step is next, when to stop.
  * LocateAnything   — all pixel-level perception: grounding inside
                       controller.py. It is a GROUNDING model, not a VQA
                       model: open-ended questions return <ref>/<box>
                       spam in ~36 s (measured), so it is never asked
                       "what do you see".
  * controller.py    — the reliable primitives (attach-first search,
                       verified address-bar sequence, two-gate wait-load,
                       title verify+retry, ...). The planner sees their
                       high-level results, never their internals.
  * the orchestrator — deterministic state for the planner: each step
                       result carries the active title and a frame-change
                       % (320x180 gray diff), and observe/verify report
                       window-manager state (active title + visible
                       windows). Page CONTENT is not visible to a text
                       planner by design — the title and the frame diff
                       are the evidence channel.

The LFM3 server is started lazily (on the first orchestrate request) by
the process that owns the LocateAnything worker, pinned to the GTX 1070
by NAME in torch's CUDA enumeration (nvidia-smi indices do not match
torch indices on this host — see entrypoint.sh). It holds ~1.4 GB of
the 8 GB card (measured) and coexists with the worker and the host's
own desktop.

Sampling follows LiquidAI's agent guidance: temp 0.1, top_k 50, thinking
disabled (chat_template_kwargs) — planning steps are short and
deterministic, and reasoning tokens would double the call latency.

Lifecycle: the server is STARTED at the beginning of an orchestrate run and
STOPPED at the end (LFM3_PERSIST=1 keeps it warm across runs). This matters
because the card is shared: the LocateAnything worker's model RELOAD (a
code-deploy event) has a ~6.8 GB fp16 peak that fits the 8 GB card only
with ~0.9 GB of headroom; a resident LFM3 server (1.25 GiB) pushes that
peak over the edge and OOMs every respawn (measured). entrypoint.sh also
kills any stray llama-server before each worker load as a safety net, and
Planner re-ensures the server if it dies mid-run.

Config (env):
  LFM3_URL          base URL of a running server (default http://127.0.0.1:8090)
  LFM3_PORT         port to spawn on (default 8090)
  LFM3_MODEL        GGUF path (default /root/models/LFM3-1.7B-Q4_K_M.gguf)
  LFM3_BIN          llama-server binary
                    (default /opt/llama.cpp/cuda-12.8/llama-server)
  LFM3_GPU          CUDA index override (default: the device named *1070*)
  LFM3_PERSIST      1 = keep the server running after the run (default 0)
  LFM3_CTX          context size (default 4096)
  LFM3_MAX_TOKENS   planner decode cap (default 256)
  LFM3_TEMPERATURE  default 0.1
  LFM3_TOP_K        default 50
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

# The planner never needs more than a step + arguments; 256 caps decode
# at ~85 tok/s -> worst case ~3 s of generation per planning call.
_TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Open Firefox on a DuckDuckGo search for the query; "
                       "the results page is then shown (top result first).",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "click",
        "description": "Left-click the UI element described by the phrase. "
                       "For a search results page the top organic result "
                       "is \"the first result in the search results\".",
        "parameters": {"type": "object",
                       "properties": {"phrase": {"type": "string"}},
                       "required": ["phrase"]}}},
    {"type": "function", "function": {
        "name": "scroll",
        "description": "Scroll the described region n notches; positive = down.",
        "parameters": {"type": "object",
                       "properties": {"phrase": {"type": "string"},
                                      "amount": {"type": "integer"}},
                       "required": ["phrase", "amount"]}}},
    {"type": "function", "function": {
        "name": "open",
        "description": "Launch or focus an app: firefox | file manager | "
                       "terminal | applications.",
        "parameters": {"type": "object",
                       "properties": {"app": {"type": "string",
                                              "enum": ["firefox",
                                                       "file manager",
                                                       "terminal",
                                                       "applications"]}},
                       "required": ["app"]}}},
    {"type": "function", "function": {
        "name": "type",
        "description": "Type text into the focused element (ASCII).",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "key",
        "description": "Press a key combo, e.g. \"enter\", \"ctrl+alt+t\".",
        "parameters": {"type": "object",
                       "properties": {"combo": {"type": "string"}},
                       "required": ["combo"]}}},
    {"type": "function", "function": {
        "name": "wait",
        "description": "Idle for a few seconds (<= 30).",
        "parameters": {"type": "object",
                       "properties": {"seconds": {"type": "number"}},
                       "required": ["seconds"]}}},
    {"type": "function", "function": {
        "name": "wait-load",
        "description": "After a click that navigates: wait until the new "
                       "page is up (title changes and the frame settles).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "observe",
        "description": "Report the current window state: the active window "
                       "title plus the list of visible windows. Use when "
                       "the reported state is unclear (a dialog appeared, "
                       "a window did not open, ...). Note: each step "
                       "result already includes a frame-change %; 0% "
                       "means the step changed nothing visible.",
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string",
                                                  "description": "What you "
                                                                 "want to "
                                                                 "know "
                                                                 "(echoed "
                                                                 "back; "
                                                                 "the "
                                                                 "answer "
                                                                 "is the "
                                                                 "window "
                                                                 "state)"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "verify",
        "description": "Check a yes/no criterion at the WINDOW level "
                       "(titles and window names): MATCH with evidence, "
                       "or the visible state when there is no window-level "
                       "evidence. Page CONTENT is not visible to this "
                       "check — only window titles/names.",
        "parameters": {"type": "object",
                       "properties": {"criterion": {"type": "string"}},
                       "required": ["criterion"]}}},
    {"type": "function", "function": {
        "name": "done",
        "description": "The goal is visibly achieved; stop. State the "
                       "evidence in the summary.",
        "parameters": {"type": "object",
                       "properties": {"summary": {"type": "string"}}}}},
]

_SYSTEM = (
    "You are the planner for a desktop computer-use system (XFCE, "
    "1280x720; the browser is Firefox, search goes through DuckDuckGo). "
    "You never act directly: each turn you emit exactly ONE tool call. "
    "A deterministic executor runs it and reports the result, including "
    "the active window title, how much of the frame changed, and "
    "occasionally a HINT about what just happened, before your next "
    "turn. Follow the HINTs: they are measured, reliable facts.\n"
    "Rules:\n"
    "- Pick the single most direct reliable step; never assume what is "
    "off-screen or unseen.\n"
    "- A result's frame-change % tells you whether the step changed "
    "anything visible: 0% with an unchanged title means the step was a "
    "no-op — do NOT repeat it; observe or change approach.\n"
    "- If a step reported FAILED, or the title looks wrong, use observe "
    "before retrying, and change approach instead of repeating the same "
    "step.\n"
    "- The goal is decomposed into numbered subgoals. Work through them in "
    "order. A subgoal counts as done when the evidence (title, frame "
    "change, window state) shows it achieved — then move to the NEXT "
    "subgoal; never redo a done subgoal or repeat its steps.\n"
    "- Call done only when the LAST subgoal is visibly achieved; state "
    "the evidence for it in the summary.\n"
    "Examples:\n"
    "History: step 1: search({\"query\": \"nvidia driver\"}) -> ok; "
    "title: 'nvidia driver at DuckDuckGo — Mozilla Firefox'; HINT: search "
    "results page is up.\n"
    "Next step: click {\"phrase\": \"the first result in the search "
    "results\"}\n"
    "History: step 2: click({\"phrase\": \"the first result in the search "
    "results\"}) -> ok; title unchanged; HINT: after a click that may "
    "navigate, use wait-load.\n"
    "Next step: wait-load {}\n"
    "History: step 5: key({\"combo\": \"enter\"}) -> ok; frame: 2.0% "
    "changed; title: 'Terminal — xfce4-terminal'\n"
    "Next step: done {\"summary\": \"...\"}"
)


class PlannerError(RuntimeError):
    """The planner produced no usable step (transport or format)."""


class Planner:
    """One planning call = one HTTP round trip to the LFM3 server.

    The full compact history is re-sent every call (goal + one line per
    executed step): at ~200 tokens/step a 15-step task is ~3K prompt
    tokens, which prefill takes ~1-2 s on the 1070 — the price of the
    planner being stateless.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url
                         or os.environ.get("LFM3_URL", "http://127.0.0.1:8090")
                         ).rstrip("/")
        self.timeout = timeout
        self.temperature = float(os.environ.get("LFM3_TEMPERATURE", "0.1"))
        self.top_k = int(os.environ.get("LFM3_TOP_K", "50"))
        self.max_tokens = int(os.environ.get("LFM3_MAX_TOKENS", "256"))
        self._proc: subprocess.Popen | None = None

    # ------------------------------------------------------------- server

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=3):
                return True
        except Exception:
            return False

    def _gpu_index(self) -> int:
        """The GTX 1070's CUDA index, found by name (see entrypoint.sh)."""
        override = os.environ.get("LFM3_GPU")
        if override:
            return int(override)
        import torch  # the serve process already has torch loaded

        for i in range(torch.cuda.device_count()):
            if "1070" in torch.cuda.get_device_properties(i).name:
                return i
        raise PlannerError(
            "no GTX 1070 in the CUDA enumeration "
            f"({[torch.cuda.get_device_properties(i).name
                 for i in range(torch.cuda.device_count())]}); "
            "set LFM3_GPU to pin a device")

    def _spawn(self) -> None:
        bin_path = os.environ.get(
            "LFM3_BIN", "/opt/llama.cpp/cuda-12.8/llama-server")
        model = os.environ.get(
            "LFM3_MODEL", "/root/models/LFM3-1.7B-Q4_K_M.gguf")
        port = os.environ.get("LFM3_PORT", "8090")
        ctx = os.environ.get("LFM3_CTX", "4096")
        if not os.path.exists(bin_path) or not os.path.exists(model):
            raise PlannerError(
                f"LFM3 artifacts missing: bin={bin_path!r} "
                f"model={model!r} (see README, orchestration)")
        # The build links against the CUDA 12.8 runtime, which this image
        # does not ship; the torch pip bundle provides the same-soname
        # 12.x libraries (libcudart.so.12, libcublas.so.12, ...).
        lib_dirs = [os.path.dirname(bin_path)]
        lib_dirs += sorted(set(
            glob.glob("/usr/local/lib/python3*/site-packages/nvidia/*/lib")
            + glob.glob("/usr/lib/python3*/site-packages/nvidia/*/lib")))
        env = {**os.environ,
               "LD_LIBRARY_PATH": os.pathsep.join(lib_dirs)}
        cmd = [
            bin_path, "--model", model,
            "--n-gpu-layers", "99",
            # --split-mode none: all layers on the main GPU. The default
            # 'layer' mode spreads layers across ALL visible GPUs — here
            # the two RTX 3060s are ~95% occupied by other workloads, so
            # a spread load OOMs on their first compute buffer (measured).
            "--split-mode", "none",
            "--main-gpu", str(self._gpu_index()),
            "-c", ctx,
            "--host", "127.0.0.1", "--port", port,
            "--no-ui",
            # jinja: the GGUF chat template drives tool-calling format.
            "--jinja",
        ]
        log = open("/tmp/lfm3-server.log", "ab")
        print(f"[planner] spawning {bin_path} (model={os.path.basename(model)}, "
              f"gpu={self._gpu_index()}, ctx={ctx}, port={port}); "
              f"log /tmp/lfm3-server.log", flush=True)
        self._proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
            start_new_session=True)

    def ensure_server(self, wait: float = 90.0) -> None:
        """Ready (health ok) or spawn and wait until it is."""
        if self.health():
            return
        if self._proc is not None and self._proc.poll() is None:
            pass  # already starting: just wait below
        else:
            self._spawn()
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.health():
                print("[planner] LFM3 server ready", flush=True)
                return
            if self._proc is not None and self._proc.poll() is not None:
                tail = ""
                try:
                    with open("/tmp/lfm3-server.log", "rb") as f:
                        tail = f.read()[-2000:].decode(errors="replace")
                except OSError:
                    pass
                raise PlannerError(
                    f"LLM server exited (code {self._proc.returncode}); "
                    f"log tail:\n{tail}")
            time.sleep(1.0)
        raise PlannerError(
            f"LLM server not ready within {wait:.0f} s; "
            "see /tmp/lfm3-server.log")

    def shutdown(self) -> None:
        """Stop the server IF we started it (free its ~1.25 GiB).

        A server we did not start (pre-existing, e.g. LFM3_PERSIST from an
        earlier run, or one started out-of-band) is left alone.
        """
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            print("[planner] LFM3 server stopped (VRAM released)", flush=True)
        self._proc = None

    def decompose(self, goal: str) -> list[str]:
        """One extra planning call: split the goal into short atomic
        subgoals (2-6, numbered).

        The 1.7B planner loses track of multi-part goals mid-run
        (measured: after the article opened — subgoal 2 of 6 — it kept
        re-clicking the result and polling observe instead of scrolling).
        A short numbered checklist in every prompt keeps it on track.
        Falls back to the whole goal as a single subgoal on any failure.
        """
        body = {
            "model": "LFM3",
            "messages": [
                {"role": "system", "content":
                 "Break the goal into 2-6 short atomic subgoals in "
                 "execution order. One numbered line per subgoal, each "
                 "under 15 words. Keep exact quoted phrases and commands "
                 "verbatim. Output only the numbered list."},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
            "max_tokens": 300,
            "temperature": 0.1,
            "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            data = self._post(body)
            text = data["choices"][0]["message"].get("content") or ""
            subs = []
            for line in text.splitlines():
                m = re.match(r"^\s*(?:\d+[\.\):]|[-*])\s*(.+)$",
                             line.strip())
                if m and m.group(1).strip():
                    subs.append(m.group(1).strip())
            if subs:
                return subs[:6]
            print(f"[planner] decompose gave no usable list "
                  f"({text[:120]!r}); using the goal as one subgoal",
                  flush=True)
        except Exception as e:
            print(f"[planner] decompose failed ({e!r}); using the goal "
                  f"as one subgoal", flush=True)
        return [goal.strip()]

    # ------------------------------------------------------------- plan

    def plan(self, goal: str, history: list[str]) -> tuple[str, dict]:
        """One planning call. Returns (tool_name, args).

        Retries: a transport failure re-checks the server (it may have
        died); a non-tool reply gets one corrective nudge.
        """
        hist = "\n".join(history) if history else "no steps executed yet"
        user = (f"Goal: {goal}\n\nHistory:\n{hist}\n\n"
                "What is the next step?")
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]
        body = {
            "model": "LFM3",
            "messages": messages,
            "tools": _TOOLS,
            "tool_choice": "required",   # every turn is a tool call
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        last_err: Exception | None = None
        for attempt in range(2):
            try:
                data = self._post(body)
            except urllib.error.HTTPError as e:
                err = e.read()[:300]
                # An old server without tool_choice support: retry auto.
                if e.code == 400 and b"tool_choice" in err:
                    body["tool_choice"] = "auto"
                    continue
                if e.code in (500, 502, 503) and self._proc is not None:
                    print("[planner] server error; re-ensuring", flush=True)
                    self._proc = None
                    self.ensure_server()
                last_err = PlannerError(
                    f"planner HTTP {e.code}: {err.decode(errors='replace')}")
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if self._proc is not None:
                    print("[planner] transport failure; re-ensuring "
                          "server", flush=True)
                    self._proc = None
                    self.ensure_server()
                last_err = PlannerError(f"planner transport error: {e}")
                continue
            except json.JSONDecodeError as e:
                last_err = PlannerError(f"planner bad JSON: {e}")
                continue

            msg = data["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # A bare-text reply: nudge once, citing what it said.
                messages.append({"role": "assistant",
                                 "content": msg.get("content") or ""})
                messages.append({"role": "user", "content":
                                 "That was not a tool call. Reply with "
                                 "exactly one tool call."})
                body["messages"] = messages
                last_err = PlannerError(
                    f"planner returned no tool call: "
                    f"{(msg.get('content') or '')[:200]!r}")
                continue

            name = tool_calls[0]["function"]["name"]
            raw = tool_calls[0]["function"].get("arguments")
            # OpenAI spec allows a JSON string; the card says json.loads()
            # it before use.
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except json.JSONDecodeError:
                args = {}
            return name, args

        raise PlannerError(f"planner failed after retries: {last_err}")

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)


class OrchestratedAgent:
    """Plan -> execute -> report loop over the controller vocabulary.

    Reuses Agent.execute for the controller verbs (grounding, search,
    open, wait-load — all the robustness machinery included); observe
    and verify are implemented here on top of the same worker (a
    free-form question to the vision model, answered in text).
    """

    def __init__(self, worker, planner: Planner,
                 dry_run: bool = False, delay: float = 0.2,
                 scope: str = "window", monitor=None, save_dir=None,
                 max_steps: int = 32, max_seconds: float = 180.0,
                 box_fallback: bool = True):
        from .controller import Agent

        self.planner = planner
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        # The inner agent is the only actor; it holds the worker and all
        # the deterministic prior machinery.
        self.agent = Agent(
            worker, dry_run=dry_run, delay=delay, max_steps=max_steps,
            scope=scope, monitor=monitor, save_dir=save_dir,
            box_fallback=box_fallback,
        )

    # ----------------------------------------------------------- sensing

    def _title(self) -> str:
        from .controller import _xdotool

        wid = _xdotool("getactivewindow")
        return _xdotool("getwindowname", wid).strip() if wid else ""

    def _windows_state(self) -> str:
        """Deterministic screen-state text: active title + visible windows.

        The LocateAnything worker is a GROUNDING model — open-ended
        questions come back as <ref>/<box> spam in ~36 s (measured), so
        observe/verify read the window manager state instead: instant,
        deterministic, and exactly what a text planner can reason about.
        (Page CONTENT is not visible here; only titles/names.)
        """
        from .controller import _window_class, _xdotool

        title = self._title()
        names: list[str] = []
        for wid in _xdotool("search", "--onlyvisible", "--name", ".").split():
            if not wid.isdigit():
                continue
            name = _xdotool("getwindowname", wid).strip()
            if name:
                names.append(name)
        # De-duplicate, keep order, cap (window-manager noise can repeat).
        seen, windows = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                windows.append(n)
            if len(windows) >= 10:
                break
        return f"active: {title!r}; windows: " + (
            "; ".join(windows) if windows else "(none visible)")

    def _observe(self, question: str) -> str:
        """Window-state report answering the planner's question (no model)."""
        return f"q: {question!r} -> {self._windows_state()}"

    def _verify(self, criterion: str) -> str:
        """Window-level yes/no check of a criterion (no model)."""
        state = self._windows_state()
        crit = criterion.lower().strip()
        # A criterion names a window/page when its text occurs verbatim
        # in a title; anything longer is content, which window state
        # cannot see.
        if crit and crit in state.lower():
            return f"window-level MATCH: criterion {criterion!r} is in " \
                   f"the visible state -> {state}"
        return (f"no window-level evidence for {criterion!r} (page "
                f"content is not visible to this check) -> {state}")

    # ------------------------------------------------------------- step

    def _exec(self, name: str, args: dict, i: int) -> tuple[bool, str]:
        """Run one planned step. Returns (ok, detail-for-history)."""
        from .controller import Step

        if name in ("click", "double-click", "right-click"):
            arg = str(args.get("phrase", "")).strip()
            if not arg:
                return False, "click needs a phrase"
            step = Step(verb=name, arg=arg, raw=f"{name} {arg}")
        elif name == "type":
            arg = str(args.get("text", ""))
            if not arg:
                return False, "type needs text"
            step = Step(verb="type", arg=arg, raw=f"type {arg}")
        elif name == "key":
            arg = str(args.get("combo", "")).strip()
            if not arg:
                return False, "key needs a combo"
            step = Step(verb="key", arg=arg, raw=f"key {arg}")
        elif name == "open":
            arg = str(args.get("app", "")).strip()
            if not arg:
                return False, "open needs an app"
            step = Step(verb="open", arg=arg, raw=f"open {arg}")
        elif name in ("search", "duckduckgo"):
            arg = str(args.get("query", "")).strip()
            # The planner copies the goal's decorative quotes into the
            # query (measured: that makes DDG phrase-search the whole
            # quoted string, changing the result set); strip one
            # surrounding pair.
            if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
                arg = arg[1:-1].strip()
            if not arg:
                return False, "search needs a query"
            step = Step(verb="search", arg=arg, raw=f"search {arg}")
        elif name == "scroll":
            arg = str(args.get("phrase", "")).strip()
            try:
                amount = int(args.get("amount", 0))
            except (TypeError, ValueError):
                amount = 0
            if not arg or amount == 0:
                return False, "scroll needs a phrase and a nonzero amount"
            step = Step(verb="scroll", arg=arg, amount=amount,
                        raw=f"scroll {arg} {amount}")
        elif name == "wait":
            try:
                seconds = float(args.get("seconds", 1.0))
            except (TypeError, ValueError):
                seconds = 1.0
            seconds = min(max(seconds, 0.0), 30.0)
            step = Step(verb="wait", arg=f"{seconds:g}", raw=f"wait {seconds:g}")
        elif name == "wait-load":
            step = Step(verb="wait-load", arg="", raw="wait-load")
        elif name == "observe":
            q = str(args.get("question", "")).strip()
            if not q:
                return False, "observe needs a question"
            return True, f"observe: {self._observe(q)}"
        elif name == "verify":
            c = str(args.get("criterion", "")).strip()
            if not c:
                return False, "verify needs a criterion"
            return True, f"verify: {self._verify(c)}"
        else:
            return False, f"unknown tool {name!r}"

        return self.agent.execute(i, step), ""

    # -------------------------------------------------------------- run

    def _scroll_amount(self, goal_block: str) -> int:
        """How many notches the goal wants scrolled (digit or word)."""
        m = re.search(r"scroll\b[^\n]*?(\d+)\s*notch", goal_block, re.I)
        if m:
            return int(m.group(1))
        words = {"one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10}
        m = re.search(r"scroll\b[^\n]*?\b(" + "|".join(words) + r")\b\s*notch",
                      goal_block, re.I)
        if m:
            return words[m.group(1).lower()]
        return 1

    def _typed_command(self, goal_block: str) -> str | None:
        """The exact command the goal says to type, if it quotes one.

        Typed text is verbatim-sensitive (it lands in a file), so when
        the goal spells out a command the deterministic layer owns the
        typing: anchored on the 'type ... command:' phrasing, ending at
        an em-dash or end of line; fallback: a shell-verb scan. None
        when the goal does not specify a command (the planner types
        freely).
        """
        m = re.search(r"type\s+(?:exactly\s+)?(?:this\s+)?command"
                      r"\s*[:\-]?\s*([^\n—]+)", goal_block, re.I)
        if m:
            cmd = m.group(1).strip()
        else:
            m = re.search(r"\b(echo|printf|cat)\b([^\n—]+)", goal_block)
            if not m:
                return None
            cmd = (m.group(1) + m.group(2)).strip()
        cmd = cmd.rstrip(".").strip("`").strip()
        words = cmd.split()
        if len(words) < 2 or not re.match(r"^[a-z0-9_./-]+$", words[0]):
            return None
        return cmd

    def _goal_block(self, goal: str, subgoals: list[str],
                    marks: list[bool]) -> str:
        """The goal as the planner sees it, on every planning call.

        Raw goal + the subgoal list with DETERMINISTIC completion marks
        (measured state, never the planner's claims). The 1.7B planner
        loses multi-part goals mid-run (measured: it re-searched after
        the scroll was done), so an external progress tracker in every
        prompt keeps it on the current subgoal.
        """
        next_n = next((n for n, m in enumerate(marks) if not m), None)
        lines = []
        for n, (s, m) in enumerate(zip(subgoals, marks), 1):
            tag = "[DONE]" if m else ("[NEXT]" if n == next_n else "[ ]")
            lines.append(f"{n}. {tag} {s}")
        block = ("Goal: " + goal + "\nSubgoals (work through them in "
                 "order):\n" + "\n".join(lines))
        if next_n is None:
            block += ("\nAll subgoals are done — call done with the "
                      "evidence (titles, frame changes, window state).")
        return block

    def _mark_subgoals(self, subgoals: list[str], marks: list[bool],
                       name: str, ok: bool, args: dict,
                       post: str) -> bool:
        """Advance the deterministic progress tracker after a step.

        Each mark fires only on measured evidence matching the subgoal's
        semantics (a DDG title for a search subgoal, a non-DDG title
        for 'open the result', the terminal title for terminal
        subgoals, a type/key-executed in the terminal for the command
        subgoals). Conservative on purpose: a missed mark lets the
        planner redo the subgoal (capped by the repeat guard); a false
        mark would make it skip one.
        """
        changed = False
        if not ok:
            return changed
        args_str = str(args).lower()
        for idx, sub in enumerate(subgoals):
            if marks[idx]:
                continue
            s = sub.lower()
            terminal_up = "terminal" in post.lower()
            if "terminal" in s:
                # 'type'/'echo' only: 'run the command' belongs to the
                # enter branch, not the type branch.
                if (("type" in s or "echo" in s)
                        and name == "type" and terminal_up):
                    marks[idx] = changed = True
                elif (("enter" in s or "execute" in s or "run" in s)
                        and name == "key" and "enter" in args_str
                        and terminal_up):
                    marks[idx] = changed = True
                elif name == "open" and terminal_up:
                    marks[idx] = changed = True
            elif "result" in s and "duckduckgo" not in post.lower():
                marks[idx] = changed = True
            # 'result' subgoals are excluded: 'open the first result
            # from the search' must not be marked by the search itself.
            elif "result" not in s and "search" in s \
                    and name in ("search", "duckduckgo") \
                    and "duckduckgo" in post.lower():
                marks[idx] = changed = True
            elif "scroll" in s and name == "scroll" \
                    and "duckduckgo" not in post.lower():
                marks[idx] = changed = True
            # The decomposer drops context from subgoals (measured:
            # 'Type exactly: `echo ...`' with no word 'terminal'), so
            # type/enter subgoals are marked by WHERE the step ran —
            # the measured terminal title — not the subgoal's words.
            elif (("type" in s or "echo" in s)
                    and name == "type" and terminal_up):
                marks[idx] = changed = True
            elif (("enter" in s or "execute" in s or "run" in s)
                    and name == "key" and "enter" in args_str
                    and terminal_up):
                marks[idx] = changed = True
        return changed

    def _suggestion(self, name: str, ok: bool, pre: str, post: str,
                    goal_block: str,
                    all_done: bool = False) -> tuple[str, tuple[str, dict] | None] | None:
        """The canonical next step, suggested by the deterministic layer.

        Returns (history_line, (tool_name, args) | None). The machine
        part is the step the system will itself execute if the planner
        deviates twice (see _run_loop): the 1.7B planner is not reliable
        on canonical micro-transitions (measured: it repeated a finished
        search 3x across runs, fixated on re-clicking the first result
        even after the article had opened, and never called done even
        with every subgoal marked [DONE]), but each suggestion only
        fires on a MEASURED state signature (a DDG title means the
        controller's verified _search finished; a terminal title means
        _open verified focus; all-done means every subgoal was marked
        on measured evidence), so following one is always safe — while
        the planner keeps full freedom to override with evidence (that
        is what makes it a planner, not a script). The chain covered:
        search->click->wait-load->scroll->open terminal->type->enter
        ->done, selected from the goal's own words (scroll count,
        'terminal', the quoted command).
        """
        if not ok:
            return None
        if all_done:
            # Every subgoal is marked on MEASURED evidence, so the goal
            # is achieved by construction; the marks ARE the done
            # evidence. The planner still gets two turns to disagree
            # (e.g. observe) before the system stops itself.
            subs = [m.group(1) for m in re.finditer(
                r"^\d+\. \[DONE\] (.+)$", goal_block, re.M)]
            summary = "all subgoals done: " + "; ".join(subs) if subs \
                else "all subgoals done"
            return ("SUGGESTED NEXT: done — every subgoal is [DONE] "
                    "(verified by titles, frame changes and window "
                    "state). Call done. Override only with evidence a "
                    "subgoal actually failed.",
                    ("done", {"summary": summary}))
        low = goal_block.lower()
        results_page = "duckduckgo" in post.lower()
        if name in ("search", "duckduckgo") and results_page:
            if re.search(r"\bresult", goal_block, re.I):
                return ("SUGGESTED NEXT: click "
                        "{\"phrase\": \"the first result in the search "
                        "results\"}, then wait-load (the results page is "
                        "up and the goal wants a result). Override only "
                        "if you have evidence the page is not a results "
                        "page.",
                        ("click",
                         {"phrase": "the first result in the search results"}))
        if name == "click":
            return ("SUGGESTED NEXT: wait-load {} (the click may be "
                    "navigating; wait for the new page before acting). "
                    "Override only if you have evidence nothing "
                    "navigated.",
                    ("wait-load", {}))
        if name == "wait-load":
            if results_page and re.search(r"\bresult", goal_block, re.I):
                return ("SUGGESTED NEXT: click "
                        "{\"phrase\": \"the first result in the search "
                        "results\"} (the results page is still up and "
                        "the goal wants a result).",
                        ("click",
                         {"phrase": "the first result in the search results"}))
            if re.search(r"\bscroll", low):
                n = self._scroll_amount(goal_block)
                args = {"phrase": "the article", "amount": n}
                return (f"SUGGESTED NEXT: scroll {json.dumps(args)} "
                        f"(the page has settled and the goal wants the "
                        f"article scrolled {n} notches).",
                        ("scroll", args))
            return ("SUGGESTED NEXT: continue with the next unfinished "
                    "subgoal (the page has settled).", None)
        if name == "scroll":
            if re.search(r"\bterminal\b", low) and "terminal" not in post.lower():
                return ("SUGGESTED NEXT: open {\"app\": \"terminal\"} "
                        "(the goal wants a terminal; scrolling is done).",
                        ("open", {"app": "terminal"}))
            return ("SUGGESTED NEXT: continue with the next unfinished "
                    "subgoal.", None)
        if name == "open" and "terminal" in post.lower():
            cmd = self._typed_command(goal_block)
            if cmd:
                return ("SUGGESTED NEXT: type "
                        + json.dumps({"text": cmd})
                        + " verbatim into the terminal (the goal gives "
                          "the exact command), then key enter.",
                        ("type", {"text": cmd}))
            return ("SUGGESTED NEXT: type the command the goal specifies "
                    "into the terminal, then key enter.", None)
        if name == "type" and "terminal" in post.lower() \
                and re.search(r"\benter\b", low):
            return ("SUGGESTED NEXT: key {\"combo\": \"enter\"} (execute "
                    "the typed command).",
                    ("key", {"combo": "enter"}))
        return ("SUGGESTED NEXT: continue with the next unfinished "
                "subgoal.", None)

    def run(self, goal: str) -> bool:
        t0 = time.time()
        print(f"[orch] goal: {goal!r} "
              f"({self.max_steps} step cap, {self.max_seconds:.0f} s cap)",
              flush=True)
        # Decompose once (one ~1.5 s call): the numbered checklist goes
        # into every planning prompt with deterministic completion marks
        # (see _goal_block / _mark_subgoals) — the 1.7B planner loses
        # multi-part goals mid-run, so progress is tracked outside it.
        subgoals = self.planner.decompose(goal)
        print("[orch] subgoals:\n"
              + "\n".join(f"  {n}. {s}"
                          for n, s in enumerate(subgoals, 1)), flush=True)
        history = [f"initial state: title: {self._title()!r}"]
        if self.agent.dry_run:
            # Dry runs never change the screen, so goal evidence can
            # never arrive: tell the planner the contract (one simulated
            # first step, then the orchestrator stops deterministically).
            history.append(
                "NOTE: this is a DRY RUN — steps are simulated and the "
                "screen will not change. Pick the single correct first "
                "step for the goal; after it runs, the orchestrator "
                "stops.")
        persist = os.environ.get("LFM3_PERSIST", "0") == "1"
        try:
            return self._run_loop(goal, subgoals, history, t0,
                                  seen_steps={})
        finally:
            # Release the planner's VRAM share unless the user opted to
            # keep it warm: the worker's model reload needs the headroom
            # (see the module docstring).
            if not persist:
                self.planner.shutdown()
            else:
                print("[orch] LFM3_PERSIST=1: keeping the LFM3 server "
                      "running (~1.25 GiB on the shared card)", flush=True)

    def _run_loop(self, goal: str, subgoals: list[str],
                  history: list[str], t0: float, seen_steps: dict) -> bool:
        consec_fail = 0
        # Suggestion protocol state: the deterministic layer's pending
        # machine suggestion (see _suggestion) and how many times the
        # planner has deviated from it.
        pending: tuple[str, dict] | None = None
        deviations = 0
        # Deterministic subgoal progress (see _mark_subgoals); rebuilt
        # into the goal block before every planning call.
        marks = [False] * len(subgoals)
        for i in range(self.max_steps):
            if time.time() - t0 > self.max_seconds:
                print(f"[orch] budget of {self.max_seconds:.0f} s exceeded; "
                      f"stopping at step {i + 1}", flush=True)
                return False
            goal_block = self._goal_block(goal, subgoals, marks)
            try:
                name, args = self.planner.plan(goal_block, history)
            except PlannerError as e:
                print(f"[orch] planner error at step {i + 1}: {e}", flush=True)
                return False
            # Suggestion protocol: after a measured state signature the
            # deterministic layer proposes a concrete next step (pending).
            # The planner may override it with evidence, but after two
            # deviations from the same pending suggestion the system
            # executes the suggested step itself — canonical transitions
            # are controller-verified, so following them is always safe
            # (measured: the 1.7B planner repeated a finished search 3x
            # across runs, ignoring prose hints). A follow resets the
            # count. 'done' is a planner decision and is never overridden.
            follow = overridden = False
            if pending is not None and name != "done":
                # Compare canonical (name, args-json) pairs: pending is
                # stored as (name, dict), so compare both sides as
                # strings (measured: comparing the dict side directly
                # made every FOLLOW count as a deviation, ratcheting the
                # override into the repeat guard — run 9).
                p_name, p_args = pending
                if (name, json.dumps(args, sort_keys=True,
                                     ensure_ascii=False)) == \
                        (p_name, json.dumps(p_args, sort_keys=True,
                                            ensure_ascii=False)):
                    follow = True
                else:
                    deviations += 1
                    if deviations >= 2:
                        overridden = True
                        p_name, p_args = pending
                        print(f"[orch] step {i + 1}: planner deviated twice "
                              f"from the suggested step; executing the "
                              f"suggestion instead ({p_name} "
                              f"{json.dumps(p_args, ensure_ascii=False)})",
                              flush=True)
                        name, args = p_name, p_args
            print(f"[orch] step {i + 1}: {name} "
                  f"{json.dumps(args, ensure_ascii=False)}", flush=True)
            if name == "done":
                # done is a TERMINATION signal, not an action: no
                # execution, no frame diff. The summary is the planner's
                # (or the suggestion's) evidence statement.
                print(f"[orch] done: {args.get('summary', '')} "
                      f"({time.time() - t0:.1f} s total, {i + 1} steps)",
                      flush=True)
                return True
            t = time.time()
            pre = self._title()
            # Repeat guard: the same step in the same screen state cannot
            # do anything new — the planner can be stuck repeating a step
            # that "succeeded" (or failed) without changing the world
            # (measured: 26x 'open terminal' in a dry run). A repeated step
            # is executed only if the state moved since the last attempt.
            key = (name, json.dumps(args, sort_keys=True,
                                    ensure_ascii=False))
            if key in seen_steps and seen_steps[key] == pre:
                blocked = True
                ok, detail = False, (
                    "repeated: the screen is identical to the last "
                    "attempt, so repeating cannot help — observe, or "
                    "move on to the next unfinished subgoal.")
                post = pre
                frame_pct = "0.0 (skipped)"
            else:
                blocked = False
                # Pre/post frame diff: the planner's "did anything
                # visible happen?" signal (0-100%, 320x180 gray diff —
                # the controller's own settle machinery, see
                # controller._frame_diff). A 0% result on an "ok" step
                # is a no-op in disguise (e.g. a click that grounded
                # but hit dead space).
                from . import screen
                from .controller import _frame_diff

                pre_cap = screen.capture(scope=self.agent.scope,
                                         monitor_index=self.agent.monitor).image
                ok, detail = self._exec(name, args, i)
                post_cap = screen.capture(scope=self.agent.scope,
                                          monitor_index=self.agent.monitor).image
                frame_pct = f"{100.0 * _frame_diff(pre_cap, post_cap) / 255.0:.1f}"
                post = self._title()
                seen_steps[key] = post
            dt = time.time() - t
            line = (f"step {i + 1}: {name}({json.dumps(args, ensure_ascii=False)}) "
                    f"-> {'ok' if ok else 'FAILED'} in {dt:.1f}s; "
                    f"frame: {frame_pct}% changed; "
                    f"title: {post[:120]!r}")
            if post == pre:
                line += " (title unchanged)"
            if detail:
                line += f" | {detail}"
            if not ok:
                line += " | NOTE: the step failed; the screen may not have " \
                        "changed; observe or try a different approach"
                consec_fail += 1
            else:
                consec_fail = 0
            history.append(line)
            print("    " + line, flush=True)
            # Update the pending suggestion from the step just run.
            if blocked:
                # The world did not move. If the blocked step WAS the
                # pending suggestion (a blocked follow or a blocked
                # override), the suggestion is proven stale in this
                # state: drop it and reset the deviation ratchet —
                # otherwise the system re-executes the same blocked step
                # every iteration until the failure budget burns
                # (measured: run 9, three blocked wait-loads in a row).
                # A blocked step that is NOT the suggestion leaves the
                # suggestion in place: the world did not move, so the
                # previous suggestion is still the right next step.
                if follow or overridden:
                    pending = None
                    deviations = 0
            elif follow or overridden or post != pre or float(frame_pct) > 0.0:
                # The state signature moved (or the suggestion was just
                # fulfilled): supersede it with a fresh one from the new
                # state. Diagnostic no-ops (observe, verify, wait, a
                # failed step) leave the pending suggestion in place.
                if follow or overridden:
                    deviations = 0
                self._mark_subgoals(subgoals, marks, name, ok, args, post)
                # The fresh block (post-step marks) is what the planner
                # will see next turn, and it is what the done-summary
                # regex reads [DONE] lines from.
                sug = self._suggestion(name, ok, pre, post,
                                       self._goal_block(goal, subgoals,
                                                        marks),
                                       all_done=all(marks))
                if sug is None:
                    pending = None
                elif sug[1] is None:
                    history.append(sug[0])
                    pending = None
                else:
                    sug_line, machine = sug
                    m_key = (machine[0], json.dumps(machine[1],
                                                    sort_keys=True,
                                                    ensure_ascii=False))
                    if seen_steps.get(m_key) == post:
                        # The suggested step was already executed in this
                        # exact state: the repeat guard would block it,
                        # so it is stale. Anchoring the planner on a step
                        # that cannot run traps it (measured: the system
                        # re-executed a blocked wait-load 3x and burned
                        # the failure budget).
                        pending = None
                        history.append(
                            "SUGGESTED NEXT: continue with the next "
                            "unfinished subgoal (no canonical transition "
                            "is pending).")
                    else:
                        history.append(sug_line)
                        pending, deviations = machine, 0

            if self.agent.dry_run and i == 0:
                # A dry run never changes the world, so "achieve the goal"
                # can never be evidenced; the measured failure mode is the
                # planner polling forever instead of accepting that. The
                # dry-run contract is deterministic: show the first move,
                # then stop (a real run is where the goal gets achieved).
                print(f"[orch] dry run: first move shown "
                      f"({name} {json.dumps(args, ensure_ascii=False)}); "
                      f"stopping (dry-run contract)", flush=True)
                return True
            if consec_fail >= 3:
                print(f"[orch] {consec_fail} consecutive failures; stopping",
                      flush=True)
                return False
        print(f"[orch] max_steps={self.max_steps} reached without done",
              flush=True)
        return False