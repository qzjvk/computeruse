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
                        % (320x180 gray diff), observe/verify report
                        window-manager state, and the `read` verb
                        closes the content gap: the OS accessibility
                        tree (atspi.py) dumps the rendered text,
                        prices and links of the current window, so a
                        text planner can see page content (measured,
                        non-LLM source).

Suggestion fast-path (the latency design): after a measured state
signature (results page up, page settled, terminal focused) the
deterministic layer proposes a concrete next machine step whose args
are extracted from the goal's own words (scroll count, "terminal", the
quoted command). When such a pending suggestion exists the system
EXECUTES IT DIRECTLY — no planning round trip — because it is
controller-verified and goal-faithful by construction (measured: the
1.7B planner deviated from nearly every canonical suggestion in the
9-step goal; consulting it per transition cost ~15 s of pure latency).
The planner keeps full authority over every turn WITHOUT a pending
suggestion — goal start, states outside the canonical chain, recovery
after a failed step — and sees every executed result in the history.
``done`` is suggested by the system once every subgoal is marked on
measured evidence (the 1.7B never calls done on its own).

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
        "name": "read",
        "description": "Dump the rendered text of the current window from "
                        "the OS accessibility tree: page title, search "
                        "results (title + URL each), body text, a "
                        "'money:' line of prices, and link names. This is "
                        "the ONLY way to see page CONTENT (titles and "
                        "frame-diff cannot). Use it after any navigation "
                        "or whenever the goal needs facts from the page. "
                        "It changes nothing on screen.",
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
    "Content and research:\n"
    "- read is the only way to see page CONTENT (window titles and the "
    "frame-diff cannot). After a navigation that should show content "
    "(a search results page, an article, a product page), the next step "
    "is usually read. Its output (title, results with URLs, body text, "
    "a 'money:' line of prices, link names) is carried in the history — "
    "use it as facts; do not re-read an unchanged page.\n"
    "- For comparison/research goals: search, read the results, open the "
    "promising sources ONE BY ONE (click a result, wait-load, read), and "
    "keep every fact with its source (e.g. 'OLX: ASUS RTX 3060 12GB 650 "
    "euro') in your memory across steps. Return to the results list with "
    "key ctrl+Left (browser Back); a new search is fine too. Compare "
    "across at least as many sources as the goal names before deciding.\n"
    "- For such goals the done summary IS the deliverable: state the "
    "answer with every compared fact and its source name, not just that "
    "you finished.\n"
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


def set_planner_ctx(ctx: int) -> None:
    """Point the next planner spawn at this context window.

    A running server keeps its own ctx (llama-server is spawned once
    and reused via health), so if one is already up it is killed first:
    the next ensure_server respawns it with LFM3_CTX set. Needed for
    research runs whose history carries read dumps (8192) on top of
    the default 4096.
    """
    os.environ["LFM3_CTX"] = str(ctx)
    port = os.environ.get("LFM3_PORT", "8090")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=2):
            pass
    except Exception:
        return  # nothing up: nothing to restart
    # The caller's own cmdline never contains 'llama-server', so the
    # pattern cannot hit us.
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(1.0)


# Keywords that put a subgoal on the mark surface (see _mark_subgoals):
# a subgoal is markable iff its words can fire at least one mark
# branch. Lines outside this surface are decomposer artifacts —
# meta-prose ('Formulate query string: ...'), alternatives the goal
# never named ('Use a command-line tool like `curl`...'), or
# verification the system already does — and no controller step can
# ever mark them, so leaving one in the checklist blocks all_done
# forever (measured: T3, runs 3 and 4 — 'Formulate query string' and
# 'Use DuckDuckGo search to find results (maybe via command...)'
# instead of atomic actions). Word-boundary matching: 'run' must not
# match 'runtime'.
_MARKABLE_RE = re.compile(
    r"\b(search|results?|scroll|terminal|type|echo|enter|press|"
    r"execute|run)\b")


def _subgoal_markable(sub: str) -> bool:
    """Whether some controller step could mark this subgoal.

    The mark branches are the system's done-surface: a line the
    system cannot verify is not an action the system can execute, so
    it is dropped at decompose time instead of blocking completion.
    """
    s = sub.lower()
    if _MARKABLE_RE.search(s):
        return True
    # The verify-article mark branch: verify/check/confirm +
    # article/page.
    return bool(re.search(r"\b(verify|check|confirm)\b", s)
                and re.search(r"\b(article|page)\b", s))


# Decomposer meta prose: the 1.7B reasons out loud INSIDE a subgoal
# line instead of writing the action (measured: the 2026-11-01
# battery re-run — 'This is ambiguous but we need to treat as an
# action: maybe use a tool like browser scroll', 'We'll assume we
# need to get the content', 'But we are told "Only the goal's own
# actions…"'). Any such line is commentary, not an atomic action.
_META_PROSE_RE = re.compile(
    r"(i\.e\.|this is ambiguous|we need to|we cannot|we can't|"
    r"we are told|we'?ll assume|the goal says|maybe use|perhaps|"
    r"one action|another action)", re.I)


def _subgoal_garbage_reason(sub: str) -> str | None:
    """Why a decompose line is decomposer garbage — None when it is a
    usable atomic action.

    A garbage line left in the checklist blocks all_done: it either
    can never be marked (meta prose carries no action) or misfires
    the mark logic on its stray words (measured: '…two lines? or
    scroll within page…' parked a scroll subgoal in the
    counted-source branch, which waits for READ source pages the task
    never reads — the run then burned its step budget re-doing a
    chain it had already completed). Two measurable signatures:
    self-referential commentary, and a line cut off mid-token by the
    decode's token budget (unbalanced parens/quotes/backticks —
    measured: 'In terminal, run `echo "T6: attach'). Dropping loses
    nothing the machine needs: the goal's full text stays in every
    prompt, the canonical chain still performs the action, and the
    command-coverage check re-adds a dropped command line.
    """
    m = _META_PROSE_RE.search(sub)
    if m:
        return f"meta prose {m.group(0)!r}"
    if (sub.count("(") != sub.count(")") or sub.count("`") % 2
            or sub.count('"') % 2):
        return "truncated mid-line"
    return None


def _is_done_directive(s: str) -> bool:
    """True if the subgoal is the 'call done' instruction itself, not an
    action. The decomposer sometimes lifts the goal's terminal clause into
    a subgoal ('Call done when everything is complete.' / 'Done when the
    file is created.'). The system calls done when every ACTION subgoal is
    marked on measured evidence, so such a line is unsatisfiable by any
    controller step and would block the all-done signal forever
    (measured: run 15).
    """
    t = s.strip().lower()
    return bool(re.search(r"\bcall\s*done\b|\bdone\s*when\b|\bmark\s*done\b", t)
                or t.startswith("done"))


def _typed_command_of(goal_text: str) -> str | None:
    """The exact command the goal says to type, if it quotes one.

    Typed text is verbatim-sensitive (it lands in a file), so when the
    goal spells out a command the deterministic layer owns the typing:
    anchored on the 'type ... command:' phrasing, ending at an em-dash
    or end of line; fallback: a shell-verb scan. None when the goal does
    not specify a command (the planner types freely).

    Module-level so the decomposer can run the same coverage check (a
    goal that quotes a command needs a subgoal that carries it).
    """
    m = re.search(r"type\s+(?:exactly\s+)?(?:this\s+)?command"
                  r"\s*[:\-]?\s*([^\n—]+)", goal_text, re.I)
    if m:
        cmd = m.group(1).strip()
    else:
        m = re.search(r"\b(echo|printf|cat)\b([^\n—]+)", goal_text)
        if not m:
            return None
        cmd = (m.group(1) + m.group(2)).strip()
    cmd = cmd.rstrip(".").strip("`").strip()
    words = cmd.split()
    if len(words) < 2 or not re.match(r"^[a-z0-9_./-]+$", words[0]):
        return None
    return cmd


def _subgoal_commands(sub: str) -> list[str]:
    """Every command a subgoal quotes, if it asks to type/run one.

    The decomposer may merge several commands into one subgoal
    (measured: T5 — both commands of the goal in a single line), so the
    mark logic checks them all. It also keeps the command verbatim but
    varies the framing (measured: T5 — 'type exactly this command: …'
    in one run, 'type exactly: …' with the word 'command' dropped in
    the next), so the primary 'type … command:' clause is backed by a
    backtick-span pass: a quoted span of two or more words with a
    shell-ish first word IS the command. Search phrases in backticks
    never reach here: the callers only invoke this from the type/enter
    mark branches, which require a type/run/enter word in the subgoal
    AND the terminal up.
    """
    cmds = []
    for m in re.finditer(r"type\s+(?:exactly\s+)?(?:this\s+)?command"
                         r"\s*[:\-]?\s*([^\n—]+)", sub, re.I):
        cmd = m.group(1).strip().rstrip(".").strip("`").strip()
        if len(cmd.split()) >= 2:
            cmds.append(cmd)
    if not cmds:
        for m in re.finditer(r"`([^`\n]+)`", sub):
            cmd = m.group(1).strip().rstrip(".").strip()
            words = cmd.split()
            if len(words) >= 2 and re.match(r"^[a-z0-9_./-]+$",
                                            words[0]):
                cmds.append(cmd)
    return cmds


def _frame_sig(img) -> str:
    """A coarse identity for a screen capture.

    md5 of a 32x18 grayscale grid quantized to 8 levels: it moves on a
    real repaint (a typed character, a page change) and holds under
    antialiasing jitter. It is the second half of the loop's state
    identity, (title, sig): a constant window title (a terminal's
    'Terminal -') says nothing about whether the screen moved —
    measured: T9's enter chain stalled because three distinct states
    (each command typed) all read as the same title, so the repeat
    guard blocked the fourth canonical enter and the planner restarted
    the chain from step one.
    """
    import hashlib

    g = img.convert("L").resize((32, 18))
    return hashlib.md5(g.point(lambda v: v // 32).tobytes()).hexdigest()


def _subgoal_typed(sub: str, typed: list[str]) -> bool:
    """Whether every command this subgoal quotes was actually typed.

    A 'type … — press Enter' subgoal is marked at the type/enter step
    only if ITS command(s) are in the typed-text history: with
    multi-command goals the first Enter must not mark the second
    command's subgoal (measured: T5 — 'echo second-line' was never
    typed, but the merged subgoal's 'press Enter' clause was marked by
    the first Enter and all_done fired with one command unrun). A
    subgoal that quotes no command passes: there the type/enter step
    in the terminal IS the evidence.
    """
    cmds = _subgoal_commands(sub)
    if not cmds:
        return True
    out = [t.lower() for t in typed]
    for c in cmds:
        c = c.lower()
        if not any(c in t
                   or all(w in t for w in c.split() if len(w) >= 3)
                   for t in out):
            return False
    return True


def _goal_commands_typed(goal: str, typed: list[str]) -> bool:
    """Whether every command the goal explicitly quotes is in the
    typed-text history.

    all_done must not fire while a quoted command is untyped, even if
    every checklist subgoal is marked: the decomposer can list fewer
    command subgoals than the goal quotes, and the marks only track
    the LISTED ones (measured necessity: the 17-command T9 stress
    goal — a truncated checklist would end the run early). Only the
    explicit 'type … command:' phrasing counts: the echo/printf/cat
    fallback can match prose ('search for the echo command') the goal
    never asks to type.
    """
    if not re.search(r"type\s+(?:exactly\s+)?(?:this\s+)?command",
                     goal, re.I):
        return True
    out = [t.lower() for t in typed]
    for m in re.finditer(r"type\s+(?:exactly\s+)?(?:this\s+)?command"
                         r"\s*[:\-]?\s*([^\n—]+)", goal, re.I):
        cmd = m.group(1).strip().rstrip(".").strip("`").strip().lower()
        if len(cmd.split()) < 2:
            continue
        if not any(cmd in t or all(w in t for w in cmd.split()
                                   if len(w) >= 3)
                   for t in out):
            return False
    return True


def _goal_commands_run(goal: str, history: list[str], typed: list[str]) -> bool:
    """Whether every command the goal explicitly quotes was typed AND —
    when the goal asks for Enter — actually executed.

    Extends _goal_commands_typed: all_done must not fire while a quoted
    command is only TYPED, not run. The decomposer can drop the "press
    Enter" clause from a command subgoal (measured: T2, the subgoal
    ended at the em-dash, so the mark logic — which marks a
    type+enter subgoal only at the Enter step — marked it at the type
    step, and the run ended with the command sitting on the terminal
    prompt, file never written). The check is positional on the EXECUTED
    history (measured ok steps, never planner claims): for each quoted
    command, in goal order, an ok Enter step between its ok type step
    and the next command's type step (unbounded for the last).
    """
    if not _goal_commands_typed(goal, typed):
        return False
    if not re.search(r"\benter\b", goal, re.I):
        return True
    events = []  # (step_no, "type"|"enter", normalized text)
    for line in history:
        m = re.match(r"^step (\d+): (type|key)\((\{.*?\})\) -> (ok|FAILED)",
                     line)
        if not m or m.group(4) != "ok":
            continue
        try:
            a = json.loads(m.group(3))
        except ValueError:
            continue
        if m.group(2) == "type":
            events.append((int(m.group(1)), "type",
                           str(a.get("text", "")).lower()))
        elif str(a.get("combo", "")).lower() == "enter":
            events.append((int(m.group(1)), "enter", None))
    cmds = []
    for m in re.finditer(r"type\s+(?:exactly\s+)?(?:this\s+)?command"
                         r"\s*[:\-]?\s*([^\n—]+)", goal, re.I):
        cmd = m.group(1).strip().rstrip(".").strip("`").strip().lower()
        if len(cmd.split()) >= 2:
            cmds.append(cmd)
    type_nos = []
    last = 0
    for cmd in cmds:
        tn = next((no for no, verb, txt in events
                   if no > last and verb == "type"
                   and (cmd in txt or all(w in txt for w in cmd.split()
                                          if len(w) >= 3))), None)
        if tn is None:
            return False
        type_nos.append(tn)
        last = tn
    enter_nos = [no for no, verb, _ in events if verb == "enter"]
    for i, tn in enumerate(type_nos):
        upper = type_nos[i + 1] if i + 1 < len(type_nos) else None
        if not any(tn < en < (upper if upper is not None else float("inf"))
                   for en in enter_nos):
            return False
    return True


def _wants_a_specific_result(goal_block: str) -> bool:
    """Does the goal ask to open ONE specific result — 'the first
    result', 'the top result', 'the result' — vs. research/comparison
    goals that want SEVERAL sources read one by one?

    Only the former may take the fast-path suggestion 'click the first
    result after a search'. A research goal must READ the results page
    first (the planner's prior does that); pre-empting it with a click
    skips the read and ends the run on one unexamined page (measured:
    the smoke run — search, click result 2, wait-load, done in 13 s,
    no facts collected).
    """
    return bool(re.search(
        r"\b(first|top|best)\b[^.\n]{0,40}\bresults?|\bthe result\b",
        goal_block, re.I))


# "open at least 3 Portuguese sources": the count the subgoal itself
# names (goal-derived; the fast path already works this way — scroll
# counts, quoted commands). N is a digit or a word number (the
# decomposer writes 'three' — measured: the T10 run) and adjectives may
# sit between it and the noun ('three Portuguese sources'), so scan a
# small window before the noun. N=0 is NOT a counted subgoal; the plain
# mark branches handle it.
_COUNT_NOUN_RE = re.compile(
    r"\b(?:sources?|results?|pages?|sites?|links?|websites?|stores?)")
_WORD_NUMS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
    "dois": 2, "três": 3, "tres": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
}


def _counted_open(s: str) -> int:
    # Every noun occurrence is checked (a subgoal can carry an
    # UNcounted noun plus the counted one — 'read the results page
    # and pick at least three sources': 'results' has no number in
    # its window, 'sources' does — measured: the T10 run), and the
    # max wins.
    best = 0
    for m in _COUNT_NOUN_RE.finditer(s):
        for tok in s[:m.start()].split()[-6:]:
            tok = tok.strip(".,;:'\"()").lower()
            if tok.isdigit():
                best = max(best, int(tok))
                break
            if tok in _WORD_NUMS:
                best = max(best, _WORD_NUMS[tok])
                break
    return best


_SOURCE_OPEN_VERB_RE = re.compile(
    r"\b(open|visit|read|pick|search|browse|check|identify|find|look)\b",
    re.I)


def _counted_sources(s: str) -> int:
    """The number a counted-source subgoal asks for — 0 when the
    subgoal is not one.

    The counted branch parks a subgoal until N distinct source pages
    are READ, so its entry condition must be the counted-source
    signature and nothing else. Two measured misfires parked other
    actions forever (battery re-run 2026-11-01: T2/T4/T6 burned their
    step budgets re-doing a chain they had already completed, because
    the branch owned their scroll subgoals): the 1.7B's meta prose
    '…two lines? or scroll within page…' (a loose 6-word number
    window caught 'two' for 'page'), and '…scroll to third page…'
    (caught 'third'). So the signature needs (a) a number bound to a
    source noun — within 3 words, close enough to count THAT noun
    ('three Portuguese sources'), far enough from a stray 'within
    page' — and (b) an opening verb, because the branch waits for
    opened/read sources: a scroll that merely mentions a page opens
    nothing. 'Identify at least three Portuguese sources' and 'pick
    3 results' pass both; the garbage fails one and falls through to
    its own action branch.
    """
    if not _SOURCE_OPEN_VERB_RE.search(s):
        return 0
    best = 0
    for m in _COUNT_NOUN_RE.finditer(s):
        for tok in s[:m.start()].split()[-3:]:
            tok = tok.strip(".,;:'\"()").lower()
            if tok.isdigit():
                best = max(best, int(tok))
                break
            if tok in _WORD_NUMS:
                best = max(best, _WORD_NUMS[tok])
                break
    return best


def _distinct_page_titles(titles_seen: set[str] | None) -> int:
    """Distinct visited PAGE titles (measured signature for a counted
    source subgoal): exclude the search-results page (DDG), the
    terminal, and the bare browser-chrome titles — about:blank's
    title is 'Mozilla Firefox' and a new tab's is 'New Tab' (measured),
    and the initial about:blank counting as a 'visited source' marked
    the return subgoal at the first results-page click (measured: the
    T10 run). Real pages carry a tab suffix ('… — Mozilla Firefox')."""
    bare = {"mozilla firefox", "new tab", ""}
    n = 0
    for t in titles_seen or set():
        tl = t.lower()
        if tl in bare or "duckduckgo" in tl or "terminal" in tl:
            continue
        n += 1
    return n


def _page_titles(titles_seen: set[str] | None) -> list[str]:
    """The visited source pages, as the planner will know them:
    bare/DDG/terminal titles dropped, the browser tab suffix
    (' — Mozilla Firefox') stripped so a title matches the result
    name in the read dump exactly (measured: the dump's result 1 is
    'rtx 3060 - anúncios classificados | OLX Portugal' and the page
    title is that plus the suffix)."""
    out = []
    for t in sorted(titles_seen or set()):
        tl = t.lower()
        if tl in {"mozilla firefox", "new tab", ""} or "duckduckgo" in tl \
                or "terminal" in tl:
            continue
        p = t.split(" — ")[0].strip()
        if p and p not in out:
            out.append(p)
    return out


# Goal-derived search query: 'Search DuckDuckGo for: X' /
# 'Search DuckDuckGo for "X"'. The deterministic escape from a stuck
# source page re-runs the goal's own search: the search verb
# self-focuses the address bar, so it works even when the page's
# consent dialog blocks browser Back (measured: the T10 run — the
# KuantoKusta modal, two dead ctrl+Left, the search recovered).
def _goal_search_query(goal: str) -> str | None:
    m = re.search(r"\bsearch\s+duckduckgo\s+for\s*[:\-]?\s*[\"']?"
                  r"(.+?)[\"']?(?:\s*\.\s|\s+and\b|$)",
                  goal, re.I | re.S)
    if not m:
        return None
    q = m.group(1).strip().strip("\"'").strip()
    return q if 3 <= len(q) <= 120 else None


def _is_source_page(title: str) -> bool:
    """A visited content page: not the results page, not the
    terminal, not the bare browser chrome (about:blank's title is
    'Mozilla Firefox', a new tab's 'New Tab' — both measured)."""
    if not title:
        return False
    tl = title.lower()
    if tl in ("mozilla firefox", "new tab") or "duckduckgo" in tl \
            or "terminal" in tl:
        return False
    return True


def _name_hits(name: str, visited: list[str]) -> bool:
    """Does a result name from the read dump name a source already
    opened? The dump's result names ARE the page titles minus the
    browser suffix (measured: the planner's click phrases were copied
    verbatim from them), so containment works both ways. Token
    overlap is NOT used: store listings share category phrases
    ('Placas Gráficas GeForce RTX 3060 …' across different shops —
    measured: a 3-token overlap would have excluded the
    PcComponentes result as 'visited' because KuantoKusta's title
    carries the same category words)."""
    n = re.sub(r"\s+", " ", name.lower()).strip()
    if not n:
        return False
    for v in visited:
        t = re.sub(r"\s+", " ", v.lower()).strip()
        if t and (n in t or t in n):
            return True
    return False


def _unopened_from_dump(dump: str, visited: list[str],
                        limit: int = 2) -> list[str]:
    """Numbered result names from the last read dump that match no
    visited source — the concrete pointer the planner's next pick
    needs (it cannot tell which sources it has already opened;
    measured: it re-clicked the first result three times in a row
    with the visited list in its history)."""
    out: list[str] = []
    for m in re.finditer(r"^\s*\d+\.\s+(.{8,80})$", dump, re.M):
        name = m.group(1).strip()
        if _name_hits(name, visited) or name in out:
            continue
        out.append(name)
        if len(out) >= limit:
            break
    return out


def _dump_result_names(dump: str) -> list[tuple[int, str]]:
    """The numbered result list of a read dump as (number, name)
    pairs, in dump order. The planner's click phrase can be a
    result NUMBER rather than its name: with a degraded title-only
    dump it has no names to copy and falls back to the number
    (measured: the T10 run — five 36 s clicks of '1.', grounding
    found the digit in the window's top strip, 0.0% frame, no
    navigation). The number resolves to a name only against this
    list."""
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"^\s*(\d+)\.\s+(.{4,80})$", dump, re.M):
        out.append((int(m.group(1)), m.group(2).strip()))
    return out


def _dump_has_results(dump: str) -> bool:
    """True if a read dump carries a numbered result list. Right
    after navigation the a11y tree is still building and a read can
    return the page title only — the lazy-tree signature (measured:
    the T10 run — the first read after the search was complete, the
    ones right after it title-only; the planner then had no result
    names and clicked the numbers)."""
    return bool(re.search(r"^\s*\d+\.\s+\S", dump, re.M))


def _parse_price(token: str) -> float | None:
    """A price token from a dump's money: line ('€334', '501€',
    '680,49€') to a float. EU/PT decimal comma wins when present
    ('680,49' -> 680.49); a bare dot is a decimal point. None when
    the token is not a price at all."""
    t = token.strip().replace("€", "").replace(" ", "")
    if not t:
        return None
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _block_commands(text: str) -> list[str]:
    """Every command quoted in `text`, in order.

    Primary: the 'type … command:' phrasing (finditer — a checklist can
    carry several). Fallback (measured: T5 — the decomposer keeps the
    command verbatim but drops the word 'command'): a shell-verb scan
    (echo/printf/cat), the same one _typed_command_of falls back on.
    """
    cmds = []
    for m in re.finditer(r"type\s+(?:exactly\s+)?(?:this\s+)?command"
                         r"\s*[:\-]?\s*([^\n—]+)", text, re.I):
        cmd = m.group(1).strip().rstrip(".").strip("`").strip()
        if len(cmd.split()) >= 2:
            cmds.append(cmd)
    if not cmds:
        for m in re.finditer(r"\b(echo|printf|cat)\b([^\n—]+)", text):
            cmd = ((m.group(1) + m.group(2)).strip().rstrip(".").strip("`")
                   .strip())
            if len(cmd.split()) >= 2:
                cmds.append(cmd)
    return cmds


class Planner:
    """One planning call = one HTTP round trip to the LFM3 server.

    The recent history is re-sent every call (the goal block's
    [DONE]/[NEXT] marks carry the earlier progress, so only the last
    ~12 lines go out): the prompt stays ~1-2K tokens, ~1 s of prefill
    on the 1070 instead of the ~3-4 s a 32-step full history would
    cost (measured 380-1480 tok/s).
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
            # main-gpu 0: with CUDA_VISIBLE_DEVICES filtering to the
            # 1070 alone (see _spawn), it is device 0 of the filtered
            # enumeration.
            "--main-gpu", "0",
            "-c", ctx,
            "--host", "127.0.0.1", "--port", port,
            "--no-ui",
            # jinja: the GGUF chat template drives tool-calling format.
            "--jinja",
        ]
        log = open("/tmp/lfm3-server.log", "ab")
        gpu_idx = self._gpu_index()
        print(f"[planner] spawning {bin_path} (model={os.path.basename(model)}, "
              f"gpu={gpu_idx} [1070], ctx={ctx}, port={port}); "
              f"log /tmp/lfm3-server.log", flush=True)
        self._proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**env,
                 # The 1070 ALONE: llama.cpp's CUDA init creates a
                 # context on every visible device. On this shared host
                 # the other GPUs (two RTX 3060s, ~95% occupied by the
                 # user's workloads) can be down to < 1 GB free, and the
                 # cudaSetDevice there OOMs and aborts the whole load —
                 # even though --main-gpu points at the 1070 (measured:
                 # worked while the 3060s had headroom, aborted at
                 # 110 MiB free). Hiding them makes the load
                 # deterministic; after the filter the 1070 is device 0.
                 "CUDA_VISIBLE_DEVICES": str(gpu_idx)},
            start_new_session=True)

    def _free_vram_mib(self) -> int:
        """Free VRAM on the 1070, via nvidia-smi, found by NAME
        (nvidia-smi indices do NOT match torch/cuda indices on this
        host — see entrypoint.sh). 0 when unreadable."""
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
        return 0

    def _wait_vram_headroom(self, need_mib: int = 2560,
                            timeout: float = 120.0) -> None:
        """Wait until the shared 1070 has headroom for the Q4 weights +
        context. llama.cpp's CUDA context init OOMs on a full card
        (measured: cudaSetDevice 'out of memory' at 0.5 s while the card
        held the worker + a desktop spike); the card is shared with the
        user's live desktop, so such spikes happen."""
        free = self._free_vram_mib()
        if not free:
            return  # nvidia-smi unreadable: proceed, the error will say
        if free < need_mib:
            print(f"[planner] 1070 free VRAM {free} MiB < ~{need_mib} "
                  f"needed; waiting for the shared card to clear...",
                  flush=True)
        deadline = time.time() + timeout
        while free < need_mib and time.time() < deadline:
            time.sleep(2)
            free = self._free_vram_mib() or 0
        if free < need_mib:
            print(f"[planner] still only {free} MiB free after "
                  f"{timeout:.0f} s; attempting the spawn anyway",
                  flush=True)
        else:
            print(f"[planner] 1070 free VRAM {free} MiB — spawning",
                  flush=True)

    def ensure_server(self, wait: float = 90.0) -> None:
        """Ready (health ok), or spawn and wait until it is.

        Retries the spawn: a cold CUDA init on the shared card can
        transiently OOM (measured) or a slow load can overrun the wait;
        each attempt first waits for VRAM headroom.
        """
        last = ""
        for attempt in range(1, 4):
            try:
                self._ensure_once(wait)
                return
            except PlannerError as e:
                last = str(e)
                self._proc = None  # dead/partial: next attempt respawns
                print(f"[planner] LFM3 spawn failed (attempt {attempt}/3): "
                      f"{str(e).splitlines()[0]}; the card is shared — "
                      f"waiting for headroom and retrying", flush=True)
                time.sleep(3)
        raise PlannerError(last)

    def _ensure_once(self, wait: float) -> None:
        if self.health():
            return
        if self._proc is not None and self._proc.poll() is None:
            pass  # already starting: just wait below
        else:
            self._wait_vram_headroom()
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
        """One (or two) planning calls: split the goal into short atomic
        subgoals (2-6, numbered).

        The 1.7B planner loses track of multi-part goals mid-run
        (measured: after the article opened — subgoal 2 of 6 — it kept
        re-clicking the result and polling observe instead of scrolling).
        A short numbered checklist in every prompt keeps it on track.
        Falls back to the whole goal as a single subgoal on any failure.

        Coverage check: a goal that quotes a command to type needs a
        subgoal that carries it. The decomposer sometimes drops exactly
        that line (measured: T1 — it wrote a meta-comment into the
        scroll subgoal and omitted the 'type exactly: `echo ...`' line;
        the run then ended at 'open terminal' and the file was never
        written, because all_done fired with the command subgoal
        absent). The decode is stochastic, so one re-decompose is a
        second draw; if the command is still missing, the subgoal is
        appended deterministically from the goal's own words.
        """
        subs = self._decompose_once(goal)
        if not subs:
            print("[planner] using the goal as one subgoal", flush=True)
            return [goal.strip()]
        # Exact duplicates: the 1.7B sometimes repeats a line verbatim
        # (measured: T3 — 'Open the first search result link.' and
        # 'Verify that the article is opened in the browser.' each
        # appeared twice). Duplicates are pure noise in the checklist,
        # and a junk phrasing can also misfire the mark logic.
        seen: set[str] = set()
        subs = [s for s in subs
                if not (s.lower() in seen or seen.add(s.lower()))]
        # Drop decomposer garbage (see _subgoal_garbage_reason): meta
        # prose and token-truncated lines are not atomic actions, and
        # left in the checklist they block all_done (measured: the
        # 2026-11-01 battery re-run — T2/T4/T6 drew 'scroll within
        # page' meta lines that parked in the counted-source branch
        # and burned the step budget re-doing completed chains).
        kept = []
        for s in subs:
            why = _subgoal_garbage_reason(s)
            if why:
                print(f"[planner] dropping garbage subgoal "
                      f"({why}): {s[:70]!r}", flush=True)
                continue
            kept.append(s)
        subs = kept
        if not subs:
            print("[planner] all subgoals garbage; using the goal as "
                  "one subgoal", flush=True)
            return [goal.strip()]
        # Drop lines outside the mark surface (see _subgoal_markable):
        # they can never be marked, so they would block all_done
        # forever. When nothing markable survives, the goal's own
        # words (which carry its actions) are the checklist.
        subs = [s for s in subs if _subgoal_markable(s)]
        if not subs:
            print("[planner] no markable subgoals; using the goal as "
                  "one subgoal", flush=True)
            return [goal.strip()]
        cmd = _typed_command_of(goal)
        if cmd:
            def _covers(sub: str) -> bool:
                t = sub.replace("`", "").lower()
                c = cmd.lower()
                if c in t:
                    return True
                # Token-level containment: every word of the command
                # appears in the subgoal (quotes/backticks may differ).
                return all(w in t for w in c.split() if len(w) >= 3)
            if not any(_covers(s) for s in subs):
                print(f"[planner] decompose dropped the typed-command "
                      f"subgoal ({cmd!r}); re-decomposing once",
                      flush=True)
                again = self._decompose_once(goal)
                if again and any(_covers(s) for s in again):
                    subs = again
                else:
                    subs.append(f"In the terminal, type exactly this "
                                f"command: {cmd} — then press Enter.")
                    print("[planner] command subgoal still missing; "
                          "appended it deterministically", flush=True)
        # Checklist cap: room for every command the goal quotes (one
        # subgoal each) plus the non-command actions. A fixed cap
        # truncates multi-command checklists, and all_done would fire
        # while quoted commands are still untyped — the marks only
        # track the LISTED subgoals (measured necessity: the 17-command
        # T9 stress goal).
        return subs[:max(8, len(_block_commands(goal)) + 4)]

    def _decompose_once(self, goal: str) -> list[str]:
        """One decompose call: the raw subgoal list ([] on failure or an
        unusable reply)."""
        body = {
            "model": "LFM3",
            "messages": [
                {"role": "system", "content":
                 "Break the goal into 2-6 short atomic subgoals in "
                 "execution order. One numbered line per subgoal, each "
                 "under 15 words. Keep exact quoted phrases and commands "
                 "verbatim. Every action in the goal gets its own line — "
                 "never drop or merge a step that types or runs a "
                 "command. Only the goal's own actions — never add "
                 "alternatives, tools the goal does not name (no curl, "
                 "no browser extensions), or verification steps (the "
                 "system verifies completion). No commentary. Output "
                 "only the numbered list."},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
            # Budget sized to the goal's command count: the list must
            # fit every quoted command (one line each); a fixed 300
            # tokens cut a 17-command list mid-line (T9).
            "max_tokens": max(300, 40 + 25 * len(_block_commands(goal))),
            "temperature": 0.1,
            "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            data = self._post(body)
        except Exception as e:
            print(f"[planner] decompose failed ({e!r})", flush=True)
            return []
        text = data["choices"][0]["message"].get("content") or ""
        subs = []
        for line in text.splitlines():
            m = re.match(r"^\s*(?:\d+[\.\):]|[-*])\s*(.+)$",
                         line.strip())
            if m and m.group(1).strip():
                subs.append(m.group(1).strip())
        # Drop subgoals that are the "call done" instruction itself
        # rather than an action (measured: run 15's decomposer lifted
        # the goal's terminal clause into a subgoal — 'Call done when
        # the file is created with that content.'). The system calls
        # done when every ACTION subgoal is marked on measured
        # evidence, so such a line has no controller step that could
        # mark it and would block the all-done signal forever.
        subs = [s for s in subs if not _is_done_directive(s)]
        if not subs:
            print(f"[planner] decompose gave no usable list "
                  f"({text[:120]!r})", flush=True)
        return subs

    # ------------------------------------------------------------- plan

    def plan(self, goal: str, history: list[str]) -> tuple[str, dict]:
        """One planning call. Returns (tool_name, args).

        Retries: a transport failure re-checks the server (it may have
        died); a non-tool reply gets one corrective nudge.
        """
        hist = ("\n".join(history[-12:]) if history
                else "no steps executed yet")
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
        # Text of the last read dump (the detail string): kept for the
        # machine's own use — the unopened-result names in the
        # revisit hints (see _unopened_from_dump).
        self._last_dump = ""
        # Every source-page read this run, as (page title, dump):
        # the measured evidence of the research report (see
        # _research_facts) — the done summary's facts when the
        # planner's own report doesn't land.
        self._source_dumps: list[tuple[str, str]] = []

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
        elif name == "read":
            # No X action: the content comes from the OS accessibility
            # tree (measured rendered text), not from the frame. The
            # detail IS the payload — it rides in history as the
            # planner's page-content evidence.
            from . import atspi
            text = atspi.dump_active()
            if not text:
                return False, ("read: no accessible content (a11y bus "
                               "off, or no registered app, or a stalled "
                               "tree) — use observe, or navigate/retry")
            return True, f"read: {text}"
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
        """The exact command the goal says to type, if it quotes one
        (see _typed_command_of)."""
        return _typed_command_of(goal_block)

    def _next_untyped_command(self, goal_block: str,
                              typed: list[str]) -> str | None:
        """The first command still quoted in the checklist that is not
        in the typed-text history (multi-command goals, T5)."""
        out = [t.lower() for t in typed]
        for c in _block_commands(goal_block):
            c_low = c.lower()
            if not any(c_low in t or all(w in t for w in c_low.split()
                                         if len(w) >= 3)
                       for t in out):
                return c
        return None

    def _research_facts(self) -> str:
        """Per-source price facts from the MEASURED read dumps — the
        content of the done summary for research goals when the
        planner's own report doesn't land. No LLM in it: every line is
        a measured dump (the page's own title, the dump's own money:
        line); 'cheapest' is the min over those measured tokens. ''
        when no source page was read."""
        by_title: dict[str, str] = {}
        for title, dump in self._source_dumps:
            by_title[title] = dump  # last read of a page wins (revisits)
        lines: list[str] = []
        best: tuple[float, str, str] | None = None
        for title, dump in by_title.items():
            m = re.search(r"^money:\s*(.+)$", dump, re.M)
            prices = m.group(1).strip() if m else ""
            name = re.sub(r"\s*—\s*Mozilla Firefox\s*$", "", title).strip()
            if prices:
                lines.append(f"{name}: {prices}")
            else:
                lines.append(f"{name}: no prices in the read dump")
            for tok in prices.split("|"):
                v = _parse_price(tok)
                if v is not None and (best is None or v < best[0]):
                    best = (v, tok.strip(), name)
        if not lines:
            return ""
        out = "sources checked: " + "; ".join(lines)
        if best:
            out += f" — cheapest price seen: {best[1]} (on {best[2]})"
        return out

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
                       post: str,
                       typed: list[str] | None = None,
                       titles_seen: set[str] | None = None) -> bool:
        """Advance the deterministic progress tracker after a step.

        Each mark fires only on measured evidence matching the subgoal's
        semantics (a DDG title for a search subgoal, a non-DDG title
        for 'open the result', the terminal title for terminal
        subgoals, a type/key-executed in the terminal for the command
        subgoals). Command subgoals additionally require their quoted
        command(s) to be in the typed-text history (`typed`): an Enter
        marks a command subgoal only if THAT command was typed, not
        just some command (measured: T5). Conservative on purpose: a
        missed mark lets the planner redo the subgoal (capped by the
        repeat guard); a false mark would make it skip one.
        """
        changed = False
        if not ok:
            return changed
        typed = typed or []
        args_str = str(args).lower()
        for idx, sub in enumerate(subgoals):
            if marks[idx]:
                continue
            s = sub.lower()
            terminal_up = "terminal" in post.lower()
            # Counted source subgoal ('open at least 3 Portuguese
            # sources'): ONE visited page must not complete it — the
            # mark needs N DISTINCT visited page titles (goal-derived
            # count; measured signature: distinct non-results,
            # non-terminal titles). Checked first so the plain
            # 'result' branch cannot mark it early (that is exactly the
            # premature-done the research runs must not hit). The gate
            # is the strict counted-source signature (see
            # _counted_sources) — meta prose that merely MENTIONS a
            # page must not park here forever (measured: the
            # 2026-11-01 battery re-run, T2/T4/T6).
            n_open = _counted_sources(s)
            if n_open >= 2:
                # The evidence is the READ source pages, not merely
                # the visited ones: the goal's own words are 'open
                # the sources one by one (… read the page)', and an
                # opened-but-unread source has no price for the
                # report (measured: the T10 run — the third source
                # was visited, all_done fired before its read, and
                # the done summary had no price for it).
                _read_titles = {t for t, _ in self._source_dumps}
                if _distinct_page_titles(_read_titles) >= n_open:
                    marks[idx] = changed = True
                continue
            if "terminal" in s:
                # 'type'/'echo' only: 'run the command' belongs to the
                # enter branch, not the type branch. A subgoal that asks
                # for typing AND pressing Enter is marked at the ENTER
                # step, never at the type step (measured: run 15's
                # decomposer merged the two — 'Type exactly: `echo ...`
                # and press Enter.' marked at the type step would leave
                # the command unexecuted when all_done fired).
                if (("type" in s or "echo" in s)
                        and "enter" not in s and "press" not in s
                        and name == "type" and terminal_up
                        and _subgoal_typed(sub, typed)):
                    marks[idx] = changed = True
                elif (("enter" in s or "press" in s
                       or "execute" in s or "run" in s)
                        and name == "key" and "enter" in args_str
                        and terminal_up
                        and _subgoal_typed(sub, typed)):
                    marks[idx] = changed = True
                elif (name == "open" and terminal_up
                        and not any(k in s for k in
                                    ("type", "echo", "enter", "press",
                                     "run", "execute"))):
                    # A subgoal that merely asks for a terminal is
                    # marked when the terminal is up; a subgoal that
                    # asks to TYPE/RUN something in it must wait for
                    # the actual type/enter step (measured: T2 — 'In
                    # the terminal, type exactly this command: …'
                    # was marked on the open, and all_done fired
                    # before the file was ever written).
                    marks[idx] = changed = True
            elif ("result" in s
                    and any(p in s for p in
                            ("return to", "go back to", "back to",
                             "come back to"))):
                # 'Return to the results page (ctrl+Left)': the generic
                # 'result' branch below marks this the moment we LEAVE
                # the results page — the opposite of the subgoal
                # (measured: the T10 run marked it at the first source
                # open, and all_done fired with 1 of 3 sources). The
                # measured signature of the return: a DDG title with at
                # least one non-results page already visited.
                if ("duckduckgo" in post.lower()
                        and _distinct_page_titles(titles_seen) >= 1):
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
            # 'Verify that the article is opened' subgoals are
            # decomposer variance (measured: T3 — the 1.7B added a
            # verify step the goal never asked for, and nothing can
            # mark a verify, so all_done never fired and the planner
            # polled observe until the failure budget). They are
            # satisfied by the same measured evidence as 'open the
            # result': the browser shows a non-results, non-terminal
            # page.
            elif (("verify" in s or "check" in s or "confirm" in s)
                    and ("article" in s or "page" in s)
                    and "duckduckgo" not in post.lower()
                    and "terminal" not in post.lower()):
                marks[idx] = changed = True
            # The decomposer drops context from subgoals (measured:
            # 'Type exactly: `echo ...`' with no word 'terminal'), so
            # type/enter subgoals are marked by WHERE the step ran —
            # the measured terminal title — not the subgoal's words.
            elif (("type" in s or "echo" in s)
                    and "enter" not in s and "press" not in s
                    and name == "type" and terminal_up
                    and _subgoal_typed(sub, typed)):
                marks[idx] = changed = True
            elif (("enter" in s or "press" in s
                   or "execute" in s or "run" in s)
                    and name == "key" and "enter" in args_str
                    and terminal_up
                    and _subgoal_typed(sub, typed)):
                marks[idx] = changed = True
        return changed

    def _settled_next(self, post: str,
                      goal_block: str, pre: str = "",
                      seen_before: set[str] | None = None,
                      last_dump: str = "") -> tuple[str, tuple[str, dict] | None] | None:
        """The canonical next step once the current page has settled.

        Shared by the wait-load branch (the wait finished) and the
        fast-path click branch (the title already moved within the
        0.2 s post-capture window, so a wait-load would only compare
        against an already-changed title and burn its full cap —
        measured 8.1 s).
        """
        low = goal_block.lower()
        # A wait-load that did NOT move the title: the navigation the
        # key/click was meant to trigger is dead (measured: the T10
        # run — the KuantoKusta consent dialog swallowed Back; read
        # still dumps the prices behind it, but the page will not
        # leave). The search verb self-focuses the address bar, so
        # re-running the goal's own search is the measured escape.
        if (not _wants_a_specific_result(goal_block) and pre == post
                and _is_source_page(post)):
            q = _goal_search_query(goal_block)
            if q:
                return ("SUGGESTED NEXT: search "
                        + json.dumps({"query": q})
                        + " (Back did not navigate — this page blocks "
                          "it; the search goes through the address bar "
                          "and lands back on the results page).",
                        ("search", {"query": q}))
        # Research goal on a source page: the goal's own recipe is
        # 'wait-load, read the page, back' — the machine runs it,
        # because the 1.7B planner fixates on re-clicking the page's
        # own title text instead (measured: the T10 run burned its
        # failure budget that way). A REVISIT (the title was already
        # seen before this step — the machine's visited set, which
        # the planner cannot track) goes straight back, with the
        # opened list AND the unopened results from the last dump, so
        # the next pick is driven by measured state.
        if not _wants_a_specific_result(goal_block) and _is_source_page(post):
            visited = _page_titles(seen_before)
            if post in (seen_before or set()):
                unopened = _unopened_from_dump(last_dump, visited)
                _extra = (" Unopened results in the last read dump: "
                          + "; ".join(f"'{u[:60]}'" for u in unopened)
                          if unopened else "")
                _opened = "; ".join(v[:50] for v in visited) or "none"
                return ("SUGGESTED NEXT: key {\"combo\": \"ctrl+Left\"} "
                        "(you just RE-opened a source you had already "
                        f"opened; opened so far: {_opened}. Back to the "
                        "results page to pick a DIFFERENT source."
                        + _extra + ")",
                        ("key", {"combo": "ctrl+Left"}))
            return ("SUGGESTED NEXT: read {} (a source page just loaded; "
                    "read it — the dump carries the page's text and "
                    "prices; after reading, browser Back returns to "
                    "the results page).",
                    ("read", {}))
        if "duckduckgo" in post.lower():
            if _wants_a_specific_result(goal_block):
                return ("SUGGESTED NEXT: click "
                        "{\"phrase\": \"the first result in the search "
                        "results\"} (the results page is still up and "
                        "the goal wants that result).",
                        ("click",
                         {"phrase": "the first result in the search results"}))
            # Research goals back on the results page (after a source):
            # re-read before picking the next source — the planner's
            # source pick is driven by the dump, not by memory of a page
            # it cannot see.
            return ("SUGGESTED NEXT: read {} (the results page is up "
                    "again; the dump lists the results NUMBERED with "
                    "their URLs — open a source you have NOT opened "
                    "yet, by name or position).",
                    ("read", {}))
        if re.search(r"\bscroll", low):
            n = self._scroll_amount(goal_block)
            args = {"phrase": "the article", "amount": n}
            return (f"SUGGESTED NEXT: scroll {json.dumps(args)} "
                    f"(the page has settled and the goal wants the "
                    f"article scrolled {n} notches).",
                    ("scroll", args))
        return ("SUGGESTED NEXT: continue with the next unfinished "
                "subgoal (the page has settled).", None)

    def _suggestion(self, name: str, ok: bool, pre: str, post: str,
                    goal_block: str,
                    all_done: bool = False,
                    typed: list[str] | None = None,
                    titles_seen: set[str] | None = None,
                    seen_before: set[str] | None = None,
                    last_dump: str = "",
                    detail: str = "") -> tuple[str, tuple[str, dict] | None] | None:
        """The canonical next step, suggested by the deterministic layer.

        Returns (history_line, (tool_name, args) | None). When the
        machine part is not None, _run_loop EXECUTES IT DIRECTLY (no
        planning round trip): each suggestion only fires on a MEASURED
        state signature (a DDG title means the controller's verified
        _search finished; a terminal title means _open verified focus;
        all-done means every subgoal was marked on measured evidence)
        and its args are extracted from the goal's own words (scroll
        count, 'terminal', the quoted command), so following it is
        always safe. The planner keeps full authority over every turn
        without a pending suggestion. The chain covered:
        search->click->wait-load->scroll->open terminal->type->enter
        ->done.
        """
        if not ok:
            return None
        if all_done:
            # Every subgoal is marked on MEASURED evidence. The final
            # turn goes to the PLANNER, not a mechanical execution:
            # its summary is the deliverable, and for research goals
            # the mechanical checklist summary cannot carry the
            # compared facts (measured: a smoke run ended with the
            # goal text parroted back, zero facts). The 1.7B planner
            # under-calls done with the marks up (measured), so the
            # HINT below is deterministic and _run_loop falls back to
            # the mechanical auto-done after two hint turns — the
            # termination guarantee is unchanged.
            _hint = ("HINT: every subgoal is [DONE] on measured "
                    "evidence. Call done NOW with a summary that "
                    "ANSWERS the goal: every fact you collected (the "
                    "read dumps in the history are your evidence) "
                    "with its source. If the goal's own completion "
                    "criterion (e.g. how many sources it names) is not "
                    "yet met, work the next unfinished part instead.")
            if not _wants_a_specific_result(goal_block):
                # The report's facts, measured, in the hint itself:
                # the planner's done summary copies them (measured:
                # without them in the prompt its summary was a
                # checklist parrot with zero facts — the T10 run).
                _facts = self._research_facts()
                if _facts:
                    _hint += (" Measured facts from the read dumps "
                              "(report them, each with its source): "
                              + _facts)
            return (_hint, None)
        low = goal_block.lower()
        results_page = "duckduckgo" in post.lower()
        if name in ("search", "duckduckgo") and results_page:
            if _wants_a_specific_result(goal_block):
                return ("SUGGESTED NEXT: click "
                        "{\"phrase\": \"the first result in the search "
                        "results\"} (the results page is up and the "
                        "goal wants that result).",
                        ("click",
                         {"phrase": "the first result in the search results"}))
            # Research/comparison goals: the canonical next step is the
            # content channel — the planner must SEE the results (names
            # + URLs + prices) before it can pick a source. Without this
            # suggestion the planner fixated on re-running the same
            # search until the failure budget (measured: the T10 run —
            # the 1.7B anchored on the goal's 'Search ...' clause). read
            # is a screen no-op, so executing it directly is safe.
            _visited = _page_titles(titles_seen)
            _extra = (" Already opened: " + "; ".join(v[:50] for v in _visited)
                      + " — pick a DIFFERENT one." if _visited else "")
            return ("SUGGESTED NEXT: read {} (the results page is up; "
                    "read is how you see its content — the result "
                    "names, URLs and prices — before picking a source."
                    + _extra + ")",
                    ("read", {}))
        if name == "click":
            if "duckduckgo" in pre.lower() and post == pre:
                # Title unchanged: the navigation may still be in
                # flight (measured slow-nav quirk: the 0.2 s
                # post-capture after the click still shows the results
                # page); wait for it to land.
                return ("SUGGESTED NEXT: wait-load {} (the click may "
                        "still be navigating; wait for the new page "
                        "before acting).",
                        ("wait-load", {}))
            if (not _wants_a_specific_result(goal_block)
                    and pre == post
                    and _is_source_page(post)):
                # No-op click on a source page (research goal): the
                # click changed nothing — dead space, or the page's
                # consent dialog swallowed it (measured: the T10 run —
                # 34 s of grounding, the click landed on the dimmed
                # area of the KuantoKusta modal, 0.0% frame). The
                # content channel still works behind the dialog (the
                # a11y tree carries the page's text + prices —
                # measured), so READ first; the read->back chain then
                # recovers.
                return ("SUGGESTED NEXT: read {} (the click changed "
                        "nothing — the page may show a consent dialog "
                        "on top of it; read still dumps the page's "
                        "text and prices behind the dialog).",
                        ("read", {}))
            # The title already moved (fast nav) or the click was not
            # on a results page: settle directly — a wait-load here
            # would compare against an already-changed title and burn
            # its full cap (measured 8.1 s).
            return self._settled_next(post, goal_block, pre=pre,
                                      seen_before=seen_before,
                                      last_dump=last_dump)
        if name == "wait-load":
            return self._settled_next(post, goal_block, pre=pre,
                                      seen_before=seen_before,
                                      last_dump=last_dump)
        if name == "read" and "duckduckgo" in post.lower() \
                and not _wants_a_specific_result(goal_block) \
                and not _dump_has_results(detail):
            # Degraded read right after navigation: the a11y tree is
            # still building, the dump carries the page title only,
            # and the planner is left with no result names to copy —
            # it falls back to clicking a result NUMBER (measured:
            # the T10 run — five 36 s clicks of '1.' after three
            # title-only reads). A re-read is usually complete by
            # then; if the tree is truly stuck, the per-state repeat
            # guard bounds the re-reads to two and hands the wheel
            # back to the planner.
            return ("SUGGESTED NEXT: read {} (the last read got only "
                    "the page title — its a11y tree is still "
                    "building; a re-read carries the numbered result "
                    "list).",
                    ("read", {}))
        if name == "read" and "duckduckgo" in post.lower() \
                and not _wants_a_specific_result(goal_block):
            # Full dump on the results page (research goal): the
            # MACHINE drives the source pick. Measured: the 1.7B
            # fixates on the first result even with the unopened
            # names listed in the hints (the T10 run — six revisits
            # of the same two sources after two distinct; the step
            # budget died on revisits, not on navigation). The next
            # source is the first UNOPENED result of the measured
            # dump, in list order — deterministic, and exactly the
            # goal's "open a source you have NOT opened yet". The
            # click phrase is the result name verbatim from the
            # dump, so grounding finds it on the page (2–4 s).
            visited = _page_titles(seen_before or set())
            unopened = _unopened_from_dump(detail, visited)
            if unopened:
                return ("SUGGESTED NEXT: click "
                        + json.dumps({"phrase": unopened[0]})
                        + " (this source has NOT been opened yet — "
                          "the machine picks the first unopened "
                          "result from the measured dump; after the "
                          "read, browser Back returns here for the "
                          "next one).",
                        ("click", {"phrase": unopened[0]}))
            # Every result in the dump is already visited: the
            # source budget is spent — the planner's wheel (the
            # final report, or a different query).
            return None
        if name == "read" and "duckduckgo" not in post.lower() \
                and "terminal" not in post.lower() \
                and not _wants_a_specific_result(goal_block):
            # Research goal: the source page has been read; the
            # canonical next step is browser Back to the results page.
            # Without this the planner fixated on re-clicking the
            # source's own title text (it IS on the page) instead of
            # navigating — 4 clicks of the product title burned the
            # failure budget (measured: the T10 run).
            return ("SUGGESTED NEXT: key {\"combo\": \"ctrl+Left\"} "
                    "(the source page is read; browser Back returns to "
                    "the results page, where the next source is picked).",
                    ("key", {"combo": "ctrl+Left"}))
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
        if (name == "key" and pre == post and _is_source_page(post)
                and not _wants_a_specific_result(goal_block)):
            # A key that changed nothing on a source page: browser
            # Back that may still be navigating (measured slow-nav) —
            # or dead, because the page's consent dialog blocks it
            # (measured: the T10 run — two ctrl+Left, the screen
            # identical, the KuantoKusta modal up). wait-load
            # disambiguates: it settles a real navigation; a
            # still-stuck wait-load lands on the search escape in
            # _settled_next.
            return ("SUGGESTED NEXT: wait-load {} (the key may still "
                    "be navigating; wait for the results page to "
                    "land).",
                    ("wait-load", {}))
        if name == "key" and "duckduckgo" in post.lower() \
                and not _wants_a_specific_result(goal_block):
            # Research goal, back on the results page: re-read before
            # the next pick — the numbered dump is what drives which
            # source to open next.
            _visited = _page_titles(titles_seen)
            _extra = (" Already opened: " + "; ".join(v[:50] for v in _visited)
                      + " — open a DIFFERENT source." if _visited else "")
            return ("SUGGESTED NEXT: read {} (back on the results page; "
                    "the dump lists the results NUMBERED with their "
                    "URLs — open a source you have NOT opened yet, by "
                    "name or position." + _extra + ")",
                    ("read", {}))
        if name == "key" and "terminal" in post.lower():
            # Multi-command goals (measured: T5): after a command has
            # run, the next command still quoted in the checklist and
            # not yet typed is the deterministic next step — no planner
            # round trip needed for what the goal already spells out
            # verbatim.
            nxt = self._next_untyped_command(goal_block, typed or [])
            if nxt:
                return ("SUGGESTED NEXT: type "
                        + json.dumps({"text": nxt})
                        + " verbatim into the terminal (the goal gives "
                          "the exact next command), then key enter.",
                        ("type", {"text": nxt}))
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
                                  seen_sig={})
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
                  history: list[str], t0: float, seen_sig: dict) -> bool:
        consec_fail = 0
        # Suggestion fast-path state: the deterministic layer's pending
        # machine suggestion (see _suggestion). When one exists the
        # system executes it DIRECTLY — no planning round trip: a
        # pending suggestion is anchored on a measured state signature
        # (results page up, page settled, terminal focused) and its
        # args are extracted from the goal's own words (scroll count,
        # the quoted command), so following it is always safe, and
        # skipping the ~1-2 s LFM3 call keeps canonical transitions at
        # executor speed. Measured: the 1.7B planner deviated from
        # nearly every canonical suggestion in the 9-step goal (fixated
        # on re-scrolling); consulting it per transition cost 15 calls
        # ≈ 15 s of pure latency for 6 transitions the controller could
        # execute directly. The planner keeps full authority over every
        # turn WITHOUT a pending suggestion — goal start, states outside
        # the canonical chain (a stale drop), recovery after a failed
        # step — and it sees every executed result in the history.
        pending: tuple[str, dict] | None = None
        # Repeat-guard retry budget (see the guard below): how many times
        # each (step, args) has actually executed IN ITS CURRENT STATE
        # (reset when the step's saved state no longer matches — see the
        # guard). A step seen in the same state twice may run one more
        # time (fresh grounding is the remedy for a stale target); from
        # the second retry on it is blocked.
        attempts: dict = {}
        # Verbatim text of every executed type step: the mark logic
        # requires a command subgoal's own command(s) in this history
        # before marking (multi-command goals — see _subgoal_typed).
        typed: list[str] = []
        # Deterministic subgoal progress (see _mark_subgoals); rebuilt
        # into the goal block before every planning call.
        marks = [False] * len(subgoals)
        # Every page title seen this run (initial + post-step): the
        # measured signature of the counted source subgoals (a counted
        # 'open N sources' subgoal marks only at N distinct page
        # titles — see _mark_subgoals).
        titles_seen: set[str] = {self._title()}
        # Planner turns spent with every subgoal marked [DONE] but no
        # done emitted (see the all-done branch in _suggestion): after
        # two, the protocol executes the mechanical auto-done — the
        # termination guarantee the fast path used to provide.
        done_tries = 0
        for i in range(self.max_steps):
            if time.time() - t0 > self.max_seconds:
                print(f"[orch] budget of {self.max_seconds:.0f} s exceeded; "
                      f"stopping at step {i + 1}", flush=True)
                return False
            goal_block = self._goal_block(goal, subgoals, marks)
            if pending is not None:
                name, args = pending
                pending = None
                print(f"[orch] step {i + 1}: protocol: executing the "
                      f"suggested step ({name} "
                      f"{json.dumps(args, ensure_ascii=False)})",
                      flush=True)
            else:
                try:
                    name, args = self.planner.plan(goal_block, history)
                except PlannerError as e:
                    print(f"[orch] planner error at step {i + 1}: {e}",
                          flush=True)
                    return False
                # A planner click whose phrase is EXACTLY a key combo is
                # the 1.7B echoing the suggestion history's key JSON
                # (measured: the T10 run — it saw 'key {"combo":
                # "ctrl+Left"}' in the history and clicked it; 31 s of
                # grounding for a phrase that is not on the page, the
                # click landed on the toolbar edge, 0.9% frame). Rewrite
                # it to the key before spending a vision pass; the
                # dead-key chain recovers if the key changes nothing.
                if (name == "click"
                        and not _wants_a_specific_result(goal)
                        and re.fullmatch(r"[a-z]+(\+[a-z]+)+",
                                         str(args.get("phrase", "")),
                                         re.I)):
                    _c = str(args.get("phrase", "")).strip()
                    print(f"[orch] planner click phrase {args.get('phrase')!r} "
                          f"is a key combo; executing it as a key",
                          flush=True)
                    name, args = "key", {"combo": _c}
                # A planner click whose phrase is EXACTLY a result
                # number ('1.') is the 1.7B without the names in
                # front of it (a degraded title-only dump) falling
                # back to the number (measured: the T10 run — five 36
                # s clicks of '1.'; the digit is in the window's top
                # strip, the click is a no-op). Resolve it to the
                # full name from the last good read dump; if no dump
                # with a numbered list is on hand, re-read — the
                # click would burn a full grounding pass for a phrase
                # that is not the page's content.
                if (name == "click"
                        and not _wants_a_specific_result(goal)
                        and re.fullmatch(r"\d{1,2}\.{0,1}",
                                         str(args.get("phrase", "")))):
                    _num = int(str(args.get("phrase")).strip().rstrip("."))
                    _full = next((n for k, n in _dump_result_names(
                                       self._last_dump) if k == _num), None)
                    if _full:
                        print(f"[orch] planner click phrase "
                              f"{args.get('phrase')!r} is a result "
                              f"number; resolved to {_full!r} from the "
                              f"last read dump", flush=True)
                        name, args = "click", {"phrase": _full}
                    else:
                        print(f"[orch] planner click phrase "
                              f"{args.get('phrase')!r} is a result "
                              f"number but the last read dump has no "
                              f"numbered results (degraded); "
                              f"re-reading instead", flush=True)
                        name, args = "read", {}
                if name != "done" and marks and all(marks):
                    # Every subgoal is [DONE] on measured evidence and
                    # the wheel is the planner's (see the all-done
                    # branch of _suggestion): its summary is the
                    # deliverable, so the run ends when IT calls done.
                    # The hint is already in the history (appended by
                    # the suggestion after the last mark). Counted are
                    # NON-DONE answers, so the planner keeps one full
                    # call to collect what is missing (a read) before
                    # the one that must produce the report (measured:
                    # the smoke run's planner read after the last mark,
                    # then needed the next call for the done). After
                    # two non-done answers the protocol executes the
                    # mechanical auto-done — the termination guarantee
                    # the fast path used to provide (the 1.7B planner
                    # under-calls done with the marks up — measured).
                    done_tries += 1
                    if done_tries >= 2:
                        subs = [m.group(1) for m in re.finditer(
                            r"^\d+\. \[DONE\] (.+)$", goal_block, re.M)]
                        summary = ("all subgoals done: " + "; ".join(subs)
                                   if subs else "all subgoals done")
                        # Research goal: the checklist parrot is not
                        # the deliverable — the report is the measured
                        # facts (sources + prices + cheapest) from the
                        # read dumps (measured: the T10 run ended on
                        # exactly this fallback with zero facts).
                        if not _wants_a_specific_result(goal):
                            _facts = self._research_facts()
                            if _facts:
                                summary = _facts
                        print(f"[orch] planner ignored the done hint "
                              f"twice; the protocol executes the "
                              f"mechanical done", flush=True)
                        print(f"[orch] done: {summary} "
                              f"({time.time() - t0:.1f} s total, "
                              f"{i + 1} steps)", flush=True)
                        return True
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
            # The visited set BEFORE this step: the revisit check in
            # _settled_next needs to know whether the post title was
            # seen earlier, and titles_seen gains the post title
            # below before the suggestion is computed.
            seen_before = set(titles_seen)
            # Repeat guard: the same step in the same screen state cannot
            # do anything new — the planner can be stuck repeating a step
            # that "succeeded" (or failed) without changing the world
            # (measured: 26x 'open terminal' in a dry run). A repeated step
            # is executed only if the state moved since the last attempt —
            # or if it has not yet been retried in this state: the first
            # retry is the correct remedy for a stale grounding on a
            # moving page (measured: T1, the DDG widget expanded during
            # the vision pass, so the click hit the widget and the fresh
            # re-grounding found the link); from the second retry on the
            # step is blocked — that is where planner fixation burns the
            # failure budget.
            key = (name, json.dumps(args, sort_keys=True,
                                    ensure_ascii=False))
            # State identity is (title, frame signature) — _frame_sig.
            # A constant title (a terminal's) does not mean a frozen
            # screen: a typed character moves the frame, so a canonical
            # re-enter is a NEW state, not a repeat (measured: T9, the
            # title-only identity blocked the fourth enter and the
            # planner restarted the whole chain from step one). A
            # STATIC screen still blocks after the second retry —
            # same title, same frame — which is the planner fixation
            # this guard exists for; a live one (blinking cursor) means
            # the repeat is not a no-op, and the failure budget + step
            # cap bound it. The signature capture short-circuits: only
            # a step already seen in this title with a spent retry
            # budget costs one extra grab.
            _ss = seen_sig.get(key)
            _same_state = _ss is not None and _ss[0] == pre
            if not _same_state:
                # The retry count is per-STATE, not global: a step
                # repeated across DIFFERENT states is legitimate — the
                # research chain re-reads the results page after every
                # return, and a global count blocked the chain's own
                # read/search steps once a few had run (measured: the
                # T10 run — blocked reads and searches burned the
                # failure budget at steps 18/25/30-32).
                attempts[key] = 0
            _repeat = _same_state and attempts.get(key, 0) >= 2
            _pre_sig = None
            if _repeat:
                from . import screen
                _pre_sig = _frame_sig(
                    screen.capture(scope=self.agent.scope,
                                   monitor_index=self.agent.monitor)
                    .image)
            if _repeat and _ss[1] == _pre_sig:
                blocked = True
                ok, detail = False, (
                    "repeated: already retried once in this exact state "
                    "and the screen is still identical — observe, or "
                    "move on to the next unfinished subgoal.")
                post = pre
                frame_pct = "0.0 (skipped)"
            else:
                blocked = False
                attempts[key] = attempts.get(key, 0) + 1
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
                if name == "type" and ok:
                    # Record the verbatim text so the mark logic can
                    # confirm THIS subgoal's command was typed (not just
                    # some command). Failed types typed nothing.
                    typed.append(str(args.get("text", "")))
                if name == "read" and ok and _dump_has_results(detail) \
                        and "duckduckgo" in pre.lower():
                    # The dump rides in history as the planner's
                    # content evidence; keep a copy for the machine's
                    # own use (the unopened-result names in the
                    # revisit hints — see _unopened_from_dump — and
                    # the number->name resolution of the planner's
                    # '1.' clicks — see _dump_result_names). Only a
                    # DDG dump that actually carries the numbered
                    # list is kept: a title-only dump is degraded
                    # (the a11y tree is still building — measured:
                    # the T10 run) and overwriting the last good
                    # list with it breaks both uses; a source page's
                    # numbered list items are not the results list.
                    self._last_dump = detail
                post_cap = screen.capture(scope=self.agent.scope,
                                          monitor_index=self.agent.monitor).image
                frame_pct = f"{100.0 * _frame_diff(pre_cap, post_cap) / 255.0:.1f}"
                post = self._title()
                if name == "read" and ok and _is_source_page(post):
                    # Evidence of the research report (see
                    # _research_facts): the dump of every source
                    # page read this run.
                    self._source_dumps.append((post, detail))
                seen_sig[key] = (post, _frame_sig(post_cap))
                titles_seen.add(post)
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
            # Update the pending suggestion from the step just run. An
            # EXECUTED step (ok or failed) recomputes the canonical next
            # step from the new state — including no-ops: the stale
            # check below (and the repeat guard) make re-proposals safe.
            # _suggestion returns None for failed steps, so a failure
            # hands the wheel back to the planner (the failure note is
            # in the history). A BLOCKED step executed nothing and moved
            # nothing; no pending exists at execution time (it was
            # consumed above), so the planner simply re-plans next turn.
            if not blocked:
                self._mark_subgoals(subgoals, marks, name, ok, args, post,
                                    typed=typed, titles_seen=titles_seen)
                # The fresh block (post-step marks) is what the planner
                # will see next turn, and it is what the done-summary
                # regex reads [DONE] lines from.
                sug = self._suggestion(name, ok, pre, post,
                                       self._goal_block(goal, subgoals,
                                                        marks),
                                       all_done=bool(marks)
                                       and all(marks)
                                       and _goal_commands_run(goal, history,
                                                                typed),
                                       typed=typed,
                                       titles_seen=titles_seen,
                                       seen_before=seen_before,
                                       last_dump=self._last_dump,
                                       detail=detail)
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
                    _ms = seen_sig.get(m_key)
                    if _ms is not None and _ms[0] == post \
                            and attempts.get(m_key, 0) >= 2 \
                            and _ms[1] == _frame_sig(post_cap):
                        # The suggested step was already executed twice in
                        # this exact state: the repeat guard would block
                        # it, so it is stale. (One execution in this state
                        # is NOT stale — the guard permits a retry, which
                        # is the right move when the first attempt
                        # grounded on a stale frame.) Anchoring on a step
                        # that cannot run traps the loop (measured: run 9,
                        # three blocked wait-loads in a row burned the
                        # failure budget).
                        pending = None
                        history.append(
                            "SUGGESTED NEXT: continue with the next "
                            "unfinished subgoal (no canonical transition "
                            "is pending).")
                    else:
                        history.append(sug_line)
                        pending = machine

            # Research goals: the visited-source state is kept OUTSIDE
            # the planner (measured: the 1.7B re-read the same dump and
            # re-opened the same first result, then spammed ctrl+Left —
            # it cannot remember what it has visited; the machine has
            # it in titles_seen, and this line is the measured fact it
            # is missing, in its own history).
            if not _wants_a_specific_result(goal):
                _pages = _page_titles(titles_seen)
                _prog = (f"PROGRESS: distinct sources opened so far: "
                         f"{len(_pages)}")
                if _pages:
                    _prog += " — " + "; ".join(p[:60] for p in _pages)
                _n = _counted_open(goal)
                if _n >= 2 and len(_pages) < _n:
                    _prog += (f"; the goal asks for at least {_n} — open a "
                              "source you have NOT opened yet (a "
                              "different numbered result from the read "
                              "dump)")
                history.append(_prog)
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