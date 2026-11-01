"""AT-SPI accessibility-tree content channel (the `read` verb).

The planner's other evidence channels (window title, frame diff) cannot
see PAGE CONTENT — and by design the vision model is grounding, not VQA.
This module closes that gap with a measured, non-LLM source: the
accessibility bus. Every app registered on the AT-SPI bus (Firefox, the
terminal, ...) exposes its live UI tree, and the rendered DOM text, link
names and even element screen coordinates are readable from it.

Design constraints (measured):
  * The traversal runs as a SUBPROCESS under a gi-capable interpreter:
    the worker's /usr/local/bin/python3 has no `gi`; /usr/bin/python3
    (distro python) does, after `python3-gi gir1.2-atspi-2.0`.
  * AT-SPI calls are synchronous D-Bus calls and CAN BLOCK (a stalled
    bridge was measured to hang a traversal > 45 s), so the subprocess
    gets a hard timeout and a node budget.
  * Firefox exposes its tree only while org.a11y.Status.IsEnabled is on
    (the entrypoint toggles it at boot, before any app starts).
  * Output is a compact dump (title / results / text / money / links),
    capped for the planner's context budget — it is carried in history
    and re-sent on later turns.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Hard cap: a traversal that needs more than this is a stalled bridge,
# not a slow page (the live tree of a results page is < 2000 nodes).
_DUMP_TIMEOUT = 8.0

# Runs under /usr/bin/python3 (distro python with gi + Atspi). The
# worker's env carries DISPLAY + DBUS_SESSION_BUS_ADDRESS, which is how
# the Atspi client finds the a11y bus (org.a11y.Bus on the session bus).
#
# The script is a standalone `python -c` payload: everything it needs
# (roles set, money regex) must be defined INSIDE the string — module
# constants of atspi.py are not visible to it (measured: a NameError
# there was swallowed by walk's except and returned a title-only dump).
_DUMP_SCRIPT = r"""
import re, sys
import gi
gi.require_version("Atspi", "2.0")
from gi.repository import Atspi

MAX_NODES = 1600
TITLE_MAX, TEXT_MAX, RESULT_MAX, LINK_MAX = 120, 700, 12, 14
# Roles whose get_text() carries rendered page content. Leaf-only text
# collection (a node is taken only when no child carries text) avoids
# the tree's name/text duplication (heading -> link -> section).
_TEXT_ROLES = frozenset(
    {"paragraph", "static", "label", "section", "heading", "list item",
     "article", "document web", "terminal", "text", "entry"})
_MONEY_RE = re.compile(
    r"(\d{1,7}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*[€$£]|[€$£]\s?"
    r"(\d{1,7}(?:[.,]\d{3})*)")

def _text(node):
    try:
        return node.get_text(0, -1) or ""
    except Exception:
        return ""

def _norm(s):
    s = s.replace("\uFFFC", " ").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()

desktop = Atspi.get_desktop(0)
app = None
for i in range(desktop.get_child_count()):
    a = desktop.get_child_at_index(i)
    if a and "firefox" in (a.get_name() or "").lower():
        app = a
        break
if app is None:  # fall back to the app with the biggest frame
    best, best_area = None, 0
    for i in range(desktop.get_child_count()):
        a = desktop.get_child_at_index(i)
        try:
            n = a.get_child_count()
        except Exception:
            continue
        for j in range(n):
            f = a.get_child_at_index(j)
            if not f:
                continue
            try:
                e = f.get_extents(Atspi.CoordType.SCREEN)
            except Exception:
                continue
            if e.width * e.height > best_area:
                best, best_area = a, e.width * e.height
    app = best
if app is None:
    sys.exit(3)  # no app registered (a11y bus off / apps predate it)

title, results, texts, links, nodes = [], [], [], [], [0]

def walk(node):
    if nodes[0] > MAX_NODES:
        return
    nodes[0] += 1
    try:
        role = node.get_role_name()
        name = _norm(node.get_name() or "")
        kids = node.get_child_count()
        child = [node.get_child_at_index(i) for i in range(kids)]
        child = [c for c in child if c is not None]
        text = _norm(_text(node)) if role in _TEXT_ROLES else ""
    except Exception:
        return
    if role == "link" and name:
        if name not in links:
            links.append(name)
    if role == "heading" and name:
        # a search-result row: heading title + (url link + snippet)
        url, snip, heading_name = "", "", ""
        def scan(n, depth):
            nonlocal url, snip, heading_name
            if nodes[0] > MAX_NODES:
                return
            nodes[0] += 1
            try:
                r = n.get_role_name()
                nm = _norm(n.get_name() or "")
            except Exception:
                return
            if r == "link" and nm.startswith("http") and not url:
                url = nm
            elif r == "link" and nm and not heading_name:
                heading_name = nm
            elif r in _TEXT_ROLES and not snip:
                t = _norm(_text(n))
                if t and t not in (heading_name, url):
                    snip = t
            for i in range(n.get_child_count()):
                c = n.get_child_at_index(i)
                if c:
                    scan(c, depth + 1)
        for c in child:
            scan(c, 1)
        h = heading_name or name
        if h and len(results) < RESULT_MAX:
            results.append(f"{len(results) + 1}. {h[:90]}"
                           + (f" | {url[:90]}" if url else "")
                           + (f" :: {snip[:110]}" if snip else ""))
    if text:
        # leaf-only: take this node's text when no child carries text,
        # which collapses the heading/link/section duplication
        if not any(_norm(_text(c)) for c in child):
            texts.append(text)
    for c in child:
        walk(c)

for i in range(app.get_child_count()):
    f = app.get_child_at_index(i)
    if f:
        try:
            title = _norm(f.get_name() or "")
        except Exception:
            pass
        walk(f)

out = []
if title:
    out.append("title: " + title[:TITLE_MAX])
if results:
    out.append("results:\n" + "\n".join(results))
    # on a results page the 'results:' lines already carry the page's
    # substance; a long 'text:' section would only eat context
    body_max = 300
else:
    body_max = TEXT_MAX
if texts:
    body = " ".join(texts)
    out.append("text: " + body[:body_max])
money = []
for s in texts:
    for m in _MONEY_RE.finditer(s):
        tok = m.group(0).strip()
        if tok not in money:
            money.append(tok)
    if len(money) >= 20:
        break
if money:
    out.append("money: " + " | ".join(money))
if links:
    out.append("links: " + " | ".join(l[:60] for l in links[:LINK_MAX]))
print("\n".join(out))
"""

_PROBE_OK = None
_PY = None


def _pick_interpreter() -> str | None:
    """The first python that can `import gi` + Atspi (cached)."""
    global _PROBE_OK, _PY
    if _PY is not None:
        return _PY
    for cand in (sys.executable, "/usr/bin/python3"):
        try:
            r = subprocess.run(
                [cand, "-c",
                 "import gi; gi.require_version('Atspi','2.0'); "
                 "from gi.repository import Atspi"],
                capture_output=True, timeout=10)
            if r.returncode == 0:
                _PY = cand
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    _PY = ""  # cached miss: don't re-probe every read
    return None


def dump_active() -> str | None:
    """Compact dump of the active app's a11y tree, or None if the
    channel is unavailable (no gi-capable python, bus off, hang)."""
    py = _pick_interpreter()
    if not py:
        return None
    try:
        r = subprocess.run([py, "-c", _DUMP_SCRIPT],
                           capture_output=True, text=True,
                           timeout=_DUMP_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None
    text = (r.stdout or "").strip()
    if not text:
        return None
    return text