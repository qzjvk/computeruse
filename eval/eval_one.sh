#!/bin/bash
# usage: eval_one.sh <id> <goalfile> <setup:none|terminal|killfirefox|preddg> <checkfile> <expected> [maxsteps]
# Runs ONE goal through the orchestration layer (LFM3 planner +
# deterministic controller) and prints a compact evidence summary.
# maxsteps: optional — pass --max-steps to the orchestrator (default 32).
export DISPLAY=:99
ID=$1; GOAL=$2; SETUP=$3; CHECKFILE=$4; EXPECTED=$5; MAXSTEPS=$6

# --- verified reset (close terminal with retry, about:blank, fresh files) ---
rm -f /root/note.txt /root/note2.txt /root/multi.txt
[ -n "$CHECKFILE" ] && rm -f "$CHECKFILE"
ok_reset=0
for i in 1 2 3; do
  xdotool search --name "Terminal" windowclose 2>/dev/null; sleep 0.5
  [ -z "$(xdotool search --name "Terminal" 2>/dev/null)" ] && { ok_reset=1; break; }
done
# The location bar is FOCUSED from the a11y tree on every attempt
# (grab_focus on the node named/roled as the bar — its name varies
# per search engine, 'Search with Google or enter address' /
# 'Search with DuckDuckGo'), not clicked and not ctrl+l'd: the live
# shared desktop moves/resizes the window (measured coordinates go
# stale — a click from an old screenshot landed in the DDG search
# box, which submitted "about:blank" as a query), a DDG page
# auto-focuses its own search box and swallows synthetic keystrokes,
# and a plain click on the bar can land on its globe icon and do
# nothing (all measured). Verify window focus before typing (a focus
# race sends the typing wherever the X input focus is). Fallback:
# ctrl+l (works off DDG pages). Verify+retry.
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
[ $ok_nav = 1 ] || { echo "$ID: RESET-FAIL($T)"; exit 2; }
[ $ok_reset = 1 ] || { echo "$ID: TERMINAL-WONT-DIE"; exit 2; }

# --- perturbation setup (handled by the orchestration layer itself) ---
case "$SETUP" in
  terminal) setsid xfce4-terminal >/dev/null 2>&1 & sleep 2.5;;
  killfirefox) pkill -9 -f "firefox-esr"; sleep 3;;
  preddg)
    xdotool search --name "Mozilla Firefox" windowactivate --sync
    sleep 0.4; xdotool key ctrl+l; sleep 0.3
    xdotool type "https://duckduckgo.com/?q=ubuntu+install+docker"; sleep 0.3
    xdotool key Return; sleep 8;;
esac

# --- the run: orchestrate kind -> OrchestratedAgent + LFM3 Planner ---
STEPS_ARG=()
[ -n "$MAXSTEPS" ] && STEPS_ARG=(--max-steps "$MAXSTEPS")
CUA_WAIT_SERVER=60 timeout 300 python -m computer_use \
  --save-dir /tmp/eval_runs --task-file "$GOAL" "${STEPS_ARG[@]}" orchestrate \
  > "/tmp/eval_${ID}.log" 2>&1
RC=$?

STEPS=$(grep -cE '^\[orch\] step [0-9]+: (search|click|read|wait-load|scroll|open|type|key|done) ' "/tmp/eval_${ID}.log")
PROTO=$(grep -c "protocol: executing" "/tmp/eval_${ID}.log")
LOOP=$(grep -oE '\([0-9.]+ s total, [0-9]+ steps\)' "/tmp/eval_${ID}.log" | tail -1)
AMT=$(grep -oE 'amount.: [0-9]+' "/tmp/eval_${ID}.log" | tail -1)
TITLE=$(xdotool search --name "Mozilla Firefox" getwindowname 2>/dev/null | head -1)
TERMPID=$(pgrep -c xfce4-terminal 2>/dev/null || true); TERMPID=${TERMPID:-0}
S1=$(grep -E "step 1: search" "/tmp/eval_${ID}.log" | grep -oE "in [0-9.]+s" | head -1)
FAILED=$(grep -c "FAILED" "/tmp/eval_${ID}.log")
# proof the orchestration layer (LFM3) was actually used this run:
LAYER=$(grep -c "LFM3 server ready" "/tmp/eval_${ID}.log")

echo "$ID: exit=$RC $LOOP steps=$STEPS planner=$((STEPS-PROTO)) lfm3_ready=$LAYER scroll_amt=$AMT s1_search=$S1 failed_lines=$FAILED term_procs=$TERMPID"
echo "   tab: $TITLE"
if [ -n "$CHECKFILE" ]; then
  if [ -f "$CHECKFILE" ]; then
    ACT=$(cat "$CHECKFILE")
    if [ "$ACT" = "$EXPECTED" ]; then
      echo "   FILE-OK (mtime $(stat -c %y "$CHECKFILE" | cut -c1-19))"
    else
      echo "   FILE-BAD:"; echo "$ACT" | head -4 | sed "s/^/     | /"
    fi
  else
    echo "   FILE-MISSING"
  fi
else
  [ -e "$CHECKFILE" ] && echo "   FILE-EXISTS-BUT-WAS-NOT-EXPECTED" || echo "   (no file expected)"
fi
DONE=$(grep -oE '^\[orch\] done: .*' "/tmp/eval_${ID}.log" | head -1 | cut -c1-140)
[ -n "$DONE" ] && echo "   $DONE"
if [ $RC -ne 0 ] || [ -n "$CHECKFILE" ] && [ ! -f "$CHECKFILE" ]; then
  tail -10 "/tmp/eval_${ID}.log" | sed "s/^/   | /"
fi