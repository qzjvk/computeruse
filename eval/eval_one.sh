#!/bin/bash
# usage: eval_one.sh <id> <goalfile> <setup:none|terminal|killfirefox|preddg> <checkfile> <expected>
# Runs ONE goal through the orchestration layer (LFM3 planner +
# deterministic controller) and prints a compact evidence summary.
export DISPLAY=:99
ID=$1; GOAL=$2; SETUP=$3; CHECKFILE=$4; EXPECTED=$5

# --- verified reset (close terminal with retry, about:blank, fresh files) ---
rm -f /root/note.txt /root/note2.txt /root/multi.txt
ok_reset=0
for i in 1 2 3; do
  xdotool search --name "Terminal" windowclose 2>/dev/null; sleep 0.5
  [ -z "$(xdotool search --name "Terminal" 2>/dev/null)" ] && { ok_reset=1; break; }
done
# ctrl+l can drop while the WM focus settles after windowclose; the
# keystrokes then land in the current page's search box (T2: "about:blank"
# typed into the DDG box). Verify+retry the navigation instead.
ok_nav=0
for i in 1 2 3; do
  xdotool search --name "Mozilla Firefox" windowactivate --sync 2>/dev/null
  sleep 0.6
  xdotool key ctrl+l 2>/dev/null; sleep 0.4
  xdotool type "about:blank" 2>/dev/null; sleep 0.3
  xdotool key Return 2>/dev/null; sleep 1.2
  T=$(xdotool getactivewindow getwindowname 2>/dev/null)
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
CUA_WAIT_SERVER=60 timeout 300 python -m computer_use \
  --save-dir /tmp/eval_runs --task-file "$GOAL" orchestrate \
  > "/tmp/eval_${ID}.log" 2>&1
RC=$?

STEPS=$(grep -cE '^\[orch\] step [0-9]+: (search|click|wait-load|scroll|open|type|key|done) ' "/tmp/eval_${ID}.log")
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