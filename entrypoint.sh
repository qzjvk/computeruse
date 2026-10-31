#!/usr/bin/env bash
# Sandbox entrypoint: virtual XFCE desktop + VNC + resident agent worker.
#
# The GPU driver's kernel module lives on the HOST. The nvidia runtime hook
# mounts the host's driver libs (libnvidia-ml.so.<ver>, libcuda.so.<ver>) at
# their original paths, but the container's own ld.so cache only knows the
# .so.1 name, so we symlink them in before anything touches CUDA.
set -e

for f in /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.[0-9]* /usr/lib/x86_64-linux-gnu/libcuda.so.[0-9]*; do
    [ -f "$f" ] || continue
    base=$(basename "$f" | sed 's/\.so\.[0-9].*/.so.1/')
    ln -sf "$f" "/usr/lib/x86_64-linux-gnu/$base"
done
ldconfig 2>/dev/null || true

export DISPLAY=:99
# Xvfb corrupts the XCB/XShmGetImage path mss 10 uses by default; the
# legacy Xlib XGetImage backend is clean (verified against `import`).
export CUA_MSS_BACKEND=xlib
# 720p, not 1080p: the full screen is then 4641 vision patches instead of
# 10549, so a full-frame step is ONE ~0.94x pass (~29 s on the 1070) instead
# of a 4-tile pass (~67 s), and the model sees the UI at near-native scale
# instead of down-scaled tiles. The edge strips (64/48 px) are unaffected:
# they are full-width crops either way.
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX &

sleep 3
eval "$(dbus-launch --sh-syntax)"
xfce4-session &
firefox-esr --no-sandbox &

sleep 2
# Pin the launcher panel in place. The image's XFCE config ships the bottom
# launcher (panel-2) with autohide-behavior=1 ("hide on focus"): whenever an
# app window (e.g. a terminal left open by a previous task) holds focus, the
# panel slides off-screen and its launcher icons disappear from the capture.
# A focus-dependent panel makes the desktop state non-deterministic for any
# task that touches it, so pin it at container start. The panel watches its
# own xfconf channel, so the change applies live; if the panel has not
# started yet, it simply reads the pinned value on startup.
for _ in $(seq 1 30); do
    if xfconf-query -c xfce4-panel -p /panels/panel-2/autohide-behavior >/dev/null 2>&1; then
        xfconf-query -c xfce4-panel -p /panels/panel-2/autohide-behavior -t uint -s 0 \
            || echo "[sandbox] warning: could not pin the launcher panel"
        break
    fi
    sleep 0.5
done
# -ncache 20: keep 20 changed sub-regions on the server side so redraws of
# small areas (cursor, panels) don't re-transfer the whole screen.
x11vnc -display :99 -forever -nopw -shared -noxdamage -ncache 20 &
websockify --web=/usr/share/novnc 6090 localhost:5900 &

echo "[sandbox] XFCE up on :99"
echo "[sandbox]   browser view: http://localhost:6090/vnc.html"
echo "[sandbox]   native VNC:   :5900 (any VNC client, e.g. vinagre/TigerVNC)"
echo "[sandbox] starting agent worker (model load ~30 s)..."

# The sandbox runs on the GTX 1070. Find it by name in CUDA's own device
# enumeration: nvidia-smi indices do NOT match torch/cuda indices on this
# host (different ordering), and the index can change across boots anyway.
# Fall back to auto-pick (most free VRAM) if the card is not present.
GPU1070=$(python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    if '1070' in torch.cuda.get_device_properties(i).name:
        print(i)
        break
" 2>/dev/null || true)

cd /app
# Run the resident worker in a respawn loop instead of exec-ing it:
# `docker exec cuasandbox pkill -f 'computer_use[ ]serve'` then restarts ONLY
# the worker (~40 s: 5 s delay + model load) without resetting the desktop,
# the VNC session or the browser. Code changes need no image rebuild —
# the container mounts the host's computer_use/ live.
while true; do
    # --quant nf4 by default: the 1070 is SHARED with the host's own
    # desktop (X server, compositor, browser run on it). fp16 holds
    # ~6.6 GB of the 8 GB card and leaves ~1.1 GB free; a model-reload
    # VRAM spike in that headroom wedged the driver and froze the host
    # machine until a reboot (measured incident, 2026-10-31). NF4 holds
    # ~3.5 GB and leaves ~4 GB for the desktop. Speed is unchanged on
    # the 1070 (measured: the 4-bit dequant kernel does not beat fp16 —
    # prefill is attention/compute-bound, and for small crops the 4-bit
    # weight stream is actually slower), so nothing is lost.
    python -m computer_use serve --model /locatenv --scope window --quant nf4 ${GPU1070:+--gpu "$GPU1070"} || rc=$?
    echo "[sandbox] worker exited (code ${rc:-0}); restarting in 5 s"
    sleep 5
done