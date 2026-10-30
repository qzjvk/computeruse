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
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX &

sleep 3
eval "$(dbus-launch --sh-syntax)"
xfce4-session &
firefox-esr --no-sandbox &

sleep 2
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
exec python -m computer_use serve --model /locatenv --scope window ${GPU1070:+--gpu "$GPU1070"}