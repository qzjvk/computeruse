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
x11vnc -display :99 -forever -nopw -shared -noxdamage &
websockify --web=/usr/share/novnc 6090 localhost:5900 &

echo "[sandbox] XFCE up on :99, desktop at http://localhost:6090/vnc.html"
echo "[sandbox] starting agent worker (model load ~30 s)..."

cd /app
exec python -m computer_use serve --model /locatenv --scope window