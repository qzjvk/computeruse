# XFCE sandbox for the computer-use agent.
#
#   docker build -t computeruse-sandbox:latest .
#   docker run -d --gpus all --device /dev/nvidiactl --device /dev/nvidia0 \
#     --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
#     -v "$PWD/locatenv:/locatenv:ro" -p 6080:6080 --name cuasandbox \
#     computeruse-sandbox:latest
#
# Or just: ./sandbox.sh up
#
# Inside: Xvfb + XFCE (1280x720), noVNC on http://localhost:6090/vnc.html,
# and a resident agent worker (`serve`) on /tmp/computeruse.sock.

FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        xfce4 \
        xfce4-terminal \
        x11-utils \
        xvfb \
        x11vnc \
        novnc \
        dbus dbus-x11 \
        xdotool \
        firefox-esr \
        # AT-SPI accessibility stack: the `read` verb (computer_use/atspi.py)
        # dumps the rendered text of the active app from the a11y tree —
        # the only page-CONTENT channel a text planner has. at-spi2-core
        # provides the bus launcher, libatk-bridge2.0-0 makes GTK apps
        # (Firefox, the terminal) export their tree, python3-gi +
        # gir1.2-atspi-2.0 let /usr/bin/python3 (distro python, which the
        # worker's /usr/local python lacks) drive the tree, python3-dbus
        # is the low-level fallback. The entrypoint starts the bus and
        # toggles org.a11y.Status IsEnabled ON before any app starts
        # (apps read it at startup).
        at-spi2-core \
        libatk-bridge2.0-0 \
        python3-gi \
        gir1.2-atspi-2.0 \
        python3-dbus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# requirements.txt first: it rarely changes, so the big pip layer stays
# cached. The code itself is live-mounted by docker-compose.yml
# (./computer_use -> /app/computer_use), so edits never force a rebuild;
# the COPY below is just a fallback for running the image standalone.
COPY requirements.txt .

# torch first (own index, big), then the rest of the stack.
RUN pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

COPY computer_use/ ./computer_use/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6090
ENTRYPOINT ["/entrypoint.sh"]