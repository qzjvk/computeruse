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
# Inside: Xvfb + XFCE (1920x1080), noVNC on http://localhost:6090/vnc.html,
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY computer_use/ ./computer_use/
COPY requirements.txt .

# torch first (own index, big), then the rest of the stack.
RUN pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6090
ENTRYPOINT ["/entrypoint.sh"]