#!/usr/bin/env bash
# Zero-friction control for the XFCE sandbox container.
#
#   ./sandbox.sh up                  # build (cached) + start the sandbox
#   ./sandbox.sh task "click ..."    # run a task through the resident worker
#   ./sandbox.sh dry "click ..."     # dry run (grounds, does not move the mouse)
#   ./sandbox.sh shell               # bash inside the container
#   ./sandbox.sh status              # container state + recent logs
#   ./sandbox.sh down                # stop and remove
#
# Desktop view: http://localhost:6090/vnc.html
set -euo pipefail
cd "$(dirname "$0")"

NAME=cuasandbox
IMG=computeruse-sandbox:latest

# The nvidia hook mounts the GPU nodes but the container runtime (runc 1.1.x
# on cgroup v2) does not learn about them; passing them as explicit --device
# flags puts them in the original OCI spec, which is what allows open().
gpu_devices() {
    local args=()
    local d
    for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia[0-9]*; do
        [ -e "$d" ] && args+=(--device "$d")
    done
    printf '%s\n' "${args[@]}"
}

cmd=${1:-up}
shift || true

case "$cmd" in
    up)
        # The sandbox owns the model now; drop any host-side resident worker
        # so both are not fighting over the same VRAM.
        pkill -f "computer_use serve" 2>/dev/null || true
        docker build -t "$IMG" .
        docker rm -f "$NAME" 2>/dev/null || true
        # shellcheck disable=SC2046
        docker run -d --name "$NAME" --gpus all --shm-size 4g \
            -v "$PWD/locatenv:/locatenv:ro" \
            -v "$PWD/runs:/app/runs" \
            -p 6090:6090 \
            $(gpu_devices) \
            "$IMG"
        echo "Sandbox is up (model loads for ~30 s)."
        echo "  Desktop:  http://localhost:6090/vnc.html"
        echo "  Task:     $0 task \"click the firefox icon; done\""
        ;;
    task)
        exec docker exec "$NAME" python -m computer_use "$@"
        ;;
    dry)
        exec docker exec "$NAME" python -m computer_use --dry-run "$@"
        ;;
    shell)
        exec docker exec -it "$NAME" bash
        ;;
    status)
        docker ps -a --filter "name=$NAME"
        docker logs --tail 20 "$NAME" 2>/dev/null || true
        ;;
    down)
        docker rm -f "$NAME"
        ;;
    *)
        echo "usage: $0 {up|down|task|dry|shell|status} [args]" >&2
        exit 2
        ;;
esac