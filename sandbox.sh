#!/usr/bin/env bash
# Zero-friction control for the XFCE sandbox (docker compose based).
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

# Prefer the docker compose plugin, fall back to the docker-compose binary.
if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
else
    COMPOSE=(docker-compose)
fi

cmd=${1:-up}
shift || true

case "$cmd" in
    up)
        # The sandbox owns the model now; drop any host-side resident
        # worker so both are not fighting over the same VRAM.
        pkill -f "computer_use serve" 2>/dev/null || true
        if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)" = "true" ]; then
            echo "Sandbox is already running."
        else
            # Remove stale/non-compose containers with the same name.
            docker rm -f "$NAME" 2>/dev/null || true
            "${COMPOSE[@]}" up -d --build
            echo "Sandbox is up (model loads for ~40 s; 'status' shows healthy when ready)."
        fi
        echo "  Desktop:  http://localhost:6090/vnc.html"
        echo "  Task:     $0 task \"click the address bar; done\""
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
        "${COMPOSE[@]}" ps
        docker logs --tail 20 "$NAME" 2>/dev/null || true
        ;;
    down)
        "${COMPOSE[@]}" down
        ;;
    *)
        echo "usage: $0 {up|down|task|dry|shell|status} [args]" >&2
        exit 2
        ;;
esac