#!/usr/bin/env bash
# Zero-friction control for the XFCE sandbox (docker compose based).
#
#   ./sandbox.sh up                  # build (cached) + start the sandbox
#   ./sandbox.sh restart             # reload the worker only (~40 s; no
#                                    # image rebuild, desktop/VNC untouched)
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
        # Drop a pre-compose (docker run) container with the same name; it
        # would block compose. Compose-managed containers pass this check.
        if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)" = "true" ] \
           && [ -z "$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$NAME" 2>/dev/null)" ]; then
            docker rm -f "$NAME" 2>/dev/null || true
        fi
        # Idempotent: no-op while running with the same config, recreates
        # when the compose file changed, builds when the image is stale.
        "${COMPOSE[@]}" up -d --build
        echo "Sandbox is up (model loads for ~40 s; 'status' shows healthy when ready)."
        echo "  Desktop (browser): http://localhost:6090/vnc.html"
        echo "  Desktop (native):  VNC :5900 (crisp 1:1, any VNC client)"
        echo "  Task:              $0 task \"click the address bar; done\""
        ;;
    restart)
        # Reload the worker process only: the entrypoint's respawn loop
        # brings serve back in ~40 s; the desktop, VNC and browser stay up.
        # (Code is mounted live, so no image rebuild is needed for edits.)
        docker exec "$NAME" pkill -f "computer_use[ ]serve" || true
        # Drop the stale socket (SIGTERM skips serve's cleanup), so the
        # health check only goes healthy when the NEW worker is ready.
        docker exec "$NAME" rm -f /tmp/computeruse.sock || true
        echo "worker restarting (~40 s: respawn delay + model load)..."
        # Poll the socket, not the container health: serve binds the socket
        # only after the model is loaded, and the health status lags a probe
        # interval behind (it read stale-healthy right after the kill).
        for i in $(seq 1 60); do
            if docker exec "$NAME" test -S /tmp/computeruse.sock 2>/dev/null; then
                echo "worker ready"; break
            fi
            sleep 2
        done
        ;;
    task)
        # CUA_WAIT_SERVER: if the worker is mid-restart, wait for the socket
        # instead of loading a second model copy on the same GPU.
        exec docker exec -e CUA_WAIT_SERVER=120 "$NAME" python -m computer_use "$@"
        ;;
    dry)
        exec docker exec -e CUA_WAIT_SERVER=120 "$NAME" python -m computer_use --dry-run "$@"
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
        echo "usage: $0 {up|down|restart|task|dry|shell|status} [args]" >&2
        exit 2
        ;;
esac