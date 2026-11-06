#!/usr/bin/env bash
# Starts the ai-console container, with `container` (macOS), podman or docker (Linux):
#  - port published on 127.0.0.1:$HOST_PORT and, if LAN_IP is set, also on
#    $LAN_IP:$HOST_PORT, to reach it from the other computers on the network;
#  - for ai-console's nodes, the host's ssh-agent if the runtime can forward it
#    (`container --ssh`), otherwise the keys in NODE_KEYS, mounted read-only and
#    copied into ~/.ssh by entrypoint.sh.
# Build the image first with build.sh.
#
# Usage: container/run.sh            ai-console in the background
#        container/run.sh bash       interactive shell instead of ai-console
# Settings for a single host (e.g. LAN_IP) go in run.env next to this script
# (see run.env.example): if it exists it is read first, by build.sh too.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f run.env ]]; then source ./run.env; fi

RUNTIME="${RUNTIME:-$(command -v container || command -v podman || command -v docker || true)}"
IMAGE="${IMAGE:-ai-console:latest}"
NAME="${NAME:-ai-console}"
HOST_PORT="${HOST_PORT:-8501}"                 # port on the host
LAN_IP="${LAN_IP:-}"                           # host IP on the LAN (empty: 127.0.0.1 only)
NODE_KEYS="${NODE_KEYS-$HOME/.ssh/id_ecdsa}"   # keys the host uses to log in to the nodes
APP_PORT=8501                                  # ai-console's port in the container (UI_PORT in app/config.json)

if [[ -z "$RUNTIME" ]]; then
    echo "run.sh: needs container (macOS), podman or docker" >&2
    exit 1
fi
rt=${RUNTIME##*/}

opts=(--rm --name "$NAME" -p "127.0.0.1:$HOST_PORT:$APP_PORT")
if [[ -n "$LAN_IP" ]]; then opts+=(-p "$LAN_IP:$HOST_PORT:$APP_PORT"); fi

# Mounts a key (and its .pub) read-only in /run/host-ssh/nodes/.
mount_key() {
    opts+=(-v "$1:/run/host-ssh/nodes/${1##*/}:ro")
    if [[ -f "$1.pub" ]]; then opts+=(-v "$1.pub:/run/host-ssh/nodes/${1##*/}.pub:ro"); fi
}

# Nodes: with the agent the keys stay on the host and the container can only ask
# it to sign. Podman cannot forward it, and on a server a session's agent does
# not last: there the keys are mounted.
if [[ $rt == container && -S "${SSH_AUTH_SOCK:-}" ]]; then
    opts+=(--ssh)
else
    for f in $NODE_KEYS; do
        f="${f/#\~/$HOME}"
        if [[ -f "$f" ]]; then mount_key "$f"; else echo "run.sh: key $f not found" >&2; fi
    done
fi

# Rootless podman: same UID inside and outside, otherwise the user cannot read the keys.
if [[ $rt == podman ]]; then opts+=(--userns=keep-id); fi

if (( $# > 0 )); then
    exec "$RUNTIME" run -it "${opts[@]}" "$IMAGE" "$@"
fi

"$RUNTIME" run -d "${opts[@]}" "$IMAGE" > /dev/null
echo "ai-console: http://127.0.0.1:$HOST_PORT"
if [[ -n "$LAN_IP" ]]; then echo "            http://$LAN_IP:$HOST_PORT"; fi
cat <<EOF
  log:      $rt logs -f $NAME
  shell:    $rt exec -it $NAME bash
  version:  $rt exec $NAME printenv AI_CONSOLE_VERSION
  stop:     $rt stop $NAME
EOF
