#!/usr/bin/env bash
# Starts the ai-console container, with `container` (macOS) or podman (Linux):
#  - port published on 127.0.0.1:$HOST_PORT and, if LAN_IP is set, also on
#    $LAN_IP:$HOST_PORT, to reach it from the other computers on the network;
#  - in ~/.ssh the keys the host uses for GitHub, mounted read-only and copied
#    by entrypoint.sh;
#  - for ai-console's nodes, the host's ssh-agent if the runtime can forward it
#    (`container --ssh`), otherwise the keys in NODE_KEYS, mounted as above.
#
# Usage: ./run.sh            ai-console in the background
#        ./run.sh bash       interactive shell instead of ai-console
# Settings for a single host (e.g. LAN_IP) go in run.env next to this script:
# if it exists it is read first, by build.sh too.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f run.env ]]; then source ./run.env; fi

RUNTIME="${RUNTIME:-$(command -v container || command -v podman || true)}"
IMAGE="${IMAGE:-pi-agent:latest}"
NAME="${NAME:-ai-console}"
HOST_PORT="${HOST_PORT:-8501}"                 # port on the host
LAN_IP="${LAN_IP:-}"                           # host IP on the LAN (empty: 127.0.0.1 only)
NODE_KEYS="${NODE_KEYS-$HOME/.ssh/id_ecdsa}"   # keys the host uses to log in to the nodes
APP_PORT=8501                                  # ai-console's port in the container (UI_PORT in config.json)

if [[ -z "$RUNTIME" ]]; then
    echo "run.sh: needs container (macOS) or podman" >&2
    exit 1
fi
rt=${RUNTIME##*/}

# Mounts a key (and its .pub) read-only in /run/host-ssh/<folder>/.
mounts=()
mount_key() {
    mounts+=(-v "$2:/run/host-ssh/$1/${2##*/}:ro")
    if [[ -f "$2.pub" ]]; then mounts+=(-v "$2.pub:/run/host-ssh/$1/${2##*/}.pub:ro"); fi
}

# GitHub: the existing IdentityFiles according to the host's ssh config.
while read -r _ f; do
    f="${f/#\~/$HOME}"
    if [[ -f "$f" ]]; then mount_key github "$f"; fi
done < <(ssh -T -G github.com | awk '$1 == "identityfile"')
if (( ${#mounts[@]} == 0 )); then
    echo "run.sh: no key for github.com in 'ssh -G github.com'" >&2
    exit 1
fi

opts=(--rm --name "$NAME" -p "127.0.0.1:$HOST_PORT:$APP_PORT")
if [[ -n "$LAN_IP" ]]; then opts+=(-p "$LAN_IP:$HOST_PORT:$APP_PORT"); fi

# Nodes: with the agent the keys stay on the host and the container can only ask
# it to sign. Podman cannot forward it, and on a server a session's agent does
# not last: there the keys are mounted.
if [[ $rt == container && -S "${SSH_AUTH_SOCK:-}" ]]; then
    opts+=(--ssh)
else
    for f in $NODE_KEYS; do
        f="${f/#\~/$HOME}"
        if [[ -f "$f" ]]; then mount_key nodes "$f"; else echo "run.sh: key $f not found" >&2; fi
    done
fi

# Rootless podman: same UID inside and outside, otherwise alvise cannot read the keys.
if [[ $rt == podman ]]; then opts+=(--userns=keep-id); fi

opts+=("${mounts[@]}")
if (( $# > 0 )); then
    exec "$RUNTIME" run -it "${opts[@]}" "$IMAGE" "$@"
fi

"$RUNTIME" run -d "${opts[@]}" "$IMAGE" > /dev/null
echo "ai-console: http://127.0.0.1:$HOST_PORT"
if [[ -n "$LAN_IP" ]]; then echo "            http://$LAN_IP:$HOST_PORT"; fi
cat <<EOF
  log:    $rt logs -f $NAME
  shell:  $rt exec -it $NAME bash
  stop:   $rt stop $NAME
EOF
