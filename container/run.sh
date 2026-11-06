#!/usr/bin/env bash
# Avvia il container pi-dev con ai-console, con `container` (macOS) o podman (Linux):
#  - porta pubblicata su 127.0.0.1:$HOST_PORT e, se LAN_IP è impostato, anche su
#    $LAN_IP:$HOST_PORT, per raggiungerla dagli altri computer della rete;
#  - in ~/.ssh le chiavi che l'host usa per GitHub, montate in sola lettura e
#    copiate da entrypoint.sh;
#  - per i nodi di ai-console l'ssh-agent dell'host se il runtime sa inoltrarlo
#    (`container --ssh`), altrimenti le chiavi in NODE_KEYS montate come sopra.
#
# Uso: ./run.sh            ai-console in background
#      ./run.sh bash       shell interattiva al posto di ai-console
# Le impostazioni di un singolo host (es. LAN_IP) vanno in run.env accanto a
# questo script: se esiste viene letto per primo, anche da build.sh.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f run.env ]]; then source ./run.env; fi

RUNTIME="${RUNTIME:-$(command -v container || command -v podman || true)}"
IMAGE="${IMAGE:-pi-agent:latest}"
NAME="${NAME:-pi-dev}"
HOST_PORT="${HOST_PORT:-8501}"                 # porta sull'host
LAN_IP="${LAN_IP:-}"                           # IP dell'host nella LAN (vuoto: solo 127.0.0.1)
NODE_KEYS="${NODE_KEYS-$HOME/.ssh/id_ecdsa}"   # chiavi con cui l'host entra nei nodi
APP_PORT=8501                                  # porta di ai-console nel container (UI_PORT di config.json)

if [[ -z "$RUNTIME" ]]; then
    echo "run.sh: serve container (macOS) o podman" >&2
    exit 1
fi
rt=${RUNTIME##*/}

# Monta in sola lettura una chiave (e la sua .pub) in /run/host-ssh/<cartella>/.
mounts=()
mount_key() {
    mounts+=(-v "$2:/run/host-ssh/$1/${2##*/}:ro")
    if [[ -f "$2.pub" ]]; then mounts+=(-v "$2.pub:/run/host-ssh/$1/${2##*/}.pub:ro"); fi
}

# GitHub: le IdentityFile esistenti secondo la config ssh dell'host.
while read -r _ f; do
    f="${f/#\~/$HOME}"
    if [[ -f "$f" ]]; then mount_key github "$f"; fi
done < <(ssh -T -G github.com | awk '$1 == "identityfile"')
if (( ${#mounts[@]} == 0 )); then
    echo "run.sh: nessuna chiave per github.com in 'ssh -G github.com'" >&2
    exit 1
fi

opts=(--rm --name "$NAME" -p "127.0.0.1:$HOST_PORT:$APP_PORT")
if [[ -n "$LAN_IP" ]]; then opts+=(-p "$LAN_IP:$HOST_PORT:$APP_PORT"); fi

# Nodi: con l'agent le chiavi restano sull'host e il container può solo chiedergli
# di firmare. Podman non sa inoltrarlo, e su un server l'agent di una sessione
# non dura: lì si montano le chiavi.
if [[ $rt == container && -S "${SSH_AUTH_SOCK:-}" ]]; then
    opts+=(--ssh)
else
    for f in $NODE_KEYS; do
        f="${f/#\~/$HOME}"
        if [[ -f "$f" ]]; then mount_key nodes "$f"; else echo "run.sh: chiave $f non trovata" >&2; fi
    done
fi

# Podman rootless: stesso UID dentro e fuori, altrimenti alvise non legge le chiavi.
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
