#!/usr/bin/env bash
# Costruisce l'immagine pi-agent dal ContainerFile, con `container` (macOS) o
# podman (Linux), con l'utente dell'host e il clone di ai-console. La chiave che
# l'host usa per GitHub è passata come build secret: serve al clone ma non resta
# nell'immagine.
#
# Uso: ./build.sh [opzioni di build, es. --no-cache]
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f run.env ]]; then source ./run.env; fi

RUNTIME="${RUNTIME:-$(command -v container || command -v podman || true)}"
IMAGE="${IMAGE:-pi-agent:latest}"
REPO="git@github.com:dorigoa/ai-console.git"

if [[ -z "$RUNTIME" ]]; then
    echo "build.sh: serve container (macOS) o podman" >&2
    exit 1
fi

# La chiave per GitHub: la prima IdentityFile esistente secondo la config ssh dell'host.
key=""
while read -r _ f; do
    f="${f/#\~/$HOME}"
    if [[ -f "$f" ]]; then key="$f"; break; fi
done < <(ssh -T -G github.com | awk '$1 == "identityfile"')
if [[ -z "$key" ]]; then
    echo "build.sh: nessuna chiave per github.com in 'ssh -G github.com'" >&2
    exit 1
fi

# Commit attuale del repo: il clone (in cache) si rifà solo quando cambia.
ref="$(git ls-remote "$REPO" HEAD | cut -f1)"

"$RUNTIME" build -f ContainerFile -t "$IMAGE" \
    --build-arg USER_NAME="$(id -un)" \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    --build-arg GIT_USER_NAME="$(git config --global user.name || true)" \
    --build-arg GIT_USER_EMAIL="$(git config --global user.email || true)" \
    --build-arg AI_CONSOLE_REPO="$REPO" \
    --build-arg AI_CONSOLE_REF="$ref" \
    --secret id=github_key,src="$key" \
    "$@" .
