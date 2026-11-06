#!/usr/bin/env bash
# Builds the ai-console image from the ContainerFile, with `container` (macOS) or
# podman (Linux), with the host user and a clone of ai-console. The key the host
# uses for GitHub is passed as a build secret: the clone needs it, but it does
# not stay in the image.
#
# Usage: ./build.sh [build options, e.g. --no-cache]
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f run.env ]]; then source ./run.env; fi

RUNTIME="${RUNTIME:-$(command -v container || command -v podman || true)}"
IMAGE="${IMAGE:-ai-console:latest}"
REPO="git@github.com:dorigoa/ai-console.git"

if [[ -z "$RUNTIME" ]]; then
    echo "build.sh: needs container (macOS) or podman" >&2
    exit 1
fi

# The GitHub key: the first existing IdentityFile according to the host's ssh config.
key=""
while read -r _ f; do
    f="${f/#\~/$HOME}"
    if [[ -f "$f" ]]; then key="$f"; break; fi
done < <(ssh -T -G github.com | awk '$1 == "identityfile"')
if [[ -z "$key" ]]; then
    echo "build.sh: no key for github.com in 'ssh -G github.com'" >&2
    exit 1
fi

# Current commit of the repo: the (cached) clone is redone only when it changes.
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
