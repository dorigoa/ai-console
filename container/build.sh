#!/usr/bin/env bash
# Builds the ai-console image from container/Containerfile, with `container`
# (macOS), podman or docker (Linux), as the host user. The build context is the
# repository root: the image holds app/ as it is in this checkout, uncommitted
# changes included, and its version (git describe) in AI_CONSOLE_VERSION.
#
# Usage: container/build.sh [build options, e.g. --no-cache]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
if [[ -f "$here/run.env" ]]; then source "$here/run.env"; fi

RUNTIME="${RUNTIME:-$(command -v container || command -v podman || command -v docker || true)}"
IMAGE="${IMAGE:-ai-console:latest}"

if [[ -z "$RUNTIME" ]]; then
    echo "build.sh: needs container (macOS), podman or docker" >&2
    exit 1
fi

version="$(git -C "$root" describe --always --dirty 2>/dev/null || echo unknown)"

"$RUNTIME" build -f "$here/Containerfile" -t "$IMAGE" \
    --build-arg USER_NAME="$(id -un)" \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    --build-arg AI_CONSOLE_VERSION="$version" \
    "$@" "$root"
echo "build.sh: $IMAGE built from ai-console $version"
