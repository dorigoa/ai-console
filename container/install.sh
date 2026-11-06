#!/usr/bin/env bash
# Linux with podman: builds the image from this checkout, installs the Quadlet
# unit and (re)starts the service, which from then on starts by itself at every
# boot. Run it again to update, after a `git pull`.
#
# Usage: container/install.sh [unit]   (default: ai-console.container -> service ai-console)
set -euo pipefail
cd "$(dirname "$0")"

unit=${1:-ai-console.container}
service=$(basename "$unit" .container).service
dir="${XDG_CONFIG_HOME:-$HOME/.config}/containers/systemd"

if ! command -v podman >/dev/null; then
    echo "install.sh: needs podman (the unit is a Quadlet: podman-systemd.unit(5))" >&2
    exit 1
fi

# Addresses published by the unit (ip:port), e.g. 127.0.0.1:8501.
mapfile -t addrs < <(awk -F'[=:]' '$1 == "PublishPort" { print $2 ":" $3 }' "$unit")
port=${addrs[0]##*:}

# The port must be free, unless the service already holds it.
if ! systemctl --user is-active --quiet "$service" && [[ -n $(ss -ltnH "sport = :$port") ]]; then
    echo "install.sh: port $port is already in use (an older service still running?): free it first" >&2
    exit 1
fi

# Quadlet runs the image from podman's store.
RUNTIME=podman ./build.sh

mkdir -p "$dir"
install -m 644 "$unit" "$dir/${unit##*/}"
systemctl --user daemon-reload
systemctl --user restart "$service"   # starts it, or restarts it with the image just built

# Without lingering the service starts at the first login instead of at boot.
if [[ $(loginctl show-user "$(id -un)" -p Linger --value) != yes ]]; then
    echo "install.sh: to start at boot without logging in, run: loginctl enable-linger" >&2
fi

for _ in {1..20}; do
    if curl -fs -o /dev/null "http://${addrs[0]}/"; then
        echo "ai-console $(podman exec "${service%.service}" printenv AI_CONSOLE_VERSION 2>/dev/null) is up:"
        printf '  http://%s\n' "${addrs[@]}"
        exit 0
    fi
    sleep 1
done
echo "install.sh: ai-console is not answering on ${addrs[0]}, see: journalctl --user-unit ${service%.service}" >&2
exit 1
