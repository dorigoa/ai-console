#!/usr/bin/env bash
# Linux with systemd, on the gateway node (the one running the ai-console
# container): installs the socat forwarders towards the LLM nodes, one
# socat-fwd@<name> instance per file in forwards/, then the Tailscale HTTPS
# endpoints (tailscale-serve.sh). Run it again after changing either: forwards
# removed from forwards/ are stopped and removed from the node too.
#
# Needs sudo, socat and tailscale. Replaces the units installed by the old
# tailgate-scripts repo (strata-fwd.service, ...), which hold the same ports.
#
# Usage: gateway/install.sh             forwarders + Tailscale serve
#        gateway/install.sh --no-serve  forwarders only
set -euo pipefail
cd "$(dirname "$0")"

conf=/etc/ai-console/forwards
units=/etc/systemd/system
template=socat-fwd@.service

for cmd in socat tailscale systemctl; do
    if ! command -v "$cmd" >/dev/null; then
        echo "install.sh: needs $cmd" >&2
        exit 1
    fi
done

# A forward with NODE=<name> must reach the address where the console finds
# that node (ip:api_port in app/nodes.json): a mismatch is reported, not fixed.
if command -v python3 >/dev/null; then
    python3 -I - ../app/nodes.json forwards/*.env <<'EOF' >&2 || true
import json, sys
try:
    nodes = json.load(open(sys.argv[1]))["NODES"]
except (OSError, ValueError, KeyError) as e:
    sys.exit(f"install.sh: cannot read {sys.argv[1]} ({e}), forwards not checked")
for path in sys.argv[2:]:
    env = {}
    for line in open(path):
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            env[key] = value
    name = env.get("NODE")
    if not name:
        continue
    node = nodes.get(name)
    if node is None:
        print(f"install.sh: warning: {path}: NODE={name} is not in app/nodes.json")
        continue
    expected = f"{node.get('ip')}:{node.get('api_port')}"
    if env.get("TARGET") != expected:
        print(f"install.sh: warning: {path}: TARGET={env.get('TARGET')}, "
              f"but app/nodes.json has {name} at {expected}")
EOF
fi

# The old tailgate-scripts units: same ports as the new instances, so they go first.
for name in freetoken-fwd llama-fwd llama-server-fwd llama-server-linux-fwd llama-server-macos-fwd \
            llama-server-webui-fwd mlx-fwd splash-fwd strata-fwd; do
    file="$units/$name.service"
    if [[ -f $file ]] && grep -q socat "$file"; then
        echo "Removing the old $name.service"
        sudo systemctl disable --now "$name.service" || true
        sudo rm -f "$file"
    fi
done

# Instances whose file is no longer in forwards/.
for file in "$conf"/*.env; do
    [[ -f $file ]] || continue
    name=$(basename "$file" .env)
    if [[ ! -f forwards/$name.env ]]; then
        echo "Removing the forward $name (not in forwards/)"
        sudo systemctl disable --now "socat-fwd@$name.service" || true
        sudo rm -f "$file"
    fi
done

sudo install -d -m 755 "$conf"
sudo install -m 644 forwards/*.env "$conf/"
sudo install -m 644 "$template" "$units/$template"
sudo systemctl daemon-reload
for file in forwards/*.env; do
    instance="socat-fwd@$(basename "$file" .env).service"
    sudo systemctl enable --quiet "$instance"
    sudo systemctl restart "$instance"   # starts it, or restarts it with a changed .env
done
systemctl --no-pager list-units 'socat-fwd@*'

if [[ ${1:-} != --no-serve ]]; then
    ./tailscale-serve.sh
fi
