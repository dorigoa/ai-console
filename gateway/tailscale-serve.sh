#!/usr/bin/env bash
# HTTPS endpoints of this node on the tailnet (Tailscale serve, tailnet only, no
# Funnel): https://<node>.<tailnet>.ts.net:<port> -> a port on 127.0.0.1.
# The LLM APIs go through the socat forwarders (gateway/forwards/*.env); the
# others are services running on this node.
#
# Run by install.sh, or alone after changing it: it starts from scratch every time.
# Without sudo it needs this user as Tailscale operator: sudo tailscale set --operator=$USER
set -euo pipefail

tailscale serve reset

# Web services on this node
tailscale serve --bg --https=443  http://127.0.0.1:8501   # ai-console (the container, see container/)
tailscale serve --bg --https=5443 http://127.0.0.1:9000   # Wake/poweroff PC Linux
tailscale serve --bg --https=6443 http://127.0.0.1:3000   # Grafana

# LLM APIs, through the forwarders: local port -> forwards/<name>.env
tailscale serve --bg --https=8443 http://127.0.0.1:8080   # ubuntulinux
tailscale serve --bg --https=9443 http://127.0.0.1:9080   # macstudio1

tailscale serve status
