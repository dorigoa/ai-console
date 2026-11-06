# container-ai-console

Container for [ai-console](https://github.com/dorigoa/ai-console), the NiceGUI console for the LLM inference services on the home nodes. The image has the same user as the host (name, UID and GID) and a clone of ai-console; the host's SSH keys never go into the image.

## Your Linux node @home

Requirements: podman; in `~/.ssh` the GitHub key (the one shown by `ssh -G github.com`) and `id_ecdsa` for the nodes; lingering enabled (`loginctl enable-linger`); port 8501 free.

```bash
git clone git@github.com:dorigoa/container-ai-console.git ~/container-ai-console
cd ~/container-ai-console && ./install.sh
```

`install.sh` builds the image, installs the `ai-console.container` unit in `~/.config/containers/systemd/` and starts the `ai-console` service. ai-console answers on http://127.0.0.1:8501 and on http://192.168.1.40:8501 (LAN's IP); the addresses are set in `ai-console.container`.

To update, after a commit in ai-console or in this repo:

```bash
cd ~/container-ai-console && git pull && ./install.sh
```

| What | Command |
|---|---|
| Status | `systemctl --user status ai-console` |
| Logs | `journalctl --user-unit ai-console -f` (or `podman logs -f ai-console`) |
| Shell in the container | `podman exec -it ai-console bash` |
| Stop / restart | `systemctl --user stop ai-console` / `systemctl --user restart ai-console` |

## Files

| File | Role |
|---|---|
| `ContainerFile` | `ai-console` image: Python 3.13 on Alpine, the host user, a clone of ai-console with its dependencies |
| `build.sh` | builds with `container` or podman; the GitHub key is a build secret used only for the clone |
| `entrypoint.sh` | at startup copies the read-only mounted keys into `~/.ssh` and writes the ssh config |
| `run.sh` | manual start; reads `run.env` if present (`LAN_IP`, `HOST_PORT`, `NODE_KEYS`, `RUNTIME`) |
| `ai-console.container` | Quadlet unit for raspy |
| `install.sh` | builds and installs or updates the service (Linux) |
