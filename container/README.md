# container-pi

Container for [ai-console](https://github.com/dorigoa/ai-console), the NiceGUI console for the LLM inference services on the home nodes. The image has the same user as the host (name, UID and GID) and a clone of ai-console; the host's SSH keys never go into the image.

- **raspy** (Linux, rootless podman): a systemd service that starts at every boot.
- **macOS** (Apple's [`container`](https://github.com/apple/container) CLI): started by hand, for development and testing.

## raspy

Requirements: podman; in `~/.ssh` the GitHub key (the one shown by `ssh -G github.com`) and `id_ecdsa` for the nodes; lingering enabled (`loginctl enable-linger`); port 8501 free.

```bash
git clone git@github.com:dorigoa/container-pi.git ~/container-pi
cd ~/container-pi && ./install.sh
```

`install.sh` builds the image, installs the `pi-dev.container` unit in `~/.config/containers/systemd/` and starts the `pi-dev` service. ai-console answers on http://127.0.0.1:8501 (for Tailscale serve) and on http://192.168.1.40:8501 (LAN); the addresses are set in `pi-dev.container`.

To update, after a commit in ai-console or in this repo:

```bash
cd ~/container-pi && git pull && ./install.sh
```

| What | Command |
|---|---|
| Status | `systemctl --user status pi-dev` |
| Logs | `journalctl --user-unit pi-dev -f` (or `podman logs -f pi-dev`) |
| Shell in the container | `podman exec -it pi-dev bash` |
| Stop / restart | `systemctl --user stop pi-dev` / `systemctl --user restart pi-dev` |

## macOS

```bash
./build.sh
./run.sh            # ai-console on http://127.0.0.1:8501
./run.sh bash       # or a shell instead of ai-console
```

For the nodes, `run.sh` forwards the Mac's ssh-agent (`container run --ssh`).

## Files

| File | Role |
|---|---|
| `ContainerFile` | `pi-agent` image: Python 3.13 on Alpine, the host user, a clone of ai-console with its dependencies |
| `build.sh` | builds with `container` or podman; the GitHub key is a build secret used only for the clone |
| `entrypoint.sh` | at startup copies the read-only mounted keys into `~/.ssh` and writes the ssh config |
| `run.sh` | manual start; reads `run.env` if present (`LAN_IP`, `HOST_PORT`, `NODE_KEYS`, `RUNTIME`) |
| `pi-dev.container` | Quadlet unit for raspy |
| `install.sh` | builds and installs or updates the service (Linux) |
