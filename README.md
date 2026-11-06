# ai-console

LLM Inference Console for the home nodes: a NiceGUI web page that starts, stops and monitors the inference services (llama-server, Strata, Splash) on the LAN nodes over SSH, packaged as a container, plus the gateway that makes the same node the single entry point to the LLM APIs, from the LAN and, through Tailscale, from outside.

One repository for everything that runs on the gateway node (raspy, `192.168.1.40`); it merges, with their history, the former [container-ai-console](https://github.com/dorigoa/container-ai-console) (now `container/`) and [tailgate-scripts](https://github.com/dorigoa/tailgate-scripts) (now `gateway/`).

```
 tailnet clients                 raspy (gateway node)                          LAN nodes
 ───────────────                 ────────────────────                          ─────────
 https://raspy…ts.net:443  ──►  tailscale serve ──► 127.0.0.1:8501  ai-console container ──ssh──►  start/stop/status
                     :1443 ──►        │          ──► 127.0.0.1:8080  socat-fwd@llama  ──────────►  192.168.1.200:8080  llama-server
                     :2443 ──►        │          ──► 127.0.0.1:8000  socat-fwd@strata ──────────►  192.168.1.200:8000  Strata
                     :4443 ──►        │          ──► 127.0.0.1:10000 socat-fwd@splash ──────────►  192.168.1.101:10000 Splash
 LAN browsers ─────────────────────────────────►  192.168.1.40:8501  ai-console
```

## Layout

| Path | What |
|---|---|
| `app/` | the console: `ai-console-gui.py` and its `gui_*` modules, `config.json` (title, port, log level), `nodes.json` (ip, user, API port of each node), `requirements.txt`; `deprecated/` holds the old `start_model.py` project, not used |
| `container/` | the `ai-console` image and how to run it: `Containerfile`, `build.sh`, `run.sh`, `entrypoint.sh`, the Quadlet unit `ai-console.container` and `install.sh` for the gateway node |
| `gateway/` | the socat forwarders towards the LLM nodes (`socat-fwd@.service`, one file per forward in `forwards/`), the Tailscale HTTPS endpoints (`tailscale-serve.sh`) and `install.sh` |
| `docs/` | reference notes: `samplers.md` (sampling parameters for local LLMs), benchmark charts and KV-cache tables in `assets/` |

## Endpoints

| Service | Runs on | On the gateway | Tailnet (HTTPS) |
|---|---|---|---|
| ai-console | raspy, container | `127.0.0.1:8501`, `192.168.1.40:8501` | `:443` |
| llama-server | PC/Ubuntu `192.168.1.200:8080` | `127.0.0.1:8080` (socat-fwd@llama) | `:1443` |
| Strata | PC/Ubuntu `192.168.1.200:8000` | `127.0.0.1:8000` (socat-fwd@strata) | `:2443` |
| Splash | MacStudio1 `192.168.1.101:10000` | `127.0.0.1:10000` (socat-fwd@splash) | `:4443` |
| Freetoken | PC/Ubuntu `192.168.1.200:1919` | `127.0.0.1:1919` (socat-fwd@freetoken) | off (`:7443`, commented out) |
| Wake/poweroff PC Linux | raspy | `127.0.0.1:9000` | `:5443` |
| Grafana | raspy | `127.0.0.1:3000` | `:6443` |

The tailnet address is `https://<node>.<tailnet>.ts.net:<port>`, e.g. an OpenAI-compatible client points to `https://raspy.<tailnet>.ts.net:1443/v1`. The forwarders listen on `127.0.0.1` only: outside the node they are reachable through Tailscale serve, which is tailnet-only (no Funnel).

## The gateway node (raspy)

Requirements: podman, socat, tailscale (logged in, with the user as operator: `sudo tailscale set --operator=$USER`); in `~/.ssh` the key `id_ecdsa` for the nodes; lingering enabled (`loginctl enable-linger`); ports 8501, 8080, 8000, 10000, 1919 free.

```bash
git clone git@github.com:dorigoa/ai-console.git ~/ai-console
cd ~/ai-console
container/install.sh    # image + ai-console service (rootless podman, Quadlet)
gateway/install.sh      # socat forwarders (sudo) + Tailscale serve
```

`container/install.sh` builds the image from the checkout, installs the `ai-console.container` unit in `~/.config/containers/systemd/` and (re)starts the `ai-console` service; ai-console answers on http://127.0.0.1:8501 and on http://192.168.1.40:8501 (the addresses are in `ai-console.container`).

`gateway/install.sh` installs the `socat-fwd@.service` template and the files of `gateway/forwards/` in `/etc/ai-console/forwards/`, enables one `socat-fwd@<name>` per file, then runs `tailscale-serve.sh`, which resets and recreates the HTTPS endpoints. It warns when a forward's `TARGET` differs from where `app/nodes.json` puts the same node (`NODE=`): the console and the gateway must point at the same servers.

### Update

```bash
cd ~/ai-console && git pull
container/install.sh    # after changes in app/ or container/ (ai-console is offline for a few seconds)
gateway/install.sh      # after changes in gateway/
```

The image is built from the checkout, not from GitHub: what is in `~/ai-console` (uncommitted changes included) is what runs. The Python dependencies are reinstalled only when `app/requirements.txt` changes.

| What | Command |
|---|---|
| Version running | `podman exec ai-console printenv AI_CONSOLE_VERSION` (`git describe` of the checkout, `-dirty` if it had local changes) |
| Status | `systemctl --user status ai-console` |
| Logs | `journalctl --user-unit ai-console -f` (or `podman logs -f ai-console`) |
| Shell in the container | `podman exec -it ai-console bash` |
| Stop / restart | `systemctl --user stop ai-console` / `systemctl --user restart ai-console` |
| Forwarders | `systemctl list-units 'socat-fwd@*'`, `journalctl -u socat-fwd@llama -f` |
| Tailscale endpoints | `tailscale serve status` |

### Adding or changing a forward

One file per forward in `gateway/forwards/<name>.env`:

```bash
NODE=llama                 # entry in app/nodes.json, for the consistency check (optional)
LISTEN_PORT=8080           # on 127.0.0.1 of the gateway
TARGET=192.168.1.200:8080  # the server on the LAN
```

Then an endpoint for it in `gateway/tailscale-serve.sh` and `gateway/install.sh` again: a file removed from `forwards/` is stopped and removed from the node too.

### Moving from the old repositories

On raspy the same `ai-console` service and image names are kept, so `container/install.sh` replaces the service built by `~/container-ai-console` in place, and `gateway/install.sh` stops and removes the units of `tailgate-scripts` (`strata-fwd.service`, `llama-server-linux-fwd.service`, ...) before starting the `socat-fwd@` instances on the same ports. After that `~/container-ai-console` and `~/tailgate-scripts` are no longer needed.

What changed in the container: the image no longer clones ai-console from GitHub, so the GitHub key is no longer a build secret, and it is no longer mounted in the running container (there is no git checkout in it to push from): only the keys for the nodes are.

## Development (macOS)

Directly on the host, with the keys of the host's ssh-agent:

```bash
cd app
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python ai-console-gui.py     # http://127.0.0.1:8501
```

Or in the container, as on the gateway node, with Apple's `container` (podman and docker work too):

```bash
container/build.sh     # image ai-console:latest from the working tree
container/run.sh       # http://127.0.0.1:8501, the host's ssh-agent forwarded for the nodes
container/run.sh bash  # a shell in the image instead of ai-console
```

`run.sh` and `build.sh` read `container/run.env` if present (per-host settings, not committed: see `container/run.env.example`).

## Configuration

| File | Keys |
|---|---|
| `app/config.json` | `UI_TITLE`, `UI_PORT` (default 8501: if changed, change `APP_PORT` in `run.sh`, `EXPOSE` and the ports in `ai-console.container`), `UI_LOG_LEVEL` (or `$AI_CONSOLE_LOG_LEVEL`); the other keys belong to the old `start_model.py` |
| `app/nodes.json` | `NODES.<name>`: `ip`, `user`, `api_port` of each node the console manages |
| `gateway/forwards/*.env` | `LISTEN_PORT`, `TARGET`, `NODE` of each forwarder |
| `gateway/tailscale-serve.sh` | the HTTPS port of each endpoint |
