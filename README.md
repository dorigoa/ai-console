# start_model.py

Command-line tool to launch and manage **`llama-server`** (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) for a model described in [`models.json`](models.json), on the local machine or on a remote host over SSH, optionally distributed across a set of **RPC nodes**.

It is the command-line engine behind the NiceGUI front-end ([`llama-console-gui.py`](llama-console-gui.py)) — the GUI shells out to this script and parses its `--json` output — but it is fully usable on its own.

---

## Purpose

One command turns *"I want to run model X"* into a live `llama-server`, taking care of everything in between:

| Concern | What the script does |
|---|---|
| **Command line** | Builds the full `llama-server` invocation from the model's entry in `models.json` (sampling, KV-cache quant, batch sizes, chat-template kwargs, MTP/speculative decoding, logging…). |
| **Where it runs** | Runs it locally, or on `LLAMA_SERVER_HOST` over SSH — the caller does not need to know which. |
| **Staying alive** | Starts the server *detached*, so it survives the SSH session and the terminal closing, then **verifies it actually stayed up**. |
| **RPC fleet** | Probes the model's RPC nodes, starts the ones that are down, discovers the resulting GPU device list, and can kill them again. |
| **Lifecycle** | `--server-status`, `--kill-server`, `--tail-log`. |
| **Automation** | `--json` gives machine-readable output for `--server-status` and `--list-models`. |

---

## Rationale

Why the script does things the way it does — each point is a failure that was hit for real.

**Model configuration belongs in data, not in shell history.**
Every model has its own sampler set, KV quantization, batch sizes, reasoning effort and RPC layout. Keeping those in `models.json` means the launch command is reproducible and reviewable, and the CLI collapses to a model name.

**A launched server must outlive the launcher — and be proven alive.**
The server is started with `nohup` (plus `setsid` where it exists — it does not on macOS) and with *all three* standard streams redirected away from the SSH channel. Redirecting only stdin left `stdout`/`stderr` wired to the SSH pipe: `ssh` kept the channel open waiting for EOF, and once the connection was torn down `llama-server` could be killed by `SIGHUP`/`SIGPIPE`. Because `ssh` returns as soon as the background shell forks, that return value proves nothing — so the script then polls `pgrep` for up to 10s before reporting success.

**Two log files, because an early crash never reaches the runtime log.**
`llama-server` opens its `--log-file` only after startup. A bad argument, a missing shared library or an OOM at load time happens *before* that and would leave no trace at all — so startup `stdout`/`stderr` go to a separate **boot log**, and the script prints its tail automatically when the process fails to stay up.

**RPC operations originate from the server host.**
Only `LLAMA_SERVER_HOST` can reach the RPC network, so probes, starts and kills are issued as *nested* SSH: `client → LLAMA_SERVER_HOST → rpc-node`.

**"The port answers" is not "the server will serve you".**
`ggml-rpc-server` handles **one client at a time**: while another `llama-server` holds the session, further connections are completed by the kernel and parked in the listen backlog, never answered. That is why `nc`/`telnet` succeed on the exact same address and port while `llama-server` reports *"Failed to connect"*. Consequently:

- a TCP probe is only used to decide whether the node needs starting;
- when an RPC-backed model gets a device list with **no `RPC*` entry**, the script stops *before* launching and inspects the node's `ESTABLISHED` connections to say whether it is **busy** (and with whom) or **wedged**. Without that check, the run would build `--device RPC0,…` from a list that has no RPC device and die remotely on the cryptic pair `Failed to connect to <ip>:<port>` + `invalid device: RPC0`, visible only in the boot log.

**Refuse a second server locally instead of failing remotely.**
A second `llama-server` cannot bind `PORT_BIND`, and it cannot get an RPC device either while the first one holds the rpc-server session. Both failures would only surface in the remote boot log, so the script checks and refuses up front.

**One source of truth for the GUI.**
`--json` returns everything a UI needs (running state, active model, ctx, temperature, quant / model catalog with sizes and samplers), so the front-end never re-parses `models.json` or re-implements the SSH plumbing.

---

## Requirements

- **Python 3.10+** (the code uses `X | Y` type syntax).
- Dependencies from [`requirements.txt`](requirements.txt):

  ```bash
  pip install -r requirements.txt
  ```

- The `llama-server` binary installed at `LLAMA_SERVER_BIN`, **locally or on the remote host**.
- For remote and RPC usage: **key-based, passwordless SSH** to `LLAMA_SERVER_HOST` and from there to every RPC node. All SSH calls use `BatchMode=yes`, so any password prompt makes the command fail.
- On the RPC nodes: `ggml-rpc-server` at the configured `bin` path, plus `nc`, `netstat` and `killall` (all present by default on Debian/macOS).

---

## Configuration

Three JSON files. Validate any of them with `jq . <file>`.

### `config.json` — global settings

Loaded by [`config_manager.py`](config_manager.py), which looks in order for:

1. the path in the `LLAMA_CONSOLE_CONFIG_FILE` environment variable;
2. `config.json` next to the script;
3. `~/llama-console-config.json`.

**Unknown keys are a fatal error** (they are almost always typos). Values are coerced to the declared type.

| Key | Default | Description |
|---|---|---|
| `ADDRESS_BIND` | `""` | Address `llama-server` binds to (`--host`). |
| `PORT_BIND` | `0` | Port `llama-server` binds to (`--port`); also queried by `--server-status` to read back the live model. |
| `LLAMA_SERVER_BIN` | `""` | Path to the `llama-server` binary on the machine that will run it. Also the `pgrep`/`pkill` pattern used to find the process. |
| `LLAMA_SERVER_HOST` | `""` | SSH host that runs the server. **Empty = everything runs locally.** |
| `LLAMA_SERVER_USER` | `""` | SSH user for `LLAMA_SERVER_HOST`. |
| `MODELS_JSON` | `None` | Model catalog (see below). |
| `RPC_JSON` | `None` | RPC node catalog (see below). |
| `LLAMA_LOG_FILE` | `""` | Runtime log (`--log-file`), followed by `--tail-log`. |
| `LLAMA_BOOT_LOG` | `""` | Startup `stdout`/`stderr` — crash diagnostics. |
| `DEFAULT_CTX` | `8192` | Context size used when `--override-ctx` is not given. |
| `SEED` | `123456789` | `--seed`. |
| `FITC` | `8192` | `-fitc`. |
| `NP` | `1` | `-np` (parallel sequences; **MTP processing requires 1**). |
| `UI_TITLE` | `"Custom UI Title"` | Title shown by the GUI. |
| `UI_PORT` | `8501` | Port the NiceGUI console listens on. |

> `MODELS_JSON` and `RPC_JSON` are used verbatim: if they are relative (as in the shipped `config.json`) they resolve against the **current working directory**, so either run the script from its own directory or make those paths absolute.

Current [`config.json`](config.json):

```json
{
	"ADDRESS_BIND": "0.0.0.0",
	"PORT_BIND": 8088,
	"UI_TITLE": "LLama.cpp Console by Alvise Dorigo (https://github.com/dorigoa/llama-console)",
	"LLAMA_SERVER_BIN": "/opt/llama.cpp/llama-server",
	"LLAMA_SERVER_HOST": "192.168.1.191",
	"LLAMA_SERVER_USER": "alvise",
	"MODELS_JSON": "models.json",
	"RPC_JSON": "rpc.json",
	"LLAMA_LOG_FILE": "/tmp/llama-server.log",
	"LLAMA_BOOT_LOG": "/tmp/llama-server.boot.log",
	"DEFAULT_CTX": 32768
}
```

### `rpc.json` — the RPC node catalog

Nodes are declared **once**, by name, and referenced by name from the models:

```json
{
    "RPC_SERVERS": {
        "mac1": {
                "ip": "192.168.30.1",
                "port": 50000,
                "cachepath": "/Volumes/Home/llama-rpc-cache",
                "bin": "/opt/llama.cpp/ggml-rpc-server",
                "remuser": "alvise"
        },
        "linux": {
                "ip": "192.168.30.3",
                "port": 50000,
                "cachepath": "/nvme/llama-rpc-cache",
                "bin": "/opt/llama.cpp/ggml-rpc-server",
                "remuser": "alvise"
        }
    }
}
```

All five attributes are **required** for every entry. `cachepath` becomes `LLAMA_CACHE` when the node is auto-started, `remuser` is the SSH user on the node.

### `models.json` — the model catalog

```json
{
  "MODEL_BASE_DIR": "/Volumes/Home/gguf_models",
  "models": {
    "GPT-OSS-120B-MXFP4": {
      "sizegb": 59,
      "RPC_SERVERS": { "ids": [ { "name": "mac1" }, { "name": "linux" } ] },
      "SAMPLERS": "1.0:1.0:0:0.0",
      "RP": 1.0,
      "PP": -1,
      "REAS": "high",
      "PRES_THK": null,
      "MMPROJ": null,
      "EXTRAS": null,
      "KVQUANT": "q8_0",
      "UB": 1024,
      "B": 2048,
      "MTP": false,
      "ext_mtp_head_file": null,
      "native_ctx": 131072
    }
  }
}
```

`MODEL_BASE_DIR` is required at top level: the model file is `MODEL_BASE_DIR/<key>.gguf` (the extension is appended when missing). The **key** is the name you pass on the command line.

| Field | Required | Meaning |
|---|---|---|
| `SAMPLERS` | ✅ | `"temp:top_p:top_k:min_p"`, as one string. A **negative `min_p`** means *"don't pass `--min-p`, keep llama-server's default"*. |
| `native_ctx` | ✅ | The model's native context length. Informational — reported by `--list-models --json` (the GUI uses it as the upper bound of the ctx slider); it does **not** set `-c`. |
| `MMPROJ` | ✅ (may be `null`) | Absolute path of the multimodal projector → `--mmproj`. |
| `KVQUANT` | ✅ (may be `null`) | KV-cache quantization → `-ctk`/`-ctv` (e.g. `q8_0`). |
| `UB`, `B` | ✅ (may be `null`) | `-ub` / `-b` (micro-batch and batch size). |
| `RP` | ✅ | `--repeat-penalty`. Always emitted — set it to `1.0` for "no penalty" rather than omitting it. |
| `PP` | – | `--presence-penalty`. Use `-1` (or omit) to leave it out. |
| `REAS` | – | `reasoning_effort` inside `--chat-template-kwargs`. |
| `PRES_THK` | – | `preserve_thinking` inside `--chat-template-kwargs`. |
| `MTP` | – | `true` enables `--spec-type draft-mtp`. |
| `ext_mtp_head_file` | – | Draft/MTP head file (resolved under `MODEL_BASE_DIR`) → `--model-draft`. |
| `RPC_SERVERS.ids[].name` | – | Names from `rpc.json`. Omit or leave empty for a local-only model. An unknown name is a fatal config error. |
| `sizegb`, `EXTRAS` | – | Not read by the loader: the real size is `stat`-ed on the host at load time. |

A model whose `.gguf` is not found (checked over SSH too) is **skipped** with a `[SKIP]` line rather than aborting the whole catalog.

To dump the parsed catalog (sizes, RPC endpoints) run the loader directly:

```bash
python model.py --nocheck
```

---

## Usage

```bash
python start_model.py [MODEL_NAME] [OPTIONS]
```

`MODEL_NAME` is the key in `models.json`, without `.gguf`. It is optional because several actions (`--list-models`, `--server-status`, `--kill-server`, `--tail-log`) do not need a model.

### What a normal launch does

```bash
python start_model.py GPT-OSS-120B-MXFP4
```

1. Loads `config.json`, `rpc.json` and `models.json`, and `stat`s the `.gguf` for the requested model.
2. Applies any `--override-*` on top of the model's entry.
3. Verifies the `llama-server` binary exists (locally, or over SSH on `LLAMA_SERVER_HOST`).
4. Refuses to continue if a `llama-server` is **already running** (it holds `PORT_BIND` and the rpc-server session).
5. For an RPC-backed model: probes each node, **starts the dead ones** over nested SSH, and polls until they answer.
6. Runs `llama-server --rpc … --list-devices` to discover the usable devices, dropping any with `0 MiB free`. If no `RPC*` device comes back, it stops and reports whether the node is busy or wedged.
7. Builds the command line and launches it **detached**, with `stdout`/`stderr` to the boot log.
8. Polls `pgrep` for ~10s: prints the PIDs on success, or the tail of the boot log and exit code 1 on a startup crash.

Example output (abridged; device names and PIDs obviously vary):

```
[I 260816 13:00:41 config_manager:51] Config override loaded from /…/config.json (11 keys).
[I 260816 13:00:44 start_model:483] All RPC servers reachable.
[I 260816 13:00:52 start_model:505] Using devices: RPC0,RPC1,Metal0
[I 260816 13:00:52 start_model:174] Starting detached llama-server on alvise@192.168.1.191 ...
[I 260816 13:00:52 start_model:175] Log file: /tmp/llama-server.log  (boot log: /tmp/llama-server.boot.log)
[I 260816 13:00:55 start_model:198] llama-server RUNNING on alvise@192.168.1.191 (pid(s): 40213).
```

### Options

| Option | Description |
|---|---|
| `MODEL_NAME` | Model key in `models.json` (without `.gguf`). |
| `--dry-run` | Print the command that *would* run. Starts nothing — no RPC probe, no device discovery, no launch. |
| `--list-models` | Print the catalog (size in GiB + number of RPC nodes) and exit. |
| `--server-status` | Report whether `llama-server` is running, and which model/ctx/temp/quant it serves. |
| `--kill-server` | Stop `llama-server` (SIGTERM, then SIGKILL after 2s) and also kill the `ggml-rpc-server`s it was using (read from the running process's `--rpc` argument), then exit. |
| `--tail-log` | Follow the runtime log with `tail -F` until Ctrl-C (survives log rotation / server restart). |
| `--tail-lines`, `-n INT` | Lines of context before following (default: 50). |
| `--only-check-rpc` | Only report which of the model's RPC nodes are unreachable. Never starts them. |
| `--only-start-rpc` | Start the model's RPC nodes and exit, without launching `llama-server`. |
| `--only-list-devices` | Show the device list (local + RPC) and exit. Does **not** start RPC nodes. |
| `--kill-rpc-server` | `killall ggml-rpc-server` on every RPC node of the model (works even when `llama-server` is not running; while it *is* running, `--kill-server` already takes care of the nodes it uses). |
| `--override-temp FLOAT` | Override the temperature. |
| `--override-top-p FLOAT` | Override top-p. |
| `--override-top-k INT` | Override top-k. |
| `--override-min-p FLOAT` | Override min-p. |
| `--override-ctx INT` | Override the context size (otherwise `DEFAULT_CTX`). |
| `--override-devices STR` | Use this device list verbatim (csv) and **skip both** RPC auto-start and device discovery. |
| `--override-rpc STR` | Use these `rpc.json` node names (csv) instead of the model's, or `none` to run without RPC. |
| `--force-no-mtp` | Disable MTP/speculative decoding even for a model that declares `MTP: true`. |
| `--json` | Machine-readable output for `--server-status` and `--list-models`. |
| `--debug` | Debug logging **and** `--verbose` on the `llama-server` command line. |

---

## Examples

### Catalog

```bash
python start_model.py --list-models
```

```
Available models:
  NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ4_NL (60 GiB - 2 RPC)
  Phi-4-reasoning-plus-UD-Q8_K_XL (16 GiB - 0 RPC)
  GPT-OSS-120B-MXFP4 (59 GiB - 2 RPC)
  ...
```

### Preview the exact command line (nothing is started)

```bash
python start_model.py GPT-OSS-120B-MXFP4 --dry-run
```

```
Dry-run command: /opt/llama.cpp/llama-server -m /Volumes/Home/gguf_models/GPT-OSS-120B-MXFP4.gguf \
  -c 32768 --rpc 192.168.30.1:50000,192.168.30.3:50000 --device 'SKIP(DRYRUN)' \
  --host 0.0.0.0 --port 8088 --split-mode layer --metrics --jinja -fa on -fit on -fitc 8192 \
  -np 1 --no-warmup --temp 1.0 --top-p 1.0 --top-k 0 \
  --chat-template-kwargs '{"reasoning_effort": "high"}' --seed 123456789 --repeat-penalty 1.0 \
  -ctk q8_0 -ctv q8_0 --alias local_AI --min-p 0.0 -ub 1024 -b 2048 \
  --log-file /tmp/llama-server.log -ctxcp 8
```

`--device 'SKIP(DRYRUN)'` is the placeholder standing in for the discovery step that a dry run deliberately skips.

### Start, watch, stop

```bash
python start_model.py Qwen3.8-27B-UD-Q8_K_XL
python start_model.py --tail-log -n 200
python start_model.py --kill-server
```

### Status

```bash
python start_model.py --server-status
```

```
llama-server is RUNNING on alvise@192.168.1.191 (pid(s): 40213)
Running model: Qwen3.8-27B-UD-Q8_K_XL - CTX Size: 32768 - Temp: 0.6 - Quant: 7
```

Machine-readable, for scripts and the GUI:

```bash
python start_model.py --server-status --json
```

```json
{"where": "alvise@192.168.1.191", "running": false, "ready": false, "pids": [], "quant": ""}
```

```bash
python start_model.py --list-models --json
```

```json
{"models": [{"name": "NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ4_NL", "size_gib": 60.05,
             "rpc_count": 2, "native_ctx": 1048576, "temperature": 0.6,
             "top_p": 0.95, "top_k": 0, "min_p": 0.0}, "..."]}
```

`running` says a process exists; `ready` says it also answers `/props` — a freshly started server is RUNNING but not ready for a while.

### Tuning a single run

```bash
python start_model.py Qwen3.8-27B-UD-Q8_K_XL \
  --override-temp 0.7 \
  --override-top-p 0.9 \
  --override-ctx 131072
```

Pin the devices by hand (skips RPC auto-start *and* discovery — the nodes must already be up):

```bash
python start_model.py GPT-OSS-120B-MXFP4 --override-devices RPC0,RPC1,Metal0
```

Run an RPC-backed model on the server host alone, e.g. to isolate an RPC problem:

```bash
python start_model.py GPT-OSS-120B-MXFP4 --override-rpc none
```

Use a different set of nodes than the ones declared in `models.json`:

```bash
python start_model.py GPT-OSS-120B-MXFP4 --override-rpc mac1
python start_model.py GPT-OSS-120B-MXFP4 --override-rpc mac1,linux
```

Rule out speculative decoding as the cause of a problem:

```bash
python start_model.py Qwen3.6-27B-MTP-UD-Q8_K_XL --force-no-mtp
```

### Managing the RPC fleet

```bash
python start_model.py GPT-OSS-120B-MXFP4 --only-check-rpc     # reachable? (never starts anything)
python start_model.py GPT-OSS-120B-MXFP4 --only-start-rpc     # bring the nodes up, then stop
python start_model.py GPT-OSS-120B-MXFP4 --only-list-devices  # what would the launch see?
python start_model.py GPT-OSS-120B-MXFP4 --kill-rpc-server    # killall ggml-rpc-server everywhere
```

Tearing the deployment down requires no model name: `--kill-server` stops `llama-server` **and** kills the `ggml-rpc-server`s it was using, identified from the running process's `--rpc` argument (so a launch made with `--override-rpc` is cleaned up correctly too). This releases the GPUs on the RPC nodes instead of leaving `ggml-rpc-server`s holding their memory for a client that is gone.

A busy RPC node is reported explicitly instead of failing later — the diagnostic looks like this:

```
Model 'GPT-OSS-120B-MXFP4' needs RPC devices but --list-devices returned none.
  Devices seen: Metal0
  The RPC port answers at TCP level (nc/telnet would succeed), but the server did not serve the request.
  192.168.30.1:50000 is BUSY — already serving:
    tcp4  0  0  192.168.30.1.50000  192.168.30.3.51872  ESTABLISHED
  Stop that client (llama-server --kill-server, or --kill-rpc-server to restart the RPC node) and retry.
```

### Troubleshooting a launch

```bash
python start_model.py GPT-OSS-120B-MXFP4 --debug
```

`--debug` prints every SSH command that is issued (including the nested ones) and adds `--verbose` to `llama-server` itself. If the server does not stay up, the tail of the boot log is printed automatically; to read more of it:

```bash
ssh alvise@192.168.1.191 'tail -n 200 /tmp/llama-server.boot.log'
```

### In a shell script

```bash
#!/usr/bin/env bash
set -e
python start_model.py --server-status --json | jq -e '.running' >/dev/null && \
  python start_model.py --kill-server
python start_model.py Qwen3.8-27B-UD-Q8_K_XL --override-ctx 65536
```

---

## Log files

| File (on the host that runs the server) | Contents |
|---|---|
| `LLAMA_LOG_FILE` (`/tmp/llama-server.log`) | **Runtime** log, written by `llama-server` via `--log-file`. Follow it with `--tail-log`. |
| `LLAMA_BOOT_LOG` (`/tmp/llama-server.boot.log`) | **Startup** `stdout`/`stderr`: bad arguments, missing libraries, OOM — everything that happens before the runtime log exists. |
| `/tmp/rpc-server.out` (on each RPC node) | Output of an auto-started `ggml-rpc-server`. |

With a remote host these are remote paths, read over SSH.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success — including `--server-status` when the server **is** running. |
| `1` | Generic failure: model not found, binary missing, server already running, RPC unreachable or busy, startup crash, `--kill-server` when a rpc-server could not be killed — and `--server-status` when the server is **not** running. |
| `2` | `LLAMA_SERVER_HOST` **unreachable over SSH** — deliberately distinct from "the server is not running". |

---

## Caveats

- **`0` is not a usable override value.** The `--override-*` options are applied with a truthiness test, so `--override-temp 0`, `--override-top-k 0`, `--override-min-p 0` and `--override-ctx 0` are silently ignored and the value from `models.json` (or `DEFAULT_CTX`) is used. Note that `top_k: 0` / `min_p: 0` set *in `models.json`* work fine — this only affects the command-line overrides.
- **`--only-check-rpc` / `--only-start-rpc` on a model with no RPC nodes falls through to a real launch.** Both options short-circuit inside the RPC branch, which is skipped entirely when the model has no `RPC_SERVERS`. Use them only with RPC-backed models; use `--dry-run` for the rest.
- **One server at a time by design.** A launch is refused while another `llama-server` is running; stop it with `--kill-server` first.
- `--tail-log` allocates a remote pty (`ssh -t`) so that Ctrl-C tears down the remote `tail` instead of orphaning it.
- Device discovery has a 60s timeout: backend enumeration (Metal/CUDA) alone can take ~10s, so it is a deadlock guard, not a latency budget.
