"""NiceGUI front-end for start_model.py.

Everything this UI knows about the *server* comes from `start_model.py --json`:
the CLI is the single source of truth, and the two communicate over a real data
structure instead of the printed text the UI used to slice by word position.

The *model list*, on the other hand, is read straight from model.load_models():
both processes would parse the same models.json anyway, so going through
`--list-models --json` only added a subprocess spawn and a lossy dict round-trip
in between. The UI now holds the Model objects themselves.
"""

import asyncio
import base64
import contextlib
import base64
import json
import os
import requests
import shlex
import signal
import sys
from datetime import datetime
from pathlib import Path

from logzero import logger
from nicegui import app, context, ui

from config_manager import get_settings
from model import Model, load_models as load_models_from_config

settings = get_settings()

# Load RPC server definitions
_rpc_config_path = Path(__file__).parent / "rpc.json"
try:
    with open(_rpc_config_path) as f:
        _rpc_config = json.load(f)
    RPC_SERVERS: dict = _rpc_config.get("RPC_SERVERS", {})
except (FileNotFoundError, json.JSONDecodeError):
    RPC_SERVERS = {}

# Load node definitions (nodes.json): each node's IP, SSH user and, for the
# cards that probe a /v1/models API, its "api_port" live here (same shape as
# rpc.json). The Strata/MLX cards are no-ops until the
# matching entry is present, so a missing/malformed file just disables them
# rather than crashing boot.
_nodes_config_path = Path(__file__).parent / "nodes.json"
try:
    with open(_nodes_config_path) as f:
        _nodes_config = json.load(f)
    NODES: dict = _nodes_config.get("NODES", {})
except (FileNotFoundError, json.JSONDecodeError):
    NODES = {}
STRATA_NODE: dict = NODES.get("strata", {})
MLX_NODE: dict = NODES.get("mlx", {})
SPLASH_NODE: dict = NODES.get("splash", {})
FREETOKEN_NODE: dict = NODES.get("freetoken", {})
LLAMA_NODE: dict = NODES.get("llama", {})
# The mlx run is a two-node (rank 0 + rank 1) mlx.launch job: rank 0 is the node the
# start script is launched on ("ip"), rank 1 is the other Mac ("ip_slave", optional;
# without it the card still works, it just only sees rank 0). The SSH user defaults to
# the rank 0 one and can be overridden with "user_slave".
MLX_SLAVE_IP: str = MLX_NODE.get("ip_slave") or ""
MLX_SLAVE_USER: str = MLX_NODE.get("user_slave") or MLX_NODE.get("user") or ""

# The Qwen "service" on the strata node: started by $HOME/Strata/start.sh, and
# counted as UP only once its HTTP API answers /v1/models (a fresh start.sh is a
# live process long before it binds the port, so "process alive" alone is not
# "service ready").
#
# start.sh is only a wrapper: its own PID is useless for stopping the service,
# because the real work happens in the python server it spawns, at
# <any-path>/Strata/Strata/serve/server.py. So status/kill ignore the pidfile
# entirely and locate that process by command line via "pgrep -f", matching
# "python ... Strata/Strata/serve/server.py". The "[s]" bracket trick keeps pgrep
# from matching the very remote shell running the check (whose argv contains the
# pattern verbatim, while the regex only matches the literal "server.py").
_STRATA_API_PORT = int(STRATA_NODE.get("api_port") or 8000)
_STRATA_SRV_PATTERN = r"python.*Strata/Strata/serve/[s]erver\.py"
_STRATA_START_CMD = (
    'cd "$HOME/Strata" && { setsid bash -c \'echo $$ > strata.pid; exec ./start.sh\' '
    '> strata.out 2>&1 </dev/null & sleep 1; echo "STARTED_PID=$(cat strata.pid)"; }'
)
_STRATA_KILL_CMD = (
    f'pids=$(pgrep -f "{_STRATA_SRV_PATTERN}"); '
    'if [ -z "$pids" ]; then echo "NO_PID"; exit 3; fi; '
    'kill -TERM $pids 2>/dev/null; sleep 2; kill -KILL $pids 2>/dev/null; '
    'echo KILLED_PIDS=$pids'
)
_STRATA_STATUS_CMD = (
    f'pids=$(pgrep -f "{_STRATA_SRV_PATTERN}" | tr "\\n" " "); '
    'if [ -z "$pids" ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi'
)
# Console log of the Qwen service: strata.out, written by _STRATA_START_CMD in the
# very directory start.sh runs from ($HOME/Strata). Fetched on demand ("Update
# log" button) as a tail, so a long-running service does not flood the browser.
_STRATA_LOG_LINES = 200
_STRATA_LOG_CMD = (
    'f="$HOME/Strata/strata.out"; '
    'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
    f'echo "--- $f: last {_STRATA_LOG_LINES} of $(wc -l < "$f") lines ---"; '
    f'tail -n {_STRATA_LOG_LINES} "$f"'
)


# The MLX-Jaccl "service" on the mlx node: same shape as the Strata card, but the
# workload is launched by ~/mlx-jaccl/mlx_jaccl-start.py with a fixed command line
# (status is purely "is the process alive"; the OpenAI-compatible /v1/models API
# it serves is only used to read the model id); the API
# port still travels in that command line as "-port <api_port>", with the
# "api_port" of the mlx entry in nodes.json (8080 = the launcher's default,
# which is what the card used to rely on silently), and the node
# is a Mac: BSD userland, so no setsid(1), no /proc, and a login shell that may well
# be zsh rather than bash.
#
# The start command below still launches the script with its fixed flags (plus
# the "-port" from nodes.json), but the
# "new session + pidfile + redirect" dance that setsid does on Linux is done here by
# a tiny python3 one-liner (python3 is a given on an MLX box): Popen's
# start_new_session=True runs setsid(2) in the child, which makes the script a
# session/group leader so "Kill Qwen" can take down the whole group (the script
# spawns the real mlx workers as children), while stdout/stderr replace
# mlx-jaccl.out ("wb", one log per run, exactly like strata.out's ">"), which the
# "Update log" button tails over the very same SSH channel.
#
# The script path keeps its "~/..." form: it is passed as python's argv[3] (so the
# *outer* shell never sees an unquoted tilde) and Popen runs it with shell=True, i.e.
# through /bin/sh -c, where POSIX tilde expansion does happen. shell=True also keeps
# the recorded PID honest: for a single simple command sh execs the script in place,
# so the pidfile holds the session leader that owns the whole worker group.
_MLX_API_PORT = int(MLX_NODE.get("api_port") or 8080)
_MLX_DIR = '$HOME/mlx-jaccl'
_MLX_PIDFILE = f'{_MLX_DIR}/mlx-jaccl.pid'
_MLX_LOGFILE = f'{_MLX_DIR}/mlx-jaccl.out'
_MLX_MAX_TOKENS = 262144
def _mlx_start_cmd(use_8bit: bool) -> str:
    """The detached start command for the MLX-Jaccl launcher, with the "-8bit"
    flag appended only when the panel's 8-bit checkbox is set (which also halves
    the max tokens: the 8-bit model leaves less memory for the KV cache, so it is
    capped at half of the full-precision default). The script lives in
    ~/mlx-jaccl and its own options use relative paths (./hosts.json, etc.),
    so the command first cds into that directory."""
    extra = "" if not use_8bit else " -8bit"
    max_tokens = _MLX_MAX_TOKENS // 2 if use_8bit else _MLX_MAX_TOKENS
    return (
        'command -v python3 >/dev/null 2>&1 || { echo "python3 not found on the mlx node"; exit 127; }; '
        'python3 -c \'import sys, subprocess; '
        'pidfile, logfile, cmd = sys.argv[1], sys.argv[2], sys.argv[3]; '
        'log = open(logfile, "wb", 0); '
        'child = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, '
        'stdin=subprocess.DEVNULL, start_new_session=True); '
        'open(pidfile, "w").write(str(child.pid) + "\\n"); '
        'print("STARTED_PID=%d" % child.pid)\' '
        f'"{_MLX_PIDFILE}" "{_MLX_LOGFILE}" '
        f'"cd $HOME/mlx-jaccl && ~/mlx-jaccl/mlx_jaccl-start.py '
        f'-port {_MLX_API_PORT} -host 0.0.0.0 -max-tokens {max_tokens} '
        f'-hostfile ./hosts.json -leak-patch ./mlx_leak_shim.py Qwen3.8-27B{extra}"'
    )
# What an MLX-Jaccl run actually is made of, as pgrep -f regexes:
#   - the start script itself (rank 0 only, and the session leader recorded in the
#     pidfile: killing it alone would leave the workers behind, hence the group kill);
#   - the distributed launcher "mlx.launch" and the leak shim, which run on *both*
#     nodes of the run (rank 0 and rank 1).
# The "[.]" bracket trick keeps pgrep from matching the remote shell running the
# check itself, whose argv contains every pattern verbatim.
_MLX_PROC_PATTERNS = (r"mlx-jaccl/mlx_jaccl-start[.]py", r"mlx[.]launch", r"mlx_leak_shim[.]py")


# One-line, de-duplicated, no leading/trailing blanks: what $pids has to look like
# before it is tested with "set -- $pids" and looped over.
_MLX_PID_NORM = 'tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -'


def _mlx_pids_snippet(patterns: tuple[str, ...]) -> str:
    """POSIX sh collecting into $pids every live process matching one of patterns.

    pgrep exits 1 on no match, so its stderr is dropped and its output simply adds
    nothing; the tr/grep/sort/paste pipeline normalises the concatenation."""
    collect = ' '.join(f'pids="$pids $(pgrep -f "{p}" 2>/dev/null)";' for p in patterns)
    return f'pids=""; {collect} pids=$(echo $pids | {_MLX_PID_NORM}); '


_MLX_STATUS_CMD = (
    _mlx_pids_snippet(_MLX_PROC_PATTERNS)
    + 'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi'
)


def _mlx_kill_cmd(kill_run_group: bool) -> str:
    """POSIX sh killing every process of the run on *one* node.

    Only the matched PIDs are signalled, never their groups: on rank 1 the shim is
    just one process of somebody else's session, and "kill -- -pgid" there could
    take down unrelated work. kill_run_group is what rank 0 adds on top: the pidfile
    holds the leader of the session _mlx_start_cmd created, so that group is exactly
    "this run" and killing it also catches workers no pattern names. TERM, two seconds
    to drain, then KILL. Exit 3 means "nothing to kill"."""
    group = ''
    if kill_run_group:
        group = (
            f'g=$(cat "{_MLX_PIDFILE}" 2>/dev/null | tr -d " "); '
            'if [ -n "$g" ] && kill -0 "$g" 2>/dev/null; then '
            'pids="$pids $g"; RUN_GROUP=$g; else RUN_GROUP=""; fi; '
            # The pidfile's leader is usually also an argv match above, so normalise
            # again: KILLED_PIDS should read as a set, not "394 395 394".
            f'pids=$(echo $pids | {_MLX_PID_NORM}); '
        )
    group_kill = ('if [ -n "$RUN_GROUP" ]; then kill -TERM -$RUN_GROUP 2>/dev/null; fi; ' if kill_run_group else '')
    group_kill2 = ('if [ -n "$RUN_GROUP" ]; then kill -KILL -$RUN_GROUP 2>/dev/null; fi; ' if kill_run_group else '')
    return (
        _mlx_pids_snippet(_MLX_PROC_PATTERNS) + group +
        'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; exit 3; fi; '
        f'for p in $pids; do kill -TERM "$p" 2>/dev/null; done; {group_kill}'
        'sleep 2; '
        f'for p in $pids; do kill -KILL "$p" 2>/dev/null; done; {group_kill2}'
        'echo "KILLED_PIDS=$(echo $pids)"'
    )


# rank 0 = the node the start script runs on (kills the run's session too),
# rank 1 = the slave Mac, where only the named processes are touched.
_MLX_KILL_CMD = _mlx_kill_cmd(kill_run_group=False)
_MLX_KILL_CMD_RANK0 = _mlx_kill_cmd(kill_run_group=True)
# Console log of the service: mlx-jaccl.out, written by _mlx_start_cmd. Fetched on
# demand as a tail, so a long-running service does not flood the browser. ("wc -l < f"
# pads its output with blanks on BSD, hence the tr.)
_MLX_LOG_LINES = 200
_MLX_LOG_CMD = (
    f'f="{_MLX_LOGFILE}"; '
    'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
    f'echo "--- $f: last {_MLX_LOG_LINES} of $(wc -l < "$f" | tr -d " ") lines ---"; '
    f'tail -n {_MLX_LOG_LINES} "$f"'
)
# Model the fixed start command loads; shown on the card (and copyable) while the
# service is up, since there is no API to ask the server for its model id.
_MLX_MODEL_NAME = "Qwen3.8-27B"


# The Splash "service" on the splash node: same shape as the MLX card — the node
# is a Mac (BSD userland: no setsid(1), no /proc, possibly zsh as login shell),
# so the "new session + pidfile + redirect" dance is again done by the python3
# one-liner described above, and the workload is launched verbatim with a fixed
# command line. Unlike MLX though, `splash serve` speaks an OpenAI-compatible
# API (on the "api_port" of the splash entry in nodes.json), so the status
# logic mirrors Strata's: RUNNING needs
# both the process alive and /v1/models to answer, a live process with a silent
# API is only STARTING.
#
# What identifies the run for pgrep is the "splash-start-30.sh" script name:
# the "[ ]"/"." class trick keeps pgrep from matching the remote shell running
# the check itself (whose argv contains the pattern verbatim, while the regex
# only matches the literal script path). The pidfile holds the session leader
# created by _SPLASH_START_CMD, so "Kill Qwen" can take down the whole group
# — the script may spawn its own workers as children.
_SPLASH_PIDFILE = '$HOME/splash.pid'
_SPLASH_LOGFILE = '$HOME/splash.out'
_SPLASH_API_PORT = int(SPLASH_NODE.get("api_port") or 10000)
_SPLASH_SERVE_CMD = (
    'bash "$HOME/splash/splash-start-30.sh"'
)
_SPLASH_START_CMD = (
    'command -v python3 >/dev/null 2>&1 || { echo "python3 not found on the splash node"; exit 127; }; '
    'python3 -c \'import sys, subprocess; '
    'pidfile, logfile, cmd = sys.argv[1], sys.argv[2], sys.argv[3]; '
    'log = open(logfile, "wb", 0); '
    'child = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, '
    'stdin=subprocess.DEVNULL, start_new_session=True); '
    'open(pidfile, "w").write(str(child.pid) + "\\n"); '
    'print("STARTED_PID=%d" % child.pid)\' '
    f'"{_SPLASH_PIDFILE}" "{_SPLASH_LOGFILE}" "{_SPLASH_SERVE_CMD}"'
)
_SPLASH_PROC_PATTERN = r"splash[-_]start[-_]30[.]sh"
_SPLASH_STATUS_CMD = (
    f'pids=$(pgrep -f "{_SPLASH_PROC_PATTERN}" 2>/dev/null | tr "\\n" " "); '
    'pids=$(echo $pids | tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -); '
    'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi'
)
# Kill = the argv matches plus the run's whole session from the pidfile (the
# leader is usually also an argv match, hence the de-dup). TERM, two seconds to
# drain, then KILL. Exit 3 means "nothing to kill".
_SPLASH_KILL_CMD = (
    f'g=$(cat "{_SPLASH_PIDFILE}" 2>/dev/null | tr -d " "); '
    f'pids=$(pgrep -f "{_SPLASH_PROC_PATTERN}" 2>/dev/null | tr "\\n" " "); '
    'pids="$pids $g"; '
    'pids=$(echo $pids | tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -); '
    'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; exit 3; fi; '
    'RUN_GROUP=$g; '
    'for p in $pids; do kill -TERM "$p" 2>/dev/null; done; '
    'if [ -n "$RUN_GROUP" ]; then kill -TERM -$RUN_GROUP 2>/dev/null; fi; '
    'sleep 2; '
    'for p in $pids; do kill -KILL "$p" 2>/dev/null; done; '
    'if [ -n "$RUN_GROUP" ]; then kill -KILL -$RUN_GROUP 2>/dev/null; fi; '
    'echo "KILLED_PIDS=$pids"'
)
# Console log of the service: splash.out, written by _SPLASH_START_CMD under
# $HOME. Fetched on demand as a tail, so a long-running service does not flood
# the browser. ("wc -l < f" pads with blanks on BSD, hence the tr.)
_SPLASH_LOG_LINES = 200
_SPLASH_LOG_CMD = (
    f'f="{_SPLASH_LOGFILE}"; '
    'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
    f'echo "--- $f: last {_SPLASH_LOG_LINES} of $(wc -l < "$f" | tr -d " ") lines ---"; '
    f'tail -n {_SPLASH_LOG_LINES} "$f"'
)
# Model the fixed serve command loads; shown on the card (and copyable) while
# the service is up and the API has not (yet) reported its own model id.
_SPLASH_MODEL_NAME = "mlx-community/Qwen3.8-27B-4bit"


# The FreeToken "service" on the freetoken node: same shape as the Strata card —
# a Linux box (so setsid(1) is available, and the remote stderr stays where ssh
# puts it) whose service is started detached, logged to a single per-run file and
# counted as UP only once its OpenAI-compatible API answers /v1/models (on port
# the "api_port" of the freetoken entry in nodes.json, where "ft serve"
# listens). The workload is a fixed "ft serve" command
# line pinned to one GPU, so status/kill do not trust the pidfile alone: what
# identifies the run for pgrep is the ft CLI path itself. The "[.]" bracket
# trick keeps pgrep from matching the very remote shell running the check, whose
# argv contains the pattern verbatim while the regex only matches the literal
# ".ft-orig".
_FT_API_PORT = int(FREETOKEN_NODE.get("api_port") or 1919)
_FT_PIDFILE = '$HOME/freetoken.pid'
_FT_LOGFILE = '$HOME/freetoken.out'
_FT_PROC_PATTERN = r"/nvme/freetoken/venv/bin/[.]ft-orig/ft serve"
# Model the fixed serve command loads, and its full command line: the GPU is
# pinned through env (rather than a bare VAR=val prefix) so the whole thing stays
# a single simple command that "exec" can replace the wrapper shell with.
_FT_MODEL_NAME = "/nvme/freetoken/gpt-oss-120b"
_FT_SERVE_CMD = (
    'env CUDA_VISIBLE_DEVICES=GPU-de110751-309a-0814-d910-623f85e22787 '
    '/nvme/freetoken/venv/bin/python /nvme/freetoken/venv/bin/.ft-orig/ft serve '
    f'--model /nvme/freetoken/gpt-oss-120b --port {_FT_API_PORT} --moe-backend auto '
    '--max-running-requests 1 --memory-ratio 0.95 --moe-cpu-threads 14 '
    '--host 0.0.0.0 '
    '--cors-origins tauri://localhost,http://tauri.localhost,http://localhost:1420'
)
# Detached start, the same dance Strata's start.sh does: setsid makes the run its
# own session/group leader, so "Kill FreeToken" can take down ft and any worker
# it spawned with one "kill -- -pgid"; "exec" keeps the pidfile honest (no
# wrapper shell left between the recorded pid and ft) and freetoken.out collects
# stdout/stderr, one file per run — the "Update log" button tails it over the
# very same SSH channel. 'cd "$HOME"' pins the cwd (hence the pidfile and the
# logfile) to the SSH user's home, whatever the login shell's startup files do.
_FT_START_CMD = (
    'cd "$HOME" && { setsid bash -c \'echo $$ > freetoken.pid; exec ' + _FT_SERVE_CMD +
    '\' > freetoken.out 2>&1 </dev/null & sleep 1; '
    'echo "STARTED_PID=$(cat freetoken.pid)"; }'
)
_FT_STATUS_CMD = (
    f'pids=$(pgrep -f "{_FT_PROC_PATTERN}" | tr "\\n" " "); '
    'pids=$(echo $pids | tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -); '
    'if [ -z "$pids" ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi'
)
# Kill = every live ft process plus the run's whole session from the pidfile (the
# leader is usually also an argv match, hence the de-dup). TERM, two seconds to
# drain, then KILL. Exit 3 means "nothing to kill".
_FT_KILL_CMD = (
    f'g=$(cat "{_FT_PIDFILE}" 2>/dev/null | tr -d " "); '
    f'pids=$(pgrep -f "{_FT_PROC_PATTERN}" | tr "\\n" " "); '
    'pids="$pids $g"; '
    'pids=$(echo $pids | tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -); '
    'if [ -z "$pids" ]; then echo "NO_PID"; exit 3; fi; '
    'RUN_GROUP=$g; '
    'for p in $pids; do kill -TERM "$p" 2>/dev/null; done; '
    'if [ -n "$RUN_GROUP" ]; then kill -TERM -$RUN_GROUP 2>/dev/null; fi; '
    'sleep 2; '
    'for p in $pids; do kill -KILL "$p" 2>/dev/null; done; '
    'if [ -n "$RUN_GROUP" ]; then kill -KILL -$RUN_GROUP 2>/dev/null; fi; '
    'echo "KILLED_PIDS=$pids"'
)
# Console log of the service: freetoken.out, the stdout/stderr of the "ft serve"
# started by _FT_START_CMD (each start truncates it, ">"). Fetched on demand as a
# tail, so a long-running service does not flood the browser.
_FT_LOG_LINES = 200
_FT_LOG_CMD = (
    f'f="{_FT_LOGFILE}"; '
    'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
    f'echo "--- $f: last {_FT_LOG_LINES} of $(wc -l < "$f") lines ---"; '
    f'tail -n {_FT_LOG_LINES} "$f"'
)
# Model the fixed serve command loads; shown on the card (and copyable) while the
# process is up and the API has not (yet) reported its own model id.
_FT_MODEL_LABEL = "GPT-OSS"


# The Llama "service": one llama-server (on the nodes.json "llama" node) fed by
# two ggml-rpc-server workers (the "rpc_nodes" list of that entry), all three
# launched with fixed command lines over the very same SSH channel the other
# cards use. Status reading mirrors Strata's: llama-server counts as RUNNING
# only once its OpenAI-compatible API answers /v1/models on the "api_port" of
# the llama entry in nodes.json; a live
# process with a silent API is only STARTING.
#
# The three nodes may be Macs or Linux boxes, so the "new session + pidfile +
# redirect" dance is done by the python3 one-liner already used by the MLX and
# Splash cards (start_new_session=True calls setsid(2) in either userland)
# rather than by Linux-only setsid(1). The workload command travels base64-
# encoded: the llama-server argv carries single-quoted JSON
# (--chat-template-kwargs) that no amount of outer shell quoting would survive,
# while base64 output is quote-proof by construction.
#
# The ggml-rpc-server command lines already pipe their own output through tee
# into rpc.log, so the console log of those launches is discarded (/dev/null);
# llama-server writes its own --log-file /tmp/llama-server.log (what the
# "Update log" button tails), so its console output goes to a separate file.
_LLAMA_API_PORT = int(LLAMA_NODE.get("api_port") or 8088)
_LLAMA_MODEL_NAME = "Gemma-4-31B-it-UD-Q8_K_XL"
_LLAMA_RPC_PIDFILE = "/tmp/llama-rpc-server.pid"
_LLAMA_SERVER_PIDFILE = "/tmp/llama-server.pid"
_LLAMA_SERVER_CONSOLE_LOG = "$HOME/llama-server.out"
# What identifies the two workloads for pgrep -f. The "[x]" bracket trick keeps
# pgrep from matching the very remote shell running the check itself, whose
# argv contains the pattern verbatim while the regex only matches the literal
# binary path.
_LLAMA_SERVER_PATTERN = r"/opt/llama[.]cpp/llama[-]server"
_LLAMA_RPC_PATTERN = r"/opt/llama[.]cpp/ggml-rpc[-]server"
# The ggml-rpc-server command lines, keyed by the IP of the node that must run
# them (cache dir and tee target differ per node); the IPs must match the
# "rpc_nodes" entries of the llama node in nodes.json.
_LLAMA_RPC_LAUNCH = {
    "192.168.1.101": (
        "LLAMA_CACHE=/Volumes/Home/llama-rpc-cache /opt/llama.cpp/ggml-rpc-server "
        "--host 0.0.0.0 --port 50000 -c 2>&1 "
        "| tee /Volumes/Home/llama-rpc-cache/llama-rpc-cache/rpc.log"
    ),
    "192.168.1.200": (
        "LLAMA_CACHE=/nvme/llama-rpc-cache /opt/llama.cpp/ggml-rpc-server "
        "--host 0.0.0.0 --port 50000 -c 2>&1 "
        "| tee /nvme/llama-rpc-cache/rpc.log"
    ),
}
# The llama-server command line, verbatim: RPC backends point at the two rpc
# nodes' storage-network IPs, model paths are on the llama node's volume.
_LLAMA_SERVER_CMD = (
    "/opt/llama.cpp/llama-server -m /Volumes/Home/gguf_models/Gemma-4-31B-it-UD-Q8_K_XL.gguf "
    "-c 262144 --rpc 192.168.20.1:50000,192.168.30.3:50000 "
    "--device MTL0,RPC0,RPC1,RPC2 -ts 28,16,15,15 "
    f"--host 0.0.0.0 --port {_LLAMA_API_PORT} --split-mode layer --metrics --no-context-shift "
    "--jinja -fa on -fitc 8192 -np 1 --no-warmup --temp 1.0 --top-p 0.95 --top-k 64 "
    "--chat-template-kwargs '{\"reasoning_effort\": \"medium\", \"preserve_thinking\": true}' "
    "--seed 123456789 --repeat-penalty 1.0 --min-p 0.0 "
    "--log-file /tmp/llama-server.log -ctxcp 8 --reasoning on"
)


def _detached_start_cmd(pidfile: str, logfile: str, cmd: str) -> str:
    """POSIX sh launching *cmd* detached: new session, pidfile, logfile.

    The python3 one-liner of the MLX/Splash cards: Popen(start_new_session=True)
    makes the workload a session/group leader (so the kill commands can take
    down the whole run with one "kill -- -pgid", tee included), its pid goes in
    the pidfile, and stdout/stderr append to logfile. *cmd* crosses the shell
    boundary base64-encoded, so no quote inside it can break the outer command
    (and no quote of the python code itself is ever needed)."""
    b64 = base64.b64encode(cmd.encode()).decode()
    return (
        'command -v python3 >/dev/null 2>&1 || { echo "python3 not found on the target node"; exit 127; }; '
        'python3 -c \'import base64,sys,subprocess; '
        'pidfile, logfile, cmd = sys.argv[1], sys.argv[2], base64.b64decode(sys.argv[3]).decode(); '
        'log = open(logfile, "ab", 0); '
        'child = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, '
        'stdin=subprocess.DEVNULL, start_new_session=True); '
        'open(pidfile, "w").write(str(child.pid) + "\\n"); '
        'print("STARTED_PID=%d" % child.pid)\' '
        f'"{pidfile}" "{logfile}" "{b64}"'
    )


def _pgrep_status_cmd(pattern: str) -> str:
    """POSIX sh echoing "ALIVE <pids>" for every live match of pattern, NO_PID otherwise."""
    return (
        f'pids=$(pgrep -f "{pattern}" 2>/dev/null | tr "\\n" " "); '
        'pids=$(echo $pids | tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -); '
        'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi'
    )


def _pgrep_kill_cmd(pattern: str, pidfile: str) -> str:
    """POSIX sh killing every match of pattern plus the run's whole session.

    The pidfile holds the session leader created by _detached_start_cmd, so the
    group is exactly "this run" (workload, tee, any worker it spawned) — but the
    pid is only trusted while it is alive (kill -0), so a stale pidfile can never
    take down an unrelated process whose pid got reused. TERM, two seconds to
    drain, then KILL. Exit 3 means "nothing to kill"; after the dust settles the
    pattern is probed again and exit 4 (STILL_ALIVE) says the run did *not* go
    down, so the caller can retry or complain."""
    return (
        f'g=$(cat "{pidfile}" 2>/dev/null | tr -d " "); '
        'if [ -n "$g" ] && kill -0 "$g" 2>/dev/null; then pids="$pids $g"; RUN_GROUP=$g; '
        'else RUN_GROUP=""; fi; '
        f'pids="$pids $(pgrep -f "{pattern}" 2>/dev/null | tr "\\n" " ")"; '
        'pids=$(echo $pids | tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -); '
        'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; exit 3; fi; '
        'for p in $pids; do kill -TERM "$p" 2>/dev/null; done; '
        'if [ -n "$RUN_GROUP" ]; then kill -TERM -$RUN_GROUP 2>/dev/null; fi; '
        'sleep 2; '
        'for p in $pids; do kill -KILL "$p" 2>/dev/null; done; '
        'if [ -n "$RUN_GROUP" ]; then kill -KILL -$RUN_GROUP 2>/dev/null; fi; '
        'sleep 1; '
        f'left=$(pgrep -f "{pattern}" 2>/dev/null | tr "\\n" " "); '
        'left=$(echo $left); '
        'if [ -n "$left" ]; then echo "STILL_ALIVE $left"; exit 4; fi; '
        'echo "KILLED_PIDS=$pids"'
    )


_LLAMA_SERVER_START_CMD = _detached_start_cmd(
    _LLAMA_SERVER_PIDFILE, _LLAMA_SERVER_CONSOLE_LOG, _LLAMA_SERVER_CMD)
_LLAMA_STATUS_CMD = _pgrep_status_cmd(_LLAMA_SERVER_PATTERN)
_LLAMA_RPC_STATUS_CMD = _pgrep_status_cmd(_LLAMA_RPC_PATTERN)
_LLAMA_KILL_CMD = _pgrep_kill_cmd(_LLAMA_SERVER_PATTERN, _LLAMA_SERVER_PIDFILE)
_LLAMA_RPC_KILL_CMD = _pgrep_kill_cmd(_LLAMA_RPC_PATTERN, _LLAMA_RPC_PIDFILE)
# Console log of the service: /tmp/llama-server.log, written by llama-server
# itself (--log-file). Fetched on demand as a tail from the llama node, so a
# long-running service does not flood the browser. ("wc -l < f" pads with
# blanks on BSD, hence the tr.)
_LLAMA_LOG_LINES = 200
_LLAMA_LOG_CMD = (
    'f="/tmp/llama-server.log"; '
    'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
    f'echo "--- $f: last {_LLAMA_LOG_LINES} of $(wc -l < "$f" | tr -d " ") lines ---"; '
    f'tail -n {_LLAMA_LOG_LINES} "$f"'
)


def _strata_api_check(ip: str, port: int = _STRATA_API_PORT) -> tuple[bool, str]:
    """(ok, model_id) from the node's OpenAI-compatible /v1/models endpoint.

    Runs in a worker thread (see update_strata_status); any failure maps to
    ok=False so a dead/unreachable API reads as "not ready", never an exception.
    Reused by the Splash and MLX cards, whose servers answer the same endpoint on
    their own ports (for MLX it is only an id lookup, never a status probe)."""
    try:
        r = requests.get(f"http://{ip}:{port}/v1/models", timeout=3)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return False, ""
    models = data.get("data") if isinstance(data, dict) else None
    if models and isinstance(models, list) and isinstance(models[0], dict):
        return True, str(models[0].get("id", ""))
    return True, ""

# Resolve start_model.py next to this file: relying on the process CWD broke as
# soon as the systemd unit was started from anywhere else.
_START_MODEL = str(Path(__file__).parent / "start_model.py")
# sys.executable, not "python3": the child must run in the same interpreter (and
# virtualenv) as the GUI, not whatever "python3" resolves to in PATH.
_PY = sys.executable

_CTX_MIN = 8192
_CTX_STEP = 1024
# ctx_budget_gb_bf16_128k is defined at 128k tokens, so scaling to any context
# is a simple ratio against it.
_KV_REF_CTX = 131072
# KV cache size multiplier relative to bf16, per quantization.
_KV_QUANT_SCALE = {"q8_0": 0.5, "q4_0": 0.25}
# ctx_budget_gb_bf16_128k is decimal GB (10^9 B) while the model size is binary
# GiB (2^30 B): convert so the whole estimate is reported in GiB.
_GB_TO_GIB = 1e9 / 1024**3
# Reasoning-effort radio options: "" (None) means "no override — keep the REAS
# value defined in models.json"; the other values override it (same convention
# as the KV quant radio).
_REAS_OPTIONS = {"": "None", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}


def _format_uptime(total_seconds: int) -> str:
    """Human-readable duration, most significant two units (3d 4h, 2h 05m, 45s)."""
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


#___________________________________________________________________________________
async def _capture(argv: list[str]) -> tuple[str, int]:
    """Run argv to completion off the event loop; return (stdout, returncode)."""
    logger.debug(f"Executing {argv}")
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if stderr:
        logger.debug(stderr.decode(errors="replace").rstrip())
    return stdout.decode(errors="replace"), proc.returncode


#___________________________________________________________________________________
async def _run_json(args: list[str]) -> dict | None:
    """Run start_model.py with --json and return the parsed stdout, or None.

    The return code is deliberately ignored: --server-status exits 1 to mean
    'not running', which is a perfectly valid answer, not a failure. What tells
    success from failure here is whether stdout carried parseable JSON.
    """
    out, rc = await _capture([_PY, _START_MODEL, *args, "--json"])
    if not out.strip():
        logger.error(f"start_model.py {' '.join(args)} produced no output (rc={rc})")
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        logger.error(f"start_model.py {' '.join(args)} returned invalid JSON: {e}")
        return None


#___________________________________________________________________________________
# Model-list cache: loading the models is slow (one SSH round-trip per model, to
# check that its file exists and how big it is), so it runs once at application
# boot and, after that, only when the user presses "Recheck models". Every page
# (re)load reads this cache instead of hitting the remote host again.
_models: dict[str, Model] | None = None
_models_lock = asyncio.Lock()


async def fetch_models(force: bool = False) -> tuple[dict[str, Model] | None, bool]:
    """Return ({name: Model}, ok). Reloads only when the cache is empty or
    force=True; a failed forced reload keeps the previous list."""
    global _models
    async with _models_lock:
        if _models is None or force:
            try:
                # load_models() is blocking (SSH), so it goes to a worker thread:
                # on the event loop it would freeze every connected browser for
                # the whole duration of the scan.
                models = await asyncio.to_thread(
                    load_models_from_config,
                    Path(settings.MODELS_JSON),
                    Path(settings.RPC_JSON),
                    remote_host=settings.LLAMA_SERVER_HOST,
                    remote_user=settings.LLAMA_SERVER_USER,
                    check_remote_file=True,
                )
            except Exception:
                # Deliberately broad, as the subprocess boundary used to be: a
                # malformed models.json or an unreachable host must leave the
                # console running (with the previous list) rather than kill it.
                logger.exception("Could not load the model list")
                return _models, False
            _models = {m.model_name: m for m in models}
        return _models, True


# Kick the fetch off in the background at boot: pages opened before it finishes
# just await the same in-flight call via the lock instead of starting their own.
app.on_startup(lambda: asyncio.create_task(fetch_models()))


#___________________________________________________________________________________
# The last tab each browser was on, so a page reload reopens that tab instead of
# always falling back to the default. Backed by app.storage.user: NiceGUI issues
# a persistent "nicegui:user" cookie on the first visit and keeps the dict in a
# local SQLite database, so the memory survives page reloads *and* GUI restarts,
# and is per browser rather than per IP (clients behind the same NAT no longer
# share a tab). The _remembered_tab/_remember_tab helpers must run in a client
# context (page handler or event handler), which is exactly where they are
# called from. The constants live above the class because build_ui() takes
# _DEFAULT_TAB as a default argument, evaluated when the class is defined.
_ACTIVE_TAB_KEY = "active_tab"

# app.storage.user requires a storage_secret in ui.run(). It is generated once
# and persisted next to this file, so the signed cookies stay valid across
# restarts; a random per-boot secret would simply log every client out.
_STORAGE_SECRET_FILE = Path(__file__).parent / ".storage_secret"


def _storage_secret() -> str:
    try:
        return _STORAGE_SECRET_FILE.read_text().strip()
    except OSError:
        secret = base64.urlsafe_b64encode(os.urandom(32)).decode()
        try:
            _STORAGE_SECRET_FILE.write_text(secret)
            os.chmod(_STORAGE_SECRET_FILE, 0o600)
        except OSError:
            pass  # still usable for this run even if the file cannot be written
        return secret


# Tab names that ui.tab_panels actually accepts ("llama" is wired but hidden).
_VALID_TABS = ("llama", "llama_cluster", "mlx", "strata", "splash", "freetoken")

# The tab selected for a first visit, and the fallback when a remembered value
# no longer matches any tab (e.g. after the old Llama.cpp card was hidden).
_DEFAULT_TAB = "llama_cluster"


def _remembered_tab() -> str:
    """The tab this browser last used, or the default for a first visit."""
    tab = app.storage.user.get(_ACTIVE_TAB_KEY, _DEFAULT_TAB)
    return tab if tab in _VALID_TABS else _DEFAULT_TAB


def _remember_tab(tab: str) -> None:
    """Persist the tab for this browser; the write also refreshes the cookie."""
    app.storage.user[_ACTIVE_TAB_KEY] = tab


#___________________________________________________________________________________
class LlamaConsoleGUI:
    """One instance per connected browser.

    The UI used to be built once at import time, so every client shared the same
    widget objects; it is now created inside an @ui.page handler (see below),
    which is also what puts UI updates on the right client context.
    """

    def __init__(self):
        self.models: dict[str, Model] = {}
        self.log_task: asyncio.Task | None = None
        self._status_busy = False
        self._start_busy = False
        self._stop_busy = False
        self.start_button = None
        self.stop_button = None

        self.strata_run_button = None
        self.strata_kill_button = None
        self.strata_status_label = None
        self.strata_model_label = None
        self.strata_model_name = ""
        self.strata_model_copy = None
        self.strata_host_label = None
        self.strata_log_update_button = None
        self.strata_log_clear_button = None
        self.strata_log_window = None
        self._strata_busy = False
        self._strata_status_busy = False
        self._strata_log_busy = False

        self.mlx_run_button = None
        self.mlx_kill_button = None
        self.mlx_status_label = None
        self.mlx_model_label = None
        self.mlx_model_name = ""
        self.mlx_model_copy = None
        self.mlx_host_label = None
        self.mlx_slave_label = None
        self.mlx_log_update_button = None
        self.mlx_log_clear_button = None
        self.mlx_log_window = None
        self._mlx_busy = False
        self._mlx_status_busy = False
        self._mlx_log_busy = False

        self.splash_run_button = None
        self.splash_kill_button = None
        self.splash_status_label = None
        self.splash_model_label = None
        self.splash_model_name = ""
        self.splash_model_copy = None
        self.splash_host_label = None
        self.splash_log_update_button = None
        self.splash_log_clear_button = None
        self.splash_log_window = None
        self._splash_busy = False
        self._splash_status_busy = False
        self._splash_log_busy = False

        self.freetoken_run_button = None
        self.freetoken_kill_button = None
        self.freetoken_status_label = None
        self.freetoken_model_label = None
        self.freetoken_model_name = ""
        self.freetoken_model_copy = None
        self.freetoken_host_label = None
        self.freetoken_log_update_button = None
        self.freetoken_log_clear_button = None
        self.freetoken_log_window = None
        self._freetoken_busy = False
        self._freetoken_status_busy = False
        self._freetoken_log_busy = False

        self.llama_run_button = None
        self.llama_kill_button = None
        self.llama_status_label = None
        self.llama_model_label = None
        self.llama_model_name = ""
        self.llama_model_copy = None
        self.llama_host_label = None
        self.llama_rpc_label = None
        self.llama_log_update_button = None
        self.llama_log_clear_button = None
        self.llama_log_window = None
        self._llama_busy = False
        self._llama_status_busy = False
        self._llama_log_busy = False

        self.status_server_label = None
        self.status_build_label = None
        self.status_started_label = None
        self.samplers_label = None
        self.samplers_tip = None
        self.status_model_label = None
        self.status_model_name = ""
        self.status_model_copy = None
        self.status_ctx_label = None
        self.status_samplers_label = None
        self.status_samplers_tip = None
        # self.status_topk_label = None
        # self.status_topp_label = None
        # self.status_minp_label = None
        self.model_dropdown = None
        self.ctx_slider = None
        self.kvquant_radio = None
        self.reas_radio = None
        self.ctx_label = None
        self.mem_label = None
        self.temp_slider = None
        self.temp_label = None
        self.log_window = None
        self.stop_log_button = None
        self.rpc_labels: dict[str, ui.label] = {}
        self.rpc_servers_row = None

    # -------------------------------------------------------------- samplers ---
    # The four values packed into the "Samplers:" label, in the order they are
    # printed, each with the one-line explanation the hover tooltip shows.
    _SAMPLER_HINTS = (
        ("temperature", "scales the logits — below 1.00 sharpens the model's distribution, "
                        "above 1.00 flattens it"),
        ("top_p",       "nucleus sampling: keeps the most likely tokens up to this cumulative "
                        "probability"),
        ("top_k",       "keeps only the k most likely tokens (0 disables the cut)"),
        ("min_p",       "drops every token below this fraction of the top token's probability"),
    )

    def _samplers_description(self, values: tuple[str, str, str, str], header: str) -> str:
        """Spell the four packed values out, one parameter per line."""
        lines = [header]
        lines += [f"{name} = {value}  —  {hint}"
                  for (name, hint), value in zip(self._SAMPLER_HINTS, values)]
        return "\n".join(lines)

    def _set_samplers(self, values: tuple[str, str, str, str] | None = None) -> None:
        """Write the "Samplers:" label together with its hover description.

        `values` holds temperature, top_p, top_k and min_p already formatted for
        display, or is None while no model has been selected yet. The tooltip
        that follows the mouse reads its text from a hidden sibling label (see
        the #cursor-tip rules and script in the head HTML), so both are written
        here: the description can never fall out of sync with the numbers.
        """
        shown = values or ("--", "--", "--", "--")
        self.samplers_label.set_text("Samplers: " + ";".join(shown))

        description = self._samplers_description(
            shown, "Sampling parameters of the selected model, in the order shown:")
        if values is None:
            description += "\n(the values appear once a model is selected)"
        self.samplers_tip.set_text(description)

    def _set_status_samplers(self, values: tuple[str, str, str, str] | None = None) -> None:
        """Same, for the status line: what the *running* server was started with.

        `values` is None whenever there is nothing to report (no server, or one
        still starting up): the line then goes blank, and the empty host is left
        with no width to hover.
        """
        if values is None:
            self.status_samplers_label.set_text("")
            self.status_samplers_tip.set_text("")
            return
        self.status_samplers_label.set_text(" - Samplers: " + ";".join(values))
        self.status_samplers_tip.set_text(self._samplers_description(
            values, "Sampling parameters the running server was started with:"))

    # ---------------------------------------------------------------- status ---
    def _update_rpc_labels( self ) -> None:
        # The selected model comes from the cache: re-reading models.json from
        # disk on every selection change (get_model_byname) bought nothing.
        selected_model = self.models.get(self.model_dropdown.value)
        # The labels depend on the selected model, so the row is rebuilt from
        # scratch on every change.
        if self.rpc_servers_row is None:
            return
        self.rpc_servers_row.clear()
        self.rpc_labels.clear()
        if selected_model is None:
            return

        with self.rpc_servers_row:
            for rpc in selected_model.rpcservers or []:
                if rpc.name in RPC_SERVERS:
                    self.rpc_labels[rpc.name] = ui.label(rpc.name)

    # ---------------------------------------------------------------- data ---
    async def load_models(self, force: bool = False) -> bool:
        models, ok = await fetch_models(force)
        if models is None:
            self.model_dropdown.options = {"": "No models found"}
            self.model_dropdown.update()
            ui.notify("Could not load the model list", type="negative")
            return False
        if not ok:
            ui.notify("Could not reload the model list — keeping the previous one",
                      type="negative")

        self.models = models
        if not self.models:
            self.model_dropdown.options = {"": "No models found"}
            self.model_dropdown.update()
            return ok

        # Dict options: the value stays the bare model name, so nothing
        # downstream has to strip the ' (60 GiB - 2 RPC)' suffix back off.
        self.model_dropdown.options = {
            name: f"{name} ({int(m.size_gib) if m.size_gib is not None else '?'} GiB"
                  f" - {len(m.rpcservers or [])} RPC)"
            for name, m in self.models.items()
        }
        self.model_dropdown.props(remove='disable')
        self.model_dropdown.props('label="Model selector"')
        # Keep the current selection across a recheck when it still exists.
        previous = self.model_dropdown.value
        selected = previous if previous in self.models else next(iter(self.models))
        self.model_dropdown.set_value(selected)
        self.model_dropdown.update()
        self._apply_model_spec(selected)
        self._update_rpc_labels()
        return ok

    async def recheck_models(self) -> None:
        """Reload the models on demand — the only trigger besides app boot."""
        self.model_dropdown.props('disable')
        self.model_dropdown.props('label="Reloading models..."')
        ui.notify("Rechecking models...")
        if await self.load_models(force=True):
            ui.notify("Model list updated", type="positive")

    # ---------------------------------------------------------------- update ---
    async def update_status(self) -> None:
        # Polled every 5 s by a ui.timer: if the previous check is still in
        # flight (slow SSH round-trip), skip this tick instead of stacking up.
        if self._status_busy:
            return
        self._status_busy = True
        try:
            await self._update_status_once()
        finally:
            self._status_busy = False

    async def _update_status_once(self) -> None:
        info = await _run_json(["--server-status"])
        if info is None or not info.get("transport", True):
            # No valid answer from the host (SSH failure or bad output): the
            # state is unknown, NOT 'not running'. Showing the latter read as
            # 'all clear, safe to start' while the real state was simply
            # unknown — the next START click then hit a live server and was
            # refused as 'already running'.
            self.status_server_label.set_text("Server Status: UNREACHABLE")
            self.status_server_label.style("color: orange;")
            self.status_model_name = ""
            self.status_model_label.set_text("")
            self.status_model_copy.classes(add='q-hidden')
            self.status_build_label.set_text("")
            self.status_started_label.set_text("")
            self.status_ctx_label.set_text("")
            self.status_samplers_label.set_text("")
            self._set_status_samplers()
            self.stop_button.classes(add='q-hidden')
            return

        running = bool(info.get("running"))
        # Starting a second model over a live server is never valid, so START
        # follows the polled status; the click handler re-checks to close the
        # window between two polls. STOP mirrors it: it only appears while a
        # model is actually live.
        if running:
            self.start_button.disable()
            self.stop_button.classes(remove='q-hidden')
        else:
            self.start_button.enable()
            self.stop_button.classes(add='q-hidden')
        color = "#00ff88" if running else "red"
        self.status_server_label.set_text(
            f"Server Status: {'RUNNING' if running else 'NOT RUNNING'}"
        )
        self.status_server_label.style(f"color: {color};")

        if running and info.get("ready"):
            build_info = str(info.get("build_info") or "").strip()
            self.status_build_label.set_text(f" - Build   : llama.cpp/{build_info}" if build_info else "")
            # Keep the bare name around: only it (not the " - Model   : "
            # prefix) is what gets copied to the clipboard.
            self.status_model_name = str(info.get("model") or "").strip()
            self.status_model_label.set_text(f" - Model   : {self.status_model_name}")
            #created = info.get("created")
            #if isinstance(created, (int, float)):
            #    started = datetime.fromtimestamp(created)
            #    uptime = _format_uptime(int((datetime.now() - started).total_seconds()))
            #    self.status_started_label.set_text(
            #        f" - Started : {started:%Y-%m-%d %H:%M:%S} ({uptime})")
            #else:
            #    self.status_started_label.set_text("")
            c = (str(info['ctx'])).strip()
            self.status_ctx_label.set_text(  f" - Context : {c} tokens")
            # Rounded: the inference engine reports the float32 round-trip of 0.6 as
            # 0.6000000238418579.
            self.status_samplers_label.set_text( f" - Samplers: {float(info['temperature']):.1f};{float(info['top_p']):.2f};{int(info['top_k'])};{float(info['min_p']):.2f}")
                                    
        elif running:
            self.status_model_name = ""
            self.status_build_label.set_text("")
            self.status_model_label.set_text("Model: (Loading...)")
            self.status_started_label.set_text("")
            self.status_ctx_label.set_text("")
            self.status_samplers_label.set_text("")
            self._set_status_samplers()
        else:
            self.status_model_name = ""
            self.status_build_label.set_text("")
            self.status_model_label.set_text("")
            self.status_started_label.set_text("")
            self.status_ctx_label.set_text("")
            self.status_samplers_label.set_text("")
            self._set_status_samplers()
        # Nothing to copy unless a model name is actually on display.
        if self.status_model_name:
            self.status_model_copy.classes(remove='q-hidden')
        else:
            self.status_model_copy.classes(add='q-hidden')

    async def refresh(self) -> None:
        await self.update_status()
        ui.notify("Status updated")

    async def _copy_model_name(self) -> None:
        """Copy the running (llama) model's name to the OS clipboard."""
        await self._copy_to_clipboard(self.status_model_name, "model name")

    async def _copy_strata_model_name(self) -> None:
        """Copy the Strata model's name to the OS clipboard."""
        await self._copy_to_clipboard(self.strata_model_name, "model name")

    async def _copy_mlx_model_name(self) -> None:
        """Copy the MLX-Jaccl model's name to the OS clipboard."""
        await self._copy_to_clipboard(self.mlx_model_name, "model name")

    async def _copy_splash_model_name(self) -> None:
        """Copy the Splash model's name to the OS clipboard."""
        await self._copy_to_clipboard(self.splash_model_name, "model name")

    async def _copy_freetoken_model_name(self) -> None:
        """Copy the FreeToken model's name to the OS clipboard."""
        await self._copy_to_clipboard(self.freetoken_model_name, "model name")

    async def _copy_llama_model_name(self) -> None:
        """Copy the Llama (Gemma) model's name to the OS clipboard."""
        await self._copy_to_clipboard(self.llama_model_name, "model name")

    async def _copy_to_clipboard(self, text: str, what: str) -> None:
        """Copy text to the OS clipboard, telling the user what happened."""
        if not text:
            ui.notify(f"No {what} to copy", type="warning")
            return
        # navigator.clipboard only exists in secure contexts (https or
        # localhost), but this console is usually reached over plain http on
        # the LAN — so fall back to the deprecated execCommand copy when the
        # browser still ships it. The result is a string: which path copied,
        # or 'unavailable' when the browser offers neither (plain HTTP on a
        # current Chrome), so the notification can say what is really wrong.
        js = f"""
        (async () => {{
            const text = {json.dumps(text)};
            if (navigator.clipboard) {{
                try {{
                    await navigator.clipboard.writeText(text);
                    return 'clipboard';
                }} catch (e) {{}}
            }}
            if (document.execCommand) {{
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                let ok = false;
                try {{ ok = document.execCommand('copy'); }} catch (e) {{}}
                document.body.removeChild(ta);
                if (ok) return 'exec';
            }}
            return 'unavailable';
        }})()
        """
        try:
            result = await ui.run_javascript(js, timeout=5)
        except Exception:
            result = None
        if result in ("clipboard", "exec"):
            ui.notify(f"{what.capitalize()} copied to clipboard", type="positive")
        elif result == "unavailable":
            ui.notify("Copying is blocked: this browser only supports the "
                      "clipboard over HTTPS or localhost", type="warning")
        else:
            ui.notify("Copy to clipboard failed", type="negative")

    # -------------------------------------------------------------- sliders ---
    def _update_memory(self) -> None:
        """Recompute the estimated memory: model size + KV cache.

        The KV cache comes from the model's ctx_budget_gb_bf16_128k (decimal GB,
        converted to GiB) scaled linearly to the context selected on the slider,
        then halved/quartered for an 8/4 bit quant. An empty radio means "keep
        the model's own KVQUANT from models.json", so that is what gets applied
        in that case. Everything is reported in GiB, like the model size.
        """
        model = self.models.get(self.model_dropdown.value)
        if model is None or model.ctx_budget_gb_bf16_128k is None:
            self.mem_label.set_text("Memory: —")
            return

        ctx = int(self.ctx_slider.value)
        quant = self.kvquant_radio.value or (model.kvquant or "")
        kv_gib = (model.ctx_budget_gb_bf16_128k * _GB_TO_GIB
                  * ctx / _KV_REF_CTX
                  * _KV_QUANT_SCALE.get(quant, 1.0))
        if model.size_gib is None:
            self.mem_label.set_text(f"Memory: ? GiB (model ? + KV cache {kv_gib:.1f} GiB)")
            return
        total = model.size_gib + kv_gib
        self.mem_label.set_text(
            f"Memory: {total:.1f} GiB  (model {model.size_gib:.1f} + KV cache {kv_gib:.1f} GiB)"
        )

    def _apply_model_spec(self, model_name: str) -> None:
        """Point both sliders at the selected model's bounds and defaults."""
        model = self.models.get(model_name)
        if model is None:
            return

        native_ctx = model.native_ctx
        # min() guards a model whose native context is below the usual floor:
        # a slider with min > max cannot be dragged at all.
        ctx_min = min(_CTX_MIN, native_ctx)
        ctx_value = settings.DEFAULT_CTX
        # element.props is a public observable dict in NiceGUI 3.x, so assigning
        # to it schedules the update by itself. The value still goes through
        # set_value(): pushing it as a 'value=' prop is what breaks dragging.
        self.ctx_slider.props['min'] = ctx_min
        self.ctx_slider.props['max'] = native_ctx
        self.ctx_slider.set_value(ctx_value)
        self.ctx_label.set_text(f"Context: {ctx_value:,}  (max: {native_ctx:,})")
        self._update_memory()

        # NOTE: unchanged semantics — the model's configured temperature doubles
        # as the slider maximum, so it can only be lowered from here.
        max_temp = model.temperature

        topp = model.top_p
        topk = model.top_k
        minp = model.min_p

        self.temp_slider.props['min'] = 0
        self.temp_slider.props['max'] = max_temp
        self.temp_slider.set_value(max_temp)
        self.temp_label.set_text(f"Temperature: {max_temp:.2f}  (max: {max_temp:.2f})")
        self._set_samplers((f"{max_temp:.2f}", f"{topp:.2f}", f"{topk}", f"{minp:.2f}"))

        logger.debug(f"Model MTP={model.mtp}")

        if model.mtp:
            self.mtp_checkbox.enable()
            self.mtp_checkbox.text="Force No-MTP (this is a mtp-enabled model)"
        else:
            self.mtp_checkbox.disable()
            self.mtp_checkbox.text="Force No-MTP"

        # Preselect the REAS level defined in models.json, when it is one of
        # the supported ones; "" means "no override" (keep the file's value).
        self.reas_radio.set_value(
            model.reasoning if model.reasoning in _REAS_OPTIONS else "")

        self._update_rpc_labels( )
        

    def _on_model_change(self, e) -> None:
        if e.value:
           self._apply_model_spec(e.value)

    def _on_ctx_change(self, e) -> None:
        self.ctx_label.set_text(f"Context: {e.value:,}")
        self._update_memory()
        
    # ------------------------------------------------------------- commands ---
    async def _stream(self, argv: list[str]) -> int:
        """Run argv and push its output into the log window line by line."""
        # start_new_session gives the child its own process group so that the
        # whole tree can be signalled at once. Terminating just the direct child
        # left start_model.py's own `ssh -t ... tail -F` orphaned onto init,
        # still holding a tail open on the remote host.
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            async for raw in proc.stdout:
                self.log_window.push(raw.decode(errors="replace").rstrip())
            return await proc.wait()
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    async def start_selected_model(self) -> None:
        if self._start_busy:
            ui.notify("A model start is already in progress", type="warning")
            return
        # A stop can run for a long time (SIGKILL escalation + per-node RPC
        # kills over nested SSH). Letting a START in during that window is what
        # produced 'already running' refusals right after a stop whose status
        # line already read NOT RUNNING.
        if self._stop_busy:
            ui.notify("The server stop is still in progress — wait for it to finish",
                      type="warning")
            return
        model = self.model_dropdown.value
        selected = self.models.get(model)
        if selected is None:
            ui.notify("Please select a model first", type="warning")
            return

        self._start_busy = True
        try:
            # Fresh check at click time: the polled status (and the START
            # button state it drives) can be up to 5 s stale.
            info = await _run_json(["--server-status"])
            if info is None or not info.get("transport", True):
                # The check itself produced no valid answer: the state is
                # unknown, not 'free'. Proceeding could launch a second server
                # over a live one, so refuse — retry once the host answers.
                ui.notify("Could not check the server status (host unreachable?) — "
                          "start aborted", type="negative")
                return
            if info.get("running"):
                current = (info.get("model") or "").strip()
                suffix = f" ({current})" if current else ""
                ui.notify(f"Inference server is already running{suffix} — "
                          "stop it before starting another model", type="negative")
                await self.update_status()
                return

            args = [model]
            ctx_value = int(self.ctx_slider.value)
            args += ["--override-ctx", str(ctx_value)]
            # The radio value IS the --override-kvquant argument; "" means
            # no override (keep the model's own KVQUANT from models.json).
            if self.kvquant_radio.value:
                args += ["--override-kvquant", self.kvquant_radio.value]

            # Same convention: "" means no override (keep the model's REAS
            # from models.json).
            if self.reas_radio.value:
                args += ["--override-reas", self.reas_radio.value]

            # Only override the temperature if the user actually moved the slider.
            temp_value = float(self.temp_slider.value)
            if abs(temp_value - selected.temperature) > 0.001:
                args += ["--override-temp", f"{temp_value:.4f}"]
            if self.mtp_checkbox.value:
                args += ["--force-no-mtp"]
            if self.no_reasoning_checkbox.value:
                args += ["--disable-thinking"]
            #args += ["--debug"]

            ui.notify(f"Starting model {model} (ctx={ctx_value})...")
            # -u keeps the child's output unbuffered so lines arrive as they happen.
            cmd = [_PY, "-u", _START_MODEL, *args]
            self.log_window.push(f"--- Starting model: {shlex.join(cmd)} ---")
            try:
                rc = await self._stream(cmd)
            except OSError as e:
                self.log_window.push(f"Launch Error: {e}")
                ui.notify(f"Error starting model: {e}", type="negative")
            else:
                if rc == 0:
                    ui.notify(f"Model {model} started successfully", type="positive")
                else:
                    ui.notify(f"Error starting model (exit {rc})", type="negative")
            await self.update_status()
        finally:
            self._start_busy = False

    async def stop_server(self) -> None:
        if self._stop_busy:
            ui.notify("A stop is already in progress", type="warning")
            return
        self._stop_busy = True
        self.stop_button.disable()
        self.start_button.disable()
        try:
            ui.notify("Stopping server...")
            out, rc = await _capture([_PY, _START_MODEL, "--kill-server"])
            if rc == 0:
                ui.notify("Server stopped", type="positive")
            else:
                ui.notify(f"Error stopping server: {out.strip()}", type="negative")
            await self.update_status()
        finally:
            self._stop_busy = False
            self.stop_button.enable()

    async def kill_rpc_servers(self) -> None:
        if self._start_busy:
            ui.notify("A model start is already in progress", type="warning")
            return
        if self._stop_busy:
            ui.notify("The server stop is still in progress — wait for it to finish",
                      type="warning")
            return
        model = self.model_dropdown.value
        if model not in self.models:
            ui.notify("Please select a model first", type="warning")
            return
        # Kill exactly the RPC servers listed for the selected model:
        # --override-rpc replaces model.rpcservers before --kill-rpc-server
        # acts on it.
        selected = [rpc.name for rpc in model.rpcservers or [] if rpc.name in RPC_SERVERS]
        if not selected:
            ui.notify("This model has no RPC servers to kill", type="warning")
            return
        ui.notify(f"Killing RPC server(s): {', '.join(selected)}...")
        out, rc = await _capture([_PY, _START_MODEL, model,
                                  "--override-rpc", ",".join(selected),
                                  "--kill-rpc-server"])
        if rc == 0:
            ui.notify("RPC servers killed", type="positive")
        else:
            ui.notify(f"Error killing RPC servers: {out.strip()}", type="negative")

    # ----------------------------------------------------------- strata ---
    async def _strata_exec(self, shell_cmd: str) -> tuple[int, str]:
        """Run shell_cmd on the strata node over SSH; return (returncode, stdout).

        rc 255 is ssh's own failure code, so it unambiguously means the node was not
        reached at all (vs. a command that ran and exited non-zero)."""
        ip, user = STRATA_NODE.get("ip"), STRATA_NODE.get("user")
        if not ip or not user:
            return 255, ""
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"{user}@{ip}", shell_cmd]
        out, rc = await _capture(argv)
        return rc, out

    async def run_strata_qwen(self) -> None:
        if self._strata_busy:
            ui.notify("A Strata action is already in progress", type="warning")
            return
        self._strata_busy = True
        try:
            # Refuse a second launch over a live one (same guard as model START): a
            # stale start.sh would only fight the running server for the port.
            rc, out = await self._strata_exec(_STRATA_STATUS_CMD)
            if rc == 0 and out.strip().startswith("ALIVE"):
                ui.notify("Qwen is already running on strata", type="warning")
                return
            ui.notify("Starting Qwen Flash Next on strata...")
            rc, out = await self._strata_exec(_STRATA_START_CMD)
            if rc == 0:
                pid = out.strip().splitlines()[-1] if out.strip() else ""
                ui.notify(f"Qwen started on strata ({pid})", type="positive")
            else:
                ui.notify(f"Error starting Qwen on strata (rc={rc}): {out.strip()}", type="negative")
            await self.update_strata_status()
        finally:
            self._strata_busy = False

    async def kill_strata_qwen(self) -> None:
        if self._strata_busy:
            ui.notify("A Strata action is already in progress", type="warning")
            return
        self._strata_busy = True
        try:
            ui.notify("Stopping Qwen on strata...")
            rc, out = await self._strata_exec(_STRATA_KILL_CMD)
            if rc == 0:
                ui.notify("Qwen stopped on strata", type="positive")
            elif rc == 3:
                ui.notify("No Qwen server process found on strata (already stopped?)", type="warning")
            else:
                ui.notify(f"Error stopping Qwen on strata (rc={rc}): {out.strip()}", type="negative")
            await self.update_strata_status()
        finally:
            self._strata_busy = False

    async def update_strata_log(self) -> None:
        """Refill the Strata log window with the tail of the remote strata.out.

        The file lives next to the start.sh launched by "Run Qwen Flash Next"
        (same directory, $HOME/Strata), so it is fetched over the very same SSH
        channel as the status/kill commands."""
        if self._strata_log_busy:
            ui.notify("A log fetch is already in progress", type="warning")
            return
        if not STRATA_NODE:
            ui.notify("Strata node not configured (nodes.json)", type="warning")
            return
        self._strata_log_busy = True
        self.strata_log_update_button.disable()
        try:
            rc, out = await self._strata_exec(_STRATA_LOG_CMD)
            self.strata_log_window.clear()
            if rc == 255:
                self.strata_log_window.push("--- strata node unreachable ---")
                ui.notify("Could not reach the strata node", type="negative")
            elif rc != 0:
                self.strata_log_window.push(f"--- log fetch failed (rc={rc}) ---")
                self.strata_log_window.push(out.rstrip())
                ui.notify(f"Error fetching strata.out (rc={rc})", type="negative")
            else:
                for line in out.splitlines():
                    self.strata_log_window.push(line)
        finally:
            self.strata_log_update_button.enable()
            self._strata_log_busy = False

    def clear_strata_log(self) -> None:
        """Empty the Strata log window.

        View only: the remote strata.out is left alone, because truncating the
        console log of a live service would throw away the only record of what it
        did (and what it crashed with). "Update log" refills the window at any
        time. Same contract as clear_mlx_log."""
        self.strata_log_window.clear()
        self.strata_log_window.push("--- log cleared; use \"Update log\" to refetch ---")
        ui.notify("Strata log window cleared")

    async def update_strata_status(self) -> None:
        # Polled every 5 s: skip a tick while the previous probe (SSH + HTTP) is still
        # in flight so a slow/unreachable node does not stack up concurrent checks.
        if self._strata_status_busy:
            return
        self._strata_status_busy = True
        try:
            if not STRATA_NODE:
                self.strata_status_label.set_text("Strata: node not configured (nodes.json)")
                self.strata_status_label.style("color: orange;")
                return
            # Node the SSH commands (and start.sh) run on, from nodes.json.
            # Shown even when the node is down: it is the configured target.
            self.strata_host_label.set_text(
                f"Strata node: {STRATA_NODE.get('user', '?')}@{STRATA_NODE.get('ip', '?')}")
            rc, out = await self._strata_exec(_STRATA_STATUS_CMD)
            if rc == 255:
                self.strata_status_label.set_text("Strata: node unreachable")
                self.strata_status_label.style("color: orange;")
                return
            proc_line = out.strip().splitlines()[-1] if out.strip() else "NO_PID"
            proc_alive = proc_line.startswith("ALIVE")
            # Ready only once the API answers, not merely when the process is alive.
            api_ok, model_id = await asyncio.to_thread(_strata_api_check, STRATA_NODE["ip"])
            if api_ok:
                self.strata_status_label.set_text("Strata: RUNNING")
                self.strata_status_label.style("color: #00ff88;")
                self.strata_kill_button.enable()
            elif proc_alive:
                self.strata_status_label.set_text("Strata: STARTING (API not answering yet)")
                self.strata_status_label.style("color: orange;")
                self.strata_kill_button.enable()
            else:
                self.strata_status_label.set_text("Strata: STOPPED")
                self.strata_status_label.style("color: red;")
                self.strata_kill_button.disable()
            # Model name on its own row (copy icon right after it), known only
            # once the API answers; the icon follows the name's visibility.
            self.strata_model_name = model_id if api_ok else ""
            self.strata_model_label.set_text(self.strata_model_name)
            if self.strata_model_name:
                self.strata_model_copy.classes(remove='q-hidden')
            else:
                self.strata_model_copy.classes(add='q-hidden')
        except Exception:
            self.strata_status_label.set_text("Strata: UNKNOWN (status check failed)")
            self.strata_status_label.style("color: orange;")
        finally:
            self._strata_status_busy = False

    # -------------------------------------------------------------- mlx ---
    async def _mlx_exec(self, shell_cmd: str, ip: str = "", user: str = "") -> tuple[int, str]:
        """Run shell_cmd on an mlx node over SSH; return (returncode, stdout).

        Defaults to rank 0 (nodes.json "mlx"); pass ip/user to target rank 1. Same
        contract as _strata_exec: rc 255 means ssh itself failed, i.e. the node was
        not reached.

        Unlike strata, the remote stderr is folded into stdout ("2>&1"): the mlx
        node is a Mac whose BSD tools phrase their errors on stderr, and _capture
        only returns stdout — without the merge a "python3 not found" or a shell
        syntax error would show up in the UI as a bare, undiagnosable rc=N."""
        ip, user = ip or MLX_NODE.get("ip", ""), user or MLX_NODE.get("user", "")
        if not ip or not user:
            return 255, ""
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"{user}@{ip}",
                f"{{ {shell_cmd}; }} 2>&1"]
        out, rc = await _capture(argv)
        return rc, out

    @staticmethod
    def _mlx_rank_state(rc: int, out: str) -> str:
        """Classify one _MLX_STATUS_CMD answer as running / stopped / unreachable / error."""
        if rc == 255:
            return "unreachable"
        if rc != 0:
            return "error"
        return "running" if out.strip().startswith("ALIVE") else "stopped"

    async def run_mlx_qwen(self) -> None:
        if self._mlx_busy:
            ui.notify("An MLX action is already in progress", type="warning")
            return
        self._mlx_busy = True
        try:
            # Refuse a second launch over a live one (same guard as Strata): a
            # second mlx_jaccl-start.py would only fight the running one for the
            # GPUs and the RPC ports listed in hosts.json. rank 1 counts too, even
            # with rank 0 gone: orphaned slave workers hold those ports just the same.
            (rc, out), (rc1, out1) = await asyncio.gather(
                self._mlx_exec(_MLX_STATUS_CMD),
                self._mlx_exec(_MLX_STATUS_CMD, MLX_SLAVE_IP, MLX_SLAVE_USER)
                if MLX_SLAVE_IP else asyncio.sleep(0, result=(255, "")))
            if rc == 0 and out.strip().startswith("ALIVE"):
                ui.notify("Qwen is already running on mlx", type="warning")
                return
            if rc1 == 0 and out1.strip().startswith("ALIVE"):
                ui.notify(f"Qwen processes still alive on rank 1 ({MLX_SLAVE_IP}); "
                          "use Kill MLX first", type="warning")
                return
            ui.notify("Starting MLX-Jaccl Qwen on the mlx node...")
            rc, out = await self._mlx_exec(_mlx_start_cmd(bool(self.mlx_8bit_checkbox.value)))
            if rc == 0:
                pid = out.strip().splitlines()[-1] if out.strip() else ""
                ui.notify(f"Qwen started on mlx ({pid})", type="positive")
            else:
                ui.notify(f"Error starting Qwen on mlx (rc={rc}): {out.strip()}", type="negative")
            await self.update_mlx_status()
        finally:
            self._mlx_busy = False

    async def kill_mlx_qwen(self) -> None:
        """Kill the run on *both* nodes of the mlx.launch job.

        rank 0 (the node "Run MLX-Jaccl Qwen" starts the script on) gets
        _MLX_KILL_CMD_RANK0: the start script, mlx.launch and mlx_leak_shim.py by
        argv, plus the whole session recorded in the pidfile. rank 1 (nodes.json
        "ip_slave") only hosts workers started remotely over there, so it gets
        _MLX_KILL_CMD, which signals the named PIDs and never a process group.
        Both probes run in parallel: a dead slave must not delay rank 0 (or the
        other way round) by a full ssh ConnectTimeout."""
        if self._mlx_busy:
            ui.notify("An MLX action is already in progress", type="warning")
            return
        self._mlx_busy = True
        try:
            ui.notify("Stopping Qwen on the mlx nodes...")
            jobs = [("rank 0", self._mlx_exec(_MLX_KILL_CMD_RANK0))]
            if MLX_SLAVE_IP:
                jobs.append((f"rank 1 ({MLX_SLAVE_IP})",
                             self._mlx_exec(_MLX_KILL_CMD, MLX_SLAVE_IP, MLX_SLAVE_USER)))
            results = await asyncio.gather(*[task for _, task in jobs])
            for (label, _), (rc, out) in zip(jobs, results):
                if rc == 0:
                    ui.notify(f"Qwen stopped on {label} ({out.strip()})", type="positive")
                elif rc == 3:
                    ui.notify(f"No Qwen process on {label} (already stopped?)", type="warning")
                elif rc == 255:
                    ui.notify(f"{label} node unreachable", type="negative")
                else:
                    ui.notify(f"Error stopping Qwen on {label} (rc={rc}): {out.strip()}",
                              type="negative")
            await self.update_mlx_status()
        finally:
            self._mlx_busy = False

    async def update_mlx_log(self) -> None:
        """Refill the MLX log window with the tail of the remote mlx-jaccl.out.

        The file is the stdout/stderr of the mlx_jaccl-start.py launched by "Run
        MLX-Jaccl Qwen" (redirected by _mlx_start_cmd), fetched over the very same
        SSH channel as the status/kill commands."""
        if self._mlx_log_busy:
            ui.notify("A log fetch is already in progress", type="warning")
            return
        if not MLX_NODE:
            ui.notify("MLX node not configured (nodes.json)", type="warning")
            return
        self._mlx_log_busy = True
        self.mlx_log_update_button.disable()
        try:
            rc, out = await self._mlx_exec(_MLX_LOG_CMD)
            self.mlx_log_window.clear()
            if rc == 255:
                self.mlx_log_window.push("--- mlx node unreachable ---")
                ui.notify("Could not reach the mlx node", type="negative")
            elif rc != 0:
                self.mlx_log_window.push(f"--- log fetch failed (rc={rc}) ---")
                self.mlx_log_window.push(out.rstrip())
                ui.notify(f"Error fetching mlx-jaccl.out (rc={rc})", type="negative")
            else:
                for line in out.splitlines():
                    self.mlx_log_window.push(line)
        finally:
            self.mlx_log_update_button.enable()
            self._mlx_log_busy = False

    def clear_mlx_log(self) -> None:
        """Empty the MLX log window.

        View only: the remote mlx-jaccl.out is left alone, because truncating the
        console log of a live run would throw away the only record of what it did
        (and what it crashed with). "Update log" refills the window at any time."""
        self.mlx_log_window.clear()
        self.mlx_log_window.push("--- log cleared; use \"Update log\" to refetch ---")
        ui.notify("MLX log window cleared")

    async def update_mlx_status(self) -> None:
        # Polled every 5 s: skip a tick while the previous SSH probe is still in
        # flight so a slow/unreachable node does not stack up concurrent checks.
        if self._mlx_status_busy:
            return
        self._mlx_status_busy = True
        try:
            if not MLX_NODE:
                self.mlx_status_label.set_text("MLX: node not configured (nodes.json)")
                self.mlx_status_label.style("color: orange;")
                return
            # Node the SSH commands (and mlx_jaccl-start.py) run on, from nodes.json.
            # Shown even when the node is down: it is the configured target.
            self.mlx_host_label.set_text(
                f"MLX rank 0: {MLX_NODE.get('user', '?')}@{MLX_NODE.get('ip', '?')}"
                + (f"   |   rank 1: {MLX_SLAVE_USER or '?'}@{MLX_SLAVE_IP}" if MLX_SLAVE_IP else ""))
            # Both ranks of the mlx.launch job are probed in parallel: the run is only
            # whole when rank 0 and rank 1 both hold their workers, and a dead slave
            # must not stall the tick behind a full ssh ConnectTimeout.
            (rc0, out0), (rc1, out1) = await asyncio.gather(
                self._mlx_exec(_MLX_STATUS_CMD),
                self._mlx_exec(_MLX_STATUS_CMD, MLX_SLAVE_IP, MLX_SLAVE_USER)
                if MLX_SLAVE_IP else asyncio.sleep(0, result=(255, "")))
            state0, state1 = self._mlx_rank_state(rc0, out0), self._mlx_rank_state(rc1, out1)
            self.mlx_slave_label.set_text(
                "rank 1: not configured (nodes.json ip_slave)" if not MLX_SLAVE_IP
                else f"rank 1 ({MLX_SLAVE_USER}@{MLX_SLAVE_IP}): {state1}")
            if state0 == "unreachable":
                self.mlx_status_label.set_text("MLX: node unreachable")
                self.mlx_status_label.style("color: orange;")
                return
            # Status stays process-based: the processes *are* the service, and
            # "running" means either rank still holds one of them — a half-dead
            # run is not "stopped", it is something Kill Qwen has to clean. (The
            # OpenAI-compatible /v1/models API on _MLX_API_PORT is only used to
            # read the model's own id, quantization included.)
            proc_alive = state0 == "running" or state1 == "running"
            if proc_alive:
                where = "+".join(n for n, s in (("rank 0", state0), ("rank 1", state1))
                                 if s == "running")
                self.mlx_status_label.set_text(f"MLX: RUNNING ({where})")
                self.mlx_status_label.style("color: #00ff88;")
                self.mlx_kill_button.enable()
            else:
                self.mlx_status_label.set_text("MLX: STOPPED")
                self.mlx_status_label.style("color: red;")
                self.mlx_kill_button.disable()
            # Model name on its own row (copy icon right after it): the API's own
            # id from /v1/models (which carries the quantization, e.g. "-8bit")
            # once it answers, else the fixed name of the start command while the
            # service is up; the icon follows the name's visibility.
            api_ok, model_id = await asyncio.to_thread(
                _strata_api_check, MLX_NODE["ip"], _MLX_API_PORT)
            if model_id:
                self.mlx_model_name = model_id
            else:
                self.mlx_model_name = _MLX_MODEL_NAME if proc_alive else ""
            self.mlx_model_label.set_text(self.mlx_model_name)
            if self.mlx_model_name:
                self.mlx_model_copy.classes(remove='q-hidden')
            else:
                self.mlx_model_copy.classes(add='q-hidden')
        except Exception:
            self.mlx_status_label.set_text("MLX: UNKNOWN (status check failed)")
            self.mlx_status_label.style("color: orange;")
        finally:
            self._mlx_status_busy = False

    # ----------------------------------------------------------- splash ---
    async def _splash_exec(self, shell_cmd: str) -> tuple[int, str]:
        """Run shell_cmd on the splash node over SSH; return (returncode, stdout).

        Same contract as _mlx_exec (the splash node is a Mac too): rc 255 means
        ssh itself failed, and the remote stderr is folded into stdout so BSD
        tool errors show up in the UI instead of vanishing behind a bare rc=N."""
        ip, user = SPLASH_NODE.get("ip", ""), SPLASH_NODE.get("user", "")
        if not ip or not user:
            return 255, ""
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"{user}@{ip}",
                f"{{ {shell_cmd}; }} 2>&1"]
        out, rc = await _capture(argv)
        return rc, out

    async def run_splash_qwen(self) -> None:
        if self._splash_busy:
            ui.notify("A Splash action is already in progress", type="warning")
            return
        self._splash_busy = True
        try:
            # Refuse a second launch over a live one (same guard as Strata/MLX):
            # a second splash serve would only fight the running one for its
            # API port (and the GPUs).
            rc, out = await self._splash_exec(_SPLASH_STATUS_CMD)
            if rc == 0 and out.strip().startswith("ALIVE"):
                ui.notify("Qwen is already running on splash", type="warning")
                return
            ui.notify("Starting Splash Qwen on the splash node...")
            rc, out = await self._splash_exec(_SPLASH_START_CMD)
            if rc == 0:
                pid = out.strip().splitlines()[-1] if out.strip() else ""
                ui.notify(f"Qwen started on splash ({pid})", type="positive")
            else:
                ui.notify(f"Error starting Qwen on splash (rc={rc}): {out.strip()}", type="negative")
            await self.update_splash_status()
        finally:
            self._splash_busy = False

    async def kill_splash_qwen(self) -> None:
        if self._splash_busy:
            ui.notify("A Splash action is already in progress", type="warning")
            return
        self._splash_busy = True
        try:
            ui.notify("Stopping Qwen on the splash node...")
            rc, out = await self._splash_exec(_SPLASH_KILL_CMD)
            if rc == 0:
                ui.notify(f"Qwen stopped on splash ({out.strip()})", type="positive")
            elif rc == 3:
                ui.notify("No Qwen process found on splash (already stopped?)", type="warning")
            elif rc == 255:
                ui.notify("Splash node unreachable", type="negative")
            else:
                ui.notify(f"Error stopping Qwen on splash (rc={rc}): {out.strip()}", type="negative")
            await self.update_splash_status()
        finally:
            self._splash_busy = False

    async def update_splash_log(self) -> None:
        """Refill the Splash log window with the tail of the remote splash.out.

        The file is the stdout/stderr of the splash serve launched by "Run
        Splash Qwen" (redirected by _SPLASH_START_CMD), fetched over the very
        same SSH channel as the status/kill commands."""
        if self._splash_log_busy:
            ui.notify("A log fetch is already in progress", type="warning")
            return
        if not SPLASH_NODE:
            ui.notify("Splash node not configured (nodes.json)", type="warning")
            return
        self._splash_log_busy = True
        self.splash_log_update_button.disable()
        try:
            rc, out = await self._splash_exec(_SPLASH_LOG_CMD)
            self.splash_log_window.clear()
            if rc == 255:
                self.splash_log_window.push("--- splash node unreachable ---")
                ui.notify("Could not reach the splash node", type="negative")
            elif rc != 0:
                self.splash_log_window.push(f"--- log fetch failed (rc={rc}) ---")
                self.splash_log_window.push(out.rstrip())
                ui.notify(f"Error fetching splash.out (rc={rc})", type="negative")
            else:
                for line in out.splitlines():
                    self.splash_log_window.push(line)
        finally:
            self.splash_log_update_button.enable()
            self._splash_log_busy = False

    def clear_splash_log(self) -> None:
        """Empty the Splash log window.

        View only: the remote splash.out is left alone, because truncating the
        console log of a live service would throw away the only record of what
        it did (and what it crashed with). "Update log" refills the window at
        any time. Same contract as clear_strata_log/clear_mlx_log."""
        self.splash_log_window.clear()
        self.splash_log_window.push("--- log cleared; use \"Update log\" to refetch ---")
        ui.notify("Splash log window cleared")

    async def update_splash_status(self) -> None:
        # Polled every 5 s: skip a tick while the previous probe (SSH + HTTP) is
        # still in flight so a slow/unreachable node does not stack up checks.
        if self._splash_status_busy:
            return
        self._splash_status_busy = True
        try:
            if not SPLASH_NODE:
                self.splash_status_label.set_text("Splash: node not configured (nodes.json)")
                self.splash_status_label.style("color: orange;")
                return
            # Node the SSH commands (and splash serve) run on, from nodes.json.
            # Shown even when the node is down: it is the configured target.
            self.splash_host_label.set_text(
                f"Splash node: {SPLASH_NODE.get('user', '?')}@{SPLASH_NODE.get('ip', '?')}")
            rc, out = await self._splash_exec(_SPLASH_STATUS_CMD)
            if rc == 255:
                self.splash_status_label.set_text("Splash: node unreachable")
                self.splash_status_label.style("color: orange;")
                return
            proc_line = out.strip().splitlines()[-1] if out.strip() else "NO_PID"
            proc_alive = proc_line.startswith("ALIVE")
            # Ready only once the API answers, not merely when the process is
            # alive (same two-phase reading as Strata: a fresh splash serve is a
            # live process long before it binds its API port).
            api_ok, model_id = await asyncio.to_thread(
                _strata_api_check, SPLASH_NODE["ip"], _SPLASH_API_PORT)
            if api_ok:
                self.splash_status_label.set_text("Splash: RUNNING")
                self.splash_status_label.style("color: #00ff88;")
                self.splash_kill_button.enable()
            elif proc_alive:
                self.splash_status_label.set_text("Splash: STARTING (API not answering yet)")
                self.splash_status_label.style("color: orange;")
                self.splash_kill_button.enable()
            else:
                self.splash_status_label.set_text("Splash: STOPPED")
                self.splash_status_label.style("color: red;")
                self.splash_kill_button.disable()
            # Model name on its own row (copy icon right after it): the API's own
            # id once it answers, else the fixed name of the serve command while
            # the process is up; the icon follows the name's visibility.
            if model_id:
                self.splash_model_name = model_id
            else:
                self.splash_model_name = _SPLASH_MODEL_NAME if proc_alive else ""
            self.splash_model_label.set_text(self.splash_model_name)
            if self.splash_model_name:
                self.splash_model_copy.classes(remove='q-hidden')
            else:
                self.splash_model_copy.classes(add='q-hidden')
        except Exception:
            self.splash_status_label.set_text("Splash: UNKNOWN (status check failed)")
            self.splash_status_label.style("color: orange;")
        finally:
            self._splash_status_busy = False

    # -------------------------------------------------------- freetoken ---
    async def _freetoken_exec(self, shell_cmd: str) -> tuple[int, str]:
        """Run shell_cmd on the freetoken node over SSH; return (returncode, stdout).

        Same contract as _strata_exec (a Linux node too, so the remote stderr is
        left alone and setsid(1) is available): rc 255 means ssh itself failed,
        i.e. the node was not reached at all."""
        ip, user = FREETOKEN_NODE.get("ip", ""), FREETOKEN_NODE.get("user", "")
        if not ip or not user:
            return 255, ""
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"{user}@{ip}", shell_cmd]
        out, rc = await _capture(argv)
        return rc, out

    async def run_freetoken(self) -> None:
        """Start the FreeToken service ("ft serve" on the freetoken node).

        Refuses a second launch over a live one (same guard as Strata/MLX/Splash):
        a second "ft serve" would only fight the running server for its API port
        and for the GPU it is pinned to."""
        if self._freetoken_busy:
            ui.notify("A FreeToken action is already in progress", type="warning")
            return
        self._freetoken_busy = True
        try:
            rc, out = await self._freetoken_exec(_FT_STATUS_CMD)
            if rc == 0 and out.strip().startswith("ALIVE"):
                ui.notify("FreeToken is already running on the freetoken node",
                          type="warning")
                return
            ui.notify(f"Starting FreeToken {_FT_MODEL_LABEL} on the freetoken node...")
            rc, out = await self._freetoken_exec(_FT_START_CMD)
            if rc == 0:
                pid = out.strip().splitlines()[-1] if out.strip() else ""
                ui.notify(f"FreeToken {_FT_MODEL_LABEL} started ({pid})", type="positive")
            else:
                ui.notify(f"Error starting FreeToken (rc={rc}): {out.strip()}",
                          type="negative")
            await self.update_freetoken_status()
        finally:
            self._freetoken_busy = False

    async def kill_freetoken(self) -> None:
        """Kill the FreeToken service: the ".ft-orig/ft" process and its session."""
        if self._freetoken_busy:
            ui.notify("A FreeToken action is already in progress", type="warning")
            return
        self._freetoken_busy = True
        try:
            ui.notify("Stopping FreeToken on the freetoken node...")
            rc, out = await self._freetoken_exec(_FT_KILL_CMD)
            if rc == 0:
                ui.notify(f"FreeToken stopped ({out.strip()})", type="positive")
            elif rc == 3:
                ui.notify("No FreeToken process found on the node (already stopped?)",
                          type="warning")
            elif rc == 255:
                ui.notify("FreeToken node unreachable", type="negative")
            else:
                ui.notify(f"Error stopping FreeToken (rc={rc}): {out.strip()}", type="negative")
            await self.update_freetoken_status()
        finally:
            self._freetoken_busy = False

    async def update_freetoken_log(self) -> None:
        """Refill the FreeToken log window with the tail of the remote freetoken.out.

        The file is the stdout/stderr of the "ft serve" started by "Run GPT-OSS"
        (redirected by _FT_START_CMD), fetched over the very same SSH channel as
        the status/kill commands."""
        if self._freetoken_log_busy:
            ui.notify("A log fetch is already in progress", type="warning")
            return
        if not FREETOKEN_NODE:
            ui.notify("FreeToken node not configured (nodes.json)", type="warning")
            return
        self._freetoken_log_busy = True
        self.freetoken_log_update_button.disable()
        try:
            rc, out = await self._freetoken_exec(_FT_LOG_CMD)
            self.freetoken_log_window.clear()
            if rc == 255:
                self.freetoken_log_window.push("--- freetoken node unreachable ---")
                ui.notify("Could not reach the freetoken node", type="negative")
            elif rc != 0:
                self.freetoken_log_window.push(f"--- log fetch failed (rc={rc}) ---")
                self.freetoken_log_window.push(out.rstrip())
                ui.notify(f"Error fetching freetoken.out (rc={rc})", type="negative")
            else:
                for line in out.splitlines():
                    self.freetoken_log_window.push(line)
        finally:
            self.freetoken_log_update_button.enable()
            self._freetoken_log_busy = False

    def clear_freetoken_log(self) -> None:
        """Empty the FreeToken log window.

        View only: the remote freetoken.out is left alone, because truncating the
        console log of a live service would throw away the only record of what it
        did (and what it crashed with). "Update log" refills the window at any
        time. Same contract as clear_strata_log/clear_mlx_log/clear_splash_log."""
        self.freetoken_log_window.clear()
        self.freetoken_log_window.push("--- log cleared; use \"Update log\" to refetch ---")
        ui.notify("FreeToken log window cleared")

    async def update_freetoken_status(self) -> None:
        # Polled every 5 s: skip a tick while the previous probe (SSH + HTTP) is
        # still in flight so a slow/unreachable node does not stack up checks.
        if self._freetoken_status_busy:
            return
        self._freetoken_status_busy = True
        try:
            if not FREETOKEN_NODE:
                self.freetoken_status_label.set_text("FreeToken: node not configured (nodes.json)")
                self.freetoken_status_label.style("color: orange;")
                return
            # Node the SSH commands (and ft serve) run on, from nodes.json. Shown
            # even when the node is down: it is the configured target.
            self.freetoken_host_label.set_text(
                f"FreeToken node: {FREETOKEN_NODE.get('user', '?')}@{FREETOKEN_NODE.get('ip', '?')}")
            rc, out = await self._freetoken_exec(_FT_STATUS_CMD)
            if rc == 255:
                self.freetoken_status_label.set_text("FreeToken: node unreachable")
                self.freetoken_status_label.style("color: orange;")
                return
            proc_line = out.strip().splitlines()[-1] if out.strip() else "NO_PID"
            proc_alive = proc_line.startswith("ALIVE")
            # Ready only once the API answers, not merely when the process is
            # alive (same two-phase reading as Strata/Splash: ft is a live process
            # long before it binds its API port).
            api_ok, model_id = await asyncio.to_thread(
                _strata_api_check, FREETOKEN_NODE["ip"], _FT_API_PORT)
            if api_ok:
                self.freetoken_status_label.set_text("FreeToken: RUNNING")
                self.freetoken_status_label.style("color: #00ff88;")
                self.freetoken_kill_button.enable()
            elif proc_alive:
                self.freetoken_status_label.set_text("FreeToken: STARTING (API not answering yet)")
                self.freetoken_status_label.style("color: orange;")
                self.freetoken_kill_button.enable()
            else:
                self.freetoken_status_label.set_text("FreeToken: STOPPED")
                self.freetoken_status_label.style("color: red;")
                self.freetoken_kill_button.disable()
            # Model name on its own row (copy icon right after it): the API's own
            # id once it answers, else the model dir of the fixed serve command
            # while the process is up; the icon follows the name's visibility.
            if model_id:
                self.freetoken_model_name = model_id
            else:
                self.freetoken_model_name = _FT_MODEL_NAME if proc_alive else ""
            self.freetoken_model_label.set_text(self.freetoken_model_name)
            if self.freetoken_model_name:
                self.freetoken_model_copy.classes(remove='q-hidden')
            else:
                self.freetoken_model_copy.classes(add='q-hidden')
        except Exception:
            self.freetoken_status_label.set_text("FreeToken: UNKNOWN (status check failed)")
            self.freetoken_status_label.style("color: orange;")
        finally:
            self._freetoken_status_busy = False

    # ------------------------------------------------------------- llama ---
    async def _llama_exec(self, shell_cmd: str, ip: str = "", user: str = "") -> tuple[int, str]:
        """Run shell_cmd on a Llama-cluster node over SSH; return (returncode, stdout).

        Same contract as _strata_exec: rc 255 means ssh itself failed, i.e. the
        node was not reached at all. Defaults to the llama node from nodes.json;
        the ggml-rpc-server nodes are reached through the very same helper."""
        ip = ip or LLAMA_NODE.get("ip", "")
        user = user or LLAMA_NODE.get("user", "")
        if not ip or not user:
            return 255, ""
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"{user}@{ip}", shell_cmd]
        out, rc = await _capture(argv)
        return rc, out

    def _llama_rpc_nodes(self) -> list[dict]:
        """The ggml-rpc-server nodes from nodes.json's llama.rpc_nodes entries.

        Each entry gets its launch command from _LLAMA_RPC_LAUNCH (keyed by IP;
        empty when nodes.json names a node the UI has no command for) and its
        SSH user, defaulting to the llama node's one."""
        user = LLAMA_NODE.get("user", "")
        nodes = []
        for n in LLAMA_NODE.get("rpc_nodes") or []:
            ip = n.get("ip", "")
            if not ip:
                continue
            nodes.append({"ip": ip, "user": n.get("user") or user,
                          "cmd": _LLAMA_RPC_LAUNCH.get(ip, "")})
        return nodes

    @staticmethod
    def _llama_proc_state(rc: int, out: str) -> str:
        """Classify one pgrep status answer as running / stopped / unreachable / error."""
        if rc == 255:
            return "unreachable"
        if rc != 0:
            return "error"
        return "running" if out.strip().startswith("ALIVE") else "stopped"

    async def _ask_rpc_choice(self, hosts: list[str]) -> str:
        """Modal dialog asking what to do with already-running ggml-rpc-servers.

        Blocks until the user answers: "keep" (tenerli), "restart" (killarli e
        rilanciarli) or "cancel" (abort the whole Run Gemma-4-31B)."""
        choice = "cancel"
        done = asyncio.Event()
        with ui.dialog() as dialog, ui.card().classes('p-4 gap-3'):
            ui.label("ggml-rpc-server is already running on: " + ", ".join(hosts)).classes('text-h6')
            ui.label("Keep the running RPC servers, or kill them and relaunch them?")
            with ui.row().classes('gap-2'):
                def pick(value: str) -> None:
                    nonlocal choice
                    choice = value
                    dialog.close()
                    done.set()
                ui.button("Keep them", on_click=lambda: pick("keep")).props('color=green')
                ui.button("Kill and relaunch", on_click=lambda: pick("restart")).props('color=red')
                ui.button("Cancel", on_click=lambda: pick("cancel")).props('outline small')
        dialog.open()
        await done.wait()
        return choice

    async def run_llama_gemma(self) -> None:
        """Start the Llama cluster: the two ggml-rpc-servers, then llama-server.

        A live llama-server refuses the whole run (same guard as Strata/FreeToken:
        a second one would only fight the running one for its API port). The
        ggml-rpc-servers get an extra courtesy: an already-live one is never
        touched silently — the user is asked whether to keep it (and skip its
        start) or kill it and relaunch it."""
        if self._llama_busy:
            ui.notify("A Llama action is already in progress", type="warning")
            return
        if not LLAMA_NODE or not LLAMA_NODE.get("ip"):
            ui.notify("Llama node not configured (nodes.json)", type="warning")
            return
        rpc_nodes = self._llama_rpc_nodes()
        if not rpc_nodes:
            ui.notify("No llama rpc_nodes configured (nodes.json)", type="warning")
            return
        self._llama_busy = True
        try:
            rc, out = await self._llama_exec(_LLAMA_STATUS_CMD)
            if rc == 0 and out.strip().startswith("ALIVE"):
                ui.notify(f"llama-server is already running on {LLAMA_NODE['ip']}; "
                          "use Kill Llama first", type="warning")
                return
            # Are any ggml-rpc-servers already up? Probe them in parallel, then
            # let the user decide about the live ones before touching anything.
            checks = await asyncio.gather(*[
                self._llama_exec(_LLAMA_RPC_STATUS_CMD, n["ip"], n["user"])
                for n in rpc_nodes])
            alive = [n for n, (rc, out) in zip(rpc_nodes, checks)
                     if rc == 0 and out.strip().startswith("ALIVE")]
            keep: set[str] = set()
            if alive:
                choice = await self._ask_rpc_choice([n["ip"] for n in alive])
                if choice == "cancel":
                    ui.notify("Run Gemma-4-31B cancelled")
                    return
                if choice == "keep":
                    keep = {n["ip"] for n in alive}
                    ui.notify("Keeping the running ggml-rpc-servers: "
                              + ", ".join(sorted(keep)))
                else:
                    ui.notify("Killing the running ggml-rpc-servers...")
                    for n in alive:
                        rc, out = await self._llama_exec(_LLAMA_RPC_KILL_CMD, n["ip"], n["user"])
                        if rc not in (0, 3):
                            ui.notify(f"Error killing ggml-rpc-server on {n['ip']} "
                                      f"(rc={rc}): {out.strip()}", type="negative")
                            return
            # Start every ggml-rpc-server that is not being kept as it is.
            for n in rpc_nodes:
                if n["ip"] in keep:
                    continue
                if not n["cmd"]:
                    ui.notify(f"No ggml-rpc-server command defined for {n['ip']} "
                              "(add it to _LLAMA_RPC_LAUNCH)", type="negative")
                    return
                ui.notify(f"Starting ggml-rpc-server on {n['ip']}...")
                start = _detached_start_cmd(_LLAMA_RPC_PIDFILE, "/dev/null", n["cmd"])
                rc, out = await self._llama_exec(start, n["ip"], n["user"])
                if rc != 0:
                    ui.notify(f"Error starting ggml-rpc-server on {n['ip']} "
                              f"(rc={rc}): {out.strip()}", type="negative")
                    return
            # And finally the model itself, on the llama node.
            ui.notify(f"Starting llama-server {_LLAMA_MODEL_NAME} on "
                      f"{LLAMA_NODE['ip']}...")
            rc, out = await self._llama_exec(_LLAMA_SERVER_START_CMD)
            if rc == 0:
                pid = out.strip().splitlines()[-1] if out.strip() else ""
                ui.notify(f"llama-server started on {LLAMA_NODE['ip']} ({pid})",
                          type="positive")
            else:
                ui.notify(f"Error starting llama-server on {LLAMA_NODE['ip']} "
                          f"(rc={rc}): {out.strip()}", type="negative")
            await self.update_llama_status()
        finally:
            self._llama_busy = False

    async def kill_llama(self) -> None:
        """Kill the whole Llama cluster: llama-server *and* both ggml-rpc-servers.

        Every node is probed and killed over its own SSH channel, in parallel; a
        dead rpc node must not hold the others behind a full ConnectTimeout. The
        kill commands already re-probe before reporting, and the whole pass is
        repeated once more on top of that: this function only returns once every
        target is verifiably gone (or after two failed attempts, loudly)."""
        if self._llama_busy:
            ui.notify("A Llama action is already in progress", type="warning")
            return
        if not LLAMA_NODE or not LLAMA_NODE.get("ip"):
            ui.notify("Llama node not configured (nodes.json)", type="warning")
            return
        self._llama_busy = True
        try:
            # (label, ip, user, kill cmd, status cmd) for every member of the run.
            targets = [(f"llama-server@{LLAMA_NODE['ip']}", "", "",
                        _LLAMA_KILL_CMD, _LLAMA_STATUS_CMD)]
            targets += [(f"ggml-rpc-server@{n['ip']}", n["ip"], n["user"],
                         _LLAMA_RPC_KILL_CMD, _LLAMA_RPC_STATUS_CMD)
                        for n in self._llama_rpc_nodes()]
            leftover: list[str] = []
            for attempt in (1, 2):
                ui.notify("Stopping llama-server and the ggml-rpc-servers..." if attempt == 1
                          else "Some processes survived, killing them again...")
                results = await asyncio.gather(*[
                    self._llama_exec(kill_cmd, ip, user)
                    for _, ip, user, kill_cmd, _ in targets])
                for (label, tip, _, _, _), (rc, out) in zip(targets, results):
                    if rc == 0:
                        ui.notify(f"{label} stopped ({out.strip()})", type="positive")
                    elif rc == 3:
                        ui.notify(f"No {label} process found (already stopped?)",
                                  type="warning")
                    elif rc == 255:
                        ui.notify(f"Node for {label} unreachable", type="negative")
                    else:
                        ui.notify(f"Error stopping {label} (rc={rc}): {out.strip()}",
                                  type="negative")
                # Verify: nothing may be left alive when we leave. The sleep lets
                # a just-SIGKILLed process disappear from the process table first.
                await asyncio.sleep(1)
                checks = await asyncio.gather(*[
                    self._llama_exec(status_cmd, ip, user)
                    for _, ip, user, _, status_cmd in targets])
                leftover = [t[0] for t, (rc, out) in zip(targets, checks)
                            if rc == 0 and out.strip().startswith("ALIVE")]
                if not leftover:
                    ui.notify("Llama: llama-server and all ggml-rpc-servers stopped",
                              type="positive")
                    break
            if leftover:
                ui.notify(f"Still running after two kill attempts: {', '.join(leftover)}",
                          type="negative")
            await self.update_llama_status()
        finally:
            self._llama_busy = False

    async def update_llama_log(self) -> None:
        """Refill the Llama log window with the tail of the remote /tmp/llama-server.log.

        The file is written by llama-server itself (--log-file), on the node the
        model runs on, and fetched over the very same SSH channel as the status
        and kill commands."""
        if self._llama_log_busy:
            ui.notify("A log fetch is already in progress", type="warning")
            return
        if not LLAMA_NODE:
            ui.notify("Llama node not configured (nodes.json)", type="warning")
            return
        self._llama_log_busy = True
        self.llama_log_update_button.disable()
        try:
            rc, out = await self._llama_exec(_LLAMA_LOG_CMD)
            self.llama_log_window.clear()
            if rc == 255:
                self.llama_log_window.push("--- llama node unreachable ---")
                ui.notify("Could not reach the llama node", type="negative")
            elif rc != 0:
                self.llama_log_window.push(f"--- log fetch failed (rc={rc}) ---")
                self.llama_log_window.push(out.rstrip())
                ui.notify(f"Error fetching /tmp/llama-server.log (rc={rc})", type="negative")
            else:
                for line in out.splitlines():
                    self.llama_log_window.push(line)
        finally:
            self.llama_log_update_button.enable()
            self._llama_log_busy = False

    def clear_llama_log(self) -> None:
        """Empty the Llama log window.

        View only: the remote /tmp/llama-server.log is left alone, because
        truncating the log of a live service would throw away the only record of
        what it did (and what it crashed with). "Update log" refills the window
        at any time. Same contract as clear_strata_log/clear_mlx_log/etc."""
        self.llama_log_window.clear()
        self.llama_log_window.push("--- log cleared; use \"Update log\" to refetch ---")
        ui.notify("Llama log window cleared")

    async def update_llama_status(self) -> None:
        # Polled every 5 s: skip a tick while the previous probe (3x SSH + HTTP)
        # is still in flight so a slow/unreachable node does not stack up checks.
        if self._llama_status_busy:
            return
        self._llama_status_busy = True
        try:
            if not LLAMA_NODE or not LLAMA_NODE.get("ip"):
                self.llama_status_label.set_text("Llama: node not configured (nodes.json)")
                self.llama_status_label.style("color: orange;")
                return
            ip = LLAMA_NODE.get("ip")
            rpc_nodes = self._llama_rpc_nodes()
            # Where the pieces of the run live, from nodes.json. Shown even when
            # a node is down: it is the configured target.
            self.llama_host_label.set_text(
                f"llama-server: {LLAMA_NODE.get('user', '?')}@{ip}"
                + ("   |   rpc: " + ", ".join(f"{n['user']}@{n['ip']}" for n in rpc_nodes)
                   if rpc_nodes else ""))
            # llama-server and the ggml-rpc-servers are probed in parallel: the
            # rpc line has its own state because a half-dead cluster (model up,
            # one worker gone) is exactly what the status must confess.
            results = await asyncio.gather(
                self._llama_exec(_LLAMA_STATUS_CMD),
                *[self._llama_exec(_LLAMA_RPC_STATUS_CMD, n["ip"], n["user"])
                  for n in rpc_nodes])
            rc, out = results[0]
            if rpc_nodes:
                self.llama_rpc_label.set_text(
                    "ggml-rpc-server: " + "   |   ".join(
                        f"{n['ip']}: {self._llama_proc_state(erc, eout)}"
                        for n, (erc, eout) in zip(rpc_nodes, results[1:])))
            if rc == 255:
                self.llama_status_label.set_text("Llama: node unreachable")
                self.llama_status_label.style("color: orange;")
                return
            proc_line = out.strip().splitlines()[-1] if out.strip() else "NO_PID"
            proc_alive = proc_line.startswith("ALIVE")
            # Ready only once the API answers, not merely when the process is
            # alive (same two-phase reading as Strata/Splash/FreeToken: a fresh
            # llama-server loads a 31B model long before it binds its API port).
            api_ok, model_id = await asyncio.to_thread(
                _strata_api_check, ip, _LLAMA_API_PORT)
            if api_ok:
                self.llama_status_label.set_text("Llama: RUNNING")
                self.llama_status_label.style("color: #00ff88;")
                self.llama_kill_button.enable()
            elif proc_alive:
                self.llama_status_label.set_text("Llama: STARTING (API not answering yet)")
                self.llama_status_label.style("color: orange;")
                self.llama_kill_button.enable()
            else:
                self.llama_status_label.set_text("Llama: STOPPED")
                self.llama_status_label.style("color: red;")
                self.llama_kill_button.disable()
            # Model name on its own row (copy icon right after it): the API's own
            # id once it answers, else the -m file stem of the fixed command
            # while the process is up; the icon follows the name's visibility.
            if model_id:
                self.llama_model_name = model_id
            else:
                self.llama_model_name = _LLAMA_MODEL_NAME if proc_alive else ""
            self.llama_model_label.set_text(self.llama_model_name)
            if self.llama_model_name:
                self.llama_model_copy.classes(remove='q-hidden')
            else:
                self.llama_model_copy.classes(add='q-hidden')
        except Exception:
            self.llama_status_label.set_text("Llama: UNKNOWN (status check failed)")
            self.llama_status_label.style("color: orange;")
        finally:
            self._llama_status_busy = False

    # ------------------------------------------------------------ log tail ---
    async def _stream_logs(self) -> None:
        argv = [_PY, "-u", _START_MODEL, "--tail-log", "-n", "1000"]
        try:
            await self._stream(argv)
        except asyncio.CancelledError:
            self.log_window.push("--- log streaming stopped ---")
            raise
        except OSError as e:
            self.log_window.push(f"Log Error: {e}")
        finally:
            if self.stop_log_button is not None:
                self.stop_log_button.disable()

    def start_log_streaming(self) -> None:
        # Without this guard every click stacked another `tail -F` that nothing
        # could ever stop.
        if self.log_task is not None and not self.log_task.done():
            ui.notify("Log streaming is already running", type="warning")
            return
        self.log_task = asyncio.create_task(self._stream_logs())
        self.stop_log_button.enable()
        ui.notify("Log streaming started")

    def stop_log_streaming(self) -> None:
        if self.log_task is None or self.log_task.done():
            ui.notify("Log streaming is not running", type="warning")
            return
        self.log_task.cancel()
        ui.notify("Log streaming stopped")

    def cleanup(self) -> None:
        """Tear down the tail when the browser tab goes away."""
        if self.log_task is not None and not self.log_task.done():
            self.log_task.cancel()

    # ------------------------------------------------------------------ UI ---
    # `active_tab` is the tab remembered for this browser across reloads
    # (see _remembered_tab); "llama_cluster" is the default for a new browser.
    def build_ui(self, active_tab: str = _DEFAULT_TAB) -> None:
        with ui.column().classes('w-full h-screen items-stretch p-0 m-0 gap-0'):
            with ui.column().classes('items-center q-mb-sm'):
                ui.label("LLM Inference Console").classes('text-h5')
                ui.label("by Alvise Dorigo").classes('text-h5')
                ui.link("https://github.com/dorigoa/ai-console",
                        "https://github.com/dorigoa/ai-console").classes('text-caption no-underline')

            with ui.row().classes('w-full flex-grow items-stretch p-0 m-0 gap-0'):
                with ui.column().classes('tab-strip items-stretch p-2'):
                    with ui.tabs().props('vertical no-caps').classes('w-full') as tabs:
                        # Explicit names (not the auto-generated ids): they are
                        # what gets remembered per browser across reloads.
                        # In recent NiceGUI the tab's value is its `name`, and the
                        # QTab value must match what ui.tab_panels expects.
                        # The old llama.cpp card is deprecated: kept fully wired
                        # (panel, buttons, status polling) but hidden from the
                        # tab strip and from its panel slot, so nothing breaks
                        # until it is finally removed. To bring it back, drop
                        # the two 'display: none' styles below.
                        tab_llama = ui.tab(name="llama", label="Llama.cpp")
                        tab_llama.style('display: none !important;')
                        tab_llama_cluster = ui.tab(name="llama_cluster", label="Llama")
                        tab_mlx = ui.tab(name="mlx", label="MLX")
                        tab_strata = ui.tab(name="strata", label="Strata")
                        tab_splash = ui.tab(name="splash", label="Splash")
                        tab_freetoken = ui.tab(name="freetoken", label="FreeToken")
                        # Remember whichever tab the user switches to, so a page
                        # reload (or a GUI restart) restores it (see _remembered_tab).
                        tabs.on_value_change(lambda e: _remember_tab(e.value))

                with ui.tab_panels(tabs, value=active_tab).classes('flex-grow p-2 overflow-auto'):
                    with ui.tab_panel(tab_llama) as llama_panel:
                        # Hidden along with its tab (see the note above): the
                        # panel is never selected while the tab is hidden, and
                        # display:none covers even a programmatic switch to it.
                        llama_panel.style('display: none !important;')
                        self._build_llama_panel()
                    with ui.tab_panel(tab_llama_cluster):
                        self._build_llama_cluster_panel()
                    with ui.tab_panel(tab_mlx):
                        self._build_mlx_panel()
                    with ui.tab_panel(tab_strata):
                        self._build_strata_panel()
                    with ui.tab_panel(tab_splash):
                        self._build_splash_panel()
                    with ui.tab_panel(tab_freetoken):
                        self._build_freetoken_panel()

    def _build_strata_panel(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("Strata — Qwen Flash Next").classes('text-h6')
                # Model name with its copy icon right after it, invisible until
                # the row is hovered (same .model-status-row/.model-copy-icon
                # rules as the llama panel); the icon only exists once a name
                # is actually known (q-hidden toggled in update_strata_status).
                # Monospace, one point above the surrounding font size.
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.strata_model_label = ui.label("")
                    self.strata_model_label.style(
                        'font-family: Consolas, Menlo, "Courier New", monospace;'
                        ' font-size: calc(1em + 1pt);')
                    self.strata_model_copy = ui.icon('content_copy').classes(
                        'model-copy-icon cursor-pointer q-hidden').tooltip('Copy model name')
                    self.strata_model_copy.on('click', self._copy_strata_model_name)
                self.strata_status_label = ui.label("Strata: checking...").classes('q-mt-sm')
                self.strata_status_label.style('font-size: 1.0rem; font-weight: 700;')
                # Where start.sh was launched: the SSH user@host from nodes.json.
                self.strata_host_label = ui.label("").classes('text-caption text-grey')
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    self.strata_run_button = ui.button(
                        "Run Qwen Flash Next", on_click=self.run_strata_qwen).props('color=green')
                    self.strata_kill_button = ui.button(
                        "Kill Strata", on_click=self.kill_strata_qwen).props('color=red')
                    self.strata_kill_button.disable()
                # Log of the service: the remote strata.out left by start.sh, pulled
                # on demand (no streaming) so it also works after the run has ended.
                with ui.row().classes('items-center gap-2 q-mt-lg w-full'):
                    ui.label("Strata log (strata.out)").classes('text-h6')
                    self.strata_log_update_button = ui.button(
                        "Update log", on_click=self.update_strata_log).props('small outline')
                    # View-only clear: the remote file is never touched (see
                    # clear_strata_log), so the button needs no confirmation.
                    self.strata_log_clear_button = ui.button(
                        "Clear log", on_click=self.clear_strata_log).props('small outline')
                    self.strata_log_clear_button.tooltip(
                        'Clear the window only; the remote strata.out is kept')
                self.strata_log_window = ui.log().classes(
                    'w-full h-[300px] bg-black text-green-400 font-mono text-xs custom-log')

    def _build_mlx_panel(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("MLX-Jaccl — Qwen-3.8-27B").classes('text-h6')
                # Model name with its copy icon right after it, invisible until
                # the row is hovered (same .model-status-row/.model-copy-icon
                # rules as the llama/strata panels); the icon only exists once a
                # name is actually known (q-hidden toggled in update_mlx_status).
                # Monospace, one point above the surrounding font size.
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.mlx_model_label = ui.label("")
                    self.mlx_model_label.style(
                        'font-family: Consolas, Menlo, "Courier New", monospace;'
                        ' font-size: calc(1em + 1pt);')
                    self.mlx_model_copy = ui.icon('content_copy').classes(
                        'model-copy-icon cursor-pointer q-hidden').tooltip('Copy model name')
                    self.mlx_model_copy.on('click', self._copy_mlx_model_name)
                self.mlx_status_label = ui.label("MLX: checking...").classes('q-mt-sm')
                self.mlx_status_label.style('font-size: 1.0rem; font-weight: 700;')
                # Where mlx_jaccl-start.py was launched: the SSH user@host from nodes.json.
                self.mlx_host_label = ui.label("").classes('text-caption text-grey')
                # rank 1 (nodes.json "ip_slave") state: the slave Mac holds no script of
                # its own, only the workers mlx.launch started there remotely, so it
                # gets its own line instead of being folded into the main status.
                self.mlx_slave_label = ui.label("").classes('text-caption text-grey')
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    self.mlx_run_button = ui.button(
                        "Run MLX-Jaccl Qwen", on_click=self.run_mlx_qwen).props('color=green')
                    self.mlx_kill_button = ui.button(
                        "Kill MLX", on_click=self.kill_mlx_qwen).props('color=red')
                    self.mlx_kill_button.disable()
                # Read at launch time in run_mlx_qwen: when set, it appends the
                # "-8bit" flag to the fixed mlx_jaccl-start.py options.
                self.mlx_8bit_checkbox = ui.checkbox("Choose 8 bit model")
                # Log of the service: the remote mlx-jaccl.out left by the start
                # script, pulled on demand (no streaming) so it also works after
                # the run has ended.
                with ui.row().classes('items-center gap-2 q-mt-lg w-full'):
                    ui.label("MLX log (mlx-jaccl.out)").classes('text-h6')
                    self.mlx_log_update_button = ui.button(
                        "Update log", on_click=self.update_mlx_log).props('small outline')
                    # View-only clear: the remote file is never touched (see
                    # clear_mlx_log), so the button needs no confirmation.
                    self.mlx_log_clear_button = ui.button(
                        "Clear log", on_click=self.clear_mlx_log).props('small outline')
                    self.mlx_log_clear_button.tooltip(
                        'Clear the window only; the remote mlx-jaccl.out is kept')
                self.mlx_log_window = ui.log().classes(
                    'w-full h-[300px] bg-black text-green-400 font-mono text-xs custom-log')

    def _build_splash_panel(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("Splash — Qwen3.8-27B (4bit)").classes('text-h6')
                # Model name with its copy icon right after it, invisible until
                # the row is hovered (same .model-status-row/.model-copy-icon
                # rules as the llama/strata/mlx panels); the icon only exists
                # once a name is actually known (q-hidden toggled in
                # update_splash_status). Monospace, one point above the
                # surrounding font size.
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.splash_model_label = ui.label("")
                    self.splash_model_label.style(
                        'font-family: Consolas, Menlo, "Courier New", monospace;'
                        ' font-size: calc(1em + 1pt);')
                    self.splash_model_copy = ui.icon('content_copy').classes(
                        'model-copy-icon cursor-pointer q-hidden').tooltip('Copy model name')
                    self.splash_model_copy.on('click', self._copy_splash_model_name)
                self.splash_status_label = ui.label("Splash: checking...").classes('q-mt-sm')
                self.splash_status_label.style('font-size: 1.0rem; font-weight: 700;')
                # Where splash serve was launched: the SSH user@host from nodes.json.
                self.splash_host_label = ui.label("").classes('text-caption text-grey')
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    self.splash_run_button = ui.button(
                        "Run Splash Qwen", on_click=self.run_splash_qwen).props('color=green')
                    self.splash_kill_button = ui.button(
                        "Kill Splash", on_click=self.kill_splash_qwen).props('color=red')
                    self.splash_kill_button.disable()
                # Log of the service: the remote splash.out left by splash serve,
                # pulled on demand (no streaming) so it also works after the run
                # has ended.
                with ui.row().classes('items-center gap-2 q-mt-lg w-full'):
                    ui.label("Splash log (splash.out)").classes('text-h6')
                    self.splash_log_update_button = ui.button(
                        "Update log", on_click=self.update_splash_log).props('small outline')
                    # View-only clear: the remote file is never touched (see
                    # clear_splash_log), so the button needs no confirmation.
                    self.splash_log_clear_button = ui.button(
                        "Clear log", on_click=self.clear_splash_log).props('small outline')
                    self.splash_log_clear_button.tooltip(
                        'Clear the window only; the remote splash.out is kept')
                self.splash_log_window = ui.log().classes(
                    'w-full h-[300px] bg-black text-green-400 font-mono text-xs custom-log')

    def _build_freetoken_panel(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("FreeToken — GPT-OSS-120B").classes('text-h6')
                # Model name with its copy icon right after it, invisible until
                # the row is hovered (same .model-status-row/.model-copy-icon
                # rules as the llama/strata/mlx/splash panels); the icon only
                # exists once a name is actually known (q-hidden toggled in
                # update_freetoken_status). Monospace, one point above the
                # surrounding font size.
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.freetoken_model_label = ui.label("")
                    self.freetoken_model_label.style(
                        'font-family: Consolas, Menlo, "Courier New", monospace;'
                        ' font-size: calc(1em + 1pt);')
                    self.freetoken_model_copy = ui.icon('content_copy').classes(
                        'model-copy-icon cursor-pointer q-hidden').tooltip('Copy model name')
                    self.freetoken_model_copy.on('click', self._copy_freetoken_model_name)
                self.freetoken_status_label = ui.label("FreeToken: checking...").classes('q-mt-sm')
                self.freetoken_status_label.style('font-size: 1.0rem; font-weight: 700;')
                # Where ft serve was launched: the SSH user@host from nodes.json.
                self.freetoken_host_label = ui.label("").classes('text-caption text-grey')
                # The one start button: "ft serve" on the GPU pinned in the
                # command line, serving GPT-OSS on its API port.
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    self.freetoken_run_button = ui.button(
                        "Run GPT-OSS", on_click=self.run_freetoken).props('color=green')
                    self.freetoken_kill_button = ui.button(
                        "Kill FreeToken", on_click=self.kill_freetoken).props('color=red')
                    self.freetoken_kill_button.disable()
                # Log of the service: the remote freetoken.out left by ft serve,
                # pulled on demand (no streaming) so it also works after the run
                # has ended.
                with ui.row().classes('items-center gap-2 q-mt-lg w-full'):
                    ui.label("FreeToken log (freetoken.out)").classes('text-h6')
                    self.freetoken_log_update_button = ui.button(
                        "Update log", on_click=self.update_freetoken_log).props('small outline')
                    # View-only clear: the remote file is never touched (see
                    # clear_freetoken_log), so the button needs no confirmation.
                    self.freetoken_log_clear_button = ui.button(
                        "Clear log", on_click=self.clear_freetoken_log).props('small outline')
                    self.freetoken_log_clear_button.tooltip(
                        'Clear the window only; the remote freetoken.out is kept')
                self.freetoken_log_window = ui.log().classes(
                    'w-full h-[300px] bg-black text-green-400 font-mono text-xs custom-log')

    def _build_llama_cluster_panel(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("Llama / RPC — Gemma-4-31B (ud-q8_k_xl)").classes('text-h6')
                # Model name with its copy icon right after it, invisible until
                # the row is hovered (same .model-status-row/.model-copy-icon
                # rules as the llama/strata panels); the icon only exists once a
                # name is actually known (q-hidden toggled in
                # update_llama_status). Monospace, one point above the
                # surrounding font size.
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.llama_model_label = ui.label("")
                    self.llama_model_label.style(
                        'font-family: Consolas, Menlo, "Courier New", monospace;'
                        ' font-size: calc(1em + 1pt);')
                    self.llama_model_copy = ui.icon('content_copy').classes(
                        'model-copy-icon cursor-pointer q-hidden').tooltip('Copy model name')
                    self.llama_model_copy.on('click', self._copy_llama_model_name)
                self.llama_status_label = ui.label("Llama: checking...").classes('q-mt-sm')
                self.llama_status_label.style('font-size: 1.0rem; font-weight: 700;')
                # Where llama-server and the ggml-rpc-servers run: the SSH
                # user@host entries from nodes.json.
                self.llama_host_label = ui.label("").classes('text-caption text-grey')
                # Per-node state of the ggml-rpc-servers: they live on their own
                # machines, so they get their own line instead of being folded
                # into the main status (which only reads llama-server).
                self.llama_rpc_label = ui.label("").classes('text-caption text-grey')
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    self.llama_run_button = ui.button(
                        "Run Gemma-4-31B", on_click=self.run_llama_gemma).props('color=green')
                    self.llama_kill_button = ui.button(
                        "Kill Llama", on_click=self.kill_llama).props('color=red')
                    self.llama_kill_button.disable()
                # Log of the service: the remote /tmp/llama-server.log written by
                # llama-server itself (--log-file), pulled on demand (no
                # streaming) so it also works after the run has ended.
                with ui.row().classes('items-center gap-2 q-mt-lg w-full'):
                    ui.label("Llama log (/tmp/llama-server.log)").classes('text-h6')
                    self.llama_log_update_button = ui.button(
                        "Update log", on_click=self.update_llama_log).props('small outline')
                    # View-only clear: the remote file is never touched (see
                    # clear_llama_log), so the button needs no confirmation.
                    self.llama_log_clear_button = ui.button(
                        "Clear log", on_click=self.clear_llama_log).props('small outline')
                    self.llama_log_clear_button.tooltip(
                        'Clear the window only; the remote /tmp/llama-server.log is kept')
                self.llama_log_window = ui.log().classes(
                    'w-full h-[300px] bg-black text-green-400 font-mono text-xs custom-log')

    def _build_llama_panel(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.column().classes('w-full max-w-2xl gap-1 q-mb-4 pr-4'):
                self.status_server_label = ui.label("Checking inference server status...")
                # Build info of the running llama-server (from /props); the
                # string is long, so this one is allowed to wrap.
                self.status_build_label = ui.label("")
                # The model name gets its own row so that a copy icon can sit
                # right after it; the icon is invisible until the row is hovered
                # (see the .model-copy-icon rules in the head HTML below).
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.status_model_label = ui.label("")
                    self.status_model_copy = ui.icon('content_copy').classes(
                        'model-copy-icon cursor-pointer q-hidden'
                    ).tooltip('Copy model name')
                    self.status_model_copy.on('click', self._copy_model_name)
                # When the running model was started (from /models 'created').
                self.status_started_label = ui.label("")
                self.status_ctx_label = ui.label("")
                self.status_samplers_label = ui.label("")
                # Same cursor-following tooltip as the label in the card below;
                # while nothing is running both stay empty, and an empty host
                # has no width left to hover. The status line only reports
                # numbers, so it gets the plain cursor (no question mark).
                with ui.element('div').classes('cursor-tip-host cursor-tip-plain'):
                    self.status_samplers_label = ui.label("")
                    self.status_samplers_tip = ui.label("").classes('cursor-tip-text')
                                
                for label in (self.status_server_label, self.status_model_label,
                              self.status_started_label, self.status_ctx_label,
                              self.status_samplers_label):
                    label.style('font-size: 0.9rem; font-weight: 600; white-space: nowrap;')
                for label in (self.status_model_label,
                              self.status_started_label,
                              self.status_ctx_label, self.status_samplers_label):
                    label.classes('font-mono').style('font-size: 0.9rem; font-weight: 600; white-space: pre;')
                # pre-wrap keeps the leading spaces (so "Build" lines up with the
                # labels below) while still letting the long build string wrap.
                self.status_build_label.classes('font-mono').style(
                    'font-size: 0.9rem; font-weight: 600; white-space: pre-wrap;')
                

                self.status_server_label.style('font-size: 1.2rem; font-weight: 850; white-space: nowrap;')

                for label in (self.status_model_label,
                              self.status_ctx_label, self.status_samplers_label):
                    label.classes('font-mono').style('font-size: 0.9rem; font-weight: 600; white-space: pre;')
                    
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    ui.button("Refresh", on_click=self.refresh).props('outline small')
                    # STOP is only meaningful while a model is live, so it stays
                    # hidden (q-hidden) until the polled status says "running".
                    self.stop_button = ui.button(
                        "STOP", on_click=self.stop_server).props('outline small color=red')
                    self.stop_button.classes('q-hidden')

            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("Model Control").classes('text-h6')

                # no-wrap: a wrapping row would let a long model name push
                # START/STOP onto the next line; with the select free to
                # shrink (min-w-0) the buttons always stay on the same row.
                with ui.row().classes('w-full items-center q-mb-md no-wrap'):
                    self.model_dropdown = ui.select(
                        options={"": "Loading models..."},
                        value="",
                        label="Loading models...",
                        on_change=self._on_model_change,
                    ).props('disable').classes('flex-grow min-w-0 model-select')

                    self.start_button = ui.button(
                        "START", on_click=self.start_selected_model).props('color=green')

                # The host div is what the cursor-following tooltip reacts to:
                # hovering it pops up the text of the .cursor-tip-text child,
                # which CSS keeps out of the page. _set_samplers() fills both.
                with ui.element('div').classes('cursor-tip-host'):
                    self.samplers_label = ui.label().classes("text-h8")
                    self.samplers_tip = ui.label().classes('cursor-tip-text')
                self._set_samplers()

                ui.button("Recheck models",
                          on_click=self.recheck_models).props('small outline')

                # RPC server labels — one per server the selected model uses.
                # _update_rpc_labels() fills the row once a model is selected,
                # which is the only place that knows the answer.
                if RPC_SERVERS:
                    with ui.row().classes('w-full items-center q-mt-sm gap-3'):
                        ui.label('RPC servers:').classes('text-subtitle1')
                        self.rpc_servers_row = ui.row().classes('items-center gap-3')
                    ui.button("Kill RPC servers",
                              on_click=self.kill_rpc_servers).props('small outline')

                with ui.row().classes('w-full items-center q-mt-sm gap-3'):
                    self.mtp_checkbox = ui.checkbox("Force No-MTP")
                    self.no_reasoning_checkbox = ui.checkbox("No reasoning")
                    with ui.row().classes('items-center gap-3'):
                        ui.label('Reasoning effort override:').classes('text-subtitle1')
                        self.reas_radio = ui.radio(_REAS_OPTIONS, value="").props('inline')

                with ui.column().classes('w-full q-mt-sm'):
                    self.ctx_label = ui.label("Context: —").classes('text-subtitle1')
                    self.ctx_slider = ui.slider(
                        min=_CTX_MIN, max=262144, value=_CTX_MIN, step=_CTX_STEP,
                        on_change=self._on_ctx_change
                    ).classes('flex-grow').props('color=green')
                    with ui.row().classes('items-center gap-3'):
                        ui.label('KV Quant:').classes('text-subtitle1')
                        self.kvquant_radio = ui.radio(
                            {"": "None", "q8_0": "8 bit", "q4_0": "4 bit"},
                            value="",
                            on_change=lambda e: self._update_memory()
                        ).props('inline')
                    # Estimated total footprint, refreshed by the context slider
                    # and the KV quant radio; the "—" state means no model
                    # selected (or its budget is unknown).
                    self.mem_label = ui.label("Memory: —").classes('text-subtitle1')

                with ui.column().classes('w-full q-mt-sm'):
                    self.temp_label = ui.label("Temperature: —").classes('text-subtitle1')
                    self.temp_slider = ui.slider(
                        min=0, max=1.0, value=1.0, step=0.01,
                        on_change=lambda e: self.temp_label.set_text(f"Temperature: {e.value:.2f}")
                    ).classes('flex-grow').props('color=orange')

            ui.label("Server Logs").classes('text-h6 q-mt-lg')
            with ui.row().classes('w-full items-center q-mb-sm'):
                ui.button("Connect to inference server logs",
                          on_click=self.start_log_streaming).props('small')
                self.stop_log_button = ui.button("Stop log streaming",
                                                 on_click=self.stop_log_streaming).props('small outline')
                self.stop_log_button.disable()
                ui.button("Clear Logs", on_click=lambda: self.log_window.clear()).props('small outline')

            self.log_window = ui.log().classes(
                'w-full h-[600px] bg-black text-green-400 font-mono text-xs custom-log')

        ui.add_head_html('''
<style>
/* The vertical tab selector: an isolated full-height strip on the left,
   separated from the panels by a border. */
.tab-strip {
    min-width: 7rem;
    border-right: 1px solid rgba(255, 255, 255, 0.18);
}
/* The selected value is a <span class="ellipsis"> inside a flex row, so like
   every flex item it carries min-width: auto and refuses to shrink — the
   ellipsis could never trigger. Freeing it lets long model names truncate
   instead of stretching the select. */
.model-select .q-field__native .ellipsis {
    min-width: 0;
}
.custom-log {
    scrollbar-width: thin !important;
    scrollbar-color: #4caf50 #1a1a1a !important;
}
.custom-log::-webkit-scrollbar {
    width: 14px !important;
    height: 14px !important;
}
.custom-log::-webkit-scrollbar-track {
    background: #1a1a1a !important;
    border-radius: 7px !important;
}
.custom-log::-webkit-scrollbar-thumb {
    background: #4caf50 !important;
    border-radius: 7px !important;
    border: 3px solid #1a1a1a !important;
}
.custom-log::-webkit-scrollbar-thumb:hover {
    background: #66bb6a !important;
}
/* Copy icon next to the model name: hidden until the row is hovered. */
.model-copy-icon {
    opacity: 0;
    transition: opacity 0.15s ease-in-out;
}
.model-status-row:hover .model-copy-icon {
    opacity: 0.75;
}
.model-copy-icon:hover {
    opacity: 1;
}
/* Cursor-following tooltip. An element opts in by carrying .cursor-tip-host
   and a .cursor-tip-text child holding the text to show; the child is hidden
   here and read by the script below, which parks #cursor-tip next to the
   pointer. fit-content keeps the hover area on the text instead of stretching
   it across the whole card row. */
.cursor-tip-host {
    width: fit-content;
    cursor: help;
}
/* Hosts that only report values keep the plain cursor (no question mark);
   the rule must come after .cursor-tip-host to override its cursor. */
.cursor-tip-plain {
    cursor: default;
}
.cursor-tip-text {
    display: none;
}
#cursor-tip {
    position: fixed;
    left: 0;
    top: 0;
    z-index: 10000;
    max-width: 30rem;
    padding: 8px 10px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 6px;
    background: rgba(28, 28, 28, 0.97);
    color: #e8e8e8;
    font-size: 0.8rem;
    line-height: 1.35;
    /* pre-line: the text arrives as one string with newlines between the
       parameters. */
    white-space: pre-line;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    /* Never let the tooltip take the mouse it is chasing: without this it
       lands under the pointer and flickers itself away. */
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.12s ease-in-out;
}
#cursor-tip.visible {
    opacity: 1;
}
</style>
<script>
(() => {
    // add_head_html() runs per page, this listener set only needs to exist once.
    if (window.__cursorTipReady) return;
    window.__cursorTipReady = true;

    const OFFSET_X = 16;
    const OFFSET_Y = 18;
    const MARGIN = 8;
    let tip = null;
    let host = null;

    // Created lazily: the head script runs before <body> exists, and Vue
    // replaces the page content afterwards anyway.
    const getTip = () => {
        if (!tip || !tip.isConnected) {
            tip = document.createElement('div');
            tip.id = 'cursor-tip';
            document.body.appendChild(tip);
        }
        return tip;
    };

    const place = (event) => {
        const box = getTip().getBoundingClientRect();
        let x = event.clientX + OFFSET_X;
        let y = event.clientY + OFFSET_Y;
        // Flip to the other side of the pointer rather than run off-screen.
        if (x + box.width > window.innerWidth - MARGIN) {
            x = event.clientX - OFFSET_X - box.width;
        }
        if (y + box.height > window.innerHeight - MARGIN) {
            y = event.clientY - OFFSET_Y - box.height;
        }
        tip.style.left = Math.max(MARGIN, x) + 'px';
        tip.style.top = Math.max(MARGIN, y) + 'px';
    };

    const hide = () => {
        host = null;
        if (tip) tip.classList.remove('visible');
    };

    document.addEventListener('mouseover', (event) => {
        const found = event.target.closest && event.target.closest('.cursor-tip-host');
        if (!found) {
            if (host) hide();
            return;
        }
        if (found === host) return;
        const source = found.querySelector('.cursor-tip-text');
        const text = source ? source.textContent.trim() : '';
        if (!text) { hide(); return; }
        host = found;
        getTip().textContent = text;
        // Position first, show second: otherwise the tooltip flashes at the
        // top-left corner on the frame before the first mousemove.
        place(event);
        tip.classList.add('visible');
    });

    document.addEventListener('mousemove', (event) => {
        if (!host) return;
        // The host can be re-rendered out of the DOM while hovered (every
        // model change rebuilds the label's text).
        if (!host.isConnected) { hide(); return; }
        place(event);
    });

    document.addEventListener('mouseout', (event) => {
        if (host && !host.contains(event.relatedTarget)) hide();
    });

    // Scrolling moves the text out from under a pointer that never moved.
    document.addEventListener('scroll', hide, true);
    window.addEventListener('blur', hide);
})();
</script>
''')

#___________________________________________________________________________________
@ui.page('/')
def index() -> None:
    ui.dark_mode().enable()
    gui = LlamaConsoleGUI()
    gui.build_ui(active_tab=_remembered_tab())
    context.client.on_disconnect(gui.cleanup)
    # Models come from the app-level cache filled at boot, so a page (re)load
    # no longer rescans them; only "Recheck models" forces it.
    ui.timer(0, gui.load_models, once=True)
    # Status polling: fires immediately on connect, then every 5 s, so the
    # user always sees whether (and what) the server is running.
    ui.timer(5.0, gui.update_status)
    # Strata service status on the same 5 s cadence; RUNNING needs both the process
    # alive and the /v1/models API to answer.
    ui.timer(5.0, gui.update_strata_status)
    # MLX-Jaccl service status on the same 5 s cadence; RUNNING only needs the
    # process alive (there is no HTTP API to answer on this one).
    ui.timer(5.0, gui.update_mlx_status)
    # Splash service status on the same 5 s cadence; like Strata, RUNNING needs
    # both the process alive and the /v1/models API (its nodes.json "api_port")
    # to answer.
    ui.timer(5.0, gui.update_splash_status)
    # FreeToken service status on the same 5 s cadence; like Strata/Splash,
    # RUNNING needs both the ft process alive and the /v1/models API (its
    # "api_port") to answer.
    ui.timer(5.0, gui.update_freetoken_status)
    # Llama cluster (llama-server + the two ggml-rpc-servers) status on the same
    # 5 s cadence; like Strata/Splash/FreeToken, RUNNING needs both the
    # llama-server process alive and its /v1/models API (its "api_port") to answer.
    ui.timer(5.0, gui.update_llama_status)


ui.run(title=settings.UI_TITLE, port=settings.UI_PORT, host="0.0.0.0",
        reload=False, show=False, storage_secret=_storage_secret())
