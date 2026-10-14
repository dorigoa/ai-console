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
import contextlib
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

# Load node definitions (nodes.json): the strata and mlx nodes' IP and SSH user
# live here (same shape as rpc.json). The Strata/MLX cards are no-ops until the
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
_STRATA_API_PORT = 8000
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
# workload is launched by ~/mlx-jaccl/mlx_jaccl-start.sh with a fixed command line
# (no HTTP API to probe, so status is purely "is the process alive"), and the node
# is a Mac: BSD userland, so no setsid(1), no /proc, and a login shell that may well
# be zsh rather than bash.
#
# The start command below still launches the script invocation *verbatim*, but the
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
_MLX_DIR = '$HOME/mlx-jaccl'
_MLX_PIDFILE = f'{_MLX_DIR}/mlx-jaccl.pid'
_MLX_LOGFILE = f'{_MLX_DIR}/mlx-jaccl.out'
_MLX_START_CMD = (
    'command -v python3 >/dev/null 2>&1 || { echo "python3 not found on the mlx node"; exit 127; }; '
    'python3 -c \'import sys, subprocess; '
    'pidfile, logfile, cmd = sys.argv[1], sys.argv[2], sys.argv[3]; '
    'log = open(logfile, "wb", 0); '
    'child = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, '
    'stdin=subprocess.DEVNULL, start_new_session=True); '
    'open(pidfile, "w").write(str(child.pid) + "\\n"); '
    'print("STARTED_PID=%d" % child.pid)\' '
    f'"{_MLX_PIDFILE}" "{_MLX_LOGFILE}" '
    '"~/mlx-jaccl/mlx_jaccl-start.sh -8bit -max-tokens 262144 '
    '-hostfile ~/mlx-jaccl/hosts.json -leak-patch ~/mlx-jaccl/mlx_leak_shim.py '
    'Qwen3.8-27B"'
)
# What an MLX-Jaccl run actually is made of, as pgrep -f regexes:
#   - the start script itself (rank 0 only, and the session leader recorded in the
#     pidfile: killing it alone would leave the workers behind, hence the group kill);
#   - the distributed launcher "mlx.launch" and the leak shim, which run on *both*
#     nodes of the run (rank 0 and rank 1).
# The "[.]" bracket trick keeps pgrep from matching the remote shell running the
# check itself, whose argv contains every pattern verbatim.
_MLX_PROC_PATTERNS = (r"mlx-jaccl/mlx_jaccl-start[.]sh", r"mlx[.]launch", r"mlx_leak_shim[.]py")


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
    holds the leader of the session _MLX_START_CMD created, so that group is exactly
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
# Console log of the service: mlx-jaccl.out, written by _MLX_START_CMD. Fetched on
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


def _strata_api_check(ip: str) -> tuple[bool, str]:
    """(ok, model_id) from the node's OpenAI-compatible /v1/models endpoint.

    Runs in a worker thread (see update_strata_status); any failure maps to
    ok=False so a dead/unreachable API reads as "not ready", never an exception."""
    try:
        r = requests.get(f"http://{ip}:{_STRATA_API_PORT}/v1/models", timeout=3)
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
            # second mlx_jaccl-start.sh would only fight the running one for the
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
                          "use Kill Qwen first", type="warning")
                return
            ui.notify("Starting MLX-Jaccl Qwen on the mlx node...")
            rc, out = await self._mlx_exec(_MLX_START_CMD)
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

        The file is the stdout/stderr of the mlx_jaccl-start.sh launched by "Run
        MLX-Jaccl Qwen" (redirected by _MLX_START_CMD), fetched over the very same
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
            # Node the SSH commands (and mlx_jaccl-start.sh) run on, from nodes.json.
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
            # No HTTP API to probe here (unlike Strata): the processes *are* the
            # service, and "running" means either rank still holds one of them — a
            # half-dead run is not "stopped", it is something Kill Qwen has to clean.
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
            # Model name on its own row (copy icon right after it), fixed by the
            # start command and shown only while the service is up; the icon
            # follows the name's visibility.
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
    def build_ui(self) -> None:
        with ui.column().classes('w-full h-screen items-stretch p-0 m-0 gap-0'):
            with ui.column().classes('items-center q-mb-sm'):
                ui.label("LLM Inference Console").classes('text-h5')
                ui.label("by Alvise Dorigo").classes('text-h5')
                ui.link("https://github.com/dorigoa/ai-console",
                        "https://github.com/dorigoa/ai-console").classes('text-caption no-underline')

            with ui.row().classes('w-full flex-grow items-stretch p-0 m-0 gap-0'):
                with ui.column().classes('tab-strip items-stretch p-2'):
                    with ui.tabs().props('vertical no-caps').classes('w-full') as tabs:
                        tab_llama = ui.tab('Llama.cpp')
                        tab_mlx = ui.tab('MLX')
                        tab_strata = ui.tab('Strata')

                with ui.tab_panels(tabs, value=tab_llama).classes('flex-grow p-2 overflow-auto'):
                    with ui.tab_panel(tab_llama):
                        self._build_llama_panel()
                    with ui.tab_panel(tab_mlx):
                        self._build_mlx_panel()
                    with ui.tab_panel(tab_strata):
                        self._build_strata_panel()

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
                        "Kill Qwen", on_click=self.kill_strata_qwen).props('color=red')
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
                ui.label("MLX — MLX-Jaccl Qwen").classes('text-h6')
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
                # Where mlx_jaccl-start.sh was launched: the SSH user@host from nodes.json.
                self.mlx_host_label = ui.label("").classes('text-caption text-grey')
                # rank 1 (nodes.json "ip_slave") state: the slave Mac holds no script of
                # its own, only the workers mlx.launch started there remotely, so it
                # gets its own line instead of being folded into the main status.
                self.mlx_slave_label = ui.label("").classes('text-caption text-grey')
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    self.mlx_run_button = ui.button(
                        "Run MLX-Jaccl Qwen", on_click=self.run_mlx_qwen).props('color=green')
                    self.mlx_kill_button = ui.button(
                        "Kill Qwen", on_click=self.kill_mlx_qwen).props('color=red')
                    self.mlx_kill_button.disable()
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
    gui.build_ui()
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


ui.run(title=settings.UI_TITLE, port=settings.UI_PORT, host="0.0.0.0", reload=False, show=False)
