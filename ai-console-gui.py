"""NiceGUI front-end of the LLM inference console.

One tab per inference service. Each tab drives a fixed workload on a remote node
over SSH (start, kill, status, log tail) and probes its OpenAI-compatible
/v1/models API; a green bullet in the tab strip flags every RUNNING service.

Nodes (IP, SSH user, API port) come from nodes.json. The legacy "Llama.cpp" tab,
a front-end for start_model.py, is still available but hidden: see
SHOW_LEGACY_LLAMA_TAB.
"""

import asyncio
import base64
import contextlib
import functools
import json
import os
import shlex
import signal
import sys
from pathlib import Path

import requests
from logzero import logger
from nicegui import app, context, ui

from config_manager import get_settings
from model import Model, load_models as load_models_from_config

settings = get_settings()
_HERE = Path(__file__).parent

# While hidden, the legacy start_model.py tab is neither built nor polled, and
# the model list it needs is never scanned. Set to True to bring it back.
SHOW_LEGACY_LLAMA_TAB = False

_POLL_SECONDS = 5.0
_REPO_URL = "https://github.com/dorigoa/ai-console"


def _json_section(filename: str, key: str) -> dict:
    """The `key` section of a JSON file next to this script ({} if missing or malformed)."""
    try:
        with open(_HERE / filename) as f:
            return json.load(f).get(key, {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


RPC_SERVERS: dict = _json_section("rpc.json", "RPC_SERVERS")
NODES: dict = _json_section("nodes.json", "NODES")


# =============================================================================
# Remote shell commands
#
# Plain POSIX sh, so they run on both the Linux and the macOS nodes. Every pgrep
# pattern uses the "[x]" bracket trick: the regex never matches its own text, so
# pgrep cannot match the remote shell running the check (whose argv contains
# the pattern verbatim).
# =============================================================================

# Normalise a list of pids: one line, de-duplicated, no blanks.
_PID_NORM = 'tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -'
_LOG_LINES = 200


def _collect_pids(var: str, patterns: tuple[str, ...]) -> str:
    """Set $var to the pids of every process matching one of patterns."""
    collect = " ".join(f'{var}="${var} $(pgrep -f "{p}" 2>/dev/null)";' for p in patterns)
    return f'{var}=""; {collect} {var}=$(echo ${var} | {_PID_NORM}); '


def _status_cmd(*patterns: str) -> str:
    """Print "ALIVE <pids>" when some process matches, "NO_PID" otherwise."""
    return (_collect_pids("pids", patterns)
            + 'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi')


def _kill_cmd(*patterns: str, pidfile: str = "", verify: bool = False) -> str:
    """Kill every process matching patterns: TERM, two seconds to drain, then KILL.

    With a pidfile, the session it records (created by the start command, so it
    is exactly "this run") is killed as a whole too, as long as its leader is
    alive. Exit 3: nothing to kill. With verify, the patterns are probed again at
    the end and exit 4 (STILL_ALIVE) means the run survived.

    The loops go over $(echo $pids), not $pids: zsh (the macOS login shell) does
    not word-split a bare $pids, so with two or more pids it would signal none."""
    group = term_group = kill_group = check = ""
    if pidfile:
        group = (f'g=$(cat "{pidfile}" 2>/dev/null | tr -d " "); '
                 'if [ -n "$g" ] && kill -0 "$g" 2>/dev/null; then '
                 'pids="$pids $g"; RUN_GROUP=$g; else RUN_GROUP=""; fi; '
                 f'pids=$(echo $pids | {_PID_NORM}); ')
        term_group = 'if [ -n "$RUN_GROUP" ]; then kill -TERM -$RUN_GROUP 2>/dev/null; fi; '
        kill_group = 'if [ -n "$RUN_GROUP" ]; then kill -KILL -$RUN_GROUP 2>/dev/null; fi; '
    if verify:
        check = ('sleep 1; ' + _collect_pids("left", patterns)
                 + 'if [ -n "$left" ]; then echo "STILL_ALIVE $left"; exit 4; fi; ')
    return (_collect_pids("pids", patterns) + group
            + 'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; exit 3; fi; '
            + 'for p in $(echo $pids); do kill -TERM "$p" 2>/dev/null; done; ' + term_group
            + 'sleep 2; '
            + 'for p in $(echo $pids); do kill -KILL "$p" 2>/dev/null; done; ' + kill_group
            + check
            + 'echo "KILLED_PIDS=$pids"')


def _log_tail_cmd(path: str) -> str:
    """Print the last _LOG_LINES lines of path ("wc -l" pads its output on BSD)."""
    return (f'f="{path}"; '
            'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
            f'echo "--- $f: last {_LOG_LINES} of $(wc -l < "$f" | tr -d " ") lines ---"; '
            f'tail -n {_LOG_LINES} "$f"')


def _setsid_start_cmd(workdir: str, name: str, cmd: str) -> str:
    """Linux detached start: cmd runs from workdir as a session leader (so a group
    kill takes the whole run down), with its pid in <name>.pid and its output in
    <name>.out, truncated at every start. "exec" keeps the recorded pid honest."""
    return (f'cd "{workdir}" && {{ setsid bash -c \'echo $$ > {name}.pid; exec {cmd}\' '
            f'> {name}.out 2>&1 </dev/null & sleep 1; echo "STARTED_PID=$(cat {name}.pid)"; }}')


def _detached_start_cmd(pidfile: str, logfile: str, cmd: str, *, node: str,
                        append: bool = True) -> str:
    """Portable detached start (macOS has no setsid(1)), done by python3.

    Popen(start_new_session=True) makes the run a session leader, so the kill
    command can take it down as a group; its pid goes to pidfile and its output
    to logfile (appended, or truncated at every start). cmd travels base64
    encoded, so no quote inside it can break the outer command line."""
    b64 = base64.b64encode(cmd.encode()).decode()
    mode = "ab" if append else "wb"
    return (
        f'command -v python3 >/dev/null 2>&1 || {{ echo "python3 not found on the {node} node"; exit 127; }}; '
        'python3 -c \'import base64,sys,subprocess; '
        'pidfile, logfile, cmd = sys.argv[1], sys.argv[2], base64.b64decode(sys.argv[3]).decode(); '
        f'log = open(logfile, "{mode}", 0); '
        'child = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, '
        'stdin=subprocess.DEVNULL, start_new_session=True); '
        'open(pidfile, "w").write(str(child.pid) + "\\n"); '
        'print("STARTED_PID=%d" % child.pid)\' '
        f'"{pidfile}" "{logfile}" "{b64}"'
    )


# =============================================================================
# Helpers
# =============================================================================

_GREEN, _ORANGE, _RED = "#00ff88", "orange", "red"
_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no")


async def _capture(argv: list[str]) -> tuple[str, int]:
    """Run argv to completion off the event loop; return (stdout, returncode)."""
    logger.debug(f"Executing {argv}")
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if stderr:
        logger.debug(stderr.decode(errors="replace").rstrip())
    return stdout.decode(errors="replace"), proc.returncode


async def _ssh(ip: str, user: str, cmd: str, merge_stderr: bool = False) -> tuple[int, str]:
    """Run cmd on user@ip; return (returncode, stdout).

    rc 255 is ssh's own failure code: the node was not reached (or has no
    address). merge_stderr folds the remote stderr into stdout, for the macOS
    nodes: without it a "python3 not found" or a shell error would reach the UI
    as a bare rc=N."""
    if not ip or not user:
        return 255, ""
    if merge_stderr:
        cmd = f"{{ {cmd}; }} 2>&1"
    out, rc = await _capture(["ssh", *_SSH_OPTS, f"{user}@{ip}", cmd])
    return rc, out


def _api_check(ip: str, port: int) -> tuple[bool, str]:
    """(ok, model_id) from an OpenAI-compatible /v1/models endpoint.

    Blocking: run it in a worker thread. Any failure reads as "not ready"."""
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


def _last_line(text: str) -> str:
    text = text.strip()
    return text.splitlines()[-1] if text else ""


def _is_alive(out: str) -> bool:
    """Whether a _status_cmd answer reports live processes (its last line, so any
    noise printed by the remote login shell is ignored)."""
    return _last_line(out).startswith("ALIVE")


def _proc_state(rc: int, out: str) -> str:
    """Classify a _status_cmd answer: running / stopped / unreachable / error."""
    if rc == 255:
        return "unreachable"
    if rc != 0:
        return "error"
    return "running" if _is_alive(out) else "stopped"


def _set_class(element: ui.element, name: str, on: bool) -> None:
    if on:
        element.classes(add=name)
    else:
        element.classes(remove=name)


async def _exclusive(lock: asyncio.Lock, busy_msg: str, button: ui.button, action) -> None:
    """Run action unless lock is already held (then just warn), with a spinner on
    button meanwhile (a loading button also ignores further clicks)."""
    if lock.locked():
        ui.notify(busy_msg, type="warning")
        return
    async with lock:
        button.props('loading')
        try:
            await action()
        finally:
            button.props(remove='loading')


async def _copy_to_clipboard(text: str, what: str) -> None:
    """Copy text to the OS clipboard, telling the user what happened.

    navigator.clipboard only exists in secure contexts (https or localhost),
    while this console is usually reached over plain http on the LAN: hence the
    fallback to the deprecated execCommand copy."""
    if not text:
        ui.notify(f"No {what} to copy", type="warning")
        return
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


# =============================================================================
# Remote services
# =============================================================================

class RemoteService:
    """One service tab, instantiated per browser: a fixed workload on a remote
    node, with Run/Kill buttons, a polled status and an on-demand log tail.

    RUNNING means the API answers on api_port; a live process whose API is still
    silent is only STARTING."""

    tab = ""                # tab name, remembered per browser
    tab_label = ""
    group_start = False     # first tab of the macOS group: a rule is drawn above it
    title = ""
    description = ""
    name = ""               # status-line prefix ("<name>: RUNNING")
    host_prefix = ""
    node: dict = {}
    node_desc = ""          # messages call the node "the <node_desc>"
    macos = False           # BSD tools report errors on stderr: merge it
    api_port = 0
    workload = ""           # what runs there, as the messages call it
    run_label = ""
    run_what = ""           # what the Run button starts, as its messages call it
    kill_label = ""
    fallback_model = ""     # shown while the process is up but the API is silent
    start_command = ""
    status_command = ""
    kill_command = ""
    log_path = ""

    def __init__(self) -> None:
        self.model_name = ""
        self.tab_element: ui.tab | None = None
        self._action_lock = asyncio.Lock()
        self._log_lock = asyncio.Lock()
        self._status_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.node.get("ip") and self.node.get("user"))

    @property
    def log_name(self) -> str:
        """The log file as the UI names it: just the file name when under $HOME."""
        if self.log_path.startswith("$HOME/"):
            return self.log_path.rsplit("/", 1)[-1]
        return self.log_path

    async def exec(self, cmd: str) -> tuple[int, str]:
        return await _ssh(self.node.get("ip", ""), self.node.get("user", ""), cmd, self.macos)

    # ------------------------------------------------------------------ UI ---
    def build(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label(self.title).classes('text-h6')
                ui.label(self.description).classes('service-description')
                # Model name, with a copy icon (shown on hover) once a name is known.
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.model_label = ui.label("").classes('model-name')
                    self.model_copy = ui.icon('content_copy').tooltip('Copy model name') \
                        .classes('model-copy-icon cursor-pointer invisible')
                    self.model_copy.on('click', lambda: _copy_to_clipboard(self.model_name, "model name"))
                self.status_label = ui.label(f"{self.name}: checking...").classes('service-status q-mt-sm')
                self.host_label = ui.label("").classes('text-caption text-grey')
                self._build_extras()
                with ui.row().classes('items-center gap-2 q-mt-md flex-nowrap'):
                    for label, action in self._run_actions():
                        self._action_button(label, 'green', action)
                    self.kill_button = self._action_button(self.kill_label, 'red', self.kill)
                    self.kill_button.disable()
                # Pulled on demand, so the log is still there after the run ended.
                with ui.row().classes('items-center gap-2 q-mt-lg w-full'):
                    ui.label(f"{self.name} log ({self.log_name})").classes('text-h6')
                    update = ui.button("Update log").props('small outline')
                    update.on_click(lambda: _exclusive(
                        self._log_lock, "A log fetch is already in progress", update, self.update_log))
                    # View only: the remote file is the only record of the run.
                    ui.button("Clear log", on_click=self.clear_log).props('small outline') \
                        .tooltip(f'Clear the window only; the remote {self.log_name} is kept')
                self.log_window = ui.log().classes(
                    'w-full h-[300px] bg-black text-green-400 font-mono text-xs custom-log')

    def _build_extras(self) -> None:
        """Widgets between the host line and the buttons (none by default)."""

    def _run_actions(self) -> list[tuple[str, object]]:
        return [(self.run_label, self.run)]

    def _action_button(self, label: str, color: str, action) -> ui.button:
        button = ui.button(label).props(f'color={color}').classes('w-48 justify-center')
        button.on_click(lambda: _exclusive(
            self._action_lock, f"A {self.name} action is already in progress", button, action))
        return button

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.set_text(f"{self.name}: {text}")
        self.status_label.style(f"color: {color}")

    def _set_model(self, name: str) -> None:
        self.model_name = name
        self.model_label.set_text(name)
        _set_class(self.model_copy, 'invisible', not name)

    def _check_configured(self) -> bool:
        if not self.configured:
            ui.notify(f"The {self.node_desc} is not configured (nodes.json)", type="warning")
        return self.configured

    # ------------------------------------------------------------- actions ---
    async def run(self) -> None:
        await self._launch(*self._start_spec())
        await self.update_status()

    def _start_spec(self) -> tuple[str, str]:
        """The start command, and what it starts as the messages call it."""
        return self.start_command, self.run_what

    async def _launch(self, start_cmd: str, what: str) -> bool:
        """Run start_cmd unless the service already runs; True on success."""
        if not self._check_configured() or not await self._can_start():
            return False
        ui.notify(f"Starting {what} on the {self.node_desc}...")
        rc, out = await self.exec(start_cmd)
        if rc == 0:
            ui.notify(f"{what} started on the {self.node_desc} ({_last_line(out)})", type="positive")
        else:
            ui.notify(f"Error starting {what} on the {self.node_desc} (rc={rc}): {out.strip()}",
                      type="negative")
        return rc == 0

    async def _can_start(self) -> bool:
        """Refuse a second launch over a live one: it would only fight the running
        one for the API port and the GPU/memory. The status poll may be stale."""
        rc, out = await self.exec(self.status_command)
        if rc == 0 and _is_alive(out):
            ui.notify(f"{self.workload} is already running on the {self.node_desc}; "
                      f"use {self.kill_label} first", type="warning")
            return False
        return True

    async def kill(self) -> None:
        if self._check_configured():
            await self._kill()
        await self.update_status()

    async def _kill(self) -> None:
        ui.notify(f"Stopping {self.workload} on the {self.node_desc}...")
        rc, out = await self.exec(self.kill_command)
        self._notify_kill(rc, out, f"the {self.node_desc}")

    def _notify_kill(self, rc: int, out: str, where: str) -> None:
        out = out.strip()
        if rc == 0:
            ui.notify(f"{self.workload} stopped on {where} ({out})", type="positive")
        elif rc == 3:
            ui.notify(f"No {self.workload} process found on {where} (already stopped?)",
                      type="warning")
        elif rc == 4:
            ui.notify(f"{self.workload} did not go down on {where}: {out}", type="negative")
        elif rc == 255:
            ui.notify(f"Could not reach {where}", type="negative")
        else:
            ui.notify(f"Error stopping {self.workload} on {where} (rc={rc}): {out}",
                      type="negative")

    # ----------------------------------------------------------------- log ---
    async def update_log(self) -> None:
        if not self._check_configured():
            return
        rc, out = await self.exec(_log_tail_cmd(self.log_path))
        self.log_window.clear()
        if rc == 255:
            self.log_window.push(f"--- {self.node_desc} unreachable ---")
            ui.notify(f"Could not reach the {self.node_desc}", type="negative")
        elif rc != 0:
            self.log_window.push(f"--- log fetch failed (rc={rc}) ---")
            self.log_window.push(out.rstrip())
            ui.notify(f"Error fetching {self.log_name} (rc={rc})", type="negative")
        else:
            self.log_window.push(out)

    def clear_log(self) -> None:
        self.log_window.clear()
        self.log_window.push('--- log cleared; use "Update log" to refetch ---')
        ui.notify(f"{self.name} log window cleared")

    # -------------------------------------------------------------- status ---
    async def update_status(self) -> None:
        """Polled by a timer: a tick is skipped while the previous probe is still
        in flight, so a slow node never stacks up checks."""
        if self._status_lock.locked():
            return
        async with self._status_lock:
            running = False
            try:
                if self.configured:
                    running = await self._probe()
                else:
                    self._set_status("node not configured (nodes.json)", _ORANGE)
            except Exception:
                logger.debug(f"{self.name} status check failed", exc_info=True)
                self._set_status("UNKNOWN (status check failed)", _ORANGE)
            finally:
                if self.tab_element is not None:
                    _set_class(self.tab_element, 'running-bullet', running)

    async def _probe(self) -> bool:
        """Refresh the status widgets; return whether the service is RUNNING."""
        self.host_label.set_text(self._host_text())
        (rc, out), (api_ok, model_id) = await asyncio.gather(
            self.exec(self.status_command), self._api_state())
        if rc == 255:
            self._set_status("node unreachable", _ORANGE)
            return False
        proc_alive = _is_alive(out)
        if api_ok:
            self._set_status("RUNNING", _GREEN)
        elif proc_alive:
            self._set_status("STARTING (API not answering yet)", _ORANGE)
        else:
            self._set_status("STOPPED", _RED)
        self.kill_button.set_enabled(api_ok or proc_alive)
        self._set_model(model_id or (self.fallback_model if proc_alive else ""))
        return api_ok

    def _host_text(self) -> str:
        return f"{self.host_prefix}: {self.node['user']}@{self.node['ip']}:{self.api_port}"

    async def _api_state(self) -> tuple[bool, str]:
        return await asyncio.to_thread(_api_check, self.node["ip"], self.api_port)


_QWEN_CODER_DESC = ("Programming, web development, front-end development, video game development: "
                    "a model specifically designed for developers.")
# The llama-server binary, on both the llama (Linux) and the vllm (macOS) node.
_LLAMA_SERVER_PATTERN = r"/opt/llama[.]cpp/llama[-]server"


class LlamaService(RemoteService):
    """llama-server on the llama node (Linux), with one of two models: they share
    the binary and the GPUs, so only one runs at a time."""

    tab, tab_label = "llama_cluster", "Llama / Linux"
    title = "Llama.cpp — gemma-4-31 | gpt-oss-120"
    description = ("Gpt: good for math&science reasoning and tool use; it writes minimalist code. "
                   "Gemma-4 is more generalist.")
    name, host_prefix = "Llama", "llama-server"
    node, node_desc = NODES.get("llama", {}), "llama node"
    workload, kill_label = "llama-server", "Kill Llama"
    api_port = int(node.get("api_port") or 8088)
    oss_api_port = int(node.get("api_port_oss") or 11000)
    ports = tuple(dict.fromkeys((api_port, oss_api_port)))
    gemma_model, oss_model = "Gemma-4-31B-it-qat", "gpt-oss-120"
    log_path = "/tmp/llama-server.log"   # written by llama-server itself (--log-file)
    pidfile = "/tmp/llama-server.pid"

    # gemma_cmd = (
    #     "/opt/llama.cpp/llama-server --model /nvme/gguf/Gemma-4-31B-it-qat/Gemma-4-31B-it-qat-UD-Q4_K_XL.gguf "
    #     "--model-draft /nvme/gguf/Gemma-4-31B-it-qat/mtp-gemma-4-31B-it-Q4_0.gguf "
    #     "--alias Gemma-4-31B-it-qat "
    #     "--spec-type draft-mtp --spec-draft-n-max 5 "
    #     "--spec-draft-device CUDA1 --spec-draft-sampling probabilistic "
    #     "--device CUDA0,CUDA1 "
    #     "--split-mode layer -ngl 999 -c 131072 -fa on -fitt 512 "
    #     "--cache-type-k q8_0 --cache-type-v q8_0 "
    #     "--parallel 1 --no-context-shift "
    #     f"--host 0.0.0.0 --port {api_port} "
    #     "--metrics --jinja --no-warmup "
    #     "--temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 --repeat-penalty 1.0 "
    #     "--seed 123456789 "
    #     '--chat-template-kwargs \'{"reasoning_effort": "medium", "preserve_thinking": true}\' '
    #     "--reasoning on "
    #     f"--log-file {log_path}"
    # )
    gemma_cmd = (
        "~/Start-Gemma-Gpt/start-gemma-4-31b.sh"
    )
    oss_cmd = (
         "~/Start-Gemma-Gpt/start-gpt-oss-120b.sh medium"
        # "/opt/llama.cpp/llama-server --device CUDA1,CUDA0 "
        # "-m /nvme/gguf/ggml.org-gpt-oss-120b/gpt-oss-120b-MXFP4.gguf "
        # "-md /nvme/gguf/ggml.org-gpt-oss-120b/eagle3-gpt-oss-120b-Q8_0.gguf "
        # "--alias gpt-oss-120b-mxfp4 "
        # "--spec-type draft-eagle3 --spec-draft-n-max 3 "
        # "-ngld all "
        # "-fa on -c 131072 -b 2048 -ub 512 "
        # "-t 16 -np 1 "
        # "--load-mode none "
        # "--moe-cache-mib 6000 "
        # f"--host 0.0.0.0 --port {oss_api_port} --fit-target 2048"
    )
    # Both share one pidfile and console log: they never run together.
    models = (
        ("Run Gemma-4-31B", gemma_model, api_port,
         _detached_start_cmd(pidfile, "$HOME/llama-server.out", gemma_cmd, node="llama")),
        ("Run gpt-oss-120B", oss_model, oss_api_port,
         _detached_start_cmd(pidfile, "$HOME/llama-server.out", oss_cmd, node="llama")),
    )
    status_command = _status_cmd(_LLAMA_SERVER_PATTERN)
    kill_command = _kill_cmd(_LLAMA_SERVER_PATTERN, pidfile=pidfile, verify=True)

    def __init__(self) -> None:
        super().__init__()
        # The model last started from this browser: what the status names while
        # the API is still loading (pgrep only sees the binary, not the model).
        self.started_model = ""
        self._live_port = self.api_port

    @property
    def fallback_model(self) -> str:
        return self.started_model or self.gemma_model

    def _run_actions(self) -> list[tuple[str, object]]:
        return [(label, functools.partial(self._run_model, model, port, cmd))
                for label, model, port, cmd in self.models]

    async def _run_model(self, model: str, port: int, cmd: str) -> None:
        if await self._launch(cmd, f"llama-server {model} (port {port})"):
            self.started_model = model
        await self.update_status()

    async def _kill(self) -> None:
        """Kill, verify, and kill once more if the server survived."""
        for attempt in (1, 2):
            ui.notify("Stopping llama-server..." if attempt == 1
                      else "Process survived, killing it again...")
            rc, out = await self.exec(self.kill_command)
            self._notify_kill(rc, out, f"the {self.node_desc}")
            await asyncio.sleep(1)
            rc, out = await self.exec(self.status_command)
            if not (rc == 0 and _is_alive(out)):
                ui.notify("Llama: llama-server stopped", type="positive")
                break
        else:
            ui.notify("llama-server still running after two kill attempts", type="negative")
        self.started_model = ""

    def _host_text(self, port: int | None = None) -> str:
        """Every API port until one answers, then just the live one."""
        ports = port or "/".join(map(str, self.ports))
        return f"{self.host_prefix}: {self.node['user']}@{self.node['ip']}:{ports}"

    async def _api_state(self) -> tuple[bool, str]:
        ip = self.node["ip"]
        results = await asyncio.gather(
            *(asyncio.to_thread(_api_check, ip, port) for port in self.ports))
        live = [port for port, (ok, _) in zip(self.ports, results) if ok]
        if live:
            self._live_port = live[0]
        return bool(live), next((model_id for _, model_id in results if model_id), "")

    async def _probe(self) -> bool:
        running = await super()._probe()
        if running:
            self.host_label.set_text(self._host_text(self._live_port))
        return running


class StrataService(RemoteService):
    """Qwen on the strata node (Linux), started by $HOME/Strata/start <variant>.

    start is only a wrapper: the service is the python server it spawns
    ("./Strata/.venv/bin/python ./Strata/serve/server.py"), which status and kill
    find by command line; the pidfile is not used."""

    tab, tab_label = "strata", "Strata / Linux"
    title = "Strata — Qwen Flash Next"
    description = _QWEN_CODER_DESC
    name, host_prefix = "Strata", "Strata node"
    node, node_desc = NODES.get("strata", {}), "strata node"
    workload, run_label, kill_label = "Qwen", "Run Qwen Flash Next", "Kill Strata"
    api_port = int(node.get("api_port") or 8000)
    log_path = "$HOME/Strata/strata.out"
    # Radio options (value: label); the first one is the default. The quant only
    # applies to the unsloth variant.
    variants = {"nvfp4": "nvfp4", "unsloth": "unsloth"}
    quants = {"ud-q4_k_xl": "Q4", "ud-q5_k_xl": "Q5"}
    pattern = r"Strata/serve/[s]erver\.py"
    status_command = _status_cmd(pattern)
    kill_command = _kill_cmd(pattern)

    def __init__(self) -> None:
        super().__init__()
        self.variant = next(iter(self.variants))
        self.quant = next(iter(self.quants))

    def _build_extras(self) -> None:
        with ui.row().classes('items-center gap-3 q-mt-sm'):
            ui.label('Model variant:').classes('text-subtitle1')
            ui.radio(self.variants, value=self.variant, on_change=self._on_variant).props('inline')
        self.quant_row = ui.row().classes('items-center gap-3 q-mt-sm')
        with self.quant_row:
            ui.label('Unsloth quant:').classes('text-subtitle1')
            ui.radio(self.quants, value=self.quant,
                     on_change=lambda e: setattr(self, 'quant', e.value)).props('inline')
        self.quant_row.visible = self.variant == "unsloth"

    def _on_variant(self, e) -> None:
        self.variant = e.value
        self.quant_row.visible = e.value == "unsloth"

    def _start_spec(self) -> tuple[str, str]:
        quant = self.quant if self.variant == "unsloth" else ""
        extra = f" --quant {quant}" if quant else ""
        cmd = _setsid_start_cmd("$HOME/Strata", "strata", f"./start {self.variant}{extra}")
        return cmd, f"Qwen Flash Next ({self.variant}{', ' + quant if quant else ''})"


class FreeTokenService(RemoteService):
    """"ft serve" with GPT-OSS on the freetoken node (Linux), pinned to one GPU."""

    tab, tab_label = "freetoken", "FreeToken / Linux"
    title = "FreeToken — gpt-oss-120"
    description = "Gpt is good for math&science reasoning and tool use; it writes minimalist code"
    name, host_prefix = "FreeToken", "FreeToken node"
    node, node_desc = NODES.get("freetoken", {}), "freetoken node"
    workload, run_label, run_what, kill_label = "FreeToken", "Run GPT-OSS", "FreeToken GPT-OSS", "Kill FreeToken"
    api_port = int(node.get("api_port") or 1919)
    fallback_model = "/nvme/freetoken/gpt-oss-120b"
    log_path = "$HOME/freetoken.out"
    # env pins the GPU while keeping a single simple command, which "exec" needs.
    serve_cmd = (
        'env CUDA_VISIBLE_DEVICES=GPU-de110751-309a-0814-d910-623f85e22787 '
        '/nvme/freetoken/venv/bin/python /nvme/freetoken/venv/bin/.ft-orig/ft serve '
        f'--model /nvme/freetoken/gpt-oss-120b --port {api_port} --moe-backend auto '
        '--max-running-requests 1 --memory-ratio 0.95 --moe-cpu-threads 14 '
        '--host 0.0.0.0 '
        '--cors-origins tauri://localhost,http://tauri.localhost,http://localhost:1420'
    )
    pattern = r"/nvme/freetoken/venv/bin/[.]ft-orig/ft serve"
    start_command = _setsid_start_cmd("$HOME", "freetoken", serve_cmd)
    status_command = _status_cmd(pattern)
    kill_command = _kill_cmd(pattern, pidfile="$HOME/freetoken.pid")


class MlxService(RemoteService):
    """Qwen on two Macs, launched by ~/mlx-jaccl/start.py as an mlx.launch job:
    rank 0 is the mlx node ("ip"), rank 1 the optional "ip_slave" (SSH user
    "user_slave", by default the rank 0 one).

    The processes are the service: it is RUNNING while either rank holds one (a
    half-dead run still needs a Kill); the API is only asked for the model id."""

    tab, tab_label, group_start = "mlx", "MLX / macOS", True
    title = "MLX-Jaccl — Qwen-3.8-27B (8bit)"
    description = _QWEN_CODER_DESC
    name, node_desc, macos = "MLX", "mlx node", True
    node = NODES.get("mlx", {})
    slave_ip = node.get("ip_slave") or ""
    slave_user = node.get("user_slave") or node.get("user") or ""
    workload, run_label, run_what, kill_label = "Qwen", "Run MLX-Jaccl Qwen", "MLX-Jaccl Qwen", "Kill MLX"
    api_port = int(node.get("api_port") or 8080)
    fallback_model = "Qwen3.8-27B"
    log_path = "$HOME/mlx-jaccl/mlx-jaccl.out"
    pidfile = "$HOME/mlx-jaccl/mlx-jaccl.pid"
    max_tokens = 262144
    # The start script (rank 0 only), mlx.launch and the leak shim (both ranks),
    # any version of it: mlx_leak_shim.py, mlx_leak_shim_v2.py (start.py's default).
    patterns = (r"mlx-jaccl/start[.]py", r"mlx[.]launch", r"mlx_leak_shim[^ /]*[.]py")
    status_command = _status_cmd(*patterns)
    # rank 0 also kills the run's session; on rank 1 only the matching pids are
    # signalled, never a group (there the shim belongs to somebody else's session).
    kill_command = _kill_cmd(*patterns, pidfile=pidfile)
    kill_command_rank1 = _kill_cmd(*patterns)

    @classmethod
    def start_cmd(cls, use_8bit: bool) -> str:
        """The 8-bit model leaves less memory for the KV cache: half the tokens.
        The script is called by its full path, so that `patterns` can find it."""
        flag = " -8bit" if use_8bit else ""
        max_tokens = cls.max_tokens // 2 if use_8bit else cls.max_tokens
        return _detached_start_cmd(
            cls.pidfile, cls.log_path,
            f"cd $HOME/mlx-jaccl && ~/mlx-jaccl/start.py{flag} -host 0.0.0.0 "
            f"-max-tokens {max_tokens} -port {cls.api_port}",
            node="mlx", append=False)

    def _start_spec(self) -> tuple[str, str]:
        return self.start_cmd(use_8bit=True), self.run_what   # always the 8-bit variant

    async def _exec_rank1(self, cmd: str) -> tuple[int, str]:
        """cmd on rank 1 (rc 255 when there is none)."""
        return await _ssh(self.slave_ip, self.slave_user, cmd, merge_stderr=True)

    def _build_extras(self) -> None:
        # rank 1 only holds the workers mlx.launch started there: its own line.
        self.slave_label = ui.label("").classes('text-caption text-grey')

    def _host_text(self) -> str:
        text = f"MLX rank 0: {self.node['user']}@{self.node['ip']}:{self.api_port}"
        if self.slave_ip:
            text += f"   |   rank 1: {self.slave_user or '?'}@{self.slave_ip}"
        return text

    async def _can_start(self) -> bool:
        # rank 1 counts too: orphaned workers there still hold the ports.
        (rc0, out0), (rc1, out1) = await asyncio.gather(
            self.exec(self.status_command), self._exec_rank1(self.status_command))
        if rc0 == 0 and _is_alive(out0):
            ui.notify("Qwen is already running on the mlx node", type="warning")
            return False
        if rc1 == 0 and _is_alive(out1):
            ui.notify(f"Qwen processes still alive on rank 1 ({self.slave_ip}); "
                      f"use {self.kill_label} first", type="warning")
            return False
        return True

    async def _kill(self) -> None:
        """Both ranks in parallel: a dead one must not delay the other."""
        ui.notify("Stopping Qwen on the mlx nodes...")
        jobs = {"rank 0": self.exec(self.kill_command)}
        if self.slave_ip:
            jobs[f"rank 1 ({self.slave_ip})"] = self._exec_rank1(self.kill_command_rank1)
        for where, (rc, out) in zip(jobs, await asyncio.gather(*jobs.values())):
            self._notify_kill(rc, out, where)

    async def _probe(self) -> bool:
        self.host_label.set_text(self._host_text())
        (rc0, out0), (rc1, out1), (_, model_id) = await asyncio.gather(
            self.exec(self.status_command), self._exec_rank1(self.status_command),
            self._api_state())
        state0, state1 = _proc_state(rc0, out0), _proc_state(rc1, out1)
        self.slave_label.set_text(
            f"rank 1 ({self.slave_user}@{self.slave_ip}): {state1}" if self.slave_ip
            else "rank 1: not configured (nodes.json ip_slave)")
        if state0 == "unreachable":
            self._set_status("node unreachable", _ORANGE)
            return False
        live = [rank for rank, state in (("rank 0", state0), ("rank 1", state1)) if state == "running"]
        if live:
            self._set_status(f"RUNNING ({'+'.join(live)})", _GREEN)
        else:
            self._set_status("STOPPED", _RED)
        self.kill_button.set_enabled(bool(live))
        # The API's own id carries the quantization (e.g. "-8bit").
        self._set_model(model_id or (self.fallback_model if live else ""))
        return bool(live)


class SplashService(RemoteService):
    """Qwen on the splash node (macOS), served by $HOME/splash/splash-start-30.sh."""

    tab, tab_label = "splash", "Splash / macOS"
    title = "Splash — Qwen3.8-27B (4bit)"
    description = _QWEN_CODER_DESC
    name, host_prefix = "Splash", "Splash node"
    node, node_desc, macos = NODES.get("splash", {}), "splash node", True
    workload, run_label, run_what, kill_label = "Qwen", "Run Splash Qwen", "Splash Qwen", "Kill Splash"
    api_port = int(node.get("api_port") or 10000)
    fallback_model = "mlx-community/Qwen3.8-27B-4bit"
    log_path = "$HOME/splash.out"
    pidfile = "$HOME/splash.pid"
    pattern = r"splash[-_]start[-_]30[.]sh"
    start_command = _detached_start_cmd(pidfile, log_path, 'bash "$HOME/splash/splash-start-30.sh"',
                                        node="splash", append=False)
    status_command = _status_cmd(pattern)
    kill_command = _kill_cmd(pattern, pidfile=pidfile)


class MacLlamaService(RemoteService):
    """llama-server on the "vllm" node (macOS), run from $HOME: the drafter path
    (-md) is relative to it."""

    tab, tab_label = "vllm", "Llama / macOS"
    title = "Llama — Muse Glimmer 30B (GGUF, dflash draft)"
    description = "Not for coding, more for generalist chats."
    name, host_prefix = "Llama", "Llama node"
    node, node_desc, macos = NODES.get("vllm", {}), "macOS llama node", True
    workload, run_label, run_what, kill_label = "llama-server", "Run Muse Glimmer", "Muse Glimmer", "Kill Llama"
    api_port = int(node.get("api_port") or 9088)
    fallback_model = "unsloth/Muse-Glimmer-30B-GGUF:UD-Q5_K_XL"
    log_path = "$HOME/llama.out"
    pidfile = "$HOME/llama.pid"
    serve_cmd = (
        'cd "$HOME" && /opt/llama.cpp/llama-server '
        '-md Muse-Glimmer-30B-GGUF-Drafter-DFlash/dflash-Muse-Glimmer-30B-Q4_K_M.gguf '
        '--offline -hf unsloth/Muse-Glimmer-30B-GGUF:UD-Q5_K_XL '
        '--spec-type draft-dflash --spec-draft-n-max 15 '
        '--reasoning on --reasoning-effort medium --jinja --metrics '
        f'--host 0.0.0.0 --port {api_port} -c 92160 -fa on'
    )
    start_command = _detached_start_cmd(pidfile, log_path, serve_cmd, node="macOS llama")
    status_command = _status_cmd(_LLAMA_SERVER_PATTERN)
    # The "cd && ..." sh wrapper never matches the pattern: the session kill gets it.
    kill_command = _kill_cmd(_LLAMA_SERVER_PATTERN, pidfile=pidfile, verify=True)


# Tab-strip order: the Linux services, then the macOS ones.
_SERVICES = (LlamaService, StrataService, FreeTokenService, MlxService, SplashService, MacLlamaService)


# =============================================================================
# Legacy "Llama.cpp" tab: front-end for start_model.py
#
# Everything it knows about the server comes from `start_model.py --json`; the
# model list is read directly through model.load_models().
# =============================================================================

# start_model.py next to this file, run by this very interpreter (and venv).
_START_MODEL = str(_HERE / "start_model.py")
_PY = sys.executable

_CTX_MIN = 8192
_CTX_STEP = 1024
# ctx_budget_gb_bf16_128k: bf16 KV cache at this context, in decimal GB.
_KV_REF_CTX = 131072
_KV_QUANT_SCALE = {"q8_0": 0.5, "q4_0": 0.25}   # KV cache size relative to bf16
_GB_TO_GIB = 1e9 / 1024**3
# Radio options where "" means "no override: keep the value from models.json".
_KVQUANT_OPTIONS = {"": "None", "q8_0": "8 bit", "q4_0": "4 bit"}
_REAS_OPTIONS = {"": "None", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}
_STREAM_MAX_LINES = 5000

# The four values of the "Samplers:" label, in display order, with the hint
# shown by its hover tooltip.
_SAMPLER_HINTS = (
    ("temperature", "scales the logits — below 1.00 sharpens the model's distribution, "
                    "above 1.00 flattens it"),
    ("top_p", "nucleus sampling: keeps the most likely tokens up to this cumulative probability"),
    ("top_k", "keeps only the k most likely tokens (0 disables the cut)"),
    ("min_p", "drops every token below this fraction of the top token's probability"),
)


def _samplers_description(values: tuple[str, ...], header: str) -> str:
    return "\n".join([header] + [f"{name} = {value}  —  {hint}"
                                 for (name, hint), value in zip(_SAMPLER_HINTS, values)])


async def _run_json(args: list[str]) -> dict | None:
    """Run start_model.py with --json and return the parsed stdout, or None.

    The return code is ignored: --server-status exits 1 to mean "not running",
    a valid answer. Only unparseable output is a failure."""
    out, rc = await _capture([_PY, _START_MODEL, *args, "--json"])
    if not out.strip():
        logger.error(f"start_model.py {' '.join(args)} produced no output (rc={rc})")
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        logger.error(f"start_model.py {' '.join(args)} returned invalid JSON: {e}")
        return None


# Loading the models is slow (an SSH round-trip to stat the model files), so it
# runs once at boot and then only on "Recheck models"; every page reads the cache.
_models: dict[str, Model] | None = None
_models_lock = asyncio.Lock()


async def fetch_models(force: bool = False) -> tuple[dict[str, Model] | None, bool]:
    """Return ({name: Model}, ok). Reloads only when the cache is empty or
    force=True; a failed reload keeps the previous list."""
    global _models
    async with _models_lock:
        if _models is None or force:
            try:
                models = await asyncio.to_thread(
                    load_models_from_config,
                    Path(settings.MODELS_JSON),
                    Path(settings.RPC_JSON),
                    remote_host=settings.LLAMA_SERVER_HOST,
                    remote_user=settings.LLAMA_SERVER_USER,
                    check_remote_file=True,
                )
            except Exception:
                # A malformed models.json or an unreachable host must not kill the console.
                logger.exception("Could not load the model list")
                return _models, False
            _models = {m.model_name: m for m in models}
        return _models, True


class LegacyLlamaPanel:
    """The original "Llama.cpp" tab: model picker, context/KV/temperature
    controls, START/STOP, RPC server kill and live server log."""

    tab, tab_label, group_start = "llama", "Llama.cpp", False

    def __init__(self) -> None:
        self.models: dict[str, Model] = {}
        self.tab_element: ui.tab | None = None
        self.log_task: asyncio.Task | None = None
        self.status_model_name = ""
        self.rpc_servers_row: ui.row | None = None
        self._status_lock = asyncio.Lock()
        self._start_busy = False
        self._stop_busy = False

    # ------------------------------------------------------------------ UI ---
    def build(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.column().classes('w-full max-w-2xl gap-1 q-mb-4 pr-4'):
                self.status_server_label = ui.label("Checking inference server status...") \
                    .style('font-size: 1.2rem; font-weight: 850; white-space: nowrap;')
                # The build string is long: this one may wrap.
                self.status_build_label = ui.label("").classes('font-mono legacy-status') \
                    .style('white-space: pre-wrap;')
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.status_model_label = ui.label("").classes('font-mono legacy-status')
                    self.status_model_copy = ui.icon('content_copy').tooltip('Copy model name') \
                        .classes('model-copy-icon cursor-pointer invisible')
                    self.status_model_copy.on(
                        'click', lambda: _copy_to_clipboard(self.status_model_name, "model name"))
                self.status_ctx_label = ui.label("").classes('font-mono legacy-status')
                # Values only: plain cursor, but the same hover description.
                with ui.element('div').classes('cursor-tip-host cursor-tip-plain'):
                    self.status_samplers_label = ui.label("").classes('font-mono legacy-status')
                    self.status_samplers_tip = ui.label("").classes('cursor-tip-text')
                with ui.row().classes('items-center gap-2 q-mt-md'):
                    ui.button("Refresh", on_click=self.refresh).props('outline small')
                    # Only while a model is live.
                    self.stop_button = ui.button("STOP", on_click=self.stop_server) \
                        .props('outline small color=red')
                    self.stop_button.set_visibility(False)

            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("Model Control").classes('text-h6')
                # no-wrap + a shrinkable select: a long name never pushes START away.
                with ui.row().classes('w-full items-center q-mb-md no-wrap'):
                    self.model_dropdown = ui.select(
                        options={"": "Loading models..."}, value="", label="Loading models...",
                        on_change=self._on_model_change,
                    ).props('disable').classes('flex-grow min-w-0 model-select')
                    self.start_button = ui.button(
                        "START", on_click=self.start_selected_model).props('color=green')

                with ui.element('div').classes('cursor-tip-host'):
                    self.samplers_label = ui.label()
                    self.samplers_tip = ui.label().classes('cursor-tip-text')
                self._set_samplers()

                ui.button("Recheck models", on_click=self.recheck_models).props('small outline')

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
                        on_change=self._refresh_ctx,
                    ).classes('flex-grow').props('color=green')
                    with ui.row().classes('items-center gap-3'):
                        ui.label('KV Quant:').classes('text-subtitle1')
                        self.kvquant_radio = ui.radio(
                            _KVQUANT_OPTIONS, value="", on_change=self._update_memory).props('inline')
                    self.mem_label = ui.label("Memory: —").classes('text-subtitle1')

                with ui.column().classes('w-full q-mt-sm'):
                    self.temp_label = ui.label("Temperature: —").classes('text-subtitle1')
                    self.temp_slider = ui.slider(
                        min=0, max=1.0, value=1.0, step=0.01, on_change=self._refresh_temp,
                    ).classes('flex-grow').props('color=orange')

            ui.label("Server Logs").classes('text-h6 q-mt-lg')
            with ui.row().classes('w-full items-center q-mb-sm'):
                ui.button("Connect to inference server logs",
                          on_click=self.start_log_streaming).props('small')
                self.stop_log_button = ui.button(
                    "Stop log streaming", on_click=self.stop_log_streaming).props('small outline')
                self.stop_log_button.disable()
                ui.button("Clear Logs", on_click=lambda: self.log_window.clear()).props('small outline')
            self.log_window = ui.log(max_lines=_STREAM_MAX_LINES).classes(
                'w-full h-[600px] bg-black text-green-400 font-mono text-xs custom-log')

    # ------------------------------------------------------------ samplers ---
    def _set_samplers(self, values: tuple[str, str, str, str] | None = None) -> None:
        """The selected model's samplers (temperature, top_p, top_k, min_p, already
        formatted), or placeholders while no model is selected."""
        shown = values or ("--", "--", "--", "--")
        self.samplers_label.set_text("Samplers: " + ";".join(shown))
        description = _samplers_description(
            shown, "Sampling parameters of the selected model, in the order shown:")
        if values is None:
            description += "\n(the values appear once a model is selected)"
        self.samplers_tip.set_text(description)

    def _set_status_samplers(self, values: tuple[str, str, str, str] | None = None) -> None:
        """What the running server was started with; blank (nothing to hover)
        while there is no ready server."""
        if values is None:
            self.status_samplers_label.set_text("")
            self.status_samplers_tip.set_text("")
            return
        self.status_samplers_label.set_text(" - Samplers: " + ";".join(values))
        self.status_samplers_tip.set_text(_samplers_description(
            values, "Sampling parameters the running server was started with:"))

    # -------------------------------------------------------------- models ---
    async def load_models(self, force: bool = False) -> bool:
        models, ok = await fetch_models(force)
        if models is None:
            self._show_no_models()
            ui.notify("Could not load the model list", type="negative")
            return False
        if not ok:
            ui.notify("Could not reload the model list — keeping the previous one",
                      type="negative")
        self.models = models
        if not models:
            self._show_no_models()
            return ok

        # The option value stays the bare model name; only the label has the size.
        options = {
            name: f"{name} ({int(m.size_gib) if m.size_gib is not None else '?'} GiB"
                  f" - {len(m.rpcservers or [])} RPC)"
            for name, m in models.items()
        }
        previous = self.model_dropdown.value
        selected = previous if previous in models else next(iter(models))
        self.model_dropdown.props(remove='disable')
        self.model_dropdown.props('label="Model selector"')
        self.model_dropdown.set_options(options, value=selected)
        if selected == previous:
            # No change event fired: apply the model spec explicitly.
            self._apply_model_spec(selected)
        return ok

    def _show_no_models(self) -> None:
        self.model_dropdown.set_options({"": "No models found"}, value="")
        self.model_dropdown.props('label="No models found"')

    async def recheck_models(self) -> None:
        """Reload the models on demand: the only trigger besides app boot."""
        self.model_dropdown.props('disable label="Reloading models..."')
        ui.notify("Rechecking models...")
        if await self.load_models(force=True):
            ui.notify("Model list updated", type="positive")

    def _on_model_change(self, e) -> None:
        if e.value:
            self._apply_model_spec(e.value)

    def _apply_model_spec(self, name: str) -> None:
        """Point the controls at the selected model's bounds and defaults."""
        model = self.models.get(name)
        if model is None:
            return
        native_ctx = model.native_ctx
        # A slider with min > max cannot be dragged at all.
        ctx_min = min(_CTX_MIN, native_ctx)
        self.ctx_slider.props['min'] = ctx_min
        self.ctx_slider.props['max'] = native_ctx
        # Through set_value(): pushing the value as a prop breaks dragging.
        self.ctx_slider.set_value(min(max(settings.DEFAULT_CTX, ctx_min), native_ctx))
        self._refresh_ctx()

        # The configured temperature is also the slider maximum: it can only be lowered.
        temp = model.temperature
        self.temp_slider.props['min'] = 0
        self.temp_slider.props['max'] = temp
        self.temp_slider.set_value(temp)
        self._refresh_temp()
        self._set_samplers((f"{temp:.2f}", f"{model.top_p:.2f}", f"{model.top_k}", f"{model.min_p:.2f}"))

        self.mtp_checkbox.set_enabled(model.mtp)
        self.mtp_checkbox.set_text(
            "Force No-MTP (this is a mtp-enabled model)" if model.mtp else "Force No-MTP")
        # Preselect the model's own REAS level when it is one of the options.
        self.reas_radio.set_value(model.reasoning if model.reasoning in _REAS_OPTIONS else "")
        self._update_rpc_labels()

    def _update_rpc_labels(self) -> None:
        """One label per RPC server (known to rpc.json) used by the selected model."""
        if self.rpc_servers_row is None:
            return
        self.rpc_servers_row.clear()
        model = self.models.get(self.model_dropdown.value)
        with self.rpc_servers_row:
            for rpc in (model.rpcservers or []) if model else []:
                if rpc.name in RPC_SERVERS:
                    ui.label(rpc.name)

    # ------------------------------------------------------------- sliders ---
    def _refresh_ctx(self) -> None:
        self.ctx_label.set_text(f"Context: {int(self.ctx_slider.value):,}"
                                f"  (max: {int(self.ctx_slider.props['max']):,})")
        self._update_memory()

    def _refresh_temp(self) -> None:
        self.temp_label.set_text(f"Temperature: {float(self.temp_slider.value):.2f}"
                                 f"  (max: {float(self.temp_slider.props['max']):.2f})")

    def _update_memory(self) -> None:
        """Estimated memory, in GiB: model size + KV cache.

        The KV cache is the model's ctx_budget_gb_bf16_128k scaled linearly to the
        selected context, then by the KV quant (the radio, or else the model's own
        KVQUANT from models.json)."""
        model = self.models.get(self.model_dropdown.value)
        if model is None or model.ctx_budget_gb_bf16_128k is None:
            self.mem_label.set_text("Memory: —")
            return
        quant = self.kvquant_radio.value or (model.kvquant or "")
        kv_gib = (model.ctx_budget_gb_bf16_128k * _GB_TO_GIB
                  * int(self.ctx_slider.value) / _KV_REF_CTX
                  * _KV_QUANT_SCALE.get(quant, 1.0))
        if model.size_gib is None:
            self.mem_label.set_text(f"Memory: ? GiB (model ? + KV cache {kv_gib:.1f} GiB)")
            return
        self.mem_label.set_text(f"Memory: {model.size_gib + kv_gib:.1f} GiB  "
                                f"(model {model.size_gib:.1f} + KV cache {kv_gib:.1f} GiB)")

    # -------------------------------------------------------------- status ---
    async def update_status(self) -> None:
        """Polled by a timer: skipped while the previous check is in flight."""
        if self._status_lock.locked():
            return
        async with self._status_lock:
            await self._update_status_once()

    async def _update_status_once(self) -> None:
        info = await _run_json(["--server-status"])
        if info is None or not info.get("transport", True):
            # No valid answer: the state is unknown, which is NOT "not running".
            self._set_server_status("UNREACHABLE", _ORANGE)
            self._show_server_details(running=False, info=None)
            self.stop_button.set_visibility(False)
            return

        running = bool(info.get("running"))
        # Never start over a live server (START re-checks at click time too).
        self.start_button.set_enabled(not running)
        self.stop_button.set_visibility(running)
        self._set_server_status("RUNNING" if running else "NOT RUNNING",
                                _GREEN if running else _RED)
        self._show_server_details(running, info if running and info.get("ready") else None)

    def _set_server_status(self, text: str, color: str) -> None:
        self.status_server_label.set_text(f"Server Status: {text}")
        self.status_server_label.style(f"color: {color}")

    def _show_server_details(self, running: bool, info: dict | None) -> None:
        """Details of a ready server (info), or of none."""
        if info is None:
            self.status_model_name = ""
            self.status_build_label.set_text("")
            self.status_model_label.set_text("Model: (Loading...)" if running else "")
            self.status_ctx_label.set_text("")
            self._set_status_samplers()
        else:
            build = str(info.get("build_info") or "").strip()
            self.status_build_label.set_text(f" - Build   : llama.cpp/{build}" if build else "")
            self.status_model_name = str(info.get("model") or "").strip()
            self.status_model_label.set_text(f" - Model   : {self.status_model_name}")
            self.status_ctx_label.set_text(f" - Context : {str(info['ctx']).strip()} tokens")
            # Rounded: the engine reports 0.6 as its float32 round-trip 0.6000000238418579.
            self._set_status_samplers((f"{float(info['temperature']):.1f}", f"{float(info['top_p']):.2f}",
                                       f"{int(info['top_k'])}", f"{float(info['min_p']):.2f}"))
        _set_class(self.status_model_copy, 'invisible', not self.status_model_name)

    async def refresh(self) -> None:
        await self.update_status()
        ui.notify("Status updated")

    # ------------------------------------------------------------ commands ---
    async def _stream(self, argv: list[str]) -> int:
        """Run argv, pushing its output into the log window line by line.

        The child gets its own process group, so stopping it also stops what it
        spawned (start_model.py's remote `ssh ... tail -F`)."""
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True)
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
        # A stop can take long (SIGKILL escalation, RPC kills over nested SSH):
        # a START in the meantime would be refused as "already running".
        if self._stop_busy:
            ui.notify("The server stop is still in progress — wait for it to finish",
                      type="warning")
            return
        name = self.model_dropdown.value
        model = self.models.get(name)
        if model is None:
            ui.notify("Please select a model first", type="warning")
            return

        self._start_busy = True
        self.start_button.props('loading')
        try:
            # Fresh check at click time: the polled status can be stale.
            info = await _run_json(["--server-status"])
            if info is None or not info.get("transport", True):
                # Unknown is not "free": starting could land over a live server.
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

            ctx_value = int(self.ctx_slider.value)
            args = [name, "--override-ctx", str(ctx_value)]
            if self.kvquant_radio.value:
                args += ["--override-kvquant", self.kvquant_radio.value]
            if self.reas_radio.value:
                args += ["--override-reas", self.reas_radio.value]
            # Only when the slider actually moved away from the model's value.
            temp_value = float(self.temp_slider.value)
            if abs(temp_value - model.temperature) > 0.001:
                args += ["--override-temp", f"{temp_value:.4f}"]
            if self.mtp_checkbox.value:
                args += ["--force-no-mtp"]
            if self.no_reasoning_checkbox.value:
                args += ["--disable-thinking"]

            ui.notify(f"Starting model {name} (ctx={ctx_value})...")
            # -u: unbuffered, so the lines arrive as they happen.
            cmd = [_PY, "-u", _START_MODEL, *args]
            self.log_window.push(f"--- Starting model: {shlex.join(cmd)} ---")
            try:
                rc = await self._stream(cmd)
            except OSError as e:
                self.log_window.push(f"Launch Error: {e}")
                ui.notify(f"Error starting model: {e}", type="negative")
            else:
                if rc == 0:
                    ui.notify(f"Model {name} started successfully", type="positive")
                else:
                    ui.notify(f"Error starting model (exit {rc})", type="negative")
            await self.update_status()
        finally:
            self._start_busy = False
            self.start_button.props(remove='loading')

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
        name = self.model_dropdown.value
        model = self.models.get(name)
        if model is None:
            ui.notify("Please select a model first", type="warning")
            return
        # Exactly the selected model's RPC servers: --override-rpc replaces its
        # list before --kill-rpc-server acts on it.
        selected = [rpc.name for rpc in model.rpcservers or [] if rpc.name in RPC_SERVERS]
        if not selected:
            ui.notify("This model has no RPC servers to kill", type="warning")
            return
        ui.notify(f"Killing RPC server(s): {', '.join(selected)}...")
        out, rc = await _capture([_PY, _START_MODEL, name, "--override-rpc", ",".join(selected),
                                  "--kill-rpc-server"])
        if rc == 0:
            ui.notify("RPC servers killed", type="positive")
        else:
            ui.notify(f"Error killing RPC servers: {out.strip()}", type="negative")

    # ------------------------------------------------------------ log tail ---
    async def _stream_logs(self) -> None:
        try:
            await self._stream([_PY, "-u", _START_MODEL, "--tail-log", "-n", "1000"])
        except asyncio.CancelledError:
            self.log_window.push("--- log streaming stopped ---")
            raise
        except OSError as e:
            self.log_window.push(f"Log Error: {e}")
        finally:
            self.stop_log_button.disable()

    def start_log_streaming(self) -> None:
        # One tail at a time: another click would stack a `tail -F` nothing can stop.
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
        """Stop the tail when the browser tab goes away."""
        if self.log_task is not None and not self.log_task.done():
            self.log_task.cancel()


# =============================================================================
# Page
# =============================================================================

_HEAD_HTML = '''
<style>
/* The vertical tab selector: a full-height strip on the left. */
.tab-strip {
    min-width: 7rem;
    border-right: 1px solid rgba(255, 255, 255, 0.18);
}
/* A thin rule above the first macOS tab splits the strip into OS groups. */
.tab-strip .q-tab.tab-group-start {
    position: relative;
    margin-top: 10px;
}
.tab-strip .q-tab.tab-group-start::before {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    top: -6px;
    height: 1px;
    background: rgba(255, 255, 255, 0.18);
}
/* Green bullet right before the label of every tab whose service is RUNNING. */
.q-tab.running-bullet .q-tab__content {
    position: relative;
}
.q-tab.running-bullet .q-tab__content::before {
    content: '';
    position: absolute;
    right: 100%;
    margin-right: 6px;
    top: 50%;
    transform: translateY(-50%);
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 4px rgba(0, 255, 136, 0.7);
}
.service-description {
    font-style: italic;
    font-size: 0.8rem;
    opacity: 0.8;
}
.service-status {
    font-size: 1rem;
    font-weight: 700;
    transition: color 0.3s ease-in-out;
}
.model-name {
    font-family: Consolas, Menlo, "Courier New", monospace;
    font-size: calc(1em + 1pt);
}
.legacy-status {
    font-size: 0.9rem;
    font-weight: 600;
    white-space: pre;
}
/* Let long model names truncate instead of stretching the select (a flex item
   refuses to shrink below its content unless told so). */
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
/* Copy icon next to a model name: hidden until the row is hovered. */
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
/* Cursor-following tooltip: a .cursor-tip-host shows the text of its hidden
   .cursor-tip-text child in #cursor-tip (see the script below). */
.cursor-tip-host {
    width: fit-content;
    cursor: help;
}
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
    white-space: pre-line;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    /* Never let the tooltip take the pointer it follows, or it flickers away. */
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
    const OFFSET_X = 16;
    const OFFSET_Y = 18;
    const MARGIN = 8;
    let tip = null;
    let host = null;

    // Created lazily: <body> does not exist yet while the head runs.
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
        // Position first, then show: no flash in the top-left corner.
        place(event);
        tip.classList.add('visible');
    });

    document.addEventListener('mousemove', (event) => {
        if (!host) return;
        if (!host.isConnected) { hide(); return; }
        place(event);
    });

    document.addEventListener('mouseout', (event) => {
        if (host && !host.contains(event.relatedTarget)) hide();
    });

    // Scrolling moves the host out from under a pointer that never moved.
    document.addEventListener('scroll', hide, true);
    window.addEventListener('blur', hide);
})();
</script>
'''
ui.add_head_html(_HEAD_HTML, shared=True)

# The tab each browser was last on, so a reload (or a GUI restart) reopens it.
# app.storage.user is per browser (a signed cookie) and persisted on disk.
_ACTIVE_TAB_KEY = "active_tab"
_VALID_TABS = tuple(s.tab for s in _SERVICES) + ((LegacyLlamaPanel.tab,) if SHOW_LEGACY_LLAMA_TAB else ())
_DEFAULT_TAB = LlamaService.tab

# app.storage.user needs a storage_secret: generated once and kept next to this
# file, so the signed cookies stay valid across restarts.
_STORAGE_SECRET_FILE = _HERE / ".storage_secret"


def _storage_secret() -> str:
    try:
        return _STORAGE_SECRET_FILE.read_text().strip()
    except OSError:
        secret = base64.urlsafe_b64encode(os.urandom(32)).decode()
        try:
            _STORAGE_SECRET_FILE.write_text(secret)
            os.chmod(_STORAGE_SECRET_FILE, 0o600)
        except OSError:
            pass  # still usable for this run
        return secret


def _remembered_tab() -> str:
    tab = app.storage.user.get(_ACTIVE_TAB_KEY, _DEFAULT_TAB)
    return tab if tab in _VALID_TABS else _DEFAULT_TAB


def _remember_tab(tab: str) -> None:
    app.storage.user[_ACTIVE_TAB_KEY] = tab


@ui.page('/')
def index() -> None:
    ui.dark_mode().enable()
    legacy = LegacyLlamaPanel() if SHOW_LEGACY_LLAMA_TAB else None
    panels = ([legacy] if legacy else []) + [cls() for cls in _SERVICES]

    with ui.column().classes('w-full h-screen items-stretch p-0 m-0 gap-0'):
        with ui.column().classes('items-center q-mb-sm'):
            ui.label("LLM Inference Console").classes('text-h5')
            ui.label("by Alvise Dorigo").classes('text-h5')
            ui.link(_REPO_URL, _REPO_URL).classes('text-caption no-underline')

        with ui.row().classes('w-full flex-grow items-stretch p-0 m-0 gap-0'):
            with ui.column().classes('tab-strip items-stretch p-2'):
                with ui.tabs(on_change=lambda e: _remember_tab(e.value)) \
                        .props('vertical no-caps').classes('w-full') as tabs:
                    for panel in panels:
                        panel.tab_element = ui.tab(panel.tab, label=panel.tab_label)
                        if panel.group_start:
                            panel.tab_element.classes('tab-group-start')

            with ui.tab_panels(tabs, value=_remembered_tab()).classes('flex-grow p-2 overflow-auto'):
                for panel in panels:
                    with ui.tab_panel(panel.tab_element):
                        panel.build()

    # Every panel is polled, not just the visible one: the tab bullets need them all.
    for panel in panels:
        ui.timer(_POLL_SECONDS, panel.update_status)
    if legacy:
        ui.timer(0, legacy.load_models, once=True)
        context.client.on_disconnect(legacy.cleanup)


if SHOW_LEGACY_LLAMA_TAB:
    # Scan the model list at boot (NiceGUI runs it as a background task): pages
    # opened before it finishes just wait for the same scan.
    app.on_startup(lambda: fetch_models())

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title=settings.UI_TITLE, port=settings.UI_PORT, host="0.0.0.0",
           reload=False, show=False, storage_secret=_storage_secret())
