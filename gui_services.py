"""The service tabs: one class per inference service the console drives.

A service is a fixed workload on a remote node (see nodes.json): the commands it
runs there come from gui_shell, they travel over gui_util.ssh_run (which logs
them), and the widgets they drive come from gui_widgets. The panel that
assembles the tabs is gui_page.

Each class is instantiated once per browser tab; everything on the class body is
static configuration, shared by all browsers.
"""

import asyncio
import functools

from nicegui import ui

from gui_config import NODES, GREEN, ORANGE, RED
from gui_logging import logger
from gui_shell import (detached_start_cmd, kill_cmd, log_tail_cmd, setsid_start_cmd,
                       status_cmd)
from gui_util import api_check, is_alive, last_line, ssh_run
from gui_widgets import copy_to_clipboard, exclusive, set_class

_QWEN_CODER_DESC = ("Programming, web development, front-end development, video game development: "
                    "a model specifically designed for developers.")
# The llama-server binary, on the llama (Linux) node.
_LLAMA_SERVER_PATTERN = r"/opt/llama[.]cpp/llama[-]server"


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
        self._status_text = ""       # last status shown, to log only the changes
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

    async def exec(self, cmd: str, *, what: str = "", routine: bool = False) -> tuple[int, str]:
        """Run cmd on this service's node (logged as <what> on <node>).

        routine=True is for the calls a timer repeats: they log at DEBUG, so a
        journal kept at INFO only shows what the user asked for."""
        return await ssh_run(self.node.get("ip", ""), self.node.get("user", ""), cmd,
                             merge_stderr=self.macos, what=what or self.name, routine=routine)

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
                    self.model_copy.on('click', lambda: copy_to_clipboard(self.model_name, "model name"))
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
                    update.on_click(lambda: exclusive(
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
        button.on_click(lambda: exclusive(
            self._action_lock, f"A {self.name} action is already in progress", button, action))
        return button

    def _set_status(self, text: str, color: str) -> None:
        # A change is news worth the journal; the polls repeating the same state
        # are not (the ssh command itself is already logged at DEBUG).
        if text != self._status_text:
            logger.info("%s: status %s -> %s", self.name, self._status_text or "(unknown)", text)
            self._status_text = text
        self.status_label.set_text(f"{self.name}: {text}")
        self.status_label.style(f"color: {color}")

    def _set_model(self, name: str) -> None:
        self.model_name = name
        self.model_label.set_text(name)
        set_class(self.model_copy, 'invisible', not name)

    def _check_configured(self) -> bool:
        if not self.configured:
            logger.warning("%s: the %s is not configured (nodes.json)", self.name, self.node_desc)
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
        logger.info("%s: starting %s on the %s", self.name, what, self.node_desc)
        ui.notify(f"Starting {what} on the {self.node_desc}...")
        rc, out = await self.exec(start_cmd, what=f"start {what}")
        if rc == 0:
            logger.info("%s: %s started (%s)", self.name, what, out.strip())
            ui.notify(f"{what} started on the {self.node_desc} ({last_line(out)})", type="positive")
        else:
            logger.error("%s: could not start %s (rc=%s): %s", self.name, what, rc, out.strip())
            ui.notify(f"Error starting {what} on the {self.node_desc} (rc={rc}): {out.strip()}",
                      type="negative")
        return rc == 0

    async def _can_start(self) -> bool:
        """Refuse a second launch over a live one: it would only fight the running
        one for the API port and the GPU/memory. The status poll may be stale."""
        rc, out = await self.exec(self.status_command, what=f"pre-start check of {self.workload}")
        if rc == 0 and is_alive(out):
            logger.warning("%s: refusing to start %s, %s is already running (%s)",
                           self.name, self.workload, self.workload, out.strip())
            ui.notify(f"{self.workload} is already running on the {self.node_desc}; "
                      f"use {self.kill_label} first", type="warning")
            return False
        return True

    async def kill(self) -> None:
        if self._check_configured():
            await self._kill()
        await self.update_status()

    async def _kill(self) -> None:
        logger.info("%s: stopping %s on the %s", self.name, self.workload, self.node_desc)
        ui.notify(f"Stopping {self.workload} on the {self.node_desc}...")
        rc, out = await self.exec(self.kill_command, what=f"kill {self.workload}")
        self._notify_kill(rc, out, f"the {self.node_desc}")

    def _notify_kill(self, rc: int, out: str, where: str) -> None:
        out = out.strip()
        if rc == 0:
            logger.info("%s: %s stopped on %s (%s)", self.name, self.workload, where, out)
            ui.notify(f"{self.workload} stopped on {where} ({out})", type="positive")
        elif rc == 3:
            logger.info("%s: no %s process to kill on %s", self.name, self.workload, where)
            ui.notify(f"No {self.workload} process found on {where} (already stopped?)",
                      type="warning")
        elif rc == 4:
            logger.error("%s: %s survived the kill on %s: %s", self.name, self.workload, where, out)
            ui.notify(f"{self.workload} did not go down on {where}: {out}", type="negative")
        elif rc == 255:
            logger.error("%s: could not reach %s to kill %s", self.name, where, self.workload)
            ui.notify(f"Could not reach {where}", type="negative")
        else:
            logger.error("%s: error stopping %s on %s (rc=%s): %s",
                         self.name, self.workload, where, rc, out)
            ui.notify(f"Error stopping {self.workload} on {where} (rc={rc}): {out}",
                      type="negative")

    # ----------------------------------------------------------------- log ---
    async def update_log(self) -> None:
        if not self._check_configured():
            return
        rc, out = await self.exec(log_tail_cmd(self.log_path), what=f"log tail of {self.log_name}")
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
                    self._set_status("node not configured (nodes.json)", ORANGE)
            except Exception:
                logger.debug("%s status check failed", self.name, exc_info=True)
                self._set_status("UNKNOWN (status check failed)", ORANGE)
            finally:
                if self.tab_element is not None:
                    set_class(self.tab_element, 'running-bullet', running)

    async def _probe(self) -> bool:
        """Refresh the status widgets; return whether the service is RUNNING."""
        self.host_label.set_text(self._host_text())
        (rc, out), (api_ok, model_id) = await asyncio.gather(
            self.exec(self.status_command, what=f"status probe of {self.workload}", routine=True),
            self._api_state())
        if rc == 255:
            self._set_status("node unreachable", ORANGE)
            return False
        proc_alive = is_alive(out)
        if api_ok:
            self._set_status("RUNNING", GREEN)
        elif proc_alive:
            self._set_status("STARTING (API not answering yet)", ORANGE)
        else:
            self._set_status("STOPPED", RED)
        self.kill_button.set_enabled(api_ok or proc_alive)
        self._set_model(model_id or (self.fallback_model if proc_alive else ""))
        return api_ok

    def _host_text(self) -> str:
        return f"{self.host_prefix}: {self.node['user']}@{self.node['ip']}:{self.api_port}"

    async def _api_state(self) -> tuple[bool, str]:
        return await asyncio.to_thread(api_check, self.node["ip"], self.api_port)


class LlamaService(RemoteService):
    """llama-server on the llama node (Linux).

    Every green button names a model and opens a small dialog asking for the
    quantization; the choice picks the script ~/Llama/start-<label>-<quant>.sh,
    launched detached on the remote node (python3 start_new_session, so the
    llama-server it spawns outlives the SSH session that started it). The
    models share the binary and the GPUs, so only one runs at a time.

    RUNNING means the API answering /v1/models on one of the candidate ports:
    the poll (gui_page's timer, every POLL_SECONDS) GETs it on the node itself
    over the LAN, and the JSON also names the model on display, which goes to
    the label under the title."""

    tab, tab_label = "llama_cluster", "Llama.cpp / Linux"
    title = "Llama.cpp — Gemma-4-31b | Qwen3.8-27b"
    description = ("Gemma-4 is more generalist; Qwen3.8 is a coder model: "
                   "programming, web and game development.")
    name, host_prefix = "Llama", "llama-server"
    node, node_desc = NODES.get("llama", {}), "llama node"
    workload, kill_label = "llama-server", "Kill Llama"
    api_port = int(node.get("api_port") or 8088)
    oss_api_port = int(node.get("api_port_oss") or 11000)
    ports = tuple(dict.fromkeys((api_port, oss_api_port)))
    log_path = "$HOME/llama-server.out"   # written by llama-server itself (--log-file)
    pidfile = "$HOME/llama-server.pid"
    launch_log = "$HOME/llama-server.out"
    # Where the start scripts live on the remote node.
    start_dir = "$HOME/Llama"

    # Green run buttons: label -> the quantizations its dialog offers. The
    # launched script is start-<label>-<quantization>.sh under start_dir.
    run_buttons = {
        "Gemma-4-31b-it": ("ud-q8_k_xl", "qat-ud-q4_k_xl"),
        "Qwen3.8-27b": ("ud-q4_k_xl", "ud-q8_k_xl"),
    }

    status_command = status_cmd(_LLAMA_SERVER_PATTERN)
    kill_command = kill_cmd(_LLAMA_SERVER_PATTERN, pidfile=pidfile, verify=True)

    def __init__(self) -> None:
        super().__init__()
        # The model last started from this browser: what the status names while
        # the API is still loading (pgrep only sees the binary, not the model).
        self.started_model = ""
        self._live_port = self.api_port

    @property
    def fallback_model(self) -> str:
        return self.started_model

    def _run_actions(self) -> list[tuple[str, object]]:
        return [(label, functools.partial(self._open_quant_dialog, label, quants))
                for label, quants in self.run_buttons.items()]

    async def _open_quant_dialog(self, label: str, quants: tuple[str, ...]) -> None:
        """Small dialog asking which quantization of <label> to start."""
        logger.debug("%s: asking which quantization of %s to run (%s)",
                     self.name, label, ", ".join(quants))
        with ui.dialog() as dialog, ui.card().classes('p-4 gap-2'):
            ui.label(f"{label}: choose the quantization").classes('text-subtitle1')
            choice = ui.radio(list(quants), value=quants[0])
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button("Cancel", on_click=dialog.close).props('flat outline')
                ui.button("Start", color="green") \
                    .on_click(lambda: self._start_quantized(dialog, label, choice))
        dialog.open()

    async def _start_quantized(self, dialog: ui.dialog, label: str, choice: ui.radio) -> None:
        """Launch start-<label>-<quantization>.sh detached, then refresh status."""
        quant = choice.value
        script = f"start-{label}-{quant}.sh"
        dialog.close()
        logger.info("%s: user chose model=%s quantization=%s -> %s/%s",
                    self.name, label, quant, self.start_dir, script)
        cmd = detached_start_cmd(self.pidfile, self.launch_log,
                                 f'bash "{self.start_dir}/{script}"', node="llama")
        if await self._launch(cmd, f"llama-server {label} ({quant})"):
            self.started_model = label
        await self.update_status()

    async def _kill(self) -> None:
        """Kill, verify, and kill once more if the server survived."""
        logger.info("%s: stopping %s on the %s (two attempts)", self.name, self.workload,
                    self.node_desc)
        for attempt in (1, 2):
            ui.notify("Stopping llama-server..." if attempt == 1
                      else "Process survived, killing it again...")
            rc, out = await self.exec(self.kill_command, what=f"kill {self.workload} (attempt {attempt})")
            self._notify_kill(rc, out, f"the {self.node_desc}")
            await asyncio.sleep(1)
            rc, out = await self.exec(self.status_command, what=f"post-kill check of {self.workload}")
            if not (rc == 0 and is_alive(out)):
                ui.notify("Llama: llama-server stopped", type="positive")
                break
        else:
            logger.error("%s: %s still running after two kill attempts", self.name, self.workload)
            ui.notify("llama-server still running after two kill attempts", type="negative")
        self.started_model = ""

    def _host_text(self, port: int | None = None) -> str:
        """Every API port until one answers, then just the live one."""
        ports = port or "/".join(map(str, self.ports))
        return f"{self.host_prefix}: {self.node['user']}@{self.node['ip']}:{ports}"

    async def _api_state(self) -> tuple[bool, str]:
        """GET /v1/models on the node, on every candidate port.

        A listening llama-server answers with the JSON of the model in
        execution: a port whose answer is that JSON is live, and the id it
        names is what the status labels as running."""
        ip = self.node["ip"]
        results = await asyncio.gather(
            *(asyncio.to_thread(api_check, ip, port) for port in self.ports))
        live = [port for port, (ok, _) in zip(self.ports, results) if ok]
        if live:
            self._live_port = live[0]
        model_id = next((mid for _, mid in results if mid), "")
        if live:
            logger.debug("%s: /v1/models answering on port %s: serving %s",
                         self.name, live[0], model_id or "(no id)")
        return bool(live), model_id

    async def _probe(self) -> bool:
        running = await super()._probe()
        if running:
            self.host_label.set_text(self._host_text(self._live_port))
        return running


class StrataService(RemoteService):
    """Qwen on the strata node (Linux), started by $HOME/Strata/run-<variant>.sh.

    the run script is only a wrapper: the service is the python server it spawns
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
    # Radio options (value: label); the first one is the default, mapped to the
    # start script of the variant.
    variants = {"nvfp4": "nvfp4", "unsloth": "unsloth"}
    scripts = {"nvfp4": "run-nvfp4-nvfp4.sh", "unsloth": "run-unsloth-ud-q4_k_xl.sh"}
    pattern = r"Strata/serve/[s]erver\.py"
    status_command = status_cmd(pattern)
    kill_command = kill_cmd(pattern)

    def __init__(self) -> None:
        super().__init__()
        self.variant = next(iter(self.variants))

    def _build_extras(self) -> None:
        with ui.row().classes('items-center gap-3 q-mt-sm'):
            ui.label('Model variant:').classes('text-subtitle1')
            ui.radio(self.variants, value=self.variant,
                     on_change=lambda e: self._set_variant(e.value)).props('inline')

    def _set_variant(self, value: str) -> None:
        """Remember the variant the user picked (the Run button acts on it)."""
        self.variant = value
        logger.info("%s: user chose variant %s", self.name, value)

    def _start_spec(self) -> tuple[str, str]:
        script = self.scripts[self.variant]
        logger.info("%s: starting variant %s through $HOME/Strata/%s", self.name, self.variant, script)
        cmd = setsid_start_cmd("$HOME/Strata", "strata", f'bash "$HOME/Strata/{script}"')
        return cmd, f"Qwen Flash Next ({self.variant})"


class SplashService(RemoteService):
    """Qwen on the splash node (macOS), served by $HOME/splash/splash-start-30.sh."""

    tab, tab_label, group_start = "splash", "Splash / macOS", True
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
    start_command = detached_start_cmd(pidfile, log_path, 'bash "$HOME/splash/splash-start-30.sh"',
                                       node="splash", append=False)
    status_command = status_cmd(pattern)
    kill_command = kill_cmd(pattern, pidfile=pidfile)


# Tab-strip order: the Linux services, then the macOS ones.
SERVICES = (LlamaService, StrataService, SplashService)
