"""The service tabs: one class per inference service running on a remote node.

Each service is a single object shared by every browser (see SERVICES): its
status is probed at most once every POLL_SECONDS, only while some page is open,
and all pages show that same status. The commands come from gui_shell.
"""

import asyncio
import functools
import time

from nicegui import background_tasks, ui

from gui_config import GREEN, NODES, ORANGE, POLL_SECONDS, RED
from gui_logging import logger
from gui_shell import detached_start_cmd, kill_cmd, log_tail_cmd, setsid_start_cmd, status_cmd
from gui_util import api_check, is_alive, last_line, ssh_run
from gui_widgets import copy_to_clipboard, exclusive, set_class

QWEN_CODER_DESC = ("Programming, web development, front-end development, video game development: "
                   "a model specifically designed for developers.")


class RemoteService:
    """A workload on a remote node: Run/Kill buttons, a status line, a log tail.

    Status: RUNNING if the API (/v1/models) answers, STARTING if only the process
    is up, STOPPED if neither. Subclasses fill in the configuration below and may
    override build_extras() and build_run_buttons().
    """

    # --- Configuration, set by the subclasses ---
    tab = ""               # tab id, remembered per browser
    tab_label = ""
    group_start = False    # separator above this tab (first macOS service)
    title = ""
    description = ""
    name = ""              # short name: status line, buttons, log lines
    node_key = ""          # entry in nodes.json
    default_port = 0       # API port if nodes.json has none
    extra_ports = ()       # more ports where the API may answer
    workload = ""          # what runs on the node, as the messages call it
    run_label = ""         # text of the green button
    start_command = ""
    status_command = ""    # gui_shell.status_cmd
    kill_command = ""      # gui_shell.kill_cmd
    log_path = ""
    fallback_model = ""    # model name shown while the process is up but the API is silent

    def __init__(self) -> None:
        node = NODES.get(self.node_key, {})
        self.ip, self.user = node.get("ip", ""), node.get("user", "")
        ports = (int(node.get("api_port") or self.default_port), *self.extra_ports)
        self.ports = tuple(dict.fromkeys(ports))  # in order, without duplicates
        self.node_desc = f"{self.node_key} node"

        # Shared state: written by refresh(), shown by every page.
        self.status, self.color = "checking...", ORANGE
        self.running = False                      # the API answers: green bullet
        self.alive = False                        # process or API up: Kill enabled
        self.model = ""
        self.live_port: int | None = None
        self.loading_model = self.fallback_model  # shown until the API names the model
        self.lock = asyncio.Lock()                # one start/kill at a time, from any browser
        self._probe_lock = asyncio.Lock()
        self._probed_at = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.ip and self.user)

    @property
    def host_text(self) -> str:
        """user@ip:port, listing every candidate port until one answers ("" if unconfigured)."""
        if not self.configured:
            return ""
        ports = self.live_port or "/".join(map(str, self.ports))
        return f"{self.name} node: {self.user}@{self.ip}:{ports}"

    @property
    def log_name(self) -> str:
        """The log file as the page names it: just the file name when under $HOME."""
        return self.log_path.rsplit("/", 1)[-1] if self.log_path.startswith("$HOME/") else self.log_path

    async def ssh(self, cmd: str, what: str, routine: bool = False) -> tuple[int, str]:
        """Run cmd on the node: (returncode, output). routine=True logs at DEBUG."""
        return await ssh_run(self.ip, self.user, cmd, what=f"{self.name}: {what}", routine=routine)

    # ------------------------------------------------------------------ page
    def build(self, tab: ui.tab) -> None:
        """Create this service's panel in the current page, kept in sync by a 1 s timer."""
        with ui.column().classes('w-full items-center p-8'), ui.card().classes('w-full max-w-2xl p-4'):
            ui.label(self.title).classes('text-h6')
            ui.label(self.description).classes('service-description')
            with ui.row().classes('model-status-row items-center gap-1'):
                model_label = ui.label().classes('model-name')
                copy_icon = ui.icon('content_copy').tooltip('Copy model name') \
                    .classes('model-copy-icon cursor-pointer')
                copy_icon.on('click', lambda: copy_to_clipboard(self.model, "model name"))
            status_label = ui.label().classes('service-status q-mt-sm')
            host_label = ui.label().classes('text-caption text-grey')
            self.build_extras()
            with ui.row().classes('items-center gap-2 q-mt-md flex-nowrap'):
                self.build_run_buttons()
                kill_button = self.action_button(f"Kill {self.name}", 'red', self.kill)
            # Fetched on demand, so the log is still readable after the run ended.
            with ui.row().classes('items-center gap-2 q-mt-lg w-full'):
                ui.label(f"{self.name} log ({self.log_name})").classes('text-h6')
                update_button = ui.button("Update log").props('small outline')
                clear_button = ui.button("Clear log").props('small outline') \
                    .tooltip(f"Clear the window only; the remote {self.log_name} is kept")
            log_window = ui.log().classes(
                'w-full h-[300px] bg-black text-green-400 font-mono text-xs custom-log')

        log_lock = asyncio.Lock()  # one log fetch at a time in this page
        update_button.on_click(lambda: exclusive(log_lock, "A log fetch is already in progress",
                                                 update_button, lambda: self.fetch_log(log_window)))

        def clear_log() -> None:
            log_window.clear()
            log_window.push('--- log cleared; use "Update log" to refetch ---')

        clear_button.on_click(clear_log)

        shown = None  # the state last drawn: if unchanged, nothing is sent to the browser

        def render() -> None:
            nonlocal shown
            state = (self.status, self.color, self.model, self.host_text, self.alive, self.running)
            if state == shown:
                return
            shown = state
            status_label.set_text(f"{self.name}: {self.status}")
            status_label.style(f"color: {self.color}")
            model_label.set_text(self.model)
            set_class(copy_icon, 'invisible', not self.model)
            host_label.set_text(self.host_text)
            kill_button.set_enabled(self.alive)
            set_class(tab, 'running-bullet', self.running)

        def tick() -> None:
            self.refresh_soon()
            render()

        tick()
        ui.timer(1.0, tick)

    def build_extras(self) -> None:
        """Widgets between the host line and the buttons (none by default)."""

    def build_run_buttons(self) -> None:
        """The green button(s): by default one, running start_command."""
        self.action_button(self.run_label, 'green', lambda: self.start(self.start_command, self.workload))

    def action_button(self, label: str, color: str, action) -> ui.button:
        """A button running action() under the service lock, with a spinner meanwhile."""
        button = ui.button(label).props(f'color={color}').classes('w-48 justify-center')
        button.on_click(lambda: exclusive(self.lock, f"A {self.name} action is already in progress",
                                          button, action))
        return button

    # ---------------------------------------------------------------- status
    def refresh_soon(self) -> None:
        """Start a background probe, unless one ran less than POLL_SECONDS ago.

        Every open page calls this each second: the first one past the deadline
        probes, the others just show the result."""
        if time.monotonic() - self._probed_at >= POLL_SECONDS:
            self._probed_at = time.monotonic()  # the other pages skip this round
            background_tasks.create(self.refresh(), name=f"refresh {self.name}")

    async def refresh(self) -> None:
        """Probe process (SSH) and API (HTTP) on the node, then update the shared state."""
        async with self._probe_lock:
            try:
                await self._probe()
            except Exception as e:  # a bad answer must not stop the polling
                logger.debug("%s: status check failed", self.name, exc_info=True)
                self.running = False
                self._set_status(f"UNKNOWN (status check failed: {e})", ORANGE)
            self._probed_at = time.monotonic()

    async def _probe(self) -> None:
        if not self.configured:
            self._set_status("node not configured (nodes.json)", ORANGE)
            return
        (rc, out), (port, model) = await asyncio.gather(
            self.ssh(self.status_command, f"status probe of {self.workload}", routine=True),
            self._probe_api())
        proc_alive = rc == 0 and is_alive(out)
        self.running, self.live_port = port is not None, port
        self.alive = self.running or proc_alive
        self.model = model or (self.loading_model if proc_alive else "")
        if self.running:
            self._set_status("RUNNING", GREEN)
        elif rc == 255:
            self._set_status("node unreachable", ORANGE)
        elif rc != 0:
            self._set_status(f"UNKNOWN (status check rc={rc})", ORANGE)
        elif proc_alive:
            self._set_status("STARTING (API not answering yet)", ORANGE)
        else:
            self._set_status("STOPPED", RED)

    async def _probe_api(self) -> tuple[int | None, str]:
        """(first port whose /v1/models answers, or None; the model it serves)."""
        answers = await asyncio.gather(*(asyncio.to_thread(api_check, self.ip, p) for p in self.ports))
        for port, (ok, model) in zip(self.ports, answers):
            if ok:
                return port, model
        return None, ""

    def _set_status(self, text: str, color: str) -> None:
        if text != self.status:  # log the changes, not every poll
            logger.info("%s: status %s -> %s", self.name, self.status, text)
        self.status, self.color = text, color

    # --------------------------------------------------------------- actions
    async def start(self, command: str, what: str, model: str = "") -> None:
        """Run a start command, unless the workload already runs (two would fight
        for the port and the GPU). model: the name to show while it loads."""
        if not self._check_configured():
            return
        rc, out = await self.ssh(self.status_command, f"pre-start check of {self.workload}")
        if rc == 255:
            logger.error("%s: not starting %s, the %s is unreachable", self.name, what, self.node_desc)
            ui.notify(f"Could not reach the {self.node_desc}", type="negative")
            return
        if rc == 0 and is_alive(out):
            logger.warning("%s: not starting %s, %s is already running", self.name, what, self.workload)
            ui.notify(f"{self.workload} is already running on the {self.node_desc}; "
                      f"use Kill {self.name} first", type="warning")
            return
        ui.notify(f"Starting {what} on the {self.node_desc}...")
        rc, out = await self.ssh(command, f"start {what}")
        if rc == 0:
            self.loading_model = model or self.fallback_model
            ui.notify(f"{what} started on the {self.node_desc} ({last_line(out)})", type="positive")
            await asyncio.sleep(2)  # let the script spawn the server, or the probe below misses it
        else:
            error = f"Error starting {what} on the {self.node_desc} (rc={rc})"
            if out.strip():
                error += f": {out.strip()}"
            logger.error("%s: %s", self.name, error)
            ui.notify(error, type="negative")
        await self.refresh()

    async def kill(self) -> None:
        """Stop the workload; once more if something survived (kill_cmd with verify)."""
        if self._check_configured():
            ui.notify(f"Stopping {self.workload} on the {self.node_desc}...")
            rc, out = await self.ssh(self.kill_command, f"kill {self.workload}")
            if rc == 4:
                ui.notify(f"{self.workload} survived, killing it again...", type="warning")
                rc, out = await self.ssh(self.kill_command, f"kill {self.workload} (2nd attempt)")
            self._notify_kill(rc, out.strip())
            self.loading_model = self.fallback_model
        await self.refresh()

    def _notify_kill(self, rc: int, out: str) -> None:
        where = f"the {self.node_desc}"
        text, kind = {
            0: (f"{self.workload} stopped on {where} ({last_line(out)})", "positive"),
            3: (f"No {self.workload} process found on {where} (already stopped?)", "warning"),
            4: (f"{self.workload} did not go down on {where}: {out}", "negative"),
            255: (f"Could not reach {where}", "negative"),
        }.get(rc, (f"Error stopping {self.workload} on {where} (rc={rc}): {out}", "negative"))
        log = logger.error if kind == "negative" else logger.info
        log("%s: %s", self.name, text)
        ui.notify(text, type=kind)

    async def fetch_log(self, log: ui.log) -> None:
        """Show the last lines of the remote log in this page's log window."""
        if not self._check_configured():
            return
        rc, out = await self.ssh(log_tail_cmd(self.log_path), f"log tail of {self.log_name}")
        log.clear()
        if rc != 0:
            reason = "node unreachable" if rc == 255 else f"rc={rc}"
            log.push(f"--- log fetch failed ({reason}) ---")
            ui.notify(f"Could not fetch {self.log_name} ({reason})", type="negative")
        log.push(out.rstrip())

    def _check_configured(self) -> bool:
        """configured, with a warning to the user when it is not."""
        if not self.configured:
            logger.warning("%s: the %s is not configured (nodes.json)", self.name, self.node_desc)
            ui.notify(f"The {self.node_desc} is not configured (nodes.json)", type="warning")
        return self.configured


class LlamaService(RemoteService):
    """llama-server on the llama node (Linux).

    One green button per model: a dialog asks for the quantization, then
    $HOME/Llama/start-<model>-<quantization>.sh is launched detached (it outlives
    the SSH session). The models share the GPUs: one runs at a time."""

    tab, tab_label = "llama_cluster", "Llama.cpp / Linux"
    title = "Llama.cpp — Gemma-4-31b | Qwen3.8-27b"
    description = ("Gemma-4 is more generalist; Qwen3.8 is a coder model: "
                   "programming, web and game development.")
    name, node_key, workload = "Llama", "llama", "llama-server"
    default_port, extra_ports = 8080, (8080,)  # the API answers on api_port or on 8080
    log_path = "$HOME/llama-server.out"  # llama-server --log-file; the launcher appends here too
    pidfile = "$HOME/llama-server.pid"
    start_dir = "$HOME/Llama"
    # Model button -> the quantizations its dialog offers (the first is preselected).
    run_buttons = {
        "Gemma-4-31b-it": ("ud-q8_k_xl", "qat-ud-q4_k_xl"),
        "Qwen3.8-27b": ("ud-q4_k_xl", "ud-q8_k_xl"),
    }
    _pattern = r"/opt/llama[.]cpp/llama[-]server"
    status_command = status_cmd(_pattern)
    kill_command = kill_cmd(_pattern, pidfile=pidfile, verify=True)

    def build_run_buttons(self) -> None:
        for label, quants in self.run_buttons.items():
            button = ui.button(label).props('color=green').classes('w-48 justify-center')
            button.on_click(functools.partial(self._choose_and_start, label, quants, button))

    async def _choose_and_start(self, label: str, quants: tuple[str, ...], button: ui.button) -> None:
        """Ask for the quantization, then start the model's script under the service lock."""
        with ui.dialog() as dialog, ui.card().classes('p-4 gap-2'):
            ui.label(f"{label}: choose the quantization").classes('text-subtitle1')
            choice = ui.radio(list(quants), value=quants[0])
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button("Cancel", on_click=dialog.close).props('flat outline')
                ui.button("Start", color="green", on_click=lambda: dialog.submit(choice.value))
        quant = await dialog  # None if cancelled or dismissed
        dialog.delete()
        if not quant:
            return
        script = f"{self.start_dir}/start-{label}-{quant}.sh"
        logger.info("%s: user chose %s (%s) -> %s", self.name, label, quant, script)
        command = detached_start_cmd(self.pidfile, self.log_path, f'bash "{script}"', node="llama")
        await exclusive(self.lock, f"A {self.name} action is already in progress", button,
                        lambda: self.start(command, f"llama-server {label} ({quant})", model=label))


class StrataService(RemoteService):
    """Qwen on the strata node (Linux), started by a run script in $HOME/Strata.

    The script is only a wrapper: status and kill find the python server it
    spawns by its command line (the pidfile is not used)."""

    tab, tab_label = "strata", "Strata / Linux"
    title = "Strata — Qwen Flash Next"
    description = QWEN_CODER_DESC
    name, node_key, workload = "Strata", "strata", "Qwen"
    default_port = 8000
    run_label = "Run Qwen Flash Next"
    workdir = "$HOME/Strata"
    log_path = f"{workdir}/strata.out"  # written by setsid_start_cmd(workdir, "strata", ...)
    # Variant (radio option; the first is the default) -> its start script in workdir.
    variants = {"nvfp4": "run-nvfp4.sh", "unsloth-iq4 (v3.1)": "run-unsloth-ud-iq4_xs-v31.sh"}
    _pattern = r"Strata/serve/[s]erver\.py"
    status_command = status_cmd(_pattern)
    kill_command = kill_cmd(_pattern)

    def __init__(self) -> None:
        super().__init__()
        self.variant = next(iter(self.variants))  # shared by all pages, like the status

    def build_extras(self) -> None:
        with ui.row().classes('items-center gap-3 q-mt-sm'):
            ui.label('Model variant:').classes('text-subtitle1')
            ui.radio(list(self.variants)).props('inline').bind_value(self, 'variant')

    def build_run_buttons(self) -> None:
        self.action_button(self.run_label, 'green', self._start_variant)

    async def _start_variant(self) -> None:
        script = f"{self.workdir}/{self.variants[self.variant]}"
        command = setsid_start_cmd(self.workdir, "strata", f'bash "{script}"')
        await self.start(command, f"Qwen Flash Next ({self.variant})")


class SplashService(RemoteService):
    """Qwen on the splash node (macOS), served by $HOME/splash/splash-start-30.sh."""

    tab, tab_label, group_start = "splash", "Splash / macOS", True
    title = "Splash — Qwen3.8-27B (4bit)"
    description = QWEN_CODER_DESC
    name, node_key, workload = "Splash", "splash", "Qwen"
    default_port = 10000
    run_label = "Run Splash Qwen"
    fallback_model = "mlx-community/Qwen3.8-27B-4bit"
    log_path = "$HOME/splash.out"  # truncated at every start
    pidfile = "$HOME/splash.pid"
    _pattern = r"splash[-_]start[-_]30[.]sh"
    start_command = detached_start_cmd(pidfile, log_path, 'bash "$HOME/splash/splash-start-30.sh"',
                                       node="splash", append=False)
    status_command = status_cmd(_pattern)
    kill_command = kill_cmd(_pattern, pidfile=pidfile)


# One shared object per service, in tab-strip order: Linux first, then macOS.
SERVICES = (LlamaService(), StrataService(), SplashService())
