"""The legacy "Llama.cpp" tab: the NiceGUI front-end of the old start_model.py.

Hidden by default (gui_config.SHOW_LEGACY_LLAMA_TAB): this module, and the old
project's modules it needs (config_manager, model, rpc, ...), are imported only
when the tab is shown. All it knows about the server comes from
`start_model.py --json`. The model list costs an SSH round-trip: it is loaded
at boot and on "Recheck models", then shared by every page.
"""

import asyncio
import contextlib
import json
import os
import shlex
import signal
import sys
from pathlib import Path

from nicegui import app, context, ui

from config_manager import get_settings
from gui_config import GREEN, HERE, ORANGE, POLL_SECONDS, RED, read_json
from gui_logging import logger
from gui_util import run
from gui_widgets import copy_to_clipboard, set_class
from model import Model, load_models as load_models_from_config

settings = get_settings()


def _local(path) -> Path:
    """A path from config.json, relative ones taken from the repo root (not the cwd)."""
    path = Path(path or "")
    return path if path.is_absolute() else HERE / path


def _start_model() -> tuple[str, dict[str, str] | None]:
    """(path of start_model.py, environment to run it with), run by this interpreter.

    From deprecated/ it cannot import its siblings (model.py, rpc.py, ...) left in
    the repo root: the root then goes on PYTHONPATH."""
    root_script = HERE / "start_model.py"
    candidates = [root_script, HERE / "deprecated" / "start_model.py"]
    for script in candidates:
        if script.is_file():
            env = None
            if script.parent != HERE:
                env = dict(os.environ,
                           PYTHONPATH=os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]))
            return str(script), env
    logger.warning("start_model.py found in none of %s: the legacy tab cannot drive a server",
                   ", ".join(map(str, candidates)))
    return str(root_script), None


START_MODEL, START_MODEL_ENV = _start_model()
PY = sys.executable
MODELS_JSON = _local(settings.MODELS_JSON)
RPC_SERVERS = read_json(_local(settings.RPC_JSON), "RPC_SERVERS")
logger.info("Legacy tab: engine %s, model catalog %s (host %s), %d RPC server(s): %s",
            START_MODEL, MODELS_JSON, settings.LLAMA_SERVER_HOST or "localhost",
            len(RPC_SERVERS), ", ".join(RPC_SERVERS) or "-")

CTX_MIN = 8192
CTX_STEP = 1024
# ctx_budget_gb_bf16_128k: bf16 KV cache at this context, in decimal GB.
KV_REF_CTX = 131072
_KV_QUANT_SCALE = {"q8_0": 0.5, "q4_0": 0.25}   # KV cache size relative to bf16
_GB_TO_GIB = 1e9 / 1024**3
# Radio options where "" means "no override: keep the value from models.json".
KVQUANT_OPTIONS = {"": "None", "q8_0": "8 bit", "q4_0": "4 bit"}
REAS_OPTIONS = {"": "None", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}
STREAM_MAX_LINES = 5000

# The four values of the "Samplers:" label, in display order, with the hint
# shown by its hover tooltip.
SAMPLER_HINTS = (
    ("temperature", "scales the logits — below 1.00 sharpens the model's distribution, "
                    "above 1.00 flattens it"),
    ("top_p", "nucleus sampling: keeps the most likely tokens up to this cumulative probability"),
    ("top_k", "keeps only the k most likely tokens (0 disables the cut)"),
    ("min_p", "drops every token below this fraction of the top token's probability"),
)


def samplers_description(values: tuple[str, ...], header: str) -> str:
    return "\n".join([header] + [f"{name} = {value}  —  {hint}"
                                 for (name, hint), value in zip(SAMPLER_HINTS, values)])


async def run_json(args: list[str], *, what: str = "", routine: bool = False) -> dict | None:
    """Run start_model.py with --json; return its parsed output, or None.

    The return code is ignored (--server-status exits 1 for "not running", a
    valid answer): only unparseable output is a failure."""
    label = what or "start_model.py " + " ".join(args)
    rc, out = await run([PY, START_MODEL, *args, "--json"], what=label, routine=routine,
                        env=START_MODEL_ENV)
    if not out.strip():
        logger.error("%s produced no output (rc=%s)", label, rc)
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        logger.error("%s returned invalid JSON: %s", label, e)
        return None


# The model catalog, shared by every page (None until loaded).
_models: dict[str, Model] | None = None
_models_lock = asyncio.Lock()


async def fetch_models(force: bool = False) -> tuple[dict[str, Model] | None, bool]:
    """Return ({name: Model}, ok). Reloads only when the cache is empty or
    force=True; a failed reload keeps the previous list."""
    global _models
    async with _models_lock:
        if _models is None or force:
            logger.info("Loading the model catalog from %s (host %s)",
                        MODELS_JSON, settings.LLAMA_SERVER_HOST or "localhost")
            try:
                models = await asyncio.to_thread(
                    load_models_from_config,
                    MODELS_JSON,
                    _local(settings.RPC_JSON),
                    remote_host=settings.LLAMA_SERVER_HOST,
                    remote_user=settings.LLAMA_SERVER_USER,
                    check_remote_file=True,
                )
            except Exception:
                # A malformed models.json or an unreachable host must not kill the console.
                logger.exception("Could not load the model list")
                return _models, False
            _models = {m.model_name: m for m in models}
            logger.info("Model catalog: %d model(s): %s", len(_models), ", ".join(_models) or "-")
        else:
            logger.debug("Model catalog served from cache (%d model(s))", len(_models))
        return _models, True


# Scanned at boot, in the background: pages opened earlier wait for the same scan.
app.on_startup(fetch_models)

# CSS of this tab, plus the cursor-following tooltip of the "Samplers:" labels:
# a .cursor-tip-host shows the text of its hidden .cursor-tip-text child.
ui.add_head_html('''
<style>
.legacy-status {
    font-size: 0.9rem;
    font-weight: 600;
    white-space: pre;
}
/* Long model names truncate instead of stretching the select. */
.model-select .q-field__native .ellipsis {
    min-width: 0;
}
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
    pointer-events: none;  /* never steal the pointer it follows */
    opacity: 0;
    transition: opacity 0.12s ease-in-out;
}
#cursor-tip.visible {
    opacity: 1;
}
</style>
<script>
(() => {
    const OFFSET_X = 16, OFFSET_Y = 18, MARGIN = 8;
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

    // Next to the pointer, flipped to the other side rather than off-screen.
    const place = (event) => {
        const box = getTip().getBoundingClientRect();
        let x = event.clientX + OFFSET_X;
        let y = event.clientY + OFFSET_Y;
        if (x + box.width > window.innerWidth - MARGIN) x = event.clientX - OFFSET_X - box.width;
        if (y + box.height > window.innerHeight - MARGIN) y = event.clientY - OFFSET_Y - box.height;
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
        place(event);  // position first, then show: no flash in the corner
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
    document.addEventListener('scroll', hide, true);  // the host moved away from the pointer
    window.addEventListener('blur', hide);
})();
</script>
''', shared=True)


class LegacyLlamaPanel:
    """The original "Llama.cpp" tab, one per page: model picker, context/KV/
    temperature controls, START/STOP, RPC server kill and live server log."""

    tab, tab_label, group_start = "llama", "Llama.cpp", False

    def __init__(self) -> None:
        self.models: dict[str, Model] = {}
        self.log_task: asyncio.Task | None = None
        self.status_model_name = ""
        self.rpc_servers_row: ui.row | None = None
        self._status_lock = asyncio.Lock()
        self._start_busy = False
        self._stop_busy = False
        self._server_state = ""      # last status shown, to log only the changes

    # ------------------------------------------------------------------ UI ---
    def build(self, tab: ui.tab) -> None:
        """Create the panel in the current page, with its timers (tab: unused, no bullet)."""
        self._build_widgets()
        ui.timer(POLL_SECONDS, self.update_status)
        ui.timer(0, self.load_models, once=True)
        context.client.on_disconnect(self.cleanup)

    def _build_widgets(self) -> None:
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
                        'click', lambda: copy_to_clipboard(self.status_model_name, "model name"))
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
                        self.reas_radio = ui.radio(REAS_OPTIONS, value="").props('inline')

                with ui.column().classes('w-full q-mt-sm'):
                    self.ctx_label = ui.label("Context: —").classes('text-subtitle1')
                    self.ctx_slider = ui.slider(
                        min=CTX_MIN, max=262144, value=CTX_MIN, step=CTX_STEP,
                        on_change=self._refresh_ctx,
                    ).classes('flex-grow').props('color=green')
                    with ui.row().classes('items-center gap-3'):
                        ui.label('KV Quant:').classes('text-subtitle1')
                        self.kvquant_radio = ui.radio(
                            KVQUANT_OPTIONS, value="", on_change=self._update_memory).props('inline')
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
            self.log_window = ui.log(max_lines=STREAM_MAX_LINES).classes(
                'w-full h-[600px] bg-black text-green-400 font-mono text-xs custom-log')

    # ------------------------------------------------------------ samplers ---
    def _set_samplers(self, values: tuple[str, str, str, str] | None = None) -> None:
        """The selected model's samplers (temperature, top_p, top_k, min_p, already
        formatted), or placeholders while no model is selected."""
        shown = values or ("--", "--", "--", "--")
        self.samplers_label.set_text("Samplers: " + ";".join(shown))
        description = samplers_description(
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
        self.status_samplers_tip.set_text(samplers_description(
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
        logger.warning("No model to offer (empty or unreadable model catalog)")
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
        logger.info("Selected model %s: ctx=%s temp=%s top_p=%s top_k=%s min_p=%s "
                    "kvquant=%s mtp=%s reasoning=%s rpc=%s size=%s GiB",
                    name, model.native_ctx, model.temperature, model.top_p, model.top_k,
                    model.min_p, model.kvquant or "(none)", model.mtp, model.reasoning or "(none)",
                    [r.name for r in model.rpcservers or []] or "-",
                    f"{model.size_gib:.1f}" if model.size_gib is not None else "?")
        native_ctx = model.native_ctx
        # A slider with min > max cannot be dragged at all.
        ctx_min = min(CTX_MIN, native_ctx)
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
        self.reas_radio.set_value(model.reasoning if model.reasoning in REAS_OPTIONS else "")
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
                  * int(self.ctx_slider.value) / KV_REF_CTX
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
        info = await run_json(["--server-status"], what="server status probe", routine=True)
        if info is None or not info.get("transport", True):
            # No valid answer: the state is unknown, which is NOT "not running".
            self._set_server_status("UNREACHABLE", ORANGE)
            self._show_server_details(running=False, info=None)
            self.stop_button.set_visibility(False)
            return

        running = bool(info.get("running"))
        # Never start over a live server (START re-checks at click time too).
        self.start_button.set_enabled(not running)
        self.stop_button.set_visibility(running)
        self._set_server_status("RUNNING" if running else "NOT RUNNING",
                                GREEN if running else RED)
        self._show_server_details(running, info if running and info.get("ready") else None)

    def _set_server_status(self, text: str, color: str) -> None:
        if text != self._server_state:
            logger.info("Inference server status %s -> %s", self._server_state or "(unknown)", text)
            self._server_state = text
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
        set_class(self.status_model_copy, 'invisible', not self.status_model_name)

    async def refresh(self) -> None:
        await self.update_status()
        ui.notify("Status updated")

    # ------------------------------------------------------------ commands ---
    async def _stream(self, argv: list[str], *, what: str = "") -> int:
        """Run argv, pushing its output into the log window line by line.

        The child gets its own process group, so stopping it also stops what it
        spawned (start_model.py's remote `ssh ... tail -F`)."""
        label = what or shlex.join(argv)
        logger.info("%s: running %s", label, shlex.join(argv))
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True, env=START_MODEL_ENV)
        try:
            async for raw in proc.stdout:
                self.log_window.push(raw.decode(errors="replace").rstrip())
            rc = await proc.wait()
            logger.info("%s: finished rc=%s", label, rc)
            return rc
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                logger.info("%s: terminated by the console (rc was still None)", label)

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
            logger.warning("Start refused: no model selected")
            ui.notify("Please select a model first", type="warning")
            return

        self._start_busy = True
        self.start_button.props('loading')
        try:
            # Fresh check at click time: the polled status can be stale.
            info = await run_json(["--server-status"], what="pre-start server status check")
            if info is None or not info.get("transport", True):
                # Unknown is not "free": starting could land over a live server.
                logger.error("Start of %s aborted: the server status is unknown", name)
                ui.notify("Could not check the server status (host unreachable?) — "
                          "start aborted", type="negative")
                return
            if info.get("running"):
                current = (info.get("model") or "").strip()
                suffix = f" ({current})" if current else ""
                logger.warning("Start of %s refused: a server is already running%s", name, suffix)
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

            # The whole parameter set the user picked, overrides included: the one
            # line to read when a run behaves differently from what was expected.
            logger.info("Starting model %s: ctx=%d (max %d, model default %s) kvquant=%s "
                        "reasoning_effort=%s temperature=%.2f (model default %.2f) "
                        "force_no_mtp=%s no_reasoning=%s -> %s",
                        name, ctx_value, int(self.ctx_slider.props['max']), model.native_ctx,
                        self.kvquant_radio.value or (model.kvquant or "(model default)"),
                        self.reas_radio.value or (model.reasoning or "(model default)"),
                        temp_value, model.temperature,
                        bool(self.mtp_checkbox.value), bool(self.no_reasoning_checkbox.value),
                        shlex.join(args))
            ui.notify(f"Starting model {name} (ctx={ctx_value})...")
            # -u: unbuffered, so the lines arrive as they happen.
            cmd = [PY, "-u", START_MODEL, *args]
            self.log_window.push(f"--- Starting model: {shlex.join(cmd)} ---")
            try:
                rc = await self._stream(cmd, what=f"start model {name}")
            except OSError as e:
                logger.exception("Could not launch %s", shlex.join(cmd))
                self.log_window.push(f"Launch Error: {e}")
                ui.notify(f"Error starting model: {e}", type="negative")
            else:
                if rc == 0:
                    logger.info("Model %s started successfully", name)
                    ui.notify(f"Model {name} started successfully", type="positive")
                else:
                    logger.error("Could not start model %s (exit %s)", name, rc)
                    ui.notify(f"Error starting model (exit {rc})", type="negative")
            await self.update_status()
        finally:
            self._start_busy = False
            self.start_button.props(remove='loading')

    async def stop_server(self) -> None:
        if self._stop_busy:
            ui.notify("A stop is already in progress", type="warning")
            return
        logger.info("Stopping the inference server")
        self._stop_busy = True
        self.stop_button.disable()
        self.start_button.disable()
        try:
            ui.notify("Stopping server...")
            rc, out = await run([PY, START_MODEL, "--kill-server"],
                                what="kill inference server", env=START_MODEL_ENV)
            if rc == 0:
                logger.info("Inference server stopped (%s)", out.strip())
                ui.notify("Server stopped", type="positive")
            else:
                logger.error("Error stopping the inference server (rc=%s): %s", rc, out.strip())
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
            logger.warning("Kill RPC refused: no model selected")
            ui.notify("Please select a model first", type="warning")
            return
        # Exactly the selected model's RPC servers: --override-rpc replaces its
        # list before --kill-rpc-server acts on it.
        selected = [rpc.name for rpc in model.rpcservers or [] if rpc.name in RPC_SERVERS]
        if not selected:
            ui.notify("This model has no RPC servers to kill", type="warning")
            return
        logger.info("Killing RPC server(s) of %s: %s", name, ", ".join(selected))
        ui.notify(f"Killing RPC server(s): {', '.join(selected)}...")
        rc, out = await run([PY, START_MODEL, name, "--override-rpc", ",".join(selected),
                             "--kill-rpc-server"], what=f"kill RPC server(s) of {name}",
                            env=START_MODEL_ENV)
        if rc == 0:
            logger.info("RPC servers killed: %s", ", ".join(selected))
            ui.notify("RPC servers killed", type="positive")
        else:
            logger.error("Error killing RPC server(s) %s (rc=%s): %s",
                         ", ".join(selected), rc, out.strip())
            ui.notify(f"Error killing RPC servers: {out.strip()}", type="negative")

    # ------------------------------------------------------------ log tail ---
    async def _stream_logs(self) -> None:
        try:
            await self._stream([PY, "-u", START_MODEL, "--tail-log", "-n", "1000"],
                               what="inference server log tail")
        except asyncio.CancelledError:
            logger.info("Inference server log streaming stopped")
            self.log_window.push("--- log streaming stopped ---")
            raise
        except OSError as e:
            logger.error("Log streaming error: %s", e)
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
