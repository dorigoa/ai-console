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
import shlex
import signal
import sys
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
        self.start_button = None

        self.status_server_label = None
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
        self.server_checkboxes: dict[str, ui.checkbox] = {}

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
    def _update_rpc_checkboxes( self ) -> None:
        # The selected model comes from the cache: re-reading models.json from
        # disk on every selection change (get_model_byname) bought nothing.
        selected_model = self.models.get(self.model_dropdown.value)
        self._uncheck_rpc_checkboxes()
        if selected_model is None:
            return

        selected_rpcs = {rpc.name for rpc in selected_model.rpcservers or []}
        for cb_name, checkbox in self.server_checkboxes.items():
            checkbox.value = cb_name in selected_rpcs

    # ---------------------------------------------------------------- status ---
    def _uncheck_rpc_checkboxes(self):
        for name in self.server_checkboxes:
            self.server_checkboxes[name].value = False

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
        self._update_rpc_checkboxes()
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
        if info is None:
            self.status_server_label.set_text("Server Status: UNKNOWN")
            self.status_server_label.style("color: orange;")
            return

        running = bool(info.get("running"))
        # Starting a second model over a live server is never valid, so START
        # follows the polled status; the click handler re-checks to close the
        # window between two polls.
        if running:
            self.start_button.disable()
        else:
            self.start_button.enable()
        color = "#00ff88" if running else "red"
        self.status_server_label.set_text(
            f"Server Status: {'RUNNING' if running else 'NOT RUNNING'}"
        )
        self.status_server_label.style(f"color: {color};")

        if running and info.get("ready"):
            # Keep the bare name around: only it (not the " - Model   : "
            # prefix) is what gets copied to the clipboard.
            self.status_model_name = str(info.get("model") or "").strip()
            self.status_model_label.set_text(f" - Model   : {self.status_model_name}")
            c = (str(info['ctx'])).strip()
            self.status_ctx_label.set_text(  f" - Context : {c} tokens")
            # Rounded: llama-server reports the float32 round-trip of 0.6 as
            # 0.6000000238418579.
            self.status_samplers_label.set_text( f" - Samplers: {float(info['temperature']):.1f};{float(info['top_p']):.2f};{int(info['top_k'])};{float(info['min_p']):.2f}")
            # self._set_status_samplers((f"{float(info['temperature']):.1f}",
            #                            f"{float(info['top_p']):.2f}",
            #                            f"{int(info['top_k'])}",
            #                            f"{float(info['min_p']):.2f}"))
            # self.status_topk_label.set_text( f" - Top-K   : {float(info['top_k'])}")
            # self.status_topp_label.set_text( f" - Top-P   : {float(info['top_p']):.2f}")
            # self.status_minp_label.set_text( f" - Min-P   : {float(info['min_p']):.2f}")
                                    
        elif running:
            self.status_model_name = ""
            self.status_model_label.set_text("Model: (Loading...)")
            self.status_ctx_label.set_text("")
            self.status_samplers_label.set_text("")
            self._set_status_samplers()
            # self.status_topp_label.set_text("")
            # self.status_topk_label.set_text("")
            # self.status_minp_label.set_text("")
        else:
            self.status_model_name = ""
            self.status_model_label.set_text("")
            self.status_ctx_label.set_text("")
            self.status_samplers_label.set_text("")
            self._set_status_samplers()
            # self.status_topp_label.set_text("")
            # self.status_topk_label.set_text("")
            # self.status_minp_label.set_text("")
        # Nothing to copy unless a model name is actually on display.
        if self.status_model_name:
            self.status_model_copy.classes(remove='q-hidden')
        else:
            self.status_model_copy.classes(add='q-hidden')

    async def refresh(self) -> None:
        await self.update_status()
        ui.notify("Status updated")

    async def _copy_model_name(self) -> None:
        """Copy the running model's name to the OS clipboard."""
        if not self.status_model_name:
            ui.notify("No model name to copy", type="warning")
            return
        # navigator.clipboard only exists in secure contexts (https or
        # localhost), but this console is usually reached over plain http on
        # the LAN — so fall back to the deprecated, yet still universally
        # working, execCommand copy.
        js = f"""
        (async () => {{
            const text = {json.dumps(self.status_model_name)};
            if (navigator.clipboard && window.isSecureContext) {{
                try {{
                    await navigator.clipboard.writeText(text);
                    return true;
                }} catch (e) {{}}
            }}
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
            return ok;
        }})()
        """
        try:
            ok = bool(await ui.run_javascript(js))
        except Exception:
            ok = False
        if ok:
            ui.notify("Model name copied to clipboard", type="positive")
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
        #ctx_value = max(ctx_min, min(int(spec["ctx"]), native_ctx))
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

        self._update_rpc_checkboxes( )
        

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
            if info is not None and info.get("running"):
                current = (info.get("model") or "").strip()
                suffix = f" ({current})" if current else ""
                ui.notify(f"llama-server is already running{suffix} — "
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

            override_rpc = []
            for cb_name in self.server_checkboxes:
                if self.server_checkboxes[cb_name].value == True:
                    override_rpc.append(cb_name)

            if len(override_rpc):
                args += ["--override-rpc", f"{','.join(override_rpc)}"]

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
        ui.notify("Stopping server...")
        out, rc = await _capture([_PY, _START_MODEL, "--kill-server"])
        if rc == 0:
            ui.notify("Server stopped", type="positive")
        else:
            ui.notify(f"Error stopping server: {out.strip()}", type="negative")
        await self.update_status()

    async def kill_rpc_servers(self) -> None:
        if self._start_busy:
            ui.notify("A model start is already in progress", type="warning")
            return
        model = self.model_dropdown.value
        if model not in self.models:
            ui.notify("Please select a model first", type="warning")
            return
        ui.notify(f"Killing RPC servers of {model}...")
        out, rc = await _capture([_PY, _START_MODEL, model, "--kill-rpc-server"])
        if rc == 0:
            ui.notify("RPC servers killed", type="positive")
        else:
            ui.notify(f"Error killing RPC servers: {out.strip()}", type="negative")

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
        with ui.column().classes('w-full items-center p-8'):
            with ui.column().classes('items-center q-mb-md'):
                ui.label("LLama.cpp Console").classes('text-h5')
                ui.label("by Alvise Dorigo").classes('text-h5')
                ui.link("https://github.com/dorigoa/llama-console",
                        "https://github.com/dorigoa/llama-console").classes('text-caption no-underline')

            with ui.column().classes('w-full max-w-2xl gap-1 q-mb-4 pr-4'):
                self.status_server_label = ui.label("Checking llama-server status...")
                # The model name gets its own row so that a copy icon can sit
                # right after it; the icon is invisible until the row is hovered
                # (see the .model-copy-icon rules in the head HTML below).
                with ui.row().classes('model-status-row items-center gap-1'):
                    self.status_model_label = ui.label("")
                    self.status_model_copy = ui.icon('content_copy').classes(
                        'model-copy-icon cursor-pointer q-hidden'
                    ).tooltip('Copy model name')
                    self.status_model_copy.on('click', self._copy_model_name)
                self.status_ctx_label = ui.label("")
                self.status_samplers_label = ui.label("")
                # Same cursor-following tooltip as the label in the card below;
                # while nothing is running both stay empty, and an empty host
                # has no width left to hover.
                with ui.element('div').classes('cursor-tip-host'):
                    self.status_samplers_label = ui.label("")
                    self.status_samplers_tip = ui.label("").classes('cursor-tip-text')
                # self.status_topk_label = ui.label("")
                # self.status_topp_label = ui.label("")
                # self.status_minp_label = ui.label("")
                                
                for label in (self.status_server_label, self.status_model_label,
                              self.status_ctx_label, self.status_samplers_label):
                    label.style('font-size: 0.9rem; font-weight: 600; white-space: nowrap;')
                for label in (self.status_model_label,
                              self.status_ctx_label, self.status_samplers_label):#, 
                    label.classes('font-mono').style('font-size: 0.9rem; font-weight: 600; white-space: pre;')
                

                self.status_server_label.style('font-size: 1.2rem; font-weight: 850; white-space: nowrap;')

                for label in (self.status_model_label,
                              self.status_ctx_label, self.status_samplers_label):
                    label.classes('font-mono').style('font-size: 0.9rem; font-weight: 600; white-space: pre;')
                    
                ui.button("Refresh", on_click=self.refresh).props('outline small').classes('q-mt-md')

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
                    ui.button("STOP", on_click=self.stop_server).props('color=red')

                # The host div is what the cursor-following tooltip reacts to:
                # hovering it pops up the text of the .cursor-tip-text child,
                # which CSS keeps out of the page. _set_samplers() fills both.
                with ui.element('div').classes('cursor-tip-host'):
                    self.samplers_label = ui.label().classes("text-h8")
                    self.samplers_tip = ui.label().classes('cursor-tip-text')
                self._set_samplers()

                ui.button("Recheck models",
                          on_click=self.recheck_models).props('small outline')

                # RPC server checkboxes — one per server defined in rpc.json.
                # They are ticked by _update_rpc_checkboxes() once a model is
                # selected, which is the only place that knows the answer.
                if RPC_SERVERS:
                    with ui.row().classes('w-full items-center q-mt-sm gap-3'):
                        ui.label('RPC servers:').classes('text-subtitle1')
                        for name in RPC_SERVERS:
                            self.server_checkboxes[name] = ui.checkbox(name)
                    ui.button("Kill RPC servers",
                              on_click=self.kill_rpc_servers).props('small outline')

                with ui.row().classes('w-full items-center q-mt-sm gap-3'):
                    self.mtp_checkbox = ui.checkbox("Force No-MTP")
                    self.no_reasoning_checkbox = ui.checkbox("No reasoning")
                    with ui.row().classes('items-center gap-3'):
                        ui.label('Reasoning effort:').classes('text-subtitle1')
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
                ui.button("Connect to llama-server logs",
                          on_click=self.start_log_streaming).props('small')
                self.stop_log_button = ui.button("Stop log streaming",
                                                 on_click=self.stop_log_streaming).props('small outline')
                self.stop_log_button.disable()
                ui.button("Clear Logs", on_click=lambda: self.log_window.clear()).props('small outline')

            self.log_window = ui.log().classes(
                'w-full h-[600px] bg-black text-green-400 font-mono text-xs custom-log')

        ui.add_head_html('''
<style>
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


ui.run(title=settings.UI_TITLE, port=settings.UI_PORT, host="0.0.0.0", reload=False, show=False)
