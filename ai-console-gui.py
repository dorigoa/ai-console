#!/usr/bin/env python3
"""NiceGUI front-end of the LLM inference console.

One tab per inference service. Each tab drives a fixed workload on a remote node
over SSH (start, kill, status, log tail) and probes its OpenAI-compatible
/v1/models API; a green bullet in the tab strip flags every RUNNING service.

Nodes (IP, SSH user, API port) come from nodes.json. The legacy "Llama.cpp" tab,
a front-end for start_model.py, is still available but hidden: see
SHOW_LEGACY_LLAMA_TAB in gui_config.

This file is only the entry point: it configures logging, then hands over.

    gui_logging    stdout/journal logging (level: UI_LOG_LEVEL or AI_CONSOLE_LOG_LEVEL)
    gui_config     settings, nodes.json/rpc.json, shared constants, storage secret
    gui_shell      the POSIX shell commands run on the remote nodes
    gui_util       subprocess/SSH/HTTP helpers (every call logs what it runs)
    gui_widgets    small NiceGUI helpers (spinner+lock, clipboard, class toggles)
    gui_theme      the console's CSS and javascript
    gui_services   RemoteService and the Llama/Strata/Splash tabs
    gui_legacy     the legacy start_model.py tab
    gui_page       the page: tab strip, panels, status timers

What happens at runtime goes to stdout, so `journalctl -u <unit>` shows it: the
remote command about to run and what it answered, the parameters the user picked
for a model, every status change. That is INFO; the periodic status probes are
DEBUG (AI_CONSOLE_LOG_LEVEL=DEBUG, or UI_LOG_LEVEL in config.json, to see them).
"""

import gui_logging

# Logging is set up before the imports below on purpose: gui_config reads
# config.json through config_manager, which logs while doing it. The environment
# (AI_CONSOLE_LOG_LEVEL) is all that is available this early; main() repeats the
# call once the configured UI_LOG_LEVEL is known.
gui_logging.setup_logging()

from nicegui import ui  # noqa: E402

import gui_page  # noqa: E402,F401  (importing it registers the "/" page)
import gui_theme  # noqa: E402
from gui_config import log_boot_config, settings, storage_secret  # noqa: E402

logger = gui_logging.logger


def main() -> None:
    # Second pass: now the level can come from config.json as well.
    gui_logging.setup_logging(settings.UI_LOG_LEVEL or None)
    log_boot_config()
    gui_theme.install()
    logger.info("Listening on http://0.0.0.0:%s", settings.UI_PORT)
    ui.run(title=settings.UI_TITLE, port=settings.UI_PORT, host="0.0.0.0",
           reload=False, show=False, storage_secret=storage_secret())


if __name__ in {"__main__", "__mp_main__"}:
    main()
