#!/usr/bin/env python3
"""LLM inference console: a NiceGUI page with one tab per inference service.

Each tab starts and stops a workload on a remote node over SSH and shows its
status; a green bullet in the tab strip marks the RUNNING services.
Run `python3 ai-console-gui.py`, then open http://<host>:<UI_PORT>.

    gui_config    settings (config.json, nodes.json) and shared constants
    gui_logging   logging to stdout, so `journalctl -u <unit>` shows it
    gui_services  the service tabs: Llama, Strata, Splash
    gui_shell     the shell commands run on the nodes
    gui_util      subprocess, SSH and HTTP helpers
    gui_widgets   small NiceGUI helpers
    gui_theme     CSS
    deprecated/   not needed by the console: the old start_model.py project and its
                  tab (gui_legacy), hidden (gui_config.SHOW_LEGACY_LLAMA_TAB)
"""

import sys

import gui_logging

# First of all, so that the imports below can log while they read the JSON files.
gui_logging.setup_logging()

from nicegui import app, ui  # noqa: E402

import gui_theme  # noqa: E402
from gui_config import (ACTIVE_TAB_KEY, HERE, REPO_URL, SHOW_LEGACY_LLAMA_TAB, UI_LOG_LEVEL,  # noqa: E402
                        UI_PORT, UI_TITLE, log_boot_config, storage_secret)
from gui_logging import logger  # noqa: E402
from gui_services import SERVICES  # noqa: E402

if SHOW_LEGACY_LLAMA_TAB:
    # gui_legacy, and the old project's modules it imports, are in deprecated/.
    sys.path.insert(0, str(HERE / "deprecated"))
    from gui_legacy import LegacyLlamaPanel  # noqa: E402


def remember_tab(event) -> None:
    """Store the tab this browser switched to (app.storage.user: a signed cookie)."""
    app.storage.user[ACTIVE_TAB_KEY] = event.value


@ui.page('/')
def index() -> None:
    """The page: header, vertical tab strip on the left, the selected panel on the right."""
    ui.dark_mode().enable()
    # The services are shared by all browsers; the legacy panel is one per page.
    panels = ([LegacyLlamaPanel()] if SHOW_LEGACY_LLAMA_TAB else []) + list(SERVICES)
    # Reopen the tab this browser was last on, if it still exists.
    current = app.storage.user.get(ACTIVE_TAB_KEY)
    if current not in {panel.tab for panel in panels}:
        current = SERVICES[0].tab

    with ui.column().classes('w-full h-screen items-stretch p-0 m-0 gap-0'):
        with ui.column().classes('items-center q-mb-sm'):
            ui.label("LLM Inference Console").classes('text-h5')
            ui.label("by Alvise Dorigo").classes('text-h5')
            ui.link(REPO_URL, REPO_URL).classes('text-caption no-underline')

        with ui.row().classes('w-full flex-grow items-stretch p-0 m-0 gap-0'):
            tab_elements = {}
            with ui.column().classes('tab-strip items-stretch p-2'):
                with ui.tabs(on_change=remember_tab).props('vertical no-caps').classes('w-full') as tabs:
                    for panel in panels:
                        tab_elements[panel.tab] = ui.tab(panel.tab, label=panel.tab_label)
                        if panel.group_start:
                            tab_elements[panel.tab].classes('tab-group-start')  # separator above it

            with ui.tab_panels(tabs, value=current).classes('flex-grow p-2 overflow-auto'):
                for panel in panels:
                    with ui.tab_panel(panel.tab):
                        panel.build(tab_elements[panel.tab])


def main() -> None:
    gui_logging.setup_logging(UI_LOG_LEVEL)  # config.json may now set the level
    log_boot_config()
    gui_theme.install()
    logger.info("Listening on http://0.0.0.0:%s", UI_PORT)
    ui.run(title=UI_TITLE, port=UI_PORT, host="0.0.0.0", reload=False, show=False,
           storage_secret=storage_secret())


if __name__ in {"__main__", "__mp_main__"}:
    main()
