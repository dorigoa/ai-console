"""The page: the vertical tab strip, one panel per service, and their timers.

Panels are built per client (their state is per browser); gui_services holds the
panels themselves, gui_theme the CSS/JS this page sends along, and the entry
point ai-console-gui.py wires everything and starts the server.
"""

from nicegui import app, context, ui

from gui_config import ACTIVE_TAB_KEY, POLL_SECONDS, REPO_URL, SHOW_LEGACY_LLAMA_TAB
from gui_legacy import LegacyLlamaPanel, fetch_models
from gui_logging import logger
from gui_services import SERVICES

# The tab each browser was last on, so a reload (or a GUI restart) reopens it.
# app.storage.user is per browser (a signed cookie) and persisted on disk.
VALID_TABS = tuple(s.tab for s in SERVICES) + ((LegacyLlamaPanel.tab,) if SHOW_LEGACY_LLAMA_TAB else ())
DEFAULT_TAB = SERVICES[0].tab


def remembered_tab() -> str:
    """The tab stored for this browser, or the default when unset/unknown (a
    service removed from the console must not open a blank page)."""
    tab = app.storage.user.get(ACTIVE_TAB_KEY, DEFAULT_TAB)
    return tab if tab in VALID_TABS else DEFAULT_TAB


def remember_tab(tab: str) -> None:
    """Store the tab this browser switched to (the tab strip's on_change)."""
    app.storage.user[ACTIVE_TAB_KEY] = tab
    logger.info("Tab switched to %s", tab)


@ui.page('/')
def index() -> None:
    ui.dark_mode().enable()
    # The legacy tab comes first and only when shown: it is the historical one.
    legacy = LegacyLlamaPanel() if SHOW_LEGACY_LLAMA_TAB else None
    panels = ([legacy] if legacy else []) + [cls() for cls in SERVICES]

    with ui.column().classes('w-full h-screen items-stretch p-0 m-0 gap-0'):
        with ui.column().classes('items-center q-mb-sm'):
            ui.label("LLM Inference Console").classes('text-h5')
            ui.label("by Alvise Dorigo").classes('text-h5')
            ui.link(REPO_URL, REPO_URL).classes('text-caption no-underline')

        with ui.row().classes('w-full flex-grow items-stretch p-0 m-0 gap-0'):
            with ui.column().classes('tab-strip items-stretch p-2'):
                with ui.tabs(on_change=lambda e: remember_tab(e.value)) \
                        .props('vertical no-caps').classes('w-full') as tabs:
                    for panel in panels:
                        panel.tab_element = ui.tab(panel.tab, label=panel.tab_label)
                        if panel.group_start:
                            panel.tab_element.classes('tab-group-start')

            with ui.tab_panels(tabs, value=remembered_tab()).classes('flex-grow p-2 overflow-auto'):
                for panel in panels:
                    with ui.tab_panel(panel.tab_element):
                        panel.build()

    # Every panel is polled, not just the visible one: the tab bullets need them all.
    for panel in panels:
        ui.timer(POLL_SECONDS, panel.update_status)
    if legacy:
        ui.timer(0, legacy.load_models, once=True)
        context.client.on_disconnect(legacy.cleanup)


# Scan the model list at boot (NiceGUI runs it as a background task): pages
# opened before it finishes just wait for the same scan.
if SHOW_LEGACY_LLAMA_TAB:
    app.on_startup(lambda: fetch_models())
