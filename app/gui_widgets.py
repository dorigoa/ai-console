"""Small NiceGUI helpers shared by the panels."""

import asyncio
import json

from nicegui import ui

from gui_logging import logger


def set_class(element: ui.element, name: str, on: bool) -> None:
    """Add or remove one CSS class."""
    if on:
        element.classes(add=name)
    else:
        element.classes(remove=name)


async def exclusive(lock: asyncio.Lock, busy_msg: str, button: ui.button, action) -> None:
    """Run action() holding lock, with a spinner on button; if lock is taken, just warn."""
    if lock.locked():
        logger.warning("Ignored a click while busy: %s", busy_msg)
        ui.notify(busy_msg, type="warning")
        return
    async with lock:
        button.props('loading')
        try:
            await action()
        finally:
            button.props(remove='loading')


async def copy_to_clipboard(text: str, what: str) -> None:
    """Copy text to the clipboard of the browser, and say whether it worked.

    navigator.clipboard exists only over https or on localhost, and the console
    is usually reached over plain http: hence the execCommand('copy') fallback."""
    if not text:
        ui.notify(f"No {what} to copy", type="warning")
        return
    js = f"""
    (async () => {{
        const text = {json.dumps(text)};
        if (navigator.clipboard) {{
            try {{ await navigator.clipboard.writeText(text); return 'ok'; }} catch (e) {{}}
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
        return ok ? 'ok' : 'unavailable';
    }})()
    """
    try:
        result = await ui.run_javascript(js, timeout=5)
    except Exception as e:  # the browser did not answer
        logger.debug("Copy to clipboard failed: %s", e)
        result = None
    if result == "ok":
        ui.notify(f"{what.capitalize()} copied to clipboard", type="positive")
    elif result == "unavailable":
        ui.notify("This browser allows copying only over HTTPS or on localhost", type="warning")
    else:
        ui.notify("Copy to clipboard failed", type="negative")
