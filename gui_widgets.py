"""Small NiceGUI interaction helpers shared by the console panels.

None of these own any state: they wrap a widget detail (a class toggle, a
spinner, the clipboard) that several panels would otherwise repeat.
"""

import asyncio
import json

from nicegui import ui

from gui_logging import logger


def set_class(element: ui.element, name: str, on: bool) -> None:
    if on:
        element.classes(add=name)
    else:
        element.classes(remove=name)


async def exclusive(lock: asyncio.Lock, busy_msg: str, button: ui.button, action) -> None:
    """Run action unless lock is already held (then just warn), with a spinner on
    button meanwhile (a loading button also ignores further clicks)."""
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
    except Exception as e:
        logger.debug("Copy to clipboard: the browser did not answer: %s", e)
        result = None
    if result in ("clipboard", "exec"):
        ui.notify(f"{what.capitalize()} copied to clipboard", type="positive")
    elif result == "unavailable":
        ui.notify("Copying is blocked: this browser only supports the "
                  "clipboard over HTTPS or localhost", type="warning")
    else:
        ui.notify("Copy to clipboard failed", type="negative")
