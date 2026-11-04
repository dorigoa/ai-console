"""Console settings and shared constants, read once at startup.

config.json  UI_TITLE, UI_PORT, UI_LOG_LEVEL (the other keys belong to the old start_model.py)
nodes.json   NODES: ip, SSH user and API port of each service node
"""

import contextlib
import json
import logging
import os
import secrets
from pathlib import Path

from gui_logging import logger

HERE = Path(__file__).parent  # the JSON files live next to the code
CONFIG_FILE = Path(os.environ.get("LLAMA_CONSOLE_CONFIG_FILE") or HERE / "config.json")


def read_json(path: Path, key: str = "") -> dict:
    """The JSON object in `path` (or its `key` section); {} if missing or malformed."""
    where = f"{path}[{key}]" if key else str(path)
    try:
        data = json.loads(path.read_text())
        if key:
            data = data[key]
    except FileNotFoundError:
        logger.warning("%s not found", path)
        return {}
    except (OSError, ValueError, LookupError, TypeError) as e:
        logger.error("Cannot read %s: %r", where, e)
        return {}
    if not isinstance(data, dict):
        logger.error("Cannot read %s: not a JSON object", where)
        return {}
    return data


_config = read_json(CONFIG_FILE)
UI_TITLE = str(_config.get("UI_TITLE") or "LLM Inference Console")  # browser tab title
UI_PORT = int(_config.get("UI_PORT") or 8501)
UI_LOG_LEVEL = str(_config.get("UI_LOG_LEVEL") or "")
NODES = read_json(HERE / "nodes.json", "NODES")

SHOW_LEGACY_LLAMA_TAB = False  # True brings back the old start_model.py tab (gui_legacy)
POLL_SECONDS = 5.0             # status probe period of each service
REPO_URL = "https://github.com/dorigoa/ai-console"
GREEN, ORANGE, RED = "#00ff88", "orange", "red"  # status colours

ACTIVE_TAB_KEY = "active_tab"  # app.storage.user key: the tab each browser was last on
STORAGE_SECRET_FILE = HERE / ".storage_secret"


def storage_secret() -> str:
    """The key signing the per-browser cookies; kept on disk so they survive restarts."""
    with contextlib.suppress(OSError):  # missing or unreadable: make a new one
        if secret := STORAGE_SECRET_FILE.read_text().strip():
            return secret
    secret = secrets.token_urlsafe(32)
    try:
        STORAGE_SECRET_FILE.touch(mode=0o600)  # private before it holds the secret
        STORAGE_SECRET_FILE.write_text(secret)
        logger.info("New storage secret saved in %s", STORAGE_SECRET_FILE)
    except OSError as e:
        logger.warning("Cannot save %s (%s): browsers will forget their tab at restart",
                       STORAGE_SECRET_FILE, e)
    return secret


def log_boot_config() -> None:
    """One summary of what this process runs with."""
    logger.info("Config %s: title %r, port %s, log level %s, legacy tab %s", CONFIG_FILE, UI_TITLE,
                UI_PORT, logging.getLevelName(logger.level),
                "shown" if SHOW_LEGACY_LLAMA_TAB else "hidden")
    logger.info("Nodes (nodes.json): %s", ", ".join(
        f"{name}={node.get('user', '?')}@{node.get('ip', '?')}" for name, node in NODES.items()) or "none")
