"""Configuration shared by the console modules, read once at import time.

Everything here is static for the life of the process: the Settings built from
config.json, the node/RPC catalogs next to the script, and the constants the
entry point, the panels and the page all need to agree on.
"""

import base64
import json
import os
from pathlib import Path

from config_manager import get_settings

from gui_logging import logger

# Every file below (config, catalogs, the storage secret) is looked up next to
# this file, however the console was started.
HERE = Path(__file__).parent

settings = get_settings()

# While hidden, the legacy start_model.py tab is neither built nor polled, and
# the model list it needs is never scanned. Set to True to bring it back.
SHOW_LEGACY_LLAMA_TAB = False

POLL_SECONDS = 5.0
REPO_URL = "https://github.com/dorigoa/ai-console"

# The three status colours the panels paint with (see gui_theme for the CSS).
GREEN, ORANGE, RED = "#00ff88", "orange", "red"

# app.storage.user key remembering the tab each browser was last on.
ACTIVE_TAB_KEY = "active_tab"
# Generated once and kept next to this file, so the signed cookies of
# app.storage.user stay valid across restarts.
STORAGE_SECRET_FILE = HERE / ".storage_secret"


def json_section(filename: str, key: str) -> dict:
    """The `key` section of a JSON file next to this script ({} if missing or malformed).

    A missing file is a supported state (the tabs still build, and report the
    node as unconfigured); a malformed one is a bug worth reading about, so it
    is logged rather than swallowed.
    """
    path = HERE / filename
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("%s not found: the services that need it will report 'not configured'", path)
        return {}
    except json.JSONDecodeError as e:
        logger.error("%s is not valid JSON (%s): treating it as empty", path, e)
        return {}
    section = data.get(key)
    if not isinstance(section, dict):
        logger.error("%s has no '%s' object (found %s): treating it as empty",
                     path, key, type(section).__name__)
        return {}
    return section


RPC_SERVERS: dict = json_section("rpc.json", "RPC_SERVERS")
NODES: dict = json_section("nodes.json", "NODES")


def storage_secret() -> str:
    """The secret NiceGUI signs the per-browser cookies with (see STORAGE_SECRET_FILE)."""
    try:
        return STORAGE_SECRET_FILE.read_text().strip()
    except OSError:
        secret = base64.urlsafe_b64encode(os.urandom(32)).decode()
        try:
            STORAGE_SECRET_FILE.write_text(secret)
            os.chmod(STORAGE_SECRET_FILE, 0o600)
        except OSError:
            pass  # still usable for this run
        logger.info("Generated a new storage secret in %s", STORAGE_SECRET_FILE)
        return secret


def log_boot_config() -> None:
    """Summarise at INFO what this process is about to run with."""
    logger.info("Console title: %s (legacy llama tab: %s)", settings.UI_TITLE,
                "shown" if SHOW_LEGACY_LLAMA_TAB else "hidden")
    logger.info("%d node(s) in nodes.json: %s", len(NODES),
                ", ".join(f"{name}={spec.get('ip', '?')}@{spec.get('user', '?')}"
                          for name, spec in NODES.items()) or "-")
    logger.info("%d RPC server(s) in rpc.json: %s", len(RPC_SERVERS),
                ", ".join(RPC_SERVERS) or "-")
    logger.info("Model catalog: %s (inference host %s)", settings.MODELS_JSON,
                settings.LLAMA_SERVER_HOST or "localhost")
