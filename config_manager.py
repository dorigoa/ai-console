
from dataclasses import dataclass, fields
from typing import Optional, Any, Dict
from logzero import logger
from pathlib import Path
import threading
import json
import sys
import os

from errors import ConfigError

_LOCAL_CONFIG = Path(__file__).parent / "config.json"
CONFIG_FILE = Path(
    os.getenv('LLAMA_CONSOLE_CONFIG_FILE') or
    (_LOCAL_CONFIG if _LOCAL_CONFIG.exists() else Path.home() / "llama-console-config.json")
)

#_________________________________________________________________________________________
@dataclass
class Settings:
    ADDRESS_BIND: str = ""
    PORT_BIND: int = 0
    UI_TITLE: str = "Custom UI Title"
    LLAMA_SERVER_BIN: str = ""
    LLAMA_SERVER_HOST: str = ""
    LLAMA_SERVER_USER: str = ""
    MODELS_JSON: Path = None
    RPC_JSON: Path = None
    LLAMA_LOG_FILE: str = ""  # shared path for polling the output
    LLAMA_BOOT_LOG: str = ""  # startup stdout/stderr (crash diagnostics)
    UI_PORT: int = 8501       # port the NiceGUI console listens on
    # Log level of the NiceGUI console (DEBUG/INFO/WARNING/ERROR); "" means "use
    # the AI_CONSOLE_LOG_LEVEL environment, else INFO".
    UI_LOG_LEVEL: str = ""
    # llama-server tunables that used to be hardcoded in command_builder.py.
    SEED: int = 123456789
    FITC: int = 8192
    NP: int = 1               # parallel sequences; 1 is required by MTP processing
    DEFAULT_CTX: int = 8192

#_________________________________________________________________________________________
def _load_overrides(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        logger.warning(f"Config override not found in {path}: using defaults.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"File {path} must contain a JSON object, found {type(data).__name__}.")
    logger.info(f"Config override loaded from {path} ({len(data)} keys).")
    for k in data:
        logger.debug(f"'{k}': '{data[k]}'")
    return data

#_________________________________________________________________________________________
def _coerce(value: Any, target_type: Any, key: str) -> Any:
    try:
        if target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if target_type in (int, float, str, Path):
            return target_type(value)
        return value
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"Invalid value for '{key}': expected {getattr(target_type, '__name__', target_type)}, "
            f"got {value!r} ({e})."
        ) from e

#_________________________________________________________________________________________
def _build_settings() -> Settings:
    s = Settings()
    overrides = _load_overrides( CONFIG_FILE )
    if not overrides:
        return s

    type_by_name = {f.name: f.type for f in fields(Settings)}
    unknown = set(overrides) - set(type_by_name)

    if unknown:
        raise ConfigError(
            f"Unknown keys in config: {sorted(unknown)}. "
            f"Valid ones are: {sorted(type_by_name)}"
        )

    for k, v in overrides.items():
        setattr(s, k, _coerce(v, type_by_name[k], k))

    return s

_settings_lock = threading.Lock()
_settings_instance: Optional[Settings] = None

#_________________________________________________________________________________________
def get_settings() -> Settings:
    """The process-wide Settings, built once from CONFIG_FILE.

    Unlike the other loaders, a ConfigError is turned into a clean message and
    SystemExit right here rather than propagated: every module calls this at
    IMPORT time (see the note at the top of start_model.py), so there is no
    caller in a position to handle it — a bare traceback would be the only
    alternative.
    """
    global _settings_instance
    with _settings_lock:
        if _settings_instance is None:
            try:
                _settings_instance = _build_settings()
            except ConfigError as e:
                logger.error(f"Configuration error: {e}")
                raise SystemExit(1) from e
    return _settings_instance

if __name__ == "__main__":
    try:
        _build_settings()
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)