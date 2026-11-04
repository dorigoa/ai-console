"""Logging: logzero's logger, written to stdout so that `journalctl -u <unit>` shows it.

Level: UI_LOG_LEVEL in config.json, else $AI_CONSOLE_LOG_LEVEL, else INFO.
INFO shows what the user asked for (commands run, status changes); DEBUG adds
the periodic status probes.
"""

import contextlib
import logging
import os
import sys

from logzero import logger

ENV_LEVEL = "AI_CONSOLE_LOG_LEVEL"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Plain text, no colours: the journal would print the escape codes.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname).1s %(asctime)s %(module)s:%(lineno)d] %(message)s",
                                        "%Y-%m-%d %H:%M:%S"))


def setup_logging(level: str = "") -> None:
    """Send the logger to stdout at `level` (see above). Safe to call again."""
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)  # each line reaches the journal at once
    for handler in list(logger.handlers):
        logger.removeHandler(handler)  # logzero's own coloured stderr handler
    logger.addHandler(_handler)

    name = (level or os.environ.get(ENV_LEVEL) or "INFO").strip().upper()
    if name not in LEVELS:
        logger.warning("Unknown log level %r, using INFO", name)
        name = "INFO"
    logger.setLevel(name)
