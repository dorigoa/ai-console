"""Journal-friendly logging for the NiceGUI console.

logzero's default logger writes to *stderr*: started as a service (or with
stdout piped anywhere) the console looked completely mute, and the events worth
reading were logged at debug level besides. This module reconfigures it:

* output goes to stdout, line-buffered, so both `journalctl -u <unit>` and a
  terminal show it interleaved with NiceGUI's own messages;
* lines are stamped "[I 2026-11-04 13:01:14 gui_services:123] message", with no
  ANSI colours (the journal would print them as escape sequences);
* the level comes from the UI_LOG_LEVEL setting, else from AI_CONSOLE_LOG_LEVEL,
  else INFO.

What is logged at which level: everything the user asks for (a remote command
about to run, the parameters picked for a model, a status change) is INFO, the
periodic chatter (a status probe every POLL_SECONDS per service) is DEBUG, so a
journal left at INFO stays readable.

Call setup_logging() before importing anything that logs at import time
(config_manager does, while reading config.json). The console entry point does
exactly that, and repeats the call once the settings are known.
"""

import logging
import os
import sys

import logzero
from logzero import logger

ENV_LEVEL = "AI_CONSOLE_LOG_LEVEL"

_FORMAT = "[%(levelname).1s %(asctime)s %(module)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LEVELS = {name: getattr(logzero, name)
           for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")}


def resolve_level(level: str | None) -> tuple[int, str]:
    """(loglevel, where it came from): the setting, else the environment, else INFO."""
    for source, value in (("UI_LOG_LEVEL", level), (ENV_LEVEL, os.environ.get(ENV_LEVEL))):
        name = (value or "").strip().upper()
        if not name:
            continue
        if (loglevel := _LEVELS.get(name)) is not None:
            return loglevel, f"{source}={name}"
        # A typo in the configuration must not cost the run its logging.
        return logzero.INFO, f"unknown {source}={name!r}, using INFO"
    return logzero.INFO, "default"


def setup_logging(level: str | None = None, *, stream=None) -> int:
    """Point the logzero logger at stdout, at the requested level; return the level.

    Safe to call more than once (the entry point does: once before the imports,
    once with the configured level): the stream handler is replaced, never
    stacked. A FileHandler added by hand is left alone.
    """
    loglevel, source = resolve_level(level)
    stream = sys.stdout if stream is None else stream
    try:
        # Line buffering, so a pipe (or the journal) sees the lines as they happen.
        stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass  # not a regular text stream (a test runner's capture, a closed stdout, ...)

    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logzero.loglevel(loglevel)  # sets the level on the logger and on its handlers

    logger.info("Logging at %s (%s) to %s", logging.getLevelName(loglevel), source,
                getattr(stream, "name", stream))
    return loglevel


def one_line(text: str) -> str:
    """text with every run of whitespace collapsed to a single space.

    Remote commands and API payloads go to the log: one entry per command keeps
    the journal greppable."""
    return " ".join(text.split())


def brief(text: str, limit: int = 200) -> str:
    """A short one-line summary of what a command printed ('no output' when silent)."""
    text = one_line(text)
    if not text:
        return "no output"
    return text if len(text) <= limit else f"{text[:limit]} ... ({len(text)} chars)"
