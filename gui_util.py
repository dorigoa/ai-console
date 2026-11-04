"""Subprocess, SSH and HTTP helpers.

Every command is logged with its outcome: at INFO for user actions, at DEBUG
for the periodic probes (routine=True), which would flood the journal.
"""

import asyncio
import shlex

import requests

from gui_logging import logger

# BatchMode: fail at once instead of waiting for a password prompt.
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no")

# trust_env=False: ignore http(s)_proxy from the environment, which would send the
# LAN probes through a proxy and report healthy servers as down.
_http = requests.Session()
_http.trust_env = False


def one_line(text: str) -> str:
    """text with every run of whitespace collapsed: one log line per command."""
    return " ".join(text.split())


def brief(text: str, limit: int = 200) -> str:
    """A short one-line summary of a command output, for the log."""
    text = one_line(text)
    if not text:
        return "no output"
    return text if len(text) <= limit else f"{text[:limit]} ... ({len(text)} chars)"


def last_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


def is_alive(out: str) -> bool:
    """Whether a gui_shell.status_cmd answer reports live processes. Only its last
    line counts: anything a login shell prints before it is ignored."""
    return last_line(out).startswith("ALIVE")


async def run(argv: list[str], *, what: str, routine: bool = False,
              env: dict[str, str] | None = None, shown: str = "") -> tuple[int, str]:
    """Run argv without blocking the UI; return (returncode, stdout).

    what: the action, as the log names it. shown: the command as logged (default: argv)."""
    log = logger.debug if routine else logger.info
    log("%s: running %s", what, shown or shlex.join(argv))
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    stdout, stderr = await proc.communicate()
    out = stdout.decode(errors="replace")
    log("%s: rc=%s (%s)", what, proc.returncode, brief(out))
    if stderr:
        logger.debug("%s: stderr: %s", what, brief(stderr.decode(errors="replace")))
    return proc.returncode, out


async def ssh_run(ip: str, user: str, cmd: str, *, what: str, routine: bool = False) -> tuple[int, str]:
    """Run cmd on user@ip; return (returncode, output).

    The remote stderr is merged into the output, so the UI shows the error and not
    just its code. rc 255 is ssh's own failure: the node was not reached."""
    if not ip or not user:
        return 255, ""
    cmd = f"{{ {cmd}; }} 2>&1"
    return await run(["ssh", *SSH_OPTS, f"{user}@{ip}", cmd], what=what, routine=routine,
                     shown=f"ssh {user}@{ip} {one_line(cmd)}")


def api_check(ip: str, port: int) -> tuple[bool, str]:
    """(answers, model id) for GET http://ip:port/v1/models. Any failure means "not up".

    Blocking: call it through asyncio.to_thread."""
    url = f"http://{ip}:{port}/v1/models"
    try:
        r = _http.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
        # llama.cpp lists the model under both keys: data[].id and models[].name.
        entries = (data.get("data") or []) + (data.get("models") or [])
    except Exception as e:  # down, refused, timeout, not the expected JSON...
        logger.debug("API check %s: %s", url, e)
        return False, ""
    for entry in entries:
        if isinstance(entry, dict):
            entry = entry.get("id") or entry.get("name") or entry.get("model") or ""
        if entry:
            logger.debug("API check %s: serving %s", url, entry)
            return True, str(entry)
    return True, ""  # up, but names no model
