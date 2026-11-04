"""Subprocess, SSH and HTTP helpers the console panels run their work through.

Every call here is a debugging checkpoint: what is about to run and what came
back go to the log (see gui_logging), at INFO for the actions the user asked for
and at DEBUG for the periodic status probes, which would otherwise flood the
journal every POLL_SECONDS.
"""

import asyncio
import json
import shlex

import requests

from gui_logging import brief, logger, one_line

# BatchMode: never block the UI waiting for a password prompt.
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no")

# Nodes found unconfigured, so the warning is printed once and not again on
# every status poll (the panels poll each other POLL_SECONDS apart).
_warned_unconfigured: set[str] = set()


async def capture(argv: list[str], *, what: str = "", routine: bool = False,
                  env: dict[str, str] | None = None) -> tuple[str, int]:
    """Run argv to completion off the event loop; return (stdout, returncode).

    `what` names the action in the log ("status probe", "start llama-server"):
    a bare argv is hard to grep for later. routine=True demotes both lines to
    DEBUG, for the calls a timer repeats. env replaces the child environment
    when it must differ from this process' (see gui_legacy.START_MODEL_ENV)."""
    label = what or argv[0]
    log = logger.debug if routine else logger.info
    log("%s: running %s", label, one_line(shlex.join(argv)))
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
    stdout, stderr = await proc.communicate()
    returncode = proc.returncode
    out = stdout.decode(errors="replace")
    log("%s: finished rc=%s (%s)", label, returncode, brief(out))
    if stderr:
        logger.debug("%s: stderr: %s", label, brief(stderr.decode(errors="replace")))
    return out, returncode


async def ssh_run(ip: str, user: str, cmd: str, *, merge_stderr: bool = False,
                  what: str = "", routine: bool = False) -> tuple[int, str]:
    """Run cmd on user@ip; return (returncode, stdout).

    rc 255 is ssh's own failure code: the node was not reached (or has no
    address). merge_stderr folds the remote stderr into stdout, for the macOS
    nodes: without it a "python3 not found" or a shell error would reach the UI
    as a bare rc=N. The full remote command line is logged, which is also what
    to paste into a shell when a node misbehaves."""
    label = what or f"{user or '?'}@{ip or '?'}"
    if not ip or not user:
        if label not in _warned_unconfigured:
            _warned_unconfigured.add(label)
            logger.warning("%s: no ip/user for this node (nodes.json), not running: %s",
                           label, one_line(cmd))
        return 255, ""

    if merge_stderr:
        cmd = f"{{ {cmd}; }} 2>&1"
    log = logger.debug if routine else logger.info
    log("%s: remote command on %s@%s: %s", label, user, ip, one_line(cmd))
    # routine=True: capture's own lines would only repeat this one.
    out, returncode = await capture(["ssh", *SSH_OPTS, f"{user}@{ip}", cmd],
                                    what=label, routine=True)
    log("%s: remote command on %s@%s finished rc=%s (%s)", label, user, ip, returncode, brief(out))
    return returncode, out


def model_id_from_body(body: str) -> tuple[bool, str]:
    """(ok, model_id) from the body of a /v1/models answer.

    Tolerates the noise a login shell may print around the JSON (the payload is
    read from the first { to the last }). llama.cpp answers with the running
    model under data[0].id:

        {"models":[{"name":"Gemma-4-31B-it (qat-ud-q4_k_xl)", ...}],
         "object":"list",
         "data":[{"id":"Gemma-4-31B-it (qat-ud-q4_k_xl)", ...}]}

    older builds list the models under "models" only (objects with a "name", or
    bare strings); an answer that is valid JSON but names no model still counts
    as "the API is up".
    """
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return False, ""
    try:
        data = json.loads(body[start:end + 1])
    except ValueError:
        logger.debug("API check: the answer is not JSON: %s", brief(body))
        return False, ""
    if not isinstance(data, dict):
        return False, ""
    model_id = ""
    for entry in (data.get("data") or []) + (data.get("models") or []):
        if isinstance(entry, dict):
            model_id = str(entry.get("id") or entry.get("name")
                           or entry.get("model") or "")
        elif isinstance(entry, str):
            model_id = entry
        if model_id:
            break
    return True, model_id


# One session per process, with trust_env off: without it requests obeys any
# http_proxy/HTTPS_PROXY in the console's environment and sends the LAN probe
# through that proxy, which reports a perfectly healthy llama-server as dead
# (ProxyError on every poll, so the tab stays "STARTING" forever).
_direct_session: requests.Session | None = None


def _direct_get(url: str, timeout: float) -> requests.Response:
    global _direct_session
    if _direct_session is None:
        _direct_session = requests.Session()
        _direct_session.trust_env = False
    return _direct_session.get(url, timeout=timeout)


def api_check(ip: str, port: int) -> tuple[bool, str]:
    """(ok, model_id) from an OpenAI-compatible /v1/models endpoint.

    This is THE readiness check: a listening llama-server answers it with the
    JSON of the model in execution, and the poll that says so runs every
    POLL_SECONDS (gui_page's timer). Blocking: run it in a worker thread.
    Any failure reads as "not ready"."""
    url = f"http://{ip}:{port}/v1/models"
    logger.info(f"Checking API {url}")
    try:
        r = _direct_get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:
        # The panels poll this every POLL_SECONDS: debug level, the status line
        # and its transition tell the user what they need.
        logger.error("API check failed: GET %s: %s", url, e)
        return False, ""
    ok, model_id = model_id_from_body(r.text)
    if ok:
        logger.info("API check ok: GET %s serving %s", url, model_id or "(no id)")
    return ok, model_id


def last_line(text: str) -> str:
    text = text.strip()
    return text.splitlines()[-1] if text else ""


def is_alive(out: str) -> bool:
    """Whether a status_cmd answer reports live processes (its last line, so any
    noise printed by the remote login shell is ignored)."""
    return last_line(out).startswith("ALIVE")


def proc_state(rc: int, out: str) -> str:
    """Classify a status_cmd answer: running / stopped / unreachable / error."""
    if rc == 255:
        return "unreachable"
    if rc != 0:
        return "error"
    return "running" if is_alive(out) else "stopped"
