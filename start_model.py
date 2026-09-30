#!/usr/bin/env python3
"""Launch llama-server for a given model defined in models.json.

When LLAMA_SERVER_HOST is set the server is **detached** (nohup + background) so
this script exits immediately.  Log output goes to a well-known log file that the
UI can poll via SSH.
"""


import shlex
import sys
import json
import time
import logzero
import argparse

from pathlib import Path
from dataclasses import dataclass
from logzero import logger

from remote_cmd_executor import remote_exec
from rpc import RpcServer, load_rpcs
import utils

# The loglevel must be set HERE, before the project imports below: command_builder
# calls get_settings() at import time, and config_manager logs at info/debug level
# while building the Settings — long before argparse gets a chance to see --debug.
# So we pre-scan sys.argv to know the flag early.
logzero.loglevel(logzero.DEBUG if "--debug" in sys.argv else logzero.INFO)

from errors import ConfigError
from model import Model, load_models
from config_manager import get_settings
from command_builder import build_command
from rpc_check import unreachable_rpc_servers, start_rpc_server, wait_for_rpc_servers, kill_rpc_server, rpc_peers


# Backend init (Metal/CUDA enumeration) alone can take ~10s, so this is deliberately
# generous: it is a deadlock guard, not a latency budget.
_LIST_DEVICES_TIMEOUT = 60

settings = get_settings()

#___________________________________________________________________________________
def report_server_status(as_json: bool = False) -> bool:
    """Print whether llama-server is running. Return True if running."""
    info = utils.server_status(settings)
    if as_json:
        print(json.dumps(info))
        return info["running"]

    if not info["running"]:
        print(f"llama-server is NOT RUNNING on {info['where']}")
        return False

    logger.debug(f"Dict INFO={info}")

    print(f"llama-server is RUNNING on {info['where']} (pid(s): {', '.join(info['pids'])})")
    if info["ready"]:
        print(f"Running model: {info['model']} - CTX Size: {info['ctx']} - Temp: {info['temperature']:.1f} - Top-k: {info['top_k']} - Top-p: {info['top_p']:.2f} - Min-p: {info['min_p']:.2f} - Quant: {info['quant']}")
    else:
        print(f"llama-server is RUNNING but not ready yet: {info['error']}")
    return True

#___________________________________________________________________________________
def _rpc_endpoints_from_cmdline(cmdline: str) -> list[str]:
    """The RPC endpoints (ip:port) of the --rpc argument of a llama-server
    command line, in order. Handles both '--rpc ip:port,ip:port' and
    '--rpc=ip:port'. Returns [] when the flag is absent (no RPC)."""
    try:
        tokens = shlex.split(cmdline)
    except ValueError:
        tokens = cmdline.split()
    for i, tok in enumerate(tokens):
        if tok == "--rpc":
            if i + 1 >= len(tokens):
                return []
            return [e for e in tokens[i + 1].split(",") if e]
        if tok.startswith("--rpc="):
            return [e for e in tok.split("=", 1)[1].split(",") if e]
    return []

#___________________________________________________________________________________
def _running_server_rpc_addrs(pids: list[str]) -> list[RpcServer]:
    """The rpc-servers the running llama-server process(es) hold, as RpcServer objects.

    Resolved from the --rpc argument of each process's command line (read with
    `ps` on the server host), NOT from models.json: the command line reflects what
    the server was actually started with, including --override-rpc, which
    models.json does not.

    Returns [] when no process was started with --rpc. Endpoints that no longer
    resolve in rpc.json (the file changed after the server started) are reported
    and skipped, because kill_rpc_server() needs the remuser that a bare
    endpoint does not carry."""
    endpoints: list[str] = []
    for pid in pids:
        try:
            r = utils.run_on_server(settings, f"ps -p {pid} -o command=")
        except utils.ServerHostUnreachable as e:
            logger.warning(f"Could not read the command line of pid {pid}: {e}")
            continue
        logger.debug(f"llama-server pid {pid} command line: {r.stdout.strip()}")
        endpoints.extend(_rpc_endpoints_from_cmdline(r.stdout))

    if not endpoints:
        return []

    if not settings.RPC_JSON:
        logger.warning(f"llama-server was using RPC endpoint(s) {', '.join(endpoints)} but no RPC_JSON is configured; cannot kill its rpc-servers")
        return []
    try:
        known = {s.endpoint(): s for s in load_rpcs(settings.RPC_JSON).values()}
    except (ConfigError, OSError) as e:
        logger.warning(f"llama-server was using RPC endpoint(s) {', '.join(endpoints)} but {settings.RPC_JSON} could not be loaded ({e}); cannot kill its rpc-servers")
        return []

    addrs: list[RpcServer] = []
    for ep in endpoints:
        if any(a.endpoint() == ep for a in addrs):
            continue  # the same node may appear for more than one process
        addr = known.get(ep)
        if addr is None:
            logger.warning(f"llama-server was using RPC endpoint {ep}, which is not in {settings.RPC_JSON}; cannot kill that rpc-server")
            continue
        addrs.append(addr)
    return addrs

#___________________________________________________________________________________
def _kill_rpc_servers(rpc_addrs: list[RpcServer], exec_host: str | None) -> list[str]:
    """Kill ggml-rpc-server on every given RPC node (killall, see
    rpc_check.kill_rpc_server). Returns the endpoints (ip:port) where the
    kill could not be confirmed."""
    failed: list[str] = []
    for addr in rpc_addrs:
        via = f"{exec_host} -> " if exec_host else ""
        logger.info(f"Killing ggml-rpc-server on {via}{addr.remuser}@{addr.IP} ...")
        if not kill_rpc_server(addr, exec_host=exec_host):
            failed.append(addr.endpoint())
    return failed

#___________________________________________________________________________________
def stop_server() -> bool:
    """Kill the llama-server process on the server — and the rpc-servers it was
    using, so the whole deployment goes down, not just the server: a leftover
    ggml-rpc-server keeps its GPU memory allocated for a client that is gone.

    The related rpc-servers are read from the command line of the running
    process(es) BEFORE it is killed (see _running_server_rpc_addrs), and killed
    afterwards, once the single-client session is freed.

    Return True if the server and all of its rpc-servers are stopped."""
    where = utils.server_location(settings)
    pids = utils.server_pids(settings)
    if not pids:
        logger.warning(f"No llama-server process found on {where}.")
        return True

    rpc_addrs = _running_server_rpc_addrs(pids)

    pattern = settings.LLAMA_SERVER_BIN#_pgrep_pattern()
    logger.debug(f"Sending SIGTERM to llama-server on {where} (pid(s): {', '.join(pids)})...")
    utils.run_on_server(settings, f"pkill -TERM -f -- '{pattern}'")

    time.sleep(2)
    pids = utils.server_pids(settings)
    if pids:
        logger.debug(f"Still running (pid(s): {', '.join(pids)}); sending SIGKILL...")
        utils.run_on_server(settings, f"pkill -KILL -f -- '{pattern}'")
        time.sleep(1)
        pids = utils.server_pids(settings)

    if pids:
        logger.error(f"Error: could not stop llama-server on {where} (pid(s): {', '.join(pids)}).")
        return False
    print(f"llama-server stopped on {where}.")

    if not rpc_addrs:
        return True

    # llama-server was the only client of these ggml-rpc-servers; with the
    # session freed, kill them so their GPUs are released.
    ssh_dest = utils.ssh_dest(settings)
    logger.info(f"Killing the {len(rpc_addrs)} rpc-server(s) of this deployment ...")
    failed = _kill_rpc_servers(rpc_addrs, ssh_dest)
    if failed:
        logger.error(f"Could not kill rpc-server on: {', '.join(failed)}")
        return False
    print(f"rpc-server(s) stopped: {', '.join(a.endpoint() for a in rpc_addrs)}.")
    return True

#___________________________________________________________________________________
def tail_log(lines: int = 50, follow: bool = True) -> int:
    """Stream llama-server's log file from the server to this terminal.

    Uses `tail` over SSH (or locally) on LLAMA_LOG_FILE. With follow=True it uses
    `tail -F` (follow by name + retry), so it survives the file being rotated or
    recreated on a server restart, and keeps streaming until the user hits Ctrl-C.

    Returns the tail/ssh exit code (0 on a clean Ctrl-C)."""
    where = utils.server_location(settings)
    ssh_dest = utils.ssh_dest(settings)

    tail_args = ["tail", "-n", str(int(lines))]
    if follow:
        tail_args.append("-F")
    tail_args.append(shlex.quote(settings.LLAMA_LOG_FILE))
    tail_cmd = " ".join(tail_args)

    if follow:
        logger.info(f"Following {settings.LLAMA_LOG_FILE} on {where} (Ctrl-C to stop)...")
    else:
        logger.info(f"Last {int(lines)} lines of {settings.LLAMA_LOG_FILE} on {where}:")

    if ssh_dest:
        # -t allocates a remote pty so Ctrl-C is forwarded and the remote `tail`
        # is torn down cleanly instead of lingering.
        argv = ["ssh", "-t", "-o", "ConnectTimeout=5", ssh_dest, tail_cmd]
    else:
        argv = ["bash", "-c", tail_cmd]

    logger.debug(f"Running command {argv}")
    try:
        r = remote_exec(argv, timeout=None, capture_output=False)
        if not r:
            raise utils.ServerHostUnreachable(f"could not tail the log on {where}")
    except KeyboardInterrupt:
        return 0
    return r.returncode

#___________________________________________________________________________________
def _run_server_action(action) -> "None":
    """Run a server-management action and exit: 0 = ok, 1 = not ok,
    2 = LLAMA_SERVER_HOST unreachable."""
    try:
        ok = action()
    except utils.ServerHostUnreachable as e:
        logger.error(f"Error: host unreachable — {e}")
        sys.exit(2)
    sys.exit(0 if ok else 1)

#___________________________________________________________________________________
def _launch_detached(cmd: list[str], ssh_dest: str | None) -> None:
    """Start llama-server in the background, detached from this session, then
    VERIFY it actually stayed up before exiting.

    Detach mechanics: redirect ALL of stdin/stdout/stderr away from the SSH
    channel. Redirecting only stdin (an older behaviour) left stdout/stderr
    wired to the SSH pipe, so ssh kept the channel open (this script would hang
    waiting for EOF) and, once the connection was torn down, llama-server could
    be killed by SIGHUP/SIGPIPE. `nohup` ignores SIGHUP; </dev/null frees the
    ssh channel so ssh returns immediately. Since ssh is invoked without a PTY
    there is no controlling terminal, so nohup alone is enough to survive the
    parent shell exiting.

    `setsid` additionally starts a brand-new session (extra isolation), but it
    is util-linux only and does NOT exist on macOS. We therefore detect it at
    runtime in the remote shell and use `setsid nohup` when present, falling
    back to plain `nohup` otherwise — portable across Linux and macOS.

    stdout/stderr go to LLAMA_BOOT_LOG (not /dev/null): an early crash (bad
    argument, missing shared library, OOM) happens before llama-server opens
    --log-file, so without a boot log it would leave no trace at all. The
    server's own runtime output still lands in LLAMA_LOG_FILE via --log-file,
    which --tail-log can follow.
    """
    server_args = " ".join(shlex.quote(a) for a in cmd)
    boot_log = shlex.quote(settings.LLAMA_BOOT_LOG)
    # D = 'setsid nohup' where setsid exists (Linux), else 'nohup' (macOS).
    # $D is intentionally unquoted so it word-splits into 1 or 2 words.
    detach = "D=nohup; command -v setsid >/dev/null 2>&1 && D='setsid nohup';"
    remote_cmd = f"{detach} $D {server_args} >{boot_log} 2>&1 </dev/null &"

    where = ssh_dest or "localhost"
    logger.info(f"Starting detached llama-server on {where} ...")
    logger.info(f"Log file: {settings.LLAMA_LOG_FILE}  (boot log: {settings.LLAMA_BOOT_LOG})")

    if ssh_dest:
        argv = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_dest, remote_cmd]
    else:
        argv = ["bash", "-c", remote_cmd]
    r = remote_exec( argv )
    if not r:
        raise utils.ServerHostUnreachable(f"timed out launching llama-server on {where}")
    if r.returncode != 0:
        logger.error(f"Failed to launch llama-server on {where}: {r.stderr.strip()}")
        sys.exit(1)

    # ssh/bash returns as soon as the background shell forks; that says nothing
    # about whether llama-server survived. Poll pgrep to confirm it is alive.
    for _ in range(10):
        time.sleep(1)
        try:
            pids = utils.server_pids(settings)
        except utils.ServerHostUnreachable as e:
            logger.warning(f"Cannot verify server yet: {e}")
            continue
        if pids:
            logger.info(f"llama-server RUNNING on {where} (pid(s): {', '.join(pids)}).")
            sys.exit(0)

    # Not alive: it crashed at startup. Show the boot log so the user knows why.
    logger.error(f"llama-server did NOT stay up on {where}. Boot log ({settings.LLAMA_BOOT_LOG}):")
    try:
        boot = utils.run_on_server(settings, f"tail -n 40 {shlex.quote(settings.LLAMA_BOOT_LOG)} 2>/dev/null || true")
    except utils.ServerHostUnreachable as e:
        # Don't let a second failure mask the launch failure we are reporting.
        logger.error(f"(boot log unavailable: {e})")
    else:
        logger.error("\n" + (boot.stdout.rstrip() if boot.stdout.strip() else "(boot log empty)"))
    sys.exit(1)

#___________________________________________________________________________________
def _list_devices(binary: str, rpc_list: str, ssh_dest: str | None) -> list[str]:
    """Device names reported by `llama-server --rpc <rpc_list> --list-devices`.

    Devices with no free memory are dropped: they cannot host any layer.
    Runs on LLAMA_SERVER_HOST when configured, since only that host reaches the
    RPC network."""
    if ssh_dest:
        argv = ["ssh", ssh_dest, binary, "--rpc", rpc_list, "--list-devices"]
    else:
        argv = [binary, "--rpc", rpc_list, "--list-devices"]

    logger.debug(f"Running list-devices: {argv}")
    
    result = remote_exec( argv, _LIST_DEVICES_TIMEOUT )
    if not result:
        logger.error(
                    f"list-devices timed out after {_LIST_DEVICES_TIMEOUT}s — the RPC server "
                    f"accepted the connection but never answered."
                )
        return []
    
    if result.stdout:
        logger.debug(result.stdout)
    if result.stderr:
        logger.debug(result.stderr)

    return [
        line.replace(":", "").split()[0]
        for line in result.stdout.splitlines()
        if line.strip()
        and "Available" not in line
        and " 0 MiB free" not in line
        and "BLAS" not in line
        and "CPU" not in line
    ]

#___________________________________________________________________________________
def _report_missing_rpc_devices(model: Model, devices: list[str], ssh_dest: str | None) -> None:
    """Explain why no RPC device came back although RPC servers are configured.

    Without this check the run would continue and build `--device RPC0,...` from a
    list that has no RPC entry, and llama-server would die remotely on the cryptic
    pair 'Failed to connect to <ip>:<port>' + 'invalid device: RPC0'.

    The usual cause is not the network: ggml-rpc-server serves ONE client at a
    time, so while another llama-server holds the session every further client
    waits unanswered in the listen backlog. That is why nc/telnet on the very same
    address and port succeed while llama-server cannot connect — the kernel
    completes the handshake regardless of whether the process ever accepts it."""
    logger.error(
        f"Model '{model.model_name}' needs RPC devices but --list-devices returned none.\n"
        f"  Devices seen: {', '.join(devices) or '(none)'}\n"
        f"  The RPC port answers at TCP level (nc/telnet would succeed), but the "
        f"server did not serve the request."
    )
    for addr in model.rpcservers:
        peers = rpc_peers(addr, exec_host=ssh_dest)
        if peers:
            logger.error(
                f"  {addr.IP}:{addr.PORT} is BUSY — already serving:\n    "
                + "\n    ".join(peers)
                + "\n  Stop that client (llama-server --kill-server, or --kill-rpc-server "
                  "to restart the RPC node) and retry."
            )
        else:
            logger.error(
                f"  {addr.IP}:{addr.PORT} has no client connected — the rpc-server is "
                f"likely wedged or was just restarted. Check its log on the RPC node."
            )

#___________________________________________________________________________________
@dataclass(frozen=True)
class LaunchOptions:
    """Everything main() collects from the command line, as one immutable bag.

    Field names match the argparse dest names exactly, so main() can build this
    with LaunchOptions(**vars(args)).
    """
    model_name: str | None = None
    dry_run: bool = False
    only_rpc: bool = False
    only_check_rpc: bool = False
    only_list_devs: bool = False
    list_models: bool = False
    kill_server: bool = False
    server_status: bool = False
    follow_log: bool = False
    tail_lines: int = 50
    kill_rpc: bool = False
    override_temp: float | None = None
    override_top_p: float | None = None
    override_top_k: int | None = None
    override_min_p: float | None = None
    override_devices: str | None = None
    override_ctx: int | None = None
    override_rpc: str | None = None
    override_kvq: str | None = None
    nomtp: bool = False
    as_json: bool = False
    debug: bool = False
    disable_thinking: bool = False

#___________________________________________________________________________________
def start_model(opts: LaunchOptions) -> None:
    if opts.server_status:
        _run_server_action(lambda: report_server_status(as_json=opts.as_json))

    if opts.kill_server:
        _run_server_action(stop_server)

    if opts.follow_log:
        try:
            rc = tail_log(lines=opts.tail_lines, follow=True)
        except utils.ServerHostUnreachable as e:
            logger.error(f"Error: host unreachable — {e}")
            sys.exit(2)
        sys.exit(rc)

    if opts.only_check_rpc or opts.only_list_devs or opts.only_rpc:
        models = load_models(Path(settings.MODELS_JSON), Path(settings.RPC_JSON), remote_host=settings.LLAMA_SERVER_HOST, remote_user=settings.LLAMA_SERVER_USER, check_remote_file=False)
    elif opts.list_models:
        models = load_models(Path(settings.MODELS_JSON), Path(settings.RPC_JSON), remote_host=settings.LLAMA_SERVER_HOST, remote_user=settings.LLAMA_SERVER_USER, check_remote_file=True)
    else:
        models = load_models(Path(settings.MODELS_JSON), Path(settings.RPC_JSON),remote_host=settings.LLAMA_SERVER_HOST, remote_user=settings.LLAMA_SERVER_USER, check_remote_file=True, check_model_name=opts.model_name)

    if opts.list_models:
        if opts.as_json:
            # Everything a UI needs about a model, in one call: without this the
            # GUI had to re-open and re-parse models.json on its own.
            print(json.dumps({"models": [
                {
                    "name": m.model_name,
                    "size_gib": m.size_gib,
                    "rpc_count": len(m.rpcservers),
                    "native_ctx": m.native_ctx,
                    "temperature": m.temperature,
                    "top_p": m.top_p,
                    "top_k": m.top_k,
                    "min_p": m.min_p,
                }
                for m in models
            ]}))
        else:
            print(
                "Available models:\n  "
                + "\n  ".join(
                    f"{m.model_name} ({int(m.size_gib) if m.size_gib is not None else '?'} GiB "
                    f"- {len(m.rpcservers)} RPC)"
                    for m in models
                )
            )
        sys.exit(0)

    if not opts.model_name:
        logger.error("Error: model_name is required")
        sys.exit(1)

    model = next((m for m in models if m.model_name == opts.model_name), None)
    if model is None:
        available = "\n  ".join(m.model_name for m in models)
        logger.error(f"Error: model '{opts.model_name}' not found in {settings.MODELS_JSON}.\n\nAvailable models:\n  {available}")
        sys.exit(1)

    if opts.override_temp:
        model.temperature = opts.override_temp
    if opts.override_top_p:
        model.top_p = opts.override_top_p
    if opts.override_top_k:
        model.top_k = opts.override_top_k
    if opts.override_min_p:
        model.min_p = opts.override_min_p
    if opts.override_kvq:
        model.kvquant = opts.override_kvq
    if opts.override_rpc:
        if opts.override_rpc != "none":
            rpcs = load_rpcs(settings.RPC_JSON)
            server_names = opts.override_rpc.split(',')
            model.rpcservers = []
            logger.debug(f"")
            for name in server_names:
                logger.debug(f"RPC Override: adding rpc {name}({rpcs[name].IP})")
                model.rpcservers.append( rpcs[name] )
        else:
            model.rpcservers = None



    binary = settings.LLAMA_SERVER_BIN
    ssh_dest = utils.ssh_dest(settings)

    if opts.only_check_rpc and model.rpcservers and len(model.rpcservers):
        # Check only: report which RPC servers are not running and exit.
        # Never start them (that's what --only-start-rpc is for).
        if not model.rpcservers:
            logger.info(f"Model '{model.model_name}' has no RPC servers configured.")
            sys.exit(0)
        logger.info(f"Checking rpc servers (via {ssh_dest or 'localhost'})...")
        dead = unreachable_rpc_servers(model.rpcservers, exec_host=ssh_dest)
        if dead:
            for addr in dead:
                logger.error(f"RPC server {addr.IP}:{addr.PORT} is NOT running")
            sys.exit(1)
        logger.info("All RPC servers reachable.")
        sys.exit(0)

    if opts.kill_rpc:
        # Kill rpc-server on every RPC node of the model (killall rpc-server),
        # originating from LLAMA_SERVER_HOST. Never starts anything.
        if not model.rpcservers:
            logger.info(f"Model '{model.model_name}' has no RPC servers configured.")
            sys.exit(0)
        failed = _kill_rpc_servers(model.rpcservers, ssh_dest)
        if failed:
            logger.error(f"Could not kill rpc-server on: {', '.join(failed)}")
            sys.exit(1)
        logger.info("rpc-server killed on all RPC nodes (or already stopped).")
        sys.exit(0)

    if ssh_dest:
        cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_dest, "test", "-f", binary]
        logger.debug(f"Executing {cmd}")
        # r = subprocess.run(
        #     cmd,
        #     capture_output=True, text=True,
        # )
        r = remote_exec (cmd)
        if not r:
            sys.exit(1)
        if r.returncode != 0:
            logger.error(f"Error: llama-server binary not found at '{binary}' on {ssh_dest}")
            sys.exit(1)
    else:
        if not Path(binary).is_file():
            logger.error(f"Error: llama-server binary not found at '{binary}'")
            sys.exit(1)

    if not opts.dry_run and not opts.only_rpc and not opts.only_list_devs:
        # A second llama-server cannot bind PORT_BIND anyway, and while the running
        # one holds the ggml-rpc-server session (one client at a time) the new one
        # cannot get an RPC device either. Both failures would only surface remotely,
        # in the boot log; refuse here instead.
        try:
            running = utils.server_pids(settings)
        except utils.ServerHostUnreachable as e:
            logger.error(f"Error: host unreachable — {e}")
            sys.exit(2)
        if running:
            logger.error(
                f"llama-server is already RUNNING on {utils.server_location(settings)} "
                f"(pid(s): {', '.join(running)}); it holds port {settings.PORT_BIND} and, "
                f"if it uses RPC, the rpc-server session too.\n"
                f"  Stop it first:  {Path(sys.argv[0]).name} --kill-server"
            )
            sys.exit(1)

    if not opts.dry_run and not opts.only_list_devs and model.rpcservers and not opts.override_devices:
        # RPC probing/starting must originate from LLAMA_SERVER_HOST: only it can
        # reach the RPC network. When ssh_dest is None (local llama-server), the
        # operations run locally as before.
        # Skipped for --only-list-devices: that option must NOT start anything.
        logger.debug(f"Checking rpc servers (via {ssh_dest or 'localhost'})...")
        dead = unreachable_rpc_servers(model.rpcservers, exec_host=ssh_dest)
        if dead:
            for addr in dead:
                via = f"{ssh_dest} -> " if ssh_dest else ""
                logger.warning(f"RPC server {addr.IP}:{addr.PORT} unreachable — starting via SSH as {via}{addr.remuser}...")
                start_rpc_server(addr, exec_host=ssh_dest)
            # No fixed sleep per server: wait_for_rpc_servers() already polls all
            # of them together and returns as soon as they answer.
            still_dead = wait_for_rpc_servers(dead, exec_host=ssh_dest)
            if still_dead:
                addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in still_dead)
                logger.error(f"Error: RPC server(s) still unreachable after start attempt: {addrs}")
                sys.exit(1)

        logger.info("All RPC servers reachable.")
        if opts.only_rpc:
            logger.warning("'--only-start-rpc' specified. Gracefully exiting.")
            sys.exit(0)

    devices = ""
    if opts.dry_run:
        devices = "SKIP(DRYRUN)"
    else:
        if not opts.override_devices:
            if model.rpcservers:
                rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
                raw_devices = _list_devices(binary, rpc_list, ssh_dest)
                # An RPC-backed model whose device list has no RPC entry means the
                # RPC server accepted the TCP connection but never served it. Stop
                # here: launching would only reproduce the failure remotely, where
                # the only trace is 'invalid device: RPC0' in the boot log.
                if not any(d.startswith("RPC") for d in raw_devices):
                    _report_missing_rpc_devices(model, raw_devices, ssh_dest)
                    if not opts.only_list_devs:
                        sys.exit(1)
                devices = ",".join(raw_devices)
                logger.info(f"Using devices: {devices or '(none)'}")
        else:
            devices = opts.override_devices

        if opts.only_list_devs:
            logger.warning("'--only-list-devices' specified. Gracefully exiting.")
            sys.exit(0)

    ctx = None
    if opts.override_ctx:
        ctx = opts.override_ctx
    cmd = build_command(binary, model, devices, ctx, opts.nomtp, opts.disable_thinking, opts.debug) # ctx == None will trigger the retrieve of the default value defined in the config.json
    logger.debug(f"Command: {cmd}")

    if opts.dry_run:
        # Still show the command with --log-file for reference
        logger.info(f"Dry-run command: {' '.join(shlex.quote(a) for a in cmd)}")
        #logger.info ("  " + )
        return

    # DETACH: start the server in the background, verify it stayed up, then exit.
    _launch_detached(cmd, ssh_dest)

#___________________________________________________________________________________
def main() -> None:
    parser = argparse.ArgumentParser(description="Launch llama-server for a model defined in models.json")
    parser.add_argument("model_name", nargs="?", help="Model name as listed in models.json (without .gguf extension)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    # dest= aligns each flag with its LaunchOptions field, so vars(args) maps 1:1.
    parser.add_argument("--only-start-rpc", dest="only_rpc", action="store_true", help="Just start the RPC remote servers and exit")
    parser.add_argument("--only-check-rpc", action="store_true", help="Just check that RPC remote servers and reachable")
    parser.add_argument("--only-list-devices", dest="only_list_devs", action="store_true", help="Just retrieve the list of GPU devices from local and remote RPC servers (does NOT start RPC servers)")
    parser.add_argument("--kill-rpc-server", dest="kill_rpc", action="store_true", help="Run 'killall rpc-server' on every RPC node of the model (via LLAMA_SERVER_HOST) and exit")
    parser.add_argument("--list-models", action="store_true", help="Print the available models and exit")
    parser.add_argument("--kill-server", action="store_true", help="Kill the llama-server process on LLAMA_SERVER_HOST (and the rpc-servers it was using) and exit")
    parser.add_argument("--server-status", action="store_true", help="Check whether llama-server is running on LLAMA_SERVER_HOST and exit")
    parser.add_argument("--tail-log", dest="follow_log", action="store_true", help=f"Follow (tail -F) llama-server's log file ({settings.LLAMA_LOG_FILE}) on LLAMA_SERVER_HOST until Ctrl-C, then exit")
    parser.add_argument("--tail-lines", "-n", type=int, default=50, metavar="INT", help="Number of trailing log lines to show before following (default: 50)")
    parser.add_argument("--override-temp", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-top-p", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-top-k", type=int, default=None, metavar="INT")
    parser.add_argument("--override-min-p", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-devices", type=str, default=None, metavar="STR")
    parser.add_argument("--override-rpc", type=str, default=None, metavar="STR")
    parser.add_argument("--override-ctx", type=int, default=None, metavar="INT")
    parser.add_argument("--override-kvquant", dest="override_kvq", type=str, default=None, metavar="STR")
    parser.add_argument("--disable-thinking", action="store_true", default=False)
    


    parser.add_argument("--json", dest="as_json", action="store_true", help="Machine-readable output for --server-status and --list-models")
    parser.add_argument("--debug", action="store_true", help="Print debug messages")
    parser.add_argument("--force-no-mtp", dest="nomtp", action="store_true", help="Disable MTP even if the model supports it")

    args = parser.parse_args()

    # The level was already set at import time (see top of the file); realign it
    # here with the authoritative value coming from argparse. INFO (not WARNING)
    # is the default on purpose: the progress messages of a launch — RPC checks,
    # device list, launch confirmation — are logged at info level, and dropping
    # them made a normal run look completely silent.
    logzero.loglevel(logzero.DEBUG if args.debug else logzero.INFO)

    start_model(LaunchOptions(**vars(args)))

#___________________________________________________________________________________
if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except utils.ServerHostUnreachable as e:
        logger.error(f"Error: host unreachable — {e}")
        sys.exit(2)
