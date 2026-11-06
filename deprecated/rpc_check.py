from concurrent.futures import ThreadPoolExecutor, as_completed
from logzero import logger
from pathlib import Path
import socket
import shlex
import time
import sys

from remote_cmd_executor import remote_exec
from model import load_models
from rpc import RpcServer
from errors import ConfigError


_TIMEOUT = 2.0

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]

_RPC_START_POLL_INTERVAL = 2
_RPC_START_TIMEOUT      = 20

#___________________________________________________________________________________
def _tcp_reachable(addr: RpcServer, exec_host: str | None = None) -> bool:
    """True if addr.IP:addr.PORT accepts a TCP connection.

    If exec_host is given, the probe is performed FROM that host over SSH
    (using bash's /dev/tcp), because only that host can reach the RPC network.
    Otherwise the probe is a local socket connection.

    CAVEAT: this answers "is something listening?", NOT "can it serve me?".
    ggml-rpc-server handles ONE client at a time (accept -> serve -> close ->
    accept), so while another client holds the session a new connection is
    completed by the kernel and parked in the listen backlog, never answered.
    The probe still succeeds — exactly like nc/telnet do — while llama-server
    reports "Failed to connect". Use rpc_peers() to tell ready from busy.
    """
    if exec_host:
        # nc, not bash's /dev/tcp: on macOS the Homebrew bash is SIGKILLed by the
        # kernel the moment it opens a non-loopback socket (loopback is fine, any
        # LAN or Internet address is not), so the probe returned 137 and reported a
        # perfectly healthy RPC server as dead. Which bash the remote login shell
        # picks up depends on its PATH, so the failure came and went with the client
        # environment. nc needs no net-redirection support and behaves the same on
        # macOS and Debian; -w bounds both the connect and the idle wait.
        probe = f"nc -z -w {int(_TIMEOUT)} {addr.IP} {addr.PORT}"

        r = remote_exec( ["ssh", *_SSH_OPTS, exec_host, probe], timeout=_TIMEOUT + 8 )
        if not r:
            return False

        return r.returncode == 0
    try:
        with socket.create_connection((addr.IP, addr.PORT), timeout=_TIMEOUT):
            return True
    except OSError:
        return False

#___________________________________________________________________________________
def unreachable_rpc_servers(servers: list[RpcServer], exec_host: str | None = None) -> list[RpcServer]:
    """Return the RpcServer entries that do not respond to a TCP ping.

    Returns [] immediately if servers is empty.
    Checks are run in parallel (one thread per server).
    If exec_host is given, probes originate from that host (see _tcp_reachable).
    """
    if not servers:
        return []

    dead: list[RpcServer] = []
    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        future_to_addr = {pool.submit(_tcp_reachable, s, exec_host): s for s in servers}
        for future in as_completed(future_to_addr):
            addr = future_to_addr[future]
            try:
                alive = future.result()
            except Exception:
                alive = False
            if not alive:
                dead.append(addr)

    return dead

#___________________________________________________________________________________
def rpc_peers(addr: RpcServer, exec_host: str | None = None) -> list[str]:
    """netstat lines of the clients currently connected to rpc-server's port.

    Asked TO the RPC node itself: since a busy ggml-rpc-server is indistinguishable
    from a ready one on the client side (see _tcp_reachable), the only reliable way
    to tell them apart is to look at who already holds an ESTABLISHED connection.
    The netstat address syntax differs between macOS (IP.PORT) and Linux (IP:PORT),
    hence the [.:] in the pattern.

    Best-effort diagnostic: returns [] if the node cannot be contacted.
    """
    remote_cmd = (
        f"netstat -an 2>/dev/null | grep ESTABLISHED | grep -E '[.:]{addr.PORT}[[:space:]]'"
    )
    if exec_host:
        inner = f"ssh {' '.join(_SSH_OPTS)} {addr.remuser}@{addr.IP} {shlex.quote(remote_cmd)}"
        argv = ["ssh", *_SSH_OPTS, exec_host, inner]
    else:
        argv = ["ssh", *_SSH_OPTS, f"{addr.remuser}@{addr.IP}", remote_cmd]

    result = remote_exec( argv, 20 )
    if not result:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

#___________________________________________________________________________________
def start_rpc_server(addr: RpcServer, exec_host: str | None = None) -> bool:
    """Start rpc-server on the RPC node via SSH in the background.

    </dev/null detaches stdin so nohup does not stay bound to the SSH session.

    If exec_host is given, the SSH to the RPC node is issued FROM that host
    (nested SSH: laptop -> exec_host -> remuser@addr.IP), because only exec_host
    has network reachability to the RPC servers. Otherwise the SSH is direct.
    Returns True if SSH exited with code 0.
    """
    remote_cmd = (
        f"LLAMA_CACHE={addr.cachepath} "
        f"nohup {addr.bin} --host 0.0.0.0 --port {addr.PORT} -c  "
        f">/tmp/rpc-server.out 2>&1 </dev/null &"
    )
    if exec_host:
        # Run the rpc-node SSH on exec_host; quote remote_cmd so the exec_host
        # shell forwards it verbatim to the inner ssh as a single argument.
        inner = f"ssh {' '.join(_SSH_OPTS)} {addr.remuser}@{addr.IP} {shlex.quote(remote_cmd)}"
        argv = ["ssh", *_SSH_OPTS, exec_host, inner]
        shown = f"{exec_host} -> {addr.remuser}@{addr.IP}"
    else:
        argv = ["ssh", *_SSH_OPTS, f"{addr.remuser}@{addr.IP}", remote_cmd]
        shown = f"{addr.remuser}@{addr.IP}"
    logger.debug(f"  SSH: {shown}  \"{remote_cmd}\"")

    result = remote_exec( argv, 20 )

    if not result:
        logger.error(f"  Timed out starting rpc-server on {shown}")
        return False

    if result.returncode != 0:
        logger.error(f"  SSH stderr: {result.stderr.strip()}")
    return result.returncode == 0

#___________________________________________________________________________________
def kill_rpc_server(addr: RpcServer, exec_host: str | None = None) -> bool:
    """Kill rpc-server on the RPC node with `killall ggml-rpc-server`.

    `killall` is available on both Linux (psmisc) and macOS/BSD, so the same
    command works regardless of the node OS. If exec_host is given, the SSH to
    the RPC node is issued FROM that host (nested SSH: laptop -> exec_host ->
    remuser@addr.IP), because only exec_host can reach the RPC network.
    Otherwise the SSH is direct.

    Returns True if the node was reached and killall ran (rc 0 = killed,
    rc 1 = no matching process / already stopped). Returns False on SSH/contact
    failure (rc 255 or other).
    """
    remote_cmd = "killall ggml-rpc-server"
    if exec_host:
        inner = f"ssh {' '.join(_SSH_OPTS)} {addr.remuser}@{addr.IP} {shlex.quote(remote_cmd)}"
        argv = ["ssh", *_SSH_OPTS, exec_host, inner]
        shown = f"{exec_host} -> {addr.remuser}@{addr.IP}"
    else:
        argv = ["ssh", *_SSH_OPTS, f"{addr.remuser}@{addr.IP}", remote_cmd]
        shown = f"{addr.remuser}@{addr.IP}"
    logger.debug(f"  SSH: {shown}  \"{remote_cmd}\"")

    result = remote_exec( argv, 20 )
    if not result:
        return False

    if result.returncode not in (0, 1):
        logger.error(f"  SSH stderr: {result.stderr.strip()}")
        return False
    return True

#___________________________________________________________________________________
def wait_for_rpc_servers(servers: list[RpcServer], exec_host: str | None = None) -> list[RpcServer]:
    """Poll until all servers become reachable or the timeout expires.

    Returns the list of servers still unreachable when the deadline is reached.
    If exec_host is given, probes originate from that host (see _tcp_reachable).
    """
    deadline = time.monotonic() + _RPC_START_TIMEOUT
    remaining = list(servers)
    while remaining and time.monotonic() < deadline:
        time.sleep(_RPC_START_POLL_INTERVAL)
        remaining = [a for a in remaining if not _tcp_reachable(a, exec_host)]
        if remaining:
            addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in remaining)
            logger.debug(f"  Still unreachable: {addrs}")
    return remaining

#___________________________________________________________________________________
if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        logger.info("Usage: python rpc_check.py <models.json> <rpc.json>")
        sys.exit(1)

    # check_remote_file=False: this tool only inspects RPC servers, so the
    # per-model SSH round-trip to stat the .gguf would be pure overhead.
    try:
        models = load_models(Path(sys.argv[1]), Path(sys.argv[2]), check_remote_file=False)
    except ConfigError as e:
        logger.error(e)
        sys.exit(1)

    for m in models:

        down = unreachable_rpc_servers(m.rpcservers)
        if not m.rpcservers:
            logger.error(f"{m.model_name}: no rpc server configured")
        elif down:
            addrs = ", ".join(f"{a.IP}:{a.PORT} ({a.remuser}@{a.bin})" for a in down)
            logger.error(f"{m.model_name}: not reachable server -> {addrs}")
        else:
            logger.info(f"{m.model_name}: all rpc servers reachable")
