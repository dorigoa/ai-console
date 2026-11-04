"""The shell commands the console runs on the nodes (over gui_util.ssh_run).

Plain POSIX sh, valid on Linux and on macOS (zsh login shell included).
The pgrep patterns use the "[x]" trick ("llama[-]server"): the regex does not
match its own text, so pgrep never finds the remote shell running the check.
"""

import base64

PID_NORM = 'tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -'  # pids on one line, no duplicates
LOG_LINES = 200


def collect_pids(var: str, patterns: tuple[str, ...]) -> str:
    """Shell code setting $var to the pids of the processes matching any pattern."""
    collect = " ".join(f'{var}="${var} $(pgrep -f "{p}" 2>/dev/null)";' for p in patterns)
    return f'{var}=""; {collect} {var}=$(echo ${var} | {PID_NORM}); '


def status_cmd(*patterns: str) -> str:
    """Prints "ALIVE <pids>" if a process matches, else "NO_PID"."""
    return (collect_pids("pids", patterns)
            + 'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi')


def kill_cmd(*patterns: str, pidfile: str = "", verify: bool = False) -> str:
    """Kills the matching processes: TERM, 2 s grace, then KILL; prints "KILLED_PIDS=<pids>".

    pidfile: also kill the process group of the run it records (see the start commands).
    verify:  check again after 1 s, exit 4 with "STILL_ALIVE <pids>" if something survived.
    Exit 3 with "NO_PID": nothing to kill.
    Loops use $(echo $pids): zsh (macOS) does not word-split a bare $pids."""
    group = term_group = kill_group = check = ""
    if pidfile:
        group = (f'g=$(cat "{pidfile}" 2>/dev/null | tr -d " "); '
                 'if [ -n "$g" ] && kill -0 "$g" 2>/dev/null; then '
                 'pids="$pids $g"; RUN_GROUP=$g; else RUN_GROUP=""; fi; '
                 f'pids=$(echo $pids | {PID_NORM}); ')
        term_group = 'if [ -n "$RUN_GROUP" ]; then kill -TERM -$RUN_GROUP 2>/dev/null; fi; '
        kill_group = 'if [ -n "$RUN_GROUP" ]; then kill -KILL -$RUN_GROUP 2>/dev/null; fi; '
    if verify:
        check = ('sleep 1; ' + collect_pids("left", patterns)
                 + 'if [ -n "$left" ]; then echo "STILL_ALIVE $left"; exit 4; fi; ')
    return (collect_pids("pids", patterns) + group
            + 'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; exit 3; fi; '
            + 'for p in $(echo $pids); do kill -TERM "$p" 2>/dev/null; done; ' + term_group
            + 'sleep 2; '
            + 'for p in $(echo $pids); do kill -KILL "$p" 2>/dev/null; done; ' + kill_group
            + check
            + 'echo "KILLED_PIDS=$pids"')


def log_tail_cmd(path: str) -> str:
    """Prints the last LOG_LINES lines of path, under a header (BSD wc pads its count)."""
    return (f'f="{path}"; '
            'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
            f'echo "--- $f: last {LOG_LINES} of $(wc -l < "$f" | tr -d " ") lines ---"; '
            f'tail -n {LOG_LINES} "$f"')


def setsid_start_cmd(workdir: str, name: str, cmd: str) -> str:
    """Linux detached start: cmd runs in workdir as a session leader, pid in <name>.pid,
    output in <name>.out (truncated at each start). exec keeps the recorded pid exact."""
    return (f'cd "{workdir}" && {{ setsid bash -c \'echo $$ > {name}.pid; exec {cmd}\' '
            f'> {name}.out 2>&1 </dev/null & sleep 1; echo "STARTED_PID=$(cat {name}.pid)"; }}')


def detached_start_cmd(pidfile: str, logfile: str, cmd: str, *, node: str,
                       append: bool = True) -> str:
    """Portable detached start, done by python3 (macOS has no setsid).

    cmd runs as a session leader (so kill_cmd can take its whole group down) and
    outlives the SSH session; pid in pidfile, output appended to logfile (or
    truncated if not append). cmd travels base64-encoded: no quoting issues."""
    b64 = base64.b64encode(cmd.encode()).decode()
    mode = "ab" if append else "wb"
    return (
        f'command -v python3 >/dev/null 2>&1 || {{ echo "python3 not found on the {node} node"; exit 127; }}; '
        'python3 -c \'import base64,sys,subprocess; '
        'pidfile, logfile, cmd = sys.argv[1], sys.argv[2], base64.b64decode(sys.argv[3]).decode(); '
        f'log = open(logfile, "{mode}", 0); '
        'child = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, '
        'stdin=subprocess.DEVNULL, start_new_session=True); '
        'open(pidfile, "w").write(str(child.pid) + "\\n"); '
        'print("STARTED_PID=%d" % child.pid)\' '
        f'"{pidfile}" "{logfile}" "{b64}"'
    )
