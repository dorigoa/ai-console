"""Builders for the remote shell commands the console runs over SSH.

Plain POSIX sh, so they run on both the Linux and the macOS nodes. Every pgrep
pattern uses the "[x]" bracket trick: the regex never matches its own text, so
pgrep cannot match the remote shell running the check (whose argv contains the
pattern verbatim).

The commands are pure functions of their arguments, built once when a service
class is defined; gui_util.ssh_run is what later executes them (and logs them).
"""

import base64

# Normalise a list of pids: one line, de-duplicated, no blanks.
PID_NORM = 'tr " " "\\n" | grep -v "^$" | sort -u | paste -sd " " -'
LOG_LINES = 200


def collect_pids(var: str, patterns: tuple[str, ...]) -> str:
    """Set $var to the pids of every process matching one of patterns."""
    collect = " ".join(f'{var}="${var} $(pgrep -f "{p}" 2>/dev/null)";' for p in patterns)
    return f'{var}=""; {collect} {var}=$(echo ${var} | {PID_NORM}); '


def status_cmd(*patterns: str) -> str:
    """Print "ALIVE <pids>" when some process matches, "NO_PID" otherwise."""
    return (collect_pids("pids", patterns)
            + 'set -- $pids; if [ $# -eq 0 ]; then echo "NO_PID"; else echo "ALIVE $pids"; fi')


def kill_cmd(*patterns: str, pidfile: str = "", verify: bool = False) -> str:
    """Kill every process matching patterns: TERM, two seconds to drain, then KILL.

    With a pidfile, the session it records (created by the start command, so it
    is exactly "this run") is killed as a whole too, as long as its leader is
    alive. Exit 3: nothing to kill. With verify, the patterns are probed again at
    the end and exit 4 (STILL_ALIVE) means the run survived.

    The loops go over $(echo $pids), not $pids: zsh (the macOS login shell) does
    not word-split a bare $pids, so with two or more pids it would signal none."""
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
    """Print the last LOG_LINES lines of path ("wc -l" pads its output on BSD)."""
    return (f'f="{path}"; '
            'if [ ! -f "$f" ]; then echo "NO_LOG_FILE: $f"; exit 0; fi; '
            f'echo "--- $f: last {LOG_LINES} of $(wc -l < "$f" | tr -d " ") lines ---"; '
            f'tail -n {LOG_LINES} "$f"')


def setsid_start_cmd(workdir: str, name: str, cmd: str) -> str:
    """Linux detached start: cmd runs from workdir as a session leader (so a group
    kill takes the whole run down), with its pid in <name>.pid and its output in
    <name>.out, truncated at every start. "exec" keeps the recorded pid honest."""
    return (f'cd "{workdir}" && {{ setsid bash -c \'echo $$ > {name}.pid; exec {cmd}\' '
            f'> {name}.out 2>&1 </dev/null & sleep 1; echo "STARTED_PID=$(cat {name}.pid)"; }}')


def detached_start_cmd(pidfile: str, logfile: str, cmd: str, *, node: str,
                       append: bool = True) -> str:
    """Portable detached start (macOS has no setsid(1)), done by python3.

    Popen(start_new_session=True) makes the run a session leader, so the kill
    command can take it down as a group; its pid goes to pidfile and its output
    to logfile (appended, or truncated at every start). cmd travels base64
    encoded, so no quote inside it can break the outer command line."""
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
