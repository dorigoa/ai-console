from remote_cmd_executor import remote_exec
from config_manager import Settings
import requests
import json

#___________________________________________________________________________________
class ServerHostUnreachable(Exception):
    """Raised when LLAMA_SERVER_HOST cannot be contacted over SSH (vs. the
    server process simply not running)."""

#___________________________________________________________________________________
def ssh_dest(settings: Settings) -> str | None:
    if not settings.LLAMA_SERVER_HOST:
        return None
    if settings.LLAMA_SERVER_USER:
        return f"{settings.LLAMA_SERVER_USER}@{settings.LLAMA_SERVER_HOST}"
    return settings.LLAMA_SERVER_HOST

#___________________________________________________________________________________
def _get_first_model_name(endpoint: str) -> tuple[str, int, float, int, float, float, str, str] | None:

    url = f"http://{endpoint}/props"
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an exception for HTTP codes 4xx, 5xx
        data = response.json()
        # Let's make sure the structure is as expected
        # if isinstance(data, dict) and 'models' in data and isinstance(data['models'], list) and len(data['models']) > 0 and 'data' in data and len(data['data']) > 0:
        #     first_model = data['models'][0]
        #     first_data  = data['data'][0]#['meta']#['n_ctx']
        #     if isinstance(first_model, dict) and 'name' in first_model and isinstance(first_data, dict) and 'meta' in first_data:
        #         return (first_model['name'], first_data['meta']['n_ctx'])
        #     else:
        #         return None
        # else:
        # logger.debug(f"data={data}")
        if (
            isinstance(data, dict)
            and 'default_generation_settings' in data
            and isinstance(data['default_generation_settings'], dict)
            and 'params' in data['default_generation_settings']
            and isinstance(data['default_generation_settings']['params'], dict)
            and 'n_ctx' in data['default_generation_settings']
            and 'temperature' in data['default_generation_settings']['params']
            and 'top_k' in data['default_generation_settings']['params']
            and 'top_p' in data['default_generation_settings']['params']
            and 'min_p' in data['default_generation_settings']['params']
            and 'model_alias' in data
        ):
            #model = data['model_alias']
            model = data['model_path'].split('/')[-1].removesuffix(".gguf")
            n_ctx = data['default_generation_settings']['n_ctx']
            params = data['default_generation_settings']['params']
            temp  = params['temperature']
            top_k = params['top_k']
            top_p = params['top_p']
            min_p = params['min_p']
            quant = data['model_ftype']
            build_info = data.get('build_info', '')
            return model, n_ctx, temp, top_k, top_p, min_p, quant, build_info
        else:
            raise ValueError("JSON response has a bad structure")
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request error: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parsing error: {e}") from e

#___________________________________________________________________________________
def _get_model_created(endpoint: str) -> float | None:
    """'created' timestamp (unix seconds) reported by /models for the loaded
    model, i.e. when it was started on the server; None when not available."""
    url = f"http://{endpoint}/models"
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an exception for HTTP codes 4xx, 5xx
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request error: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parsing error: {e}") from e

    if isinstance(data, dict) and isinstance(data.get('data'), list):
        for m in data['data']:
            if isinstance(m, dict) and isinstance(m.get('created'), (int, float)):
                return m['created']
    return None

#___________________________________________________________________________________
def server_location(settings: Settings) -> str:
    return ssh_dest(settings) or "localhost"

#___________________________________________________________________________________
def server_pids(settings: Settings) -> list[str]:
    """PIDs of the running llama-server process(es), or [] if none.

    May raise ServerHostUnreachable (propagated from _run_on_server)."""
    #r = _run_on_server(f"pgrep -f -- '{_pgrep_pattern()}'")
    r = run_on_server(settings, f"pgrep -f -- '{settings.LLAMA_SERVER_BIN}'")
    return [p for p in r.stdout.split() if p.strip().isdigit()]

#___________________________________________________________________________________
def run_on_server(settings: Settings, shell_cmd: str, timeout: int = 15):# -> subprocess.CompletedProcess:
    """Run shell_cmd on LLAMA_SERVER_HOST via SSH if configured, else locally.

    Raises ServerHostUnreachable if SSH itself fails (connection refused,
    timeout, auth/host error). SSH reserves exit code 255 for its own failures,
    while pgrep/pkill only ever return 0/1/2/3, so 255 unambiguously means the
    host was not reached rather than 'no process found'."""
    ssh_destination = ssh_dest(settings)
    if ssh_destination:
        argv = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", ssh_destination, shell_cmd]
    else:
        argv = ["bash", "-c", shell_cmd]
    r = remote_exec( argv, timeout )

    if not r:
        raise ServerHostUnreachable(f"{server_location(settings)} did not answer within {timeout}s")

    if ssh_dest and r.returncode == 255:
        detail = r.stderr.strip() or "connection failed"
        raise ServerHostUnreachable(f"SSH error contacting {ssh_dest}: {detail}")
    return r

#___________________________________________________________________________________
def server_status(settings: Settings) -> dict:
    """Structured status of llama-server: the single source of truth behind both
    the human-readable report and --json.

    'running' says a process exists; 'ready' says it also answers /props (a
    freshly started server is RUNNING but not yet ready for a while)."""
    info = {"where": server_location(settings), "running": False, "ready": False, "pids": [], "quant": "", "build_info": ""}
    pids = server_pids(settings)
    if not pids:
        return info

    info["running"] = True
    info["pids"] = pids
    info["quant"] = 'BOH'
    try:
        model, ctxsize, temp, top_k, top_p, min_p, quant, build_info = _get_first_model_name(f"{settings.LLAMA_SERVER_HOST}:{settings.PORT_BIND}")
    except (RuntimeError, ValueError) as e:
        # ValueError too: _get_first_model_name raises it on an unexpected JSON
        # shape, and it is not a subclass of RuntimeError.
        info["error"] = str(e)
    else:
        info.update(ready=True, model=model, ctx=ctxsize, temperature=temp, top_k=top_k, top_p=top_p, min_p=min_p, quant=quant, build_info=build_info)
        try:
            created = _get_model_created(f"{settings.LLAMA_SERVER_HOST}:{settings.PORT_BIND}")
        except (RuntimeError, ValueError):
            # A /models that refuses to answer only costs the "started" line,
            # not the whole status.
            created = None
        if created is not None:
            info["created"] = created
    return info