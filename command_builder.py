from config_manager import get_settings
from pathlib import Path
import subprocess
import sys
from model import Model
import json
from logzero import logger
#import psutil

settings = get_settings()

#___________________________________________________________________________________
def build_command(binary: str, model: Model, devices: str = "", ctx: int | None = None, nomtp: bool = False, disablethink: bool = False, verbose: bool = False) -> list[str]:
    cmd = [binary, "-m", str(model.model_path), "-c", str(ctx if ctx is not None else settings.DEFAULT_CTX)]

    #if model.fitt:
    #    cmd += ["-fitt", model.fitt]

    if model.rpcservers and len(model.rpcservers):
        rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
        cmd += ["--rpc", rpc_list]

    if devices:
        cmd += ["--device", devices]

    if model.mmproj_path and str(model.mmproj_path).lower() not in ("none", "null", ""):
        cmd += ["--mmproj", str(model.mmproj_path)]

#    data = None
    data = {}
    if model.reasoning:
        data['reasoning_effort'] = model.reasoning

    if model.preserv_think:
        if data:
            data['preserve_thinking'] = model.preserv_think
        else:
            data = {"preserve_thinking": model.preserv_think}

    cmd += ["--host", settings.ADDRESS_BIND]
    cmd += ["--port", str(settings.PORT_BIND)]
    cmd += ["--split-mode", "layer"]
    cmd += ["--metrics"]
    cmd += ["--no-context-shift"]
    cmd += ["--jinja"]
    cmd += ["-fa", "on"]
    cmd += ["-fit", "on"] # Using "on" makes the rpc/Vulkan on PC with 2 NVidia cards crash
    cmd += ["-fitc", str(settings.FITC)]
    cmd += ["-np", str(settings.NP)] # NP=1 is required by MTP processing
    cmd += ["--no-warmup"]
    cmd += ["--temp", str(model.temperature)]
    cmd += ["--top-p", str(model.top_p)]
    cmd += ["--top-k", str(model.top_k)]
    if data:
        cmd += ["--chat-template-kwargs", json.dumps(data)]

    cmd += ["--seed", str(settings.SEED)]
    cmd += ["--repeat-penalty", str(model.rep_pen)]
    if model.pres_pen:
        cmd += ["--presence-penalty", str(model.pres_pen) ]
    if model.kvquant:
        cmd += ["-ctk", model.kvquant]
        cmd += ["-ctv", model.kvquant]
    #cmd += ["--alias", "local_AI"]
    # A negative min_p in SAMPLERS means "leave llama-server's default alone".
    if model.min_p >= 0:
        cmd += ["--min-p", str(model.min_p)]

    if model.ub:
        cmd += ["-ub", str(model.ub)]
    if model.b:
        cmd += ["-b", str(model.b)]
    if not nomtp:
        if model.mtp:
            cmd += ["--spec-type", "draft-mtp"]
            if model.ext_mtp_head_file:
                cmd += ["--model-draft", str(model.ext_mtp_head_file)]
                cmd += ["--spec-draft-n-min", "2"]
    cmd += ["--log-file", settings.LLAMA_LOG_FILE]

    if verbose:
        cmd += ["--verbose"]
    cmd += ["-ctxcp", "8"]
    #cmd += ["--reasoning-preserve"]
    cmd += ["--reasoning", "on"]
    cmd += ["--load-mode", "mmap+mlock"]
    
    ct = Path(f"{Path('./chat-templates') / model.model_name}.jinja")
    logger.debug(f"Checking existance of file {ct}")
    if  ct.exists():
        # SCP chat template on remote host
        scpcmd = ["scp", "-p", "-o", "BatchMode=yes", str(ct), f"{settings.LLAMA_SERVER_HOST}:/tmp/"]
        logger.debug(f"Executing command {scpcmd}")
        res = subprocess.run(scpcmd, capture_output=False, text=True, timeout=30)
        cmd += ["--chat-template-file", f'{Path("/tmp") / model.model_name}.jinja']
    
    if disablethink:
        cmd += ["--reasoning", "off"]
    else:
        cmd += ["--reasoning", "on"]

    return cmd
