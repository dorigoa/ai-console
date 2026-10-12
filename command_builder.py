from config_manager import get_settings
from pathlib import Path
from model import Model
import json
from logzero import logger

settings = get_settings()

#___________________________________________________________________________________
def build_command(binary: str, 
                  model: Model, 
                  #devices: str | list[str] | None = None,
                  ctx: int | None = None, 
                  nomtp: bool = False, 
                  disablethink: bool = False, 
                  verbose: bool = False, 
                  reasoning: str | None = None) -> list[str]:
    
    cmd = [binary, "-m", str(model.model_path), "-c", str(ctx if ctx is not None else settings.DEFAULT_CTX)]

    if model.rpcservers and len(model.rpcservers):
        rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
        cmd += ["--rpc", rpc_list]

    
    # `devices` is the runtime-resolved list (from --list-devices / --override-devices)
    # and wins over the static DEVICES of models.json; a str is already comma-joined.
    # if isinstance(devices, (list, tuple)):
    #     devices = ",".join(str(d) for d in devices)
    # device_arg = devices or model.get_devices()
    if model.get_devices():
        cmd += ["--device", str(model.get_devices())]

    # -ts takes one comma-separated string, and only makes sense together with the
    # device list: an empty list means "no split", and passing "" would be rejected.
    #tensplit_arg = 
#    if model.get_tensplit():
#        cmd += ["-ts", model.get_tensplit()]

    if model.mmproj_path and str(model.mmproj_path).lower() not in ("none", "null", ""):
        cmd += ["--mmproj", str(model.mmproj_path)]

    data = {}
    # reasoning is the REAS override (CLI/GUI); None means "keep the value
    # defined in models.json".
    reas = reasoning if reasoning is not None else model.reasoning
    if reas:
        data['reasoning_effort'] = reas

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
    
    ct = Path(f"{Path('./chat-templates') / model.model_name}.jinja")
    logger.debug(f"Checking existance of file {ct}")
    
    if disablethink:
        cmd += ["--reasoning", "off"]
    else:
        cmd += ["--reasoning", "on"]

    return cmd
