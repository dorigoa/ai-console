from dataclasses import dataclass
from logzero import logger
from pathlib import Path
import shlex
import json
import sys

from remote_cmd_executor import remote_exec
from rpc import RpcServer,load_rpcs
from errors import ConfigError

#___________________________________________________________________________________
@dataclass
class Model:
    model_name: str
    model_path: Path
    size_gib: float | None
    mmproj_path: Path | None
    # temperature / top_p / top_k / min_p are NOT stored fields: they are the
    # four values packed inside `samplers`, exposed by the properties further
    # down. Keeping them as fields too would duplicate the same data in two
    # places, free to drift apart.
    samplers: str
    reasoning: str | None
    preserv_think: bool | None
    last_started: int
    rpcservers: list[RpcServer] | None
    ub: int
    b: int
    kvquant: str | None
    mtp: bool
    ext_mtp_head_file: Path | None
    native_ctx: int
    # KV cache size, in decimal GB (10^9 B), at 131072 tokens with a bf16 cache:
    # the GUI converts it to GiB and scales it with context and KV quant.
    ctx_budget_gb_bf16_128k: float | None
    rep_pen: float
    pres_pen: float | None

    def rpc_endpoints( self ):
        endpoints = []
        for R in self.rpcservers:
            endpoints.append( R.endpoint() )
        return ','.join(endpoints)

    def get_samplers( self ):
        """The sampler values as strings, from SAMPLERS ('temp:top_p:top_k:min_p')."""
        parts = self.samplers.split(':')
        if len(parts) < 4:
            raise ValueError(
                f"model '{self.model_name}': SAMPLERS must be "
                f"'temp:top_p:top_k:min_p', got '{self.samplers}'"
            )
        return parts

    def _set_sampler( self, index: int, value ) -> None:
        parts = self.get_samplers()
        parts[index] = str(value)
        self.samplers = ':'.join(parts)

    # The four sampler values live packed in a single SAMPLERS string, but both
    # build_command() and the --override-* options need them individually. These
    # properties read from and write back into that string: without the setters,
    # `model.temperature = x` would just create a stray attribute nobody reads,
    # and the override would be silently dropped.
    @property
    def temperature( self ) -> float:
        return float(self.get_samplers()[0])

    @temperature.setter
    def temperature( self, value: float ) -> None:
        self._set_sampler(0, value)

    @property
    def top_p( self ) -> float:
        return float(self.get_samplers()[1])

    @top_p.setter
    def top_p( self, value: float ) -> None:
        self._set_sampler(1, value)

    @property
    def top_k( self ) -> int:
        return int(self.get_samplers()[2])

    @top_k.setter
    def top_k( self, value: int ) -> None:
        self._set_sampler(2, value)

    @property
    def min_p( self ) -> float:
        return float(self.get_samplers()[3])

    @min_p.setter
    def min_p( self, value: float ) -> None:
        self._set_sampler(3, value)

#___________________________________________________________________________________
def _file_sizes_gib(paths: list[Path], remote_host: str = "", remote_user: str = "") -> dict[Path, float | None]:
    """Return {path: size in GiB, or None if the file does not exist} for all `paths`.

    Existence check and size are one operation: a missing file simply maps to None.
    On a remote host ALL paths are resolved in a SINGLE ssh round-trip: the stat
    commands (one line per path) are streamed over the connection's stdin to a
    remote `sh -s`, and the output lines map back to `paths` by position. This
    replaces the old per-model _file_exists + _file_size_gib pair, which cost
    two ssh connections per model and made --list-models crawl.

    Portable across Linux and macOS: tries GNU `stat -c %s`, then falls back to
    BSD `stat -f %z`; a path where both fail prints MISSING. Raises RuntimeError
    when the ssh transport itself fails."""
    if not remote_host:
        sizes: dict[Path, float | None] = {}
        for p in paths:
            try:
                sizes[p] = p.stat().st_size / (1024 ** 3)
            except OSError:
                sizes[p] = None
        return sizes

    dest = f"{remote_user}@{remote_host}" if remote_user else remote_host
    script = "".join(
        f"stat -c %s {shlex.quote(str(p))} 2>/dev/null || "
        f"stat -f %z {shlex.quote(str(p))} 2>/dev/null || echo MISSING\n"
        for p in paths
    )
    args = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            dest, "sh -s"]
    result = remote_exec( args, input=script )
    # Every script line ends in `|| echo MISSING`, so the batch itself always
    # exits 0: any non-zero rc (or a timeout, returned as None) is the ssh
    # transport failing, not a missing file.
    if result is None or result.returncode != 0:
        rc = "timeout" if result is None else f"rc={result.returncode}"
        stderr = (result.stderr.strip() if result else "") or "(no stderr)"
        raise RuntimeError(f"SSH to {dest} failed ({rc}): {stderr}")

    lines = result.stdout.splitlines()
    if len(lines) != len(paths):
        raise RuntimeError(
            f"batched stat on {dest}: expected {len(paths)} output lines, got {len(lines)}"
        )
    return {
        p: int(out) / (1024 ** 3) if out.strip().isdigit() else None
        for p, out in zip(paths, lines)
    }

#___________________________________________________________________________________
def load_models(config_path: Path, 
                rpc_config_path: Path,
                remote_host: str = "", 
                remote_user: str = "", 
                check_remote_file: bool = True, 
                check_model_name: str | None = None) -> list[Model]:

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    base_dir = Path(config["MODEL_BASE_DIR"])
    models_section = config.get("models", {})
    
    models: list[Model] = []

    rpcs = load_rpcs( rpc_config_path )

    def _model_path(name: str) -> Path:
        filename = name if name.endswith(".gguf") else f"{name}.gguf"
        return base_dir / filename

    def _needs_check(name: str) -> bool:
        # When check_model_name is set, only check the remote file for that model;
        # all others skip the check.
        return check_remote_file and (check_model_name is None or name == check_model_name)

    # Resolve existence + size of every file to check in ONE batched call
    # (a single ssh round-trip when remote), instead of two ssh per model.
    paths_to_check = [_model_path(n) for n in models_section if _needs_check(n)]
    sizes = _file_sizes_gib(paths_to_check, remote_host, remote_user) if paths_to_check else {}

    for name, spec in models_section.items():
        model_path = _model_path(name)

        if _needs_check(name):
            size_gib = sizes[model_path]
            if size_gib is None:
                logger.error(f"[SKIP] Model '{name}': file not found: {model_path}")
                continue
        else:
            size_gib = None

        mtp = False
        ext_mtp_file = None
        if 'MTP' in spec and spec['MTP'] == True:
            mtp = True
            if 'ext_mtp_head_file' in spec and spec['ext_mtp_head_file']:
                ext_mtp_file = str(spec['ext_mtp_head_file'])
        pp = None
        if spec.get("PP") and spec.get("PP") > -1:
            pp = float( spec.get("PP") )

        rp = None
        if spec.get("RP"):
            rp = float( spec.get("RP") )

        reas_eff = None
        if spec.get("REAS"):
            reas_eff = str(spec["REAS"])

        preserv_think = None
        if spec.get("PRES_THK"):
            preserv_think = spec["PRES_THK"]

        rpcs_for_this_model = []
        for n in (spec.get('RPC_SERVERS') or {}).get('ids') or []:
            rpc_name = n['name']
            if rpc_name not in rpcs:
                raise ConfigError(
                    f"model '{name}' references RPC server '{rpc_name}', which is not "
                    f"defined in {rpc_config_path}. Known ones: {', '.join(sorted(rpcs)) or '(none)'}"
                )
            rpcs_for_this_model.append( rpcs[rpc_name] )

        samplers = spec.get("SAMPLERS")
        if not samplers or len(str(samplers).split(':')) < 4:
            raise ConfigError(
                f"model '{name}': SAMPLERS must be 'temp:top_p:top_k:min_p', "
                f"got {samplers!r}"
            )


        models.append(
            Model(
                model_name=name,
                model_path=model_path,
                size_gib=size_gib,
                mmproj_path=(Path(base_dir) / spec["MMPROJ"]) if spec["MMPROJ"] is not None else None,
                samplers=spec["SAMPLERS"],
                reasoning=reas_eff,
                preserv_think=preserv_think,
                last_started=0,
                rpcservers=rpcs_for_this_model,
                kvquant=spec["KVQUANT"],
                ub=spec["UB"],
                b=spec["B"],
                mtp=mtp,
                ext_mtp_head_file=base_dir / ext_mtp_file if ext_mtp_file else None,
                native_ctx=int(spec.get("native_ctx")), # this has to be defined in the models.json !!
                ctx_budget_gb_bf16_128k=float(spec["ctx_budget_gb_bf16_128k"])
                    if spec.get("ctx_budget_gb_bf16_128k") else None,
                rep_pen=rp,
                pres_pen=pp
            )
        )
    return models

def get_model_byname( name: str, config_path: Path, rpc_config_path: Path ) -> Model | None:
    models = load_models(config_path, 
                            rpc_config_path,
                            "", 
                            "", 
                            False, 
                            None)

    m = None
    for m in models:
        if m.model_name == name:
            return m
    return m

#___________________________________________________________________________________
if __name__ == "__main__":
    import argparse
    from config_manager import get_settings

    parser = argparse.ArgumentParser(description="List models from models config file")

    parser.add_argument(
        "--master-host", default=None,
        help="SSH host for file existence check (overrides config.json)",
    )
    parser.add_argument(
        "--master-user", default=None,
        help="SSH user for file existence check (overrides config.json)",
    )
    parser.add_argument(
        "--models-config", type=Path, default=Path(__file__).parent / "models.json",
        help="path of models json description file"
    )
    parser.add_argument(
        "--rpc-config", type=Path, default=Path(__file__).parent / "rpc.json",
        help="path of rpc servers json description file"
    )
    parser.add_argument(
            "--nocheck", action="store_true", default=False,
            help="fast mode only for debug"
        )
    parser.add_argument(
                "--get-model-byname", type=str, required=False,
                help="Get a model specifying its name"
            )

    args = parser.parse_args()

    if not args.models_config.exists():
        logger.error(f"Error: config file '{args.models_config}' not found")
        sys.exit(1)

    if not args.rpc_config.exists():
        logger.error(f"Error: config file '{args.rpc_config}' not found")
        sys.exit(1)
    
    settings = get_settings()
    master_host = args.master_host if args.master_host is not None else settings.LLAMA_SERVER_HOST
    master_user = args.master_user if args.master_user is not None else settings.LLAMA_SERVER_USER

    if args.get_model_byname:
        m = get_model_byname( args.get_model_byname, args.models_config, args.rpc_config)
        logger.info(f"Model={m}")
        sys.exit(0)

    try:
        ms = load_models(config_path=args.models_config,
                         rpc_config_path=args.rpc_config,
                         remote_host=master_host,
                         remote_user=master_user,
                         check_remote_file=not args.nocheck)
    except ConfigError as e:
        logger.error(e)
        sys.exit(1)


    logger.debug(f"{len(ms)} models loaded (host: {master_host or 'local'})")
    for m in ms:
        size = f"{m.size_gib:.2f} GiB" if m.size_gib is not None else "n/a"
        logger.debug(f"  name={m.model_name:45} - size={size:>11s} rpc={m.rpc_endpoints()} mmproj={m.mmproj_path}")
