from dataclasses import dataclass
from logzero import logger
from pathlib import Path
import json
import sys

from errors import ConfigError

# Every RPC server entry in rpc.json must carry all of these.
_REQUIRED_KEYS = ("ip", "port", "cachepath", "bin", "remuser")

#___________________________________________________________________________________
@dataclass
class RpcServer:
    IP: str
    PORT: int
    cachepath: Path
    bin: Path
    remuser: str
    name: str

    def endpoint( self ) -> str:
        return f"{self.IP}:{self.PORT}"


#___________________________________________________________________________________
def load_rpcs(config_path: Path):
    with config_path.open(encoding="utf-8") as f:
        rpcs = json.load(f)

    rpc_section = rpcs.get("RPC_SERVERS", {})
    if not rpc_section:
        raise ConfigError(f"Malformed json file {config_path}: missing section 'RPC_SERVERS'")

    rpc = {}

    for name, spec in rpc_section.items():
        missing = [k for k in _REQUIRED_KEYS if not spec.get(k)]
        if missing:
            raise ConfigError(
                f"Malformed json file {config_path}: RPC server '{name}' is missing "
                f"attribute(s): {', '.join(missing)}"
            )
        rpc[name] = RpcServer(
                IP=spec["ip"],
                PORT=int(spec["port"]),
                cachepath=spec["cachepath"],
                bin=spec["bin"],
                remuser=spec["remuser"],
                name=name
            )
    return rpc

#___________________________________________________________________________________
if __name__ == "__main__":
    import argparse
    from config_manager import get_settings

    parser = argparse.ArgumentParser(description="List RPC servers from rpc config file")

    parser.add_argument(
        "--models-config", default=Path(__file__).parent / "rpc.json",
        help="path of models json description file"
    )
    
    args = parser.parse_args()

    config_path = Path(args.models_config)

    if not config_path.exists():
        logger.error(f"Error: config file '{config_path}' not found")
        sys.exit(1)
    
    settings = get_settings()

    try:
        rpcs = load_rpcs(config_path=config_path)
    except ConfigError as e:
        logger.error(e)
        sys.exit(1)
    logger.debug(f"{len(rpcs)} RPC servers loaded")
    logger.debug(f"RPC={rpcs}")