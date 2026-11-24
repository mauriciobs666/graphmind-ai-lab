"""Test launcher: runs the real `Holder` in its own process with injected paths and trusted uid
(production `keyholder.main` hard-wires `/etc`, `/var/lib` and uid 0). Prints READY once bound."""

import json
import os
import pathlib
import signal
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llmgateway.config import ConfigSource  # noqa: E402
from llmgateway.keyholder import Holder, HolderSettings, HolderStartError  # noqa: E402

spec = json.loads(sys.argv[1])
settings = HolderSettings(
    config_source=ConfigSource(spec["config"], {spec["uid"]}, anchor=spec["anchor"]),
    state_root=spec["state_root"],
    socket_path=spec["socket"],
    read_timeout=spec.get("read_timeout", 5.0),
    max_frame_bytes=spec.get("max_frame_bytes", 4 << 20),
    max_connections=spec.get("max_connections", 64),
)
try:
    holder = Holder(settings)
    holder.bind()
except HolderStartError as exc:
    print(f"REFUSED {exc}", flush=True)
    sys.exit(2)
signal.signal(signal.SIGTERM, lambda *_: holder.stop())
print("READY", flush=True)
holder.serve_forever()
holder.close()
os._exit(0)
