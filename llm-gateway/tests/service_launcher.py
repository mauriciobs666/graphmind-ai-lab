"""Test launcher: runs the real `Service` in its own process with an injected configuration
source, credential path and socket (production `service.main` hard-wires `/etc` and uid 0).
Prints READY once bound. `holder_own_uid=-1` lets the test holder run under the test's own uid."""

import json
import os
import pathlib
import signal
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llmgateway.config import ConfigSource  # noqa: E402
from llmgateway.service import Service, ServiceSettings, ServiceStartError  # noqa: E402

spec = json.loads(sys.argv[1])
settings = ServiceSettings(
    config_source=ConfigSource(spec["config"], {spec["uid"]}, anchor=spec["anchor"]),
    socket_path=spec["socket"],
    state_dir=spec["state_dir"],
    credential_path=spec.get("credential"),
    holder_own_uid=-1,
    read_timeout=spec.get("read_timeout", 5.0),
)
try:
    service = Service(settings)
    service.bind()
except ServiceStartError as exc:
    print(f"REFUSED {exc}", flush=True)
    sys.exit(2)
signal.signal(signal.SIGTERM, lambda *_: service.stop())
print("READY", flush=True)
service.serve_forever()
service.close()
os._exit(0)
