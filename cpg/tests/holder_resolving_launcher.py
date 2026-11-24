"""Test launcher: `llm-gateway/tests/holder_launcher.py` plus a resolver that maps every provider name to 127.0.0.1.

A hosted entry of a ccg manifest must name a dotted host (the manifest schema's `remoteEndpoint`) and the broker's static
check refuses a loopback IP literal there, so the real-subprocess tests give the provider a name (`*.example.test`) that
only this holder resolves, to the logging stand-in on loopback. Nothing in `llm-gateway/` is edited.
Prints READY once bound.
"""
import json
import os
import pathlib
import signal
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm-gateway" / "src"))

from llmgateway.config import ConfigSource  # noqa: E402
from llmgateway.keyholder import Holder, HolderSettings, HolderStartError  # noqa: E402

spec = json.loads(sys.argv[1])
settings = HolderSettings(
    config_source=ConfigSource(spec["config"], {spec["uid"]}, anchor=spec["anchor"]),
    state_root=spec["state_root"],
    socket_path=spec["socket"],
    resolver=lambda host, port: ["127.0.0.1"],
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
