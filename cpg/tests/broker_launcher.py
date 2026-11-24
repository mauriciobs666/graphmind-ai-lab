"""Test launcher: runs the real `ccg.broker.server.main` as its own process with the test-only keyword seams
(`trusted_uids`, `check_hosted_entries`), which no option or environment variable of the product reaches.

    python -I broker_launcher.py '{"argv": [...], "trusted": [0, 1000], "check_hosted": true, "fake_uid": true}'

It skips `reexec_if_needed` exactly as a test calling `main` directly does (the re-exec would drop the injection).
`fake_uid` makes the `--holder-uid != own uid` rule see another uid while the arguments are parsed, so a real holder owned
by the test's own account can be named: the production rule itself is exercised by `python -m ccg.broker.server` tests.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccg.broker import server  # noqa: E402

spec = json.loads(sys.argv[1])
if spec.get("fake_uid"):
    _parse = server._parse

    def parse(argv):
        real, os.getuid = os.getuid, lambda: 2**31 - 5
        try:
            return _parse(argv)
        finally:
            os.getuid = real

    server._parse = parse
sys.exit(server.main(spec["argv"], trusted_uids=frozenset(spec["trusted"]),
                     check_hosted_entries=spec.get("check_hosted", True)))
