"""K-095: `FALKORCHAT_AGENT_NAME` belongs to `scripts/seed_demo.sh` alone.

The server never reads it, so `config` carries no copy and the bring-up scripts hand it to
`seed_demo.sh` (the seeded `Agent`'s display name) but no longer export it to uvicorn.
The scripts are driven for real in a throwaway tree copy through the stub harnesses of
`test_opencode_config_resolution.py` (`start_server.sh`) and `test_start_demo_script.py`
(`start_demo.sh`); no FalkorDB, no LM Studio, no network.

`test_every_config_key_is_read` guards the item's done-condition ("no unread config key")
for every public module-level name `config.py` assigns. It counts AST reads, not text, across
`falkorchat/**/*.py`: a bare name in load position (`config.py`'s own functions included), an
attribute in load position (`config.X`, `_config.X`), and a `from .config import X as Y`
alias that is itself loaded somewhere. Comments, docstrings, string literals and the
definition itself (a store) never count. It is a bound, not a proof: an unrelated variable
or attribute that happens to share a key's name also counts as a read, and a read spelled
`getattr(config, "X")` does not (that direction fails red, not falsely green).
"""

from __future__ import annotations

import ast
from pathlib import Path

from test_opencode_config_resolution import Tree
from test_start_demo_script import _run_demo

_PACKAGE = Path(__file__).resolve().parents[1] / "falkorchat"


def _agent_names_seen_by(calls: list[dict], argv0: str) -> list[str | None]:
    seen = [c["agent_name"] for c in calls if c["argv0"] == argv0]
    assert seen, f"{argv0} never ran: {calls}"
    return seen


def _start_server(tmp_path: Path, **extra: str) -> list[dict]:
    tree = Tree(tmp_path, "start_server.sh")
    out = tree.run(tmp_path, "/explicit/opencode.json", **extra)
    assert out.returncode == 0, out.stdout + out.stderr
    return tree.calls()


def _start_demo(tmp_path: Path, **extra: str) -> list[dict]:
    out, calls = _run_demo(tmp_path, **extra)
    assert out.returncode == 0, out.stdout + out.stderr
    return calls


def test_start_server_seeds_the_default_name_without_exporting_it(tmp_path):
    calls = _start_server(tmp_path)
    assert _agent_names_seen_by(calls, "seed_demo.sh") == ["Assistant"]
    assert _agent_names_seen_by(calls, "uvicorn") == [None]


def test_start_server_hands_a_caller_name_to_seed_demo(tmp_path):
    calls = _start_server(tmp_path, FALKORCHAT_AGENT_NAME="Helper")
    assert _agent_names_seen_by(calls, "seed_demo.sh") == ["Helper"]


def test_start_demo_seeds_the_default_name_without_exporting_it(tmp_path):
    calls = _start_demo(tmp_path)
    assert _agent_names_seen_by(calls, "seed_demo.sh") == ["Assistant"]
    assert _agent_names_seen_by(calls, "uvicorn") == [None]


def test_start_demo_hands_a_caller_name_to_seed_demo(tmp_path):
    calls = _start_demo(tmp_path, FALKORCHAT_AGENT_NAME="Helper")
    assert _agent_names_seen_by(calls, "seed_demo.sh") == ["Helper"]


def _config_keys() -> set[str]:
    tree = ast.parse((_PACKAGE / "config.py").read_text())
    keys: set[str] = set()
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        keys.update(t.id for t in targets
                    if isinstance(t, ast.Name) and not t.id.startswith("_"))
    return keys


def _loaded_names() -> set[str]:
    loaded: set[str] = set()
    aliases: dict[str, str] = {}  # local alias -> name imported from config
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                loaded.add(node.attr)
            elif isinstance(node, ast.ImportFrom) and (
                    (node.level > 0 and node.module == "config")
                    or node.module == "falkorchat.config"):
                aliases.update((a.asname or a.name, a.name) for a in node.names)
    return loaded | {name for alias, name in aliases.items() if alias in loaded}


def test_every_config_key_is_read():
    unread = sorted(_config_keys() - _loaded_names())
    assert unread == [], f"config keys no code in falkorchat/ reads: {unread}"
