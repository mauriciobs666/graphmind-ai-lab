"""K-083: how the bring-up scripts resolve `FALKORCHAT_OPENCODE_CONFIG` when the caller
left it unset — one definition, `scripts/shlib/opencode_config.sh`, sourced by
`start_server.sh`, `start_demo.sh`, `ensure_demo_models.sh` and `start_agent_team.sh`:

  1. explicit env var -> used as-is, no message;
  2. else `$HOME/.config/opencode/opencode.local.json` if it exists -> one INFO line;
  3. else the shared `$HOME/.config/opencode/opencode.json` -> a WARNING.

`HOME` always points at a tmp dir, so the real `~/.config/opencode/` is never read or
written. The entry-point scripts are driven for real, but in a throwaway copy of the
`falkor-chat/` tree: each script is copied byte-for-byte (with the real helper) into
`<tmp>/falkor-chat/scripts/`, and everything it would call — sibling scripts, the venv's
`pip`/`uvicorn`, `redis-cli` on PATH — is a stub that records the env it got. No FalkorDB,
no LM Studio, no network, no real venv; never touches `reference`.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../falkor-chat
_SCRIPTS = _REPO_ROOT / "scripts"
_HELPER = _SCRIPTS / "shlib" / "opencode_config.sh"
_MODELS_JSON = _REPO_ROOT / "config" / "models.json"

ENTRY_POINTS = ["start_server.sh", "start_demo.sh", "ensure_demo_models.sh",
                "start_agent_team.sh"]

INFO = "INFO: FALKORCHAT_OPENCODE_CONFIG not set"
WARNING = "WARNING: FALKORCHAT_OPENCODE_CONFIG not set"


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _opencode(port: int) -> str:
    return json.dumps({"provider": {"lmstudio": {
        "npm": "@ai-sdk/openai-compatible", "name": "stub",
        "options": {"baseURL": f"http://127.0.0.1:{port}"},
    }}})


@pytest.fixture()
def home(tmp_path):
    """A tmp HOME with the shared opencode.json present (port 1111) and no local file."""
    h = tmp_path / "home"
    (h / ".config" / "opencode").mkdir(parents=True)
    (h / ".config" / "opencode" / "opencode.json").write_text(_opencode(1111))
    return h


def _shared(home: Path) -> Path:
    return home / ".config" / "opencode" / "opencode.json"


def _local(home: Path) -> Path:
    return home / ".config" / "opencode" / "opencode.local.json"


def _add_local(home: Path) -> Path:
    _local(home).write_text(_opencode(2222))
    return _local(home)


# ── the helper itself ───────────────────────────────────────────────────────────────

def _resolve(home: Path, env_value: str | None) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if env_value is not None:
        env["FALKORCHAT_OPENCODE_CONFIG"] = env_value
    script = (f'set -euo pipefail; . "{_HELPER}"; resolve_opencode_config; '
              'printf "%s" "$FALKORCHAT_OPENCODE_CONFIG"')
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True,
                          text=True, timeout=30)


def test_helper_env_var_set_is_used_as_is_silently(home):
    _add_local(home)  # present, but the explicit var still wins
    out = _resolve(home, "/somewhere/explicit.json")
    assert out.returncode == 0, out.stderr
    assert out.stdout == "/somewhere/explicit.json"
    assert out.stderr == ""


def test_helper_local_present_wins_with_an_info_line(home):
    local = _add_local(home)
    out = _resolve(home, None)
    assert out.returncode == 0, out.stderr
    assert out.stdout == str(local)
    assert INFO in out.stderr and str(local) in out.stderr
    assert "WARNING" not in out.stderr


def test_helper_local_absent_falls_back_to_shared_with_a_warning(home):
    out = _resolve(home, None)
    assert out.returncode == 0, out.stderr
    assert out.stdout == str(_shared(home))
    assert WARNING in out.stderr and str(_local(home)) in out.stderr
    assert "INFO" not in out.stderr


def test_helper_empty_env_var_counts_as_unset(home):
    local = _add_local(home)
    out = _resolve(home, "")
    assert out.stdout == str(local) and INFO in out.stderr


@pytest.mark.parametrize("tier", ["env", "local", "shared"])
def test_helper_exports_the_resolved_value_to_children(home, tier):
    # m2: a child process (create_workspace.sh -> pin) must see the same file.
    if tier == "local":
        _add_local(home)
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if tier == "env":
        env["FALKORCHAT_OPENCODE_CONFIG"] = "/explicit.json"
    script = (f'set -euo pipefail; FALKORCHAT_OPENCODE_CONFIG="${{FALKORCHAT_OPENCODE_CONFIG:-}}"; '
              f'export -n FALKORCHAT_OPENCODE_CONFIG; . "{_HELPER}"; resolve_opencode_config; '
              'bash -c \'printf "%s" "${FALKORCHAT_OPENCODE_CONFIG:-<not exported>}"\'')
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True,
                         timeout=30)
    assert out.returncode == 0, out.stderr
    expected = {"env": "/explicit.json", "local": str(_local(home)),
                "shared": str(_shared(home))}[tier]
    assert out.stdout == expected


def test_helper_with_home_unset_and_env_var_set_does_not_abort():
    # n1: under set -u, an explicit value must not need HOME at all.
    script = (f'set -euo pipefail; unset HOME; . "{_HELPER}"; resolve_opencode_config; '
              'printf "%s" "$FALKORCHAT_OPENCODE_CONFIG"')
    out = subprocess.run(["bash", "-c", script],
                         env={"PATH": "/usr/bin:/bin", "FALKORCHAT_OPENCODE_CONFIG": "/x.json"},
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout == "/x.json"


def test_helper_never_writes_the_opencode_dir(home):
    before = sorted(p.name for p in _shared(home).parent.iterdir())
    _resolve(home, None)
    assert sorted(p.name for p in _shared(home).parent.iterdir()) == before


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_every_entry_point_sources_the_one_definition(script):
    # No second copy of the order: the default is not spelled inline anywhere.
    text = (_SCRIPTS / script).read_text()
    assert "shlib/opencode_config.sh" in text and "resolve_opencode_config" in text
    assert "FALKORCHAT_OPENCODE_CONFIG:-$HOME" not in text
    assert "opencode.local.json\"" not in text  # no inline path assignment


# ── ensure_demo_models.sh, end to end (--print-set shows which file was read) ──────

def _print_set(home: Path, env_value: str | None) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin",
           "FALKORCHAT_MODEL_CONFIG": str(_MODELS_JSON)}
    if env_value is not None:
        env["FALKORCHAT_OPENCODE_CONFIG"] = env_value
    return subprocess.run(["bash", str(_SCRIPTS / "ensure_demo_models.sh"), "--print-set"],
                          env=env, capture_output=True, text=True, timeout=60)


def test_ensure_env_set(home, tmp_path):
    _add_local(home)
    explicit = tmp_path / "explicit.json"
    explicit.write_text(_opencode(3333))
    out = _print_set(home, str(explicit))
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["api_root"] == "http://127.0.0.1:3333"
    assert INFO not in out.stderr and WARNING not in out.stderr


def test_ensure_local_present(home):
    _add_local(home)
    out = _print_set(home, None)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["api_root"] == "http://127.0.0.1:2222"
    assert INFO in out.stderr


def test_ensure_local_absent(home):
    out = _print_set(home, None)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["api_root"] == "http://127.0.0.1:1111"
    assert WARNING in out.stderr


# ── start_demo.sh / start_server.sh / start_agent_team.sh in a stubbed tree copy ───

_RECORD_ENV = f'''#!{sys.executable}
import json, os, sys
with open(os.environ["STUB_RECORD"], "a") as fh:
    fh.write(json.dumps({{"argv0": os.path.basename(sys.argv[0]),
                         "opencode": os.environ.get("FALKORCHAT_OPENCODE_CONFIG")}}) + "\\n")
sys.exit(int(os.environ.get("STUB_RC_" + os.path.basename(sys.argv[0]).replace(".", "_"), "0")))
'''

_REDIS_CLI = '#!/bin/sh\necho PONG\n'


class Tree:
    """`<tmp>/falkor-chat/` holding a real copy of one entry-point script + the helper;
    every callee is a stub appending `{argv0, opencode}` to STUB_RECORD."""

    def __init__(self, tmp: Path, script: str):
        self.root = tmp / "falkor-chat"
        self.script = self.root / "scripts" / script
        (self.root / "scripts" / "shlib").mkdir(parents=True)
        shutil.copy2(_SCRIPTS / script, self.script)
        shutil.copy2(_HELPER, self.root / "scripts" / "shlib" / "opencode_config.sh")
        (self.root / "config").mkdir()
        (self.root / "config" / "logging.json").write_text("{}")
        for callee in ("create_workspace.sh", "seed_demo.sh", "seed_workflows.sh",
                       "seed_agent_team.sh", "start_falkordb.sh", "ensure_demo_models.sh"):
            if callee != script:
                _exe(self.root / "scripts" / callee, _RECORD_ENV)
        venv = self.root / "server" / ".venv" / "bin"
        _exe(venv / "pip", _RECORD_ENV)
        _exe(venv / "uvicorn", _RECORD_ENV)
        self.bin = tmp / "stubbin"
        _exe(self.bin / "redis-cli", _REDIS_CLI)
        self.record = tmp / "record.jsonl"

    def run(self, home: Path, env_value: str | None = None,
            **extra: str) -> subprocess.CompletedProcess:
        env = {"HOME": str(home), "PATH": f"{self.bin}:/usr/bin:/bin",
               "STUB_RECORD": str(self.record), **extra}
        if env_value is not None:
            env["FALKORCHAT_OPENCODE_CONFIG"] = env_value
        return subprocess.run(["bash", str(self.script)], env=env, capture_output=True,
                              text=True, timeout=60)

    def calls(self) -> list[dict]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def config_seen_by(self, argv0: str) -> str | None:
        seen = [c["opencode"] for c in self.calls() if c["argv0"] == argv0]
        assert seen, f"{argv0} never ran: {self.calls()}"
        return seen[-1]


# start_demo.sh hands the config to its ensure step (step 3); the stub exits 3 there, so
# the run stops before any seeding. start_server.sh / start_agent_team.sh export it to
# uvicorn, which the stub records.
_RECEIVER = {"start_demo.sh": "ensure_demo_models.sh", "start_server.sh": "uvicorn",
             "start_agent_team.sh": "uvicorn"}
_STOP_EARLY = {"STUB_RC_ensure_demo_models_sh": "3"}


@pytest.mark.parametrize("script", sorted(_RECEIVER))
def test_entry_point_env_set(script, home, tmp_path):
    _add_local(home)
    tree = Tree(tmp_path, script)
    out = tree.run(home, "/explicit/opencode.json", **_STOP_EARLY)
    assert tree.config_seen_by(_RECEIVER[script]) == "/explicit/opencode.json", out.stderr
    assert INFO not in out.stderr and WARNING not in out.stderr


@pytest.mark.parametrize("script", sorted(_RECEIVER))
def test_entry_point_local_present(script, home, tmp_path):
    local = _add_local(home)
    tree = Tree(tmp_path, script)
    out = tree.run(home, **_STOP_EARLY)
    assert tree.config_seen_by(_RECEIVER[script]) == str(local), out.stderr
    assert INFO in out.stderr and WARNING not in out.stderr


@pytest.mark.parametrize("script", sorted(_RECEIVER))
def test_entry_point_local_absent(script, home, tmp_path):
    tree = Tree(tmp_path, script)
    out = tree.run(home, **_STOP_EARLY)
    assert tree.config_seen_by(_RECEIVER[script]) == str(_shared(home)), out.stderr
    assert WARNING in out.stderr and INFO not in out.stderr


# ── start_demo.sh's step-3 exit-code branches (K-083 leftover, review P2-n2) ─────────

def test_start_demo_exit_3_offers_the_bypass_and_stops_before_seeding(home, tmp_path):
    tree = Tree(tmp_path, "start_demo.sh")
    out = tree.run(home, **{"STUB_RC_ensure_demo_models_sh": "3"})
    assert out.returncode == 1, out.stdout + out.stderr
    assert "Not an LM Studio server" in out.stderr
    assert "DEMO_ENSURE_MODELS=0 skips this step" in out.stderr
    assert "Fix it as the message says" not in out.stderr
    ran = [c["argv0"] for c in tree.calls()]
    assert "ensure_demo_models.sh" in ran
    assert not {"create_workspace.sh", "seed_demo.sh", "uvicorn"} & set(ran)


def test_start_demo_exit_1_never_offers_the_bypass(home, tmp_path):
    tree = Tree(tmp_path, "start_demo.sh")
    out = tree.run(home, **{"STUB_RC_ensure_demo_models_sh": "1"})
    assert out.returncode == 1, out.stdout + out.stderr
    assert "Fix it as the message says" in out.stderr
    assert "DEMO_ENSURE_MODELS" not in out.stderr
    assert not {"create_workspace.sh", "seed_demo.sh", "uvicorn"} & {
        c["argv0"] for c in tree.calls()}


# ── m2: children run BEFORE the late export see the resolved file too ──────────────

# start_demo.sh: ensure must pass (rc 0) to reach step 4's create_workspace.sh, whose stub
# then fails so the run stops there. start_server.sh / start_agent_team.sh run it before
# uvicorn with no stop needed.
_REACH_CREATE_WORKSPACE = {
    "start_demo.sh": {"STUB_RC_ensure_demo_models_sh": "0",
                      "STUB_RC_create_workspace_sh": "1"},
    "start_server.sh": {}, "start_agent_team.sh": {},
}


@pytest.mark.parametrize("script", sorted(_REACH_CREATE_WORKSPACE))
def test_create_workspace_sees_the_resolved_config(script, home, tmp_path):
    local = _add_local(home)
    tree = Tree(tmp_path, script)
    out = tree.run(home, **_REACH_CREATE_WORKSPACE[script])
    assert tree.config_seen_by("create_workspace.sh") == str(local), out.stdout + out.stderr
