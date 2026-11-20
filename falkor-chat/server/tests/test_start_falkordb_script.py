"""`start_falkordb.sh`'s module-arg composition (K-098, `falkor-chat/docs/plans/graph-config-defaults-graph.md`
§5 Steps 1-3): the script builds the image's `FALKORDB_ARGS` from `config/falkordb.env`, which
REPLACES the image's own string, and `compose.yaml` must render the same string.

The real script runs in a throwaway copy of the `falkor-chat/` tree with `docker` stubbed on
`PATH` (the stub records its argv), so no container starts and no FalkorDB is needed. The
compose-parity tests call the real `docker compose ... config`, which only renders the file
(no daemon call); they skip when the `docker` CLI is absent.
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
_SCRIPT = _REPO_ROOT / "scripts" / "start_falkordb.sh"
_ENV_FILE = _REPO_ROOT / "config" / "falkordb.env"
_COMPOSE = _REPO_ROOT / "compose.yaml"

# The note's falkordb-dev values (§3, §4.1, §5 Step 1), as the image receives them.
_EXPECTED_ARGS = ("THREAD_COUNT 8 OMP_THREAD_COUNT 1 CACHE_SIZE 25 MAX_QUEUED_QUERIES 320 "
                  "QUERY_MEM_CAPACITY 67108864 ASYNC_DELETE yes TIMEOUT 1000 RESULTSET_SIZE 10000")
_IMAGE_ARGS = "MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"
_DATA_DIR = "/var/lib/falkordb/data"

_DOCKER_STUB = f'''#!{sys.executable}
import json, os, sys
with open(os.environ["STUB_RECORD"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
'''


def _tree(tmp: Path, env_text: str | None = None) -> Path:
    root = tmp / "falkor-chat"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(_SCRIPT, root / "scripts" / "start_falkordb.sh")
    (root / "config").mkdir()
    if env_text is None:
        shutil.copy2(_ENV_FILE, root / "config" / "falkordb.env")
    elif env_text != "<absent>":
        (root / "config" / "falkordb.env").write_text(env_text)
    stub = tmp / "stubbin" / "docker"
    stub.parent.mkdir()
    stub.write_text(_DOCKER_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return root


def _run(tmp: Path, *args: str, env_text: str | None = None,
         **extra: str) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    root = _tree(tmp, env_text)
    record = tmp / "record.jsonl"
    env = {"HOME": str(tmp), "PATH": f"{tmp / 'stubbin'}:/usr/bin:/bin",
           "STUB_RECORD": str(record), **extra}
    out = subprocess.run(["bash", str(root / "scripts" / "start_falkordb.sh"), *args],
                         env=env, capture_output=True, text=True, timeout=30)
    calls = ([json.loads(line) for line in record.read_text().splitlines()]
             if record.exists() else [])
    return out, calls


def _docker_run(out, calls) -> list[str]:
    assert out.returncode == 0, out.stdout + out.stderr
    assert len(calls) == 1 and calls[0][0] == "run", calls
    return calls[0]


def _flag(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


@pytest.mark.parametrize("mode", [("-d",), ()], ids=["detached", "foreground"])
def test_default_passes_the_env_file_string_in_both_branches(tmp_path, mode):
    argv = _docker_run(*_run(tmp_path, *mode))
    assert _flag(argv, "-e") == [f"FALKORDB_ARGS={_EXPECTED_ARGS}"]
    assert _flag(argv, "-v") == [f"falkordb-data:{_DATA_DIR}"]
    assert ("-d" in argv) == bool(mode)


def test_banner_echoes_module_args(tmp_path):
    out, _ = _run(tmp_path, "-d")
    assert f"Module args:             {_EXPECTED_ARGS}" in out.stdout


def test_thread_count_override_changes_only_that_token(tmp_path):
    argv = _docker_run(*_run(tmp_path, "-d", FALKORDB_THREAD_COUNT="4"))
    assert _flag(argv, "-e") == [
        "FALKORDB_ARGS=" + _EXPECTED_ARGS.replace("THREAD_COUNT 8", "THREAD_COUNT 4", 1)]


def test_module_args_override_wins_over_file(tmp_path):
    argv = _docker_run(*_run(tmp_path, "-d", FALKORDB_MODULE_ARGS="TIMEOUT 2000"))
    assert _flag(argv, "-e") == ["FALKORDB_ARGS=THREAD_COUNT 8 TIMEOUT 2000"]


def test_full_falkordb_args_used_as_is_without_env_file(tmp_path):
    argv = _docker_run(*_run(tmp_path, "-d", env_text="<absent>", FALKORDB_ARGS=_IMAGE_ARGS))
    assert _flag(argv, "-e") == [f"FALKORDB_ARGS={_IMAGE_ARGS}"]


def test_volume_override(tmp_path):
    argv = _docker_run(*_run(tmp_path, "-d", FALKORDB_VOLUME="/scratch/dir"))
    assert _flag(argv, "-v") == [f"/scratch/dir:{_DATA_DIR}"]


def test_env_file_override_path(tmp_path):
    other = tmp_path / "prod.env"
    other.write_text("FALKORDB_THREAD_COUNT=2\nFALKORDB_MODULE_ARGS=MAX_QUEUED_QUERIES 128\n")
    argv = _docker_run(*_run(tmp_path, "-d", FALKORDB_ENV_FILE=str(other)))
    assert _flag(argv, "-e") == ["FALKORDB_ARGS=THREAD_COUNT 2 MAX_QUEUED_QUERIES 128"]


@pytest.mark.parametrize("env_text, needle", [
    ("<absent>", "not found"),
    ("FALKORDB_THREAD_COUNT=8\n", "FALKORDB_MODULE_ARGS"),
    ("FALKORDB_MODULE_ARGS=TIMEOUT 1000\n", "FALKORDB_THREAD_COUNT"),
    ("FALKORDB_THREAD_COUNT=8\nFALKORDB_MODULE_ARGS=TIMEOUT 1000\nFALKORDB_PORT=6380\n",
     "unknown key 'FALKORDB_PORT'"),
], ids=["missing-file", "missing-module-args", "missing-thread-count", "unknown-key"])
def test_bad_env_file_fails_before_docker_run(tmp_path, env_text, needle):
    out, calls = _run(tmp_path, "-d", env_text=env_text)
    assert out.returncode != 0
    assert needle in out.stderr
    assert calls == []


def test_env_file_is_parsed_not_sourced(tmp_path):
    """`$(…)` is outside the env-file grammar: refused before `docker run`, and never executed."""
    marker = tmp_path / "pwned"
    text = f"FALKORDB_THREAD_COUNT=8\nFALKORDB_MODULE_ARGS=TIMEOUT 1000 $(touch {marker})"
    out, calls = _run(tmp_path, "-d", env_text=text)
    assert out.returncode != 0 and calls == [], out.stdout + out.stderr
    assert not marker.exists()


def test_subscript_injection_key_is_refused_unevaluated(tmp_path):
    marker = tmp_path / "pwned"
    text = f"a[$(touch {marker})]=1\nFALKORDB_THREAD_COUNT=8\nFALKORDB_MODULE_ARGS=TIMEOUT 1000\n"
    out, calls = _run(tmp_path, "-d", env_text=text)
    assert out.returncode != 0 and calls == [] and "unknown key" in out.stderr
    assert not marker.exists()


def test_crlf_env_file_is_tolerated(tmp_path):
    text = "# c\r\n\r\nFALKORDB_THREAD_COUNT=8\r\nFALKORDB_MODULE_ARGS=TIMEOUT 1000\r\n"
    argv = _docker_run(*_run(tmp_path, "-d", env_text=text))
    assert _flag(argv, "-e") == ["FALKORDB_ARGS=THREAD_COUNT 8 TIMEOUT 1000"]


def test_unreadable_env_file_says_not_readable(tmp_path):
    out, calls = _run(tmp_path, "-d", FALKORDB_ENV_FILE=str(tmp_path))  # a directory
    assert out.returncode != 0 and calls == []
    assert "not found or not readable" in out.stderr


def test_shipped_env_file_booleans_are_yes_no():
    """`ASYNC_DELETE 1` / `CMD_INFO 1` abort the server start (note §3.6): booleans are yes/no."""
    values = dict(line.split("=", 1) for line in _ENV_FILE.read_text().splitlines()
                  if line and not line.startswith("#"))
    tokens = values["FALKORDB_MODULE_ARGS"].split()
    pairs = dict(zip(tokens[::2], tokens[1::2]))
    for key in ("ASYNC_DELETE", "CMD_INFO"):
        if key in pairs:
            assert pairs[key] in ("yes", "no"), (key, pairs[key])


_docker = shutil.which("docker")
needs_compose = pytest.mark.skipif(_docker is None, reason="docker CLI not installed")


def _compose(*args: str, **extra: str) -> subprocess.CompletedProcess:
    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin:" + str(Path(_docker).parent),
           **extra}
    return subprocess.run([_docker, "compose", "-f", str(_COMPOSE), *args, "config",
                           "--format", "json"], cwd=_REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=60)


@needs_compose
def test_compose_renders_the_same_string_as_the_script(tmp_path):
    out = _compose("--env-file", str(_ENV_FILE))
    assert out.returncode == 0, out.stderr
    rendered = json.loads(out.stdout)["services"]["falkordb"]["environment"]["FALKORDB_ARGS"]
    script_argv = _docker_run(*_run(tmp_path, "-d"))
    assert f"FALKORDB_ARGS={rendered}" == _flag(script_argv, "-e")[0]
    assert rendered == _EXPECTED_ARGS


@needs_compose
@pytest.mark.parametrize("present, missing", [
    ({}, "FALKORDB_THREAD_COUNT"),
    ({"FALKORDB_MODULE_ARGS": "TIMEOUT 1000"}, "FALKORDB_THREAD_COUNT"),
    ({"FALKORDB_THREAD_COUNT": "8"}, "FALKORDB_MODULE_ARGS"),
], ids=["neither", "no-thread-count", "no-module-args"])
def test_compose_refuses_without_each_var(present, missing):
    """Each of the two variables is required on its own: no default ever lets compose start
    FalkorDB on a partial string."""
    out = _compose(**present)
    assert out.returncode != 0
    assert f"required variable {missing} is missing" in out.stderr
    assert "--env-file config/falkordb.env" in out.stderr


# Every grammar axis on which compose's dotenv reader and the script's reader could differ
# (review m2, `falkor-chat/docs/reviews/graph-config-defaults-impl.md`). Each shape must be either
# refused by the script before `docker run`, or rendered identically by both. "same" rows also
# guard against a reader so strict that it refuses legitimate files.
_T, _M = "FALKORDB_THREAD_COUNT", "FALKORDB_MODULE_ARGS"
_GRAMMAR_AXES = [
    ("plain", f"{_T}=8\n{_M}=TIMEOUT 1000 ASYNC_DELETE yes\n", "same"),
    ("comments-and-blank-lines", f"# a\n\n{_T}=8\n# b\n{_M}=TIMEOUT 1000\n", "same"),
    ("crlf-with-blank-and-comment", f"# a\r\n\r\n{_T}=8\r\n{_M}=TIMEOUT 1000\r\n", "same"),
    ("no-final-newline", f"{_T}=8\n{_M}=TIMEOUT 1000", "same"),
    ("dotted-and-dashed-words", f"{_T}=8\n{_M}=X 1.5 Y -1\n", "same"),
    ("repeated-key", f"{_T}=8\n{_M}=TIMEOUT 1000\n{_T}=4\n", "refuse"),
    ("double-quotes", f'{_T}=8\n{_M}="TIMEOUT 1000"\n', "refuse"),
    ("single-quotes", f"{_T}=8\n{_M}='TIMEOUT 1000'\n", "refuse"),
    ("inline-comment", f"{_T}=8 # cores\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("hash-in-word", f"{_T}=8\n{_M}=TIMEOUT 1000#x\n", "refuse"),
    ("dollar-var", f"{_T}=8\n{_M}=TIMEOUT $HOME\n", "refuse"),
    ("dollar-brace", f"{_T}=8\n{_M}=TIMEOUT ${{HOME}}\n", "refuse"),
    ("trailing-equals", f"{_T}=8\n{_M}=TIMEOUT 1000=\n", "refuse"),
    ("export-prefix", f"export {_T}=8\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("leading-whitespace-line", f"  {_T}=8\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("indented-comment", f"  # c\n{_T}=8\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("space-before-equals", f"{_T} =8\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("leading-space-in-value", f"{_T}= 8\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("trailing-space-in-value", f"{_T}=8 \n{_M}=TIMEOUT 1000\n", "refuse"),
    ("double-space-in-value", f"{_T}=8\n{_M}=TIMEOUT  1000\n", "refuse"),
    ("tab-in-value", f"{_T}=8\n{_M}=TIMEOUT\t1000\n", "refuse"),
    ("backslash", f"{_T}=8\n{_M}=TIMEOUT 1000\\\n", "refuse"),
    ("yaml-colon-form", f"{_T}: 8\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("no-equals", f"{_T}\n{_M}=TIMEOUT 1000\n", "refuse"),
    ("empty-value", f"{_T}=\n{_M}=TIMEOUT 1000\n", "refuse"),
]


@pytest.mark.parametrize("text, expect", [a[1:] for a in _GRAMMAR_AXES],
                         ids=[a[0] for a in _GRAMMAR_AXES])
def test_env_file_grammar_axis_refused_or_identical_to_compose(tmp_path, text, expect):
    out, calls = _run(tmp_path, "-d", env_text=text)
    if expect == "refuse":
        assert out.returncode != 0, f"accepted: {calls}"
        assert calls == []
        return
    argv = _docker_run(out, calls)
    if _docker is None:
        pytest.skip("docker CLI not installed: compose side of the parity check not run")
    env_file = tmp_path / "falkor-chat" / "config" / "falkordb.env"
    rendered = _compose("--env-file", str(env_file))
    assert rendered.returncode == 0, rendered.stderr
    value = json.loads(rendered.stdout)["services"]["falkordb"]["environment"]["FALKORDB_ARGS"]
    assert _flag(argv, "-e") == [f"FALKORDB_ARGS={value}"]
