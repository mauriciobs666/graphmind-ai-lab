"""K-066 (1): `scripts/ensure_demo_models.sh` — the demo bring-up step that ensures LM
Studio has the storefront demo's models loaded, and never unloads or reloads.

Subprocess-integration convention (`test_create_workspace_script.py`): this drives the
real wrapper + Python script end to end. LM Studio is replaced by two stubs:

  * a stub `/api/v0/models` HTTP server on 127.0.0.1 (a thread in this process) that
    serves a JSON state file, reached through a throwaway `FALKORCHAT_OPENCODE_CONFIG`
    whose `lmstudio` baseURL points at it — so the script's real derivation path
    (`ModelGateway` + `SALESPERSON_DEF`) is what finds the URL and the model ids;
  * a stub `lms` CLI (`LMS_CLI`) that appends its argv to a log and, on `load`, marks
    the model loaded in the state file at the requested `-c`; on `ps --json` it lists
    the loaded instances with their real context (`_ps_ctx`, which the API side
    misreports as 2048 for embedders — K-099's advisory embedder-context check).

No FalkorDB, no network beyond loopback, no real LM Studio. Never touches `reference`.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../falkor-chat
_SCRIPT = _REPO_ROOT / "scripts" / "ensure_demo_models.sh"
_MODELS_JSON = _REPO_ROOT / "config" / "models.json"

CHAT = "mistralai/ministral-3-3b"
EMBED = "text-embedding-qwen3-embedding-0.6b"
QWEN = "qwen/qwen3-4b-2507"


def _model(mid: str, *, loaded: bool, ctx: int | None = None, mtype: str = "llm") -> dict:
    m = {"id": mid, "type": mtype, "state": "loaded" if loaded else "not-loaded"}
    if loaded:
        m["loaded_context_length"] = ctx
    return m


def _settle_one_read(state_file: Path) -> bytes:
    """Serve the state, then count down any `_loading_reads` (a model that reports
    `loading` for that many more API reads before turning `loaded`)."""
    body = state_file.read_bytes()
    data = json.loads(body)
    changed = False
    for m in data.get("data", []) if isinstance(data, dict) else []:
        if isinstance(m, dict) and "_loading_reads" in m:
            changed = True
            m["_loading_reads"] -= 1
            if m["_loading_reads"] <= 0:
                del m["_loading_reads"]
                m["state"] = "loaded"
    if changed:
        state_file.write_text(json.dumps(data))
    return body


class _Handler(BaseHTTPRequestHandler):
    state_file: Path

    def do_GET(self):  # noqa: N802
        if self.path != "/api/v0/models":
            self.send_response(404)
            self.end_headers()
            return
        srv = self.server
        if srv.mode == "close":  # accept, read the request, answer nothing, close
            self.close_connection = True
            return
        if srv.mode == "garbage":  # a non-HTTP status line -> BadStatusLine
            self.wfile.write(b"garbage\r\n\r\n")
            self.close_connection = True
            return
        if srv.status != 200:  # type: ignore[attr-defined]
            self.send_response(srv.status)  # type: ignore[attr-defined]
            self.end_headers()
            return
        if srv.raw_body is not None:  # type: ignore[attr-defined]
            body = srv.raw_body  # type: ignore[attr-defined]
        else:
            body = _settle_one_read(srv.state_file)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        # truncate: promise more bytes than are sent, then close -> IncompleteRead
        extra = 1000 if srv.truncate else 0  # type: ignore[attr-defined]
        self.send_header("Content-Length", str(len(body) + extra))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


_STUB_LMS = r'''#!{python}
import json, sys
state_file, log_file = {state!r}, {log!r}
with open(log_file, "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\n")
args = sys.argv[1:]
if args and args[0] == "load":
    mid = args[1]
    ctx = int(args[args.index("-c") + 1]) if "-c" in args else 4096
    data = json.load(open(state_file))
    for m in data["data"]:
        if m["id"] == mid:
            m["state"] = "loaded"
            # LM Studio's API misreports embedders' context as 2048; `lms ps` does not
            m["loaded_context_length"] = 2048 if m.get("type") == "embeddings" else ctx
            m["_ps_ctx"] = ctx
    json.dump(data, open(state_file, "w"))
    print("Model loaded successfully.")
elif args[:2] == ["ps", "--json"]:
    data = json.load(open(state_file))
    print(json.dumps([{{"identifier": m["id"],
                        "contextLength": m.get("_ps_ctx", m.get("loaded_context_length"))}}
                       for m in data["data"] if m.get("state") == "loaded"]))
sys.exit(0)
'''


class Rig:
    def __init__(self, tmp: Path, port: int):
        self.tmp = tmp
        self.state = tmp / "state.json"
        self.log = tmp / "lms.log"
        self.lms = tmp / "bin" / "lms"
        self.port = port
        self.opencode = tmp / "opencode.json"
        self.opencode.write_text(json.dumps({"provider": {"lmstudio": {
            "npm": "@ai-sdk/openai-compatible", "name": "stub",
            "options": {"baseURL": f"http://127.0.0.1:{port}"},
        }}}))
        self.lms.parent.mkdir()
        self.lms.write_text(_STUB_LMS.format(python=sys.executable, state=str(self.state),
                                             log=str(self.log)))
        self.lms.chmod(self.lms.stat().st_mode | stat.S_IXUSR)

    def set_models(self, models: list[dict]) -> None:
        self.state.write_text(json.dumps({"object": "list", "data": models}))

    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def run(self, *, lms: bool = True, opencode: Path | None = None,
            settle_s: float | str = 2.0,
            settle_interval_s: float | str = 0.05) -> subprocess.CompletedProcess:
        env = {
            # Post-load settle wait, shrunk from the operator default so the suite stays fast.
            "ENSURE_DEMO_MODELS_SETTLE_S": str(settle_s),
            "ENSURE_DEMO_MODELS_SETTLE_INTERVAL_S": str(settle_interval_s),
            "HOME": os.environ.get("HOME", str(self.tmp)),
            # Minimal PATH: the dev box's Windows interop PATH carries lms.exe.
            "PATH": "/usr/bin:/bin",
            "FALKORCHAT_OPENCODE_CONFIG": str(opencode or self.opencode),
            "FALKORCHAT_MODEL_CONFIG": str(_MODELS_JSON),
        }
        if lms:
            env["LMS_CLI"] = str(self.lms)
        return subprocess.run(["bash", str(_SCRIPT)], env=env, capture_output=True,
                              text=True, timeout=60)


@pytest.fixture()
def rig(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    r = Rig(tmp_path, server.server_address[1])
    server.state_file = r.state  # type: ignore[attr-defined]
    server.status = 200  # type: ignore[attr-defined]
    server.raw_body = None  # type: ignore[attr-defined]
    server.truncate = False  # type: ignore[attr-defined]
    server.mode = None  # type: ignore[attr-defined]
    r.server = server
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield r
    server.shutdown()
    server.server_close()


def _catalog(*, chat=None, embed=None, embed_ps=4096, extra=()) -> list[dict]:
    """chat/embed: None = not loaded, else the context the API reports; embed_ps: the
    embedder's real context, as `lms ps --json` shows it."""
    embedder = _model(EMBED, loaded=embed is not None, ctx=embed, mtype="embeddings")
    if embed is not None:
        embedder["_ps_ctx"] = embed_ps
    return [
        _model(CHAT, loaded=chat is not None, ctx=chat),
        embedder,
        _model(QWEN, loaded=False),
        *extra,
    ]


def _loads(calls):
    return [c for c in calls if c and c[0] == "load"]


def _unloads(calls):
    return [c for c in calls if c and c[0] == "unload"]


def _mutating(calls):
    """Every CLI call except the read-only `ps --json` of the embedder-context check."""
    return [c for c in calls if c[:2] != ["ps", "--json"]]


def test_derives_the_demo_set_from_its_owners(rig):
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", ""),
           "FALKORCHAT_OPENCODE_CONFIG": str(rig.opencode),
           "FALKORCHAT_MODEL_CONFIG": str(_MODELS_JSON)}
    out = subprocess.run(["bash", str(_SCRIPT), "--print-set"], env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"api_root": f"http://127.0.0.1:{rig.port}",
                                      "chat": [CHAT], "embedding": EMBED}


def test_both_loaded_is_a_noop(rig):
    # The embedder reported at 2048 (the API's known misreport) must not be rejected.
    rig.set_models(_catalog(chat=8192, embed=2048))
    out = rig.run()
    assert out.returncode == 0, out.stderr
    assert "nothing to load" in out.stdout
    assert _mutating(rig.calls()) == []


def test_both_loaded_needs_no_cli(rig):
    rig.set_models(_catalog(chat=8192, embed=2048))
    out = rig.run(lms=False)
    assert out.returncode == 0, out.stderr
    assert "nothing to load" in out.stdout


def test_one_missing_loads_exactly_it_with_demo_flags(rig):
    rig.set_models(_catalog(chat=8192, embed=None))
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    # K-099: the embedder loads at its own 4096, not the chat model's 8192.
    assert _mutating(rig.calls()) == [["load", EMBED, "-c", "4096", "--gpu", "max",
                                       "--identifier", EMBED, "-y"]]
    assert "demo models ready" in out.stdout
    assert "WARNING" not in out.stderr  # loaded at 4096 -> the ps check agrees


def test_missing_chat_model_loads_at_8192(rig):
    rig.set_models(_catalog(chat=None, embed=2048))
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert _loads(rig.calls()) == [["load", CHAT, "-c", "8192", "--gpu", "max",
                                    "--identifier", CHAT, "-y"]]


def test_both_missing_loads_both_and_second_run_is_idempotent(rig):
    rig.set_models(_catalog())
    first = rig.run()
    assert first.returncode == 0, first.stdout + first.stderr
    assert [c[1] for c in _loads(rig.calls())] == [CHAT, EMBED]
    assert [c[2:6] for c in _loads(rig.calls())] == [["-c", "8192", "--gpu", "max"],
                                                    ["-c", "4096", "--gpu", "max"]]
    second = rig.run()
    assert second.returncode == 0
    assert "nothing to load" in second.stdout
    assert len(_loads(rig.calls())) == 2  # no new load
    assert _unloads(rig.calls()) == []


def test_foreign_model_loaded_stops_without_load_or_unload(rig):
    models = _catalog(chat=8192, embed=None)
    models[2] = _model(QWEN, loaded=True, ctx=8192)
    rig.set_models(models)
    out = rig.run()
    assert out.returncode == 1
    assert QWEN in out.stderr and "never unloads" in out.stderr
    assert f"lms unload {QWEN}" in out.stderr  # told what to do, not done
    assert rig.calls() == []


def test_foreign_instance_of_a_demo_model_under_another_identifier_stops(rig):
    rig.set_models(_catalog(chat=8192, embed=2048,
                            extra=[_model("ministral-half", loaded=True, ctx=8192)]))
    out = rig.run()
    assert out.returncode == 1
    assert "ministral-half" in out.stderr
    assert rig.calls() == []


def test_chat_model_with_wrong_context_stops(rig):
    rig.set_models(_catalog(chat=4096, embed=2048))
    out = rig.run()
    assert out.returncode == 1
    assert "context 4096" in out.stderr and "8192" in out.stderr
    assert rig.calls() == []


def test_wrong_context_stops_even_when_the_other_model_is_missing(rig):
    rig.set_models(_catalog(chat=4096, embed=None))
    out = rig.run()
    assert out.returncode == 1
    assert rig.calls() == []  # refusals come before any load


def test_cli_absent_stops_with_instructions(rig):
    rig.set_models(_catalog(chat=8192, embed=None))
    out = rig.run(lms=False)
    assert out.returncode == 1
    assert "lms CLI was not found" in out.stderr
    assert f"lms load {EMBED} -c 4096 --gpu max" in out.stderr
    assert f"{EMBED} 4096" in out.stderr  # the LM Studio app hint names its own context
    assert "LMS_CLI" in out.stderr


def test_api_unreachable_stops(rig, tmp_path):
    dead = tmp_path / "dead-opencode.json"
    dead.write_text(json.dumps({"provider": {"lmstudio": {
        "npm": "@ai-sdk/openai-compatible", "name": "dead",
        "options": {"baseURL": "http://127.0.0.1:9"},  # discard port: refused
    }}}))
    out = rig.run(opencode=dead)
    assert out.returncode == 1
    assert "unreachable" in out.stderr
    assert rig.calls() == []


def test_load_that_does_not_show_up_in_the_api_stops(rig):
    # A CLI wired to a different LM Studio than the server's URL: load "succeeds",
    # the API still does not list it.
    rig.set_models(_catalog(chat=8192, embed=None))
    rig.lms.write_text(f"#!{sys.executable}\nimport json,sys\n"
                       f"open({str(rig.log)!r},'a').write(json.dumps(sys.argv[1:])+'\\n')\n")
    out = rig.run()
    assert out.returncode == 1
    assert "still does not list" in out.stderr
    assert len(_loads(rig.calls())) == 1 and _unloads(rig.calls()) == []


def _stub_lms_with_side_effect(rig, body: str) -> None:
    """Replace the stub CLI: log argv, then run `body` against the state on `load`."""
    rig.lms.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"state_file, log_file = {str(rig.state)!r}, {str(rig.log)!r}\n"
        "open(log_file, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "args = sys.argv[1:]\n"
        "if args and args[0] == 'load':\n"
        "    data = json.load(open(state_file))\n"
        "    mid = args[1]\n"
        + "".join(f"    {line}\n" for line in body.splitlines())
        + "    json.dump(data, open(state_file, 'w'))\n"
    )


def test_foreign_model_appearing_during_the_load_stops_after_it(rig):
    # m1: the check-then-load window — e.g. :8200 JIT-loading qwen while `lms load` runs.
    rig.set_models(_catalog(chat=8192, embed=None))
    _stub_lms_with_side_effect(rig, (
        "for m in data['data']:\n"
        "    if m['id'] == mid: m.update(state='loaded', loaded_context_length=2048)\n"
        f"    if m['id'] == {QWEN!r}: m.update(state='loaded', loaded_context_length=8192)"
    ))
    out = rig.run()
    assert out.returncode == 1, out.stdout
    assert QWEN in out.stderr and "outside the demo set" in out.stderr
    assert "demo models ready" not in out.stdout
    assert len(_loads(rig.calls())) == 1 and _unloads(rig.calls()) == []


def test_chat_model_landing_at_wrong_context_stops_after_load(rig):
    # m1: the CLI ignored -c (or another loader won the race) — chat ends up at 4096.
    rig.set_models(_catalog(chat=None, embed=2048))
    _stub_lms_with_side_effect(rig, (
        "for m in data['data']:\n"
        "    if m['id'] == mid: m.update(state='loaded', loaded_context_length=4096)"
    ))
    out = rig.run()
    assert out.returncode == 1, out.stdout
    assert "context 4096" in out.stderr
    assert _unloads(rig.calls()) == []


def test_not_lm_studio_exits_3_with_the_bypass_hint(rig, tmp_path):
    # m3: a server that answers but has no /api/v0 — the one case the bypass is for.
    other = tmp_path / "other-opencode.json"
    other.write_text(json.dumps({"provider": {"lmstudio": {
        "npm": "@ai-sdk/openai-compatible", "name": "other",
        "options": {"baseURL": f"http://127.0.0.1:{rig.port}/not-lmstudio"},
    }}}))
    out = rig.run(opencode=other)
    assert out.returncode == 3
    assert "DEMO_ENSURE_MODELS=0" in out.stderr
    assert rig.calls() == []


@pytest.mark.parametrize("case", ["foreign", "wrong_context", "cli_absent", "unreachable"])
def test_enforcing_refusals_exit_1_and_never_offer_the_bypass(rig, tmp_path, case):
    # m3: the bypass must not be advertised on a refusal the step exists to enforce.
    lms = True
    opencode = None
    if case == "foreign":
        rig.set_models(_catalog(chat=8192, embed=2048,
                                extra=[_model("ministral-half", loaded=True, ctx=8192)]))
    elif case == "wrong_context":
        rig.set_models(_catalog(chat=4096, embed=2048))
    elif case == "cli_absent":
        rig.set_models(_catalog(chat=8192, embed=None))
        lms = False
    else:
        opencode = tmp_path / "dead.json"
        opencode.write_text(json.dumps({"provider": {"lmstudio": {
            "npm": "@ai-sdk/openai-compatible", "name": "dead",
            "options": {"baseURL": "http://127.0.0.1:9"}}}}))
    out = rig.run(lms=lms, opencode=opencode)
    assert out.returncode == 1
    assert "DEMO_ENSURE_MODELS" not in out.stderr + out.stdout


def test_unreachable_message_names_the_fix(rig, tmp_path):
    # m4
    dead = tmp_path / "dead.json"
    dead.write_text(json.dumps({"provider": {"lmstudio": {
        "npm": "@ai-sdk/openai-compatible", "name": "dead",
        "options": {"baseURL": "http://127.0.0.1:9"}}}}))
    out = rig.run(opencode=dead)
    assert "FALKORCHAT_OPENCODE_CONFIG=$HOME/.config/opencode/opencode.local.json" in out.stderr


def test_demo_model_still_loading_stops_without_load(rig):
    # m5: `loading` is not ready, and not "context None".
    models = _catalog(chat=None, embed=None)
    models[0] = {"id": CHAT, "type": "vlm", "state": "loading"}
    rig.set_models(models)
    out = rig.run()
    assert out.returncode == 1
    assert "still loading" in out.stderr and "context None" not in out.stderr
    assert rig.calls() == []


def test_malformed_model_entries_are_a_refusal_not_a_traceback(rig):
    # n3
    rig.state.write_text(json.dumps({"data": ["oops", {"state": "loaded"}]}))
    out = rig.run()
    assert out.returncode == 1
    assert "malformed" in out.stderr and "Traceback" not in out.stderr
    assert rig.calls() == []


def _stub_lms_leaving_loading(rig, reads: int) -> None:
    """`lms load` returns while the API still reports the model `loading` for `reads`
    more reads (P2-m1: whether LM Studio does this is unknown)."""
    _stub_lms_with_side_effect(rig, (
        "for m in data['data']:\n"
        "    if m['id'] == mid: m.update(state='loading', loaded_context_length=8192,"
        f" _loading_reads={reads})"
    ))


def test_model_still_loading_right_after_load_is_waited_for(rig):
    # P2-m1: a post-load `loading` settles within the bound -> ready, not a refusal.
    rig.set_models(_catalog(chat=None, embed=2048))
    _stub_lms_leaving_loading(rig, reads=3)
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert "waiting for" in out.stdout  # the settle wait actually ran, after the load
    assert "demo models ready" in out.stdout
    assert len(_loads(rig.calls())) == 1 and _unloads(rig.calls()) == []


def test_settle_wait_runs_after_a_load_only_never_before(rig):
    # K-083 (review pass 3): a demo model already `loading` BEFORE any load is someone
    # else's load in flight -> an immediate refusal, with no wait, even though it would
    # settle within the (generous) bound. A wait moved before the load would instead
    # wait it out and succeed.
    models = _catalog(chat=None, embed=2048)
    models[0] = {"id": CHAT, "type": "llm", "state": "loading",
                 "loaded_context_length": 8192, "_loading_reads": 2}
    rig.set_models(models)
    out = rig.run(settle_s=30)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "still loading" in out.stderr
    assert "waiting for" not in out.stdout
    assert rig.calls() == []


def test_model_never_leaving_loading_after_load_stops_when_the_wait_runs_out(rig):
    # P2-m1: bounded — a load that never settles is refused with the still-loading message.
    rig.set_models(_catalog(chat=None, embed=2048))
    _stub_lms_leaving_loading(rig, reads=10**9)
    out = rig.run(settle_s=0.5)
    assert out.returncode == 1, out.stdout
    assert "still loading" in out.stderr
    assert "demo models ready" not in out.stdout
    assert len(_loads(rig.calls())) == 1 and _unloads(rig.calls()) == []


@pytest.mark.parametrize(("status", "rc"), [(404, 3), (405, 3), (500, 1)])
def test_only_404_and_405_mean_not_lm_studio(rig, status, rc):
    # P2-m2: a sick LM Studio (5xx) must not be offered the bypass.
    rig.server.status = status
    out = rig.run()
    assert out.returncode == rc, out.stderr
    assert f"HTTP {status}" in out.stderr
    assert ("DEMO_ENSURE_MODELS=0" in out.stderr) is (rc == 3)
    assert rig.calls() == []


def test_200_with_a_non_json_body_is_not_lm_studio(rig):
    # P2-n1: an HTML catch-all host is reachable, just not LM Studio.
    rig.server.raw_body = b"<!doctype html><html><body>shop</body></html>"
    out = rig.run()
    assert out.returncode == 3, out.stderr
    assert "unreachable" not in out.stderr
    assert "DEMO_ENSURE_MODELS=0" in out.stderr
    assert rig.calls() == []


@pytest.mark.parametrize(("knob", "value"), [
    ("settle_s", "abc"), ("settle_s", "nan"), ("settle_s", "-1"), ("settle_s", "inf"),
    ("settle_interval_s", "abc"), ("settle_interval_s", "0"),
])
def test_bad_settle_knob_is_a_refusal_not_a_traceback(rig, knob, value):
    # K-083: was a ValueError at import. Refuses (exit 1) naming the knob, before any
    # read or load.
    rig.set_models(_catalog(chat=None, embed=None))
    out = rig.run(**{knob: value})
    assert out.returncode == 1, out.stdout + out.stderr
    assert "Traceback" not in out.stderr
    env_name = {"settle_s": "ENSURE_DEMO_MODELS_SETTLE_S",
                "settle_interval_s": "ENSURE_DEMO_MODELS_SETTLE_INTERVAL_S"}[knob]
    assert f"{env_name}={value!r}" in out.stderr
    assert rig.calls() == []


def test_valid_zero_settle_is_accepted(rig):
    rig.set_models(_catalog(chat=8192, embed=2048))
    out = rig.run(settle_s="0")
    assert out.returncode == 0, out.stderr


def test_answer_cut_off_mid_body_is_a_refusal_not_a_traceback(rig):
    # K-083: http.client.IncompleteRead (an HTTPException, not an OSError) used to escape.
    # Exit 1, not 3: a broken answer is no proof the server isn't LM Studio.
    rig.set_models(_catalog(chat=8192, embed=2048))
    rig.server.truncate = True
    out = rig.run()
    assert out.returncode == 1, out.stdout + out.stderr
    assert "Traceback" not in out.stderr
    assert "cut off or malformed" in out.stderr and "IncompleteRead" in out.stderr
    assert "DEMO_ENSURE_MODELS" not in out.stderr
    assert rig.calls() == []


def test_connection_closed_before_any_answer_is_unreachable_with_the_config_hint(rig):
    # Review m1: RemoteDisconnected is an OSError AND an HTTPException; nothing was
    # answered, so it is the wrong-URL symptom and must keep the config-fix hint.
    rig.server.mode = "close"
    out = rig.run()
    assert out.returncode == 1, out.stdout + out.stderr
    assert "unreachable" in out.stderr
    assert "opencode.local.json" in out.stderr
    assert "cut off" not in out.stderr and "Traceback" not in out.stderr
    assert "DEMO_ENSURE_MODELS" not in out.stderr
    assert rig.calls() == []


def test_garbage_status_line_is_cut_off_or_malformed(rig):
    # Review m1: BadStatusLine (not a RemoteDisconnected) stays on the "cut off" path.
    rig.server.mode = "garbage"
    out = rig.run()
    assert out.returncode == 1, out.stdout + out.stderr
    assert "cut off or malformed" in out.stderr and "BadStatusLine" in out.stderr
    assert "unreachable" not in out.stderr and "Traceback" not in out.stderr
    assert "DEMO_ENSURE_MODELS" not in out.stderr
    assert rig.calls() == []


# ── K-099: the embedder's real context, checked advisorily via `lms ps --json` ─────────

@pytest.mark.parametrize(("ps_ctx", "effect"), [
    (8192, "only costs VRAM"), (2048, "first 2048 tokens"),
])
def test_resident_embedder_at_another_context_warns_and_is_never_reloaded(rig, ps_ctx, effect):
    # The contract "never unloads or reloads" holds for the embedder too: a mismatch
    # (e.g. one loaded at 8192 before K-099) is a WARNING with the reload command, exit 0.
    rig.set_models(_catalog(chat=8192, embed=2048, embed_ps=ps_ctx))
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"loaded with context {ps_ctx}, not 4096" in out.stderr and effect in out.stderr
    assert f"lms unload {EMBED} && lms load {EMBED} -c 4096" in out.stderr  # told, not done
    assert _mutating(rig.calls()) == []
    assert rig.calls() == [["ps", "--json"]]


def test_resident_embedder_at_4096_passes_quietly(rig):
    rig.set_models(_catalog(chat=8192, embed=2048, embed_ps=4096))
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert "embedder context 4096" in out.stdout and "WARNING" not in out.stderr


def test_embedder_context_unchecked_without_cli_is_a_note_not_a_refusal(rig):
    rig.set_models(_catalog(chat=8192, embed=2048, embed_ps=8192))
    out = rig.run(lms=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "embedder context not checked (no lms CLI)" in out.stdout


def test_unreadable_lms_ps_is_a_note_not_a_refusal(rig):
    rig.set_models(_catalog(chat=8192, embed=2048))
    rig.lms.write_text(f"#!{sys.executable}\nprint('not json')\n")
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert "`lms ps --json` unreadable" in out.stdout and "Traceback" not in out.stderr


def test_post_load_path_warns_on_a_resident_embedder_at_8192(rig):
    # The likeliest real case after K-099: an embedder loaded at 8192 before the change is
    # still resident and ministral is missing -> load only ministral, then warn.
    rig.set_models(_catalog(chat=None, embed=2048, embed_ps=8192))
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert [c[1] for c in _loads(rig.calls())] == [CHAT] and _unloads(rig.calls()) == []
    assert "loaded with context 8192, not 4096" in out.stderr


def test_manual_instructions_name_each_models_own_context(rig):
    rig.set_models(_catalog(chat=None, embed=None))
    out = rig.run(lms=False)
    assert out.returncode == 1
    assert f"lms load {CHAT} -c 8192 --gpu max" in out.stderr
    assert f"lms load {EMBED} -c 4096 --gpu max" in out.stderr
    assert f"{CHAT} 8192" in out.stderr and f"{EMBED} 4096" in out.stderr


def test_bad_lms_cli_on_the_noop_path_is_a_note_not_a_refusal(rig, tmp_path):
    # A bad LMS_CLI only blocks loads; a ready demo must stay exit 0.
    rig.set_models(_catalog(chat=8192, embed=2048))
    rig.lms = tmp_path / "nonexistent-lms"
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert "embedder context not checked" in out.stdout and "is not an executable" in out.stdout


def _stub_lms_ps(rig, instances: list[dict]) -> None:
    rig.lms.write_text(f"#!{sys.executable}\nimport json\nprint(json.dumps({instances!r}))\n")


@pytest.mark.parametrize("instances", [
    [],
    [{"identifier": EMBED, "contextLength": 2048}, {"identifier": EMBED, "contextLength": 2048}],
])
def test_not_exactly_one_embedder_instance_is_a_note_not_a_verdict(rig, instances):
    rig.set_models(_catalog(chat=8192, embed=2048))
    _stub_lms_ps(rig, instances)
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"lists {len(instances)} instance(s)" in out.stdout and "WARNING" not in out.stderr


@pytest.mark.parametrize("ctx", [None, "4096", True])
def test_non_integer_embedder_context_is_a_note_not_a_verdict(rig, ctx):
    rig.set_models(_catalog(chat=8192, embed=2048))
    _stub_lms_ps(rig, [{"identifier": EMBED, "contextLength": ctx}])
    out = rig.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert "embedder context not checked (`lms ps --json` contextLength" in out.stdout
    assert "WARNING" not in out.stderr and "None tokens" not in out.stderr
