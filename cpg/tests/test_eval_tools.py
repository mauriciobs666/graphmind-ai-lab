"""Tests for cpg/eval/tools/{export_kit,strip_gold,audit}.py (S48, P0-41).

Run from cpg/:  .venv/bin/python -m pytest -q tests/test_eval_tools.py

Fixtures: ``tests/fixtures/headless/capture.*`` are S55's REAL capture. Everything else under that
directory (``derived.*``, ``twoattempt*``) is derived from it by the builders in this file;
``python tests/test_eval_tools.py --write-fixtures`` regenerates them and
``test_derived_fixtures_match_builders`` pins the committed bytes to the builders.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "headless"
TOOLS = CPG / "eval" / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit")
export_kit = _load("export_kit")
strip_gold = _load("strip_gold")

PY = sys.executable

# =============================================================================================
# fixture builders (derived from S55's real capture)
# =============================================================================================

CAP_STREAM = FIX / "capture.1.stream.jsonl"
CAP_TRANSCRIPT = FIX / "capture.transcript.jsonl"
CAP_INDEX = FIX / "capture.index.jsonl"


def _read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _dump(recs):
    return "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in recs)


_INIT = _read(CAP_STREAM)[0]
CAP_ROOT = _INIT["cwd"]
ROLES_DIR = CAP_ROOT.rsplit("/", 1)[0]
SESSION = _INIT["session_id"]
REPO_AGENTS = "/home/mauricio/prg/graphmind-ai-lab/AGENTS.md"
KEEP_TYPES = {"agent-setting", "queue-operation", "user", "assistant", "last-prompt", "tools-used", "cost-state"}


def reduced_transcript():
    """The real transcript minus the bulky request blobs/attachments; every assistant/user line kept."""
    return [r for r in _read(CAP_TRANSCRIPT) if r["type"] in KEEP_TYPES]


def _tool_pair(stream_asst, stream_user, t_asst, t_user, tid, name, inp, refused):
    """Clones of a real tool_use/tool_result pair re-pointed at (tid, name, inp)."""
    sa, su, ta, tu = (copy.deepcopy(x) for x in (stream_asst, stream_user, t_asst, t_user))
    for rec in (sa, ta):
        blk = rec["message"]["content"][0]
        blk.update(id=tid, name=name, input=inp)
        rec["message"]["id"] = "msg_" + tid.replace("toolu_", "")        # one message per injected call
        rec.pop("wire_tool_inputs", None)
        rec.pop("wireToolInputs", None)
        rec["uuid"] = f"a-{tid}"
    for rec in (su, tu):
        blk = rec["message"]["content"][0]
        blk["tool_use_id"] = tid
        blk["content"] = "guard-paths: denied, outside the session root" if refused else "ok"
        if refused:
            blk["is_error"] = True
        rec["uuid"] = f"u-{tid}"
    su["tool_result_meta"] = [{"id": tid, "permission_decision":
                               {"decision": "deny" if refused else "accept", "source": "hook"}}]
    su.pop("tool_use_result", None)
    tu["toolUseResult"] = {"type": "text"}
    tu["permissionDecision"] = {"decision": "deny" if refused else "accept", "source": "hook"}
    tu["sourceToolAssistantUUID"] = ta["uuid"]
    return (sa, su), (ta, tu)


def with_call(name, inp, tid, refused=False):
    """(stream records, transcript records) = the capture with one extra call+result before the
    final assistant text, in BOTH logs."""
    st = _read(CAP_STREAM)
    tr = reduced_transcript()
    s_idx = [i for i, r in enumerate(st) if r["type"] == "assistant"][-1]
    t_idx = [i for i, r in enumerate(tr) if r["type"] == "assistant"][-1]
    s_pairs = [r for r in st if r["type"] in ("assistant", "user")]
    t_pairs = [r for r in tr if r["type"] in ("assistant", "user") and "toolUseResult" in r or
               (r["type"] == "assistant" and r["message"]["content"][0]["type"] == "tool_use")]
    (sa, su), (ta, tu) = _tool_pair(s_pairs[0], s_pairs[1], t_pairs[0], t_pairs[1], tid, name, inp, refused)
    st[s_idx:s_idx] = [sa, su]
    tr[t_idx:t_idx] = [ta, tu]
    return st, tr


def p(rel):                       # a path inside the capture's role root
    return f"{CAP_ROOT}/{rel}"


# kind -> (tool name, input, refused)
VIOLATIONS = {
    "inroot-read": ("Read", {"file_path": p("kit-src/README.md")}, False),
    "inroot-relative-read": ("Read", {"file_path": "kit-src/../kit-src/README.md"}, False),
    "abs-repo-read": ("Read", {"file_path": REPO_AGENTS}, False),
    "relative-dotdot-read": ("Read", {"file_path": "../../../../AGENTS.md"}, False),
    "dotdot-grep": ("Grep", {"pattern": "x", "path": "../../.."}, False),
    "sibling-root-read": ("Read", {"file_path": f"{ROLES_DIR}/KA-flow/out/KA.jsonl"}, False),
    "mcp-call": ("mcp__cypher__query", {"graph": "g", "cypher": "RETURN 1"}, False),
    "disallowed-bash": ("Bash", {"command": "ls"}, False),
    "write-outside-out": ("Write", {"file_path": p("BRIEF.md"), "content": "x"}, False),
    "refused-abs-read": ("Read", {"file_path": REPO_AGENTS}, True),
}


def build_violation(kind):
    name, inp, refused = VIOLATIONS[kind]
    return with_call(name, inp, "toolu_PLANTED" + re.sub("[^A-Za-z0-9]", "", kind) + "0001", refused)


def build_init_extra_tool():
    st = _read(CAP_STREAM)
    st[0]["tools"] = st[0]["tools"] + ["Bash"]
    return st, reduced_transcript()


# --- the two-attempt session: start, kill with a truncated last line, resume ------------------

def _index_two(second_exit=0):
    base = _read(CAP_INDEX)
    start1, end1 = base[0], base[1]
    end1 = dict(end1, exit=124, end="2026-11-05T16:14:21Z")
    start2 = dict(start1, mode="resume", n=2, start="2026-11-05T16:20:00Z",
                  stream=f"{SESSION}.2.stream.jsonl")
    end2 = dict(end1, n=2, exit=second_exit, end="2026-11-05T16:20:09Z")
    return [start1, end1, start2, end2]


def _t_pairs(tr):
    return [r for r in tr if (r["type"] == "assistant" and r["message"]["content"][0]["type"] == "tool_use")
            or (r["type"] == "user" and "toolUseResult" in r)]


def build_twoattempt(variant="clean"):
    """variant: clean (cut mid tool_use input, id present) | cutid (cut before the id) |
    planted (attempt 2 makes an out-of-root read) | planted-truncated (the call in the cut-off line
    is itself an out-of-root read: it must be judged from the transcript alone)."""
    st = _read(CAP_STREAM)
    tr = reduced_transcript()
    init, a1, rl, u1, a2 = st[0], st[1], st[2], st[3], st[4]
    if variant == "planted-truncated":
        for rec in (a2, ):
            rec["message"]["content"][0]["input"] = {"file_path": REPO_AGENTS, "content": "x"}
            rec["message"]["content"][0]["name"] = "Read"
        for rec in tr:
            if rec["type"] == "assistant" and rec["message"]["content"][0]["type"] == "tool_use" \
                    and rec["message"]["content"][0]["id"] == a2["message"]["content"][0]["id"]:
                rec["message"]["content"][0]["input"] = {"file_path": REPO_AGENTS}
                rec["message"]["content"][0]["name"] = "Read"
    line = json.dumps(a2, separators=(",", ":"), ensure_ascii=False)
    if variant == "cutid":
        cut = line.index('"content"') + 5           # after the message id, before the tool_use block
    elif variant in ("cutearly", "midmissing-early"):
        cut = line.index('"model"') + 10            # before the message id: nothing recoverable
    else:
        cut = line.index('"input"') + 12            # inside the input object, after the id
    s1 = [init, a1, rl, u1]
    t_mid = []
    if variant == "x-before-tail":
        # no stream-matched message follows X: only the tail's own message may be excused
        _, (xa, xu) = _tool_pair(a1, u1, *_t_pairs(tr)[:2], "toolu_MIDMISSING0001", "mcp__cypher__query",
                                 {"graph": "g", "cypher": "RETURN 1"}, False)
        t_mid = [xa, xu]
    if variant.startswith("midmissing"):
        # attempt 1 also has a stream-matched Read (M) AFTER a call X that only the transcript holds
        (_, _), (xa, xu) = _tool_pair(a1, u1, *_t_pairs(tr)[:2], "toolu_MIDMISSING0001", "mcp__cypher__query",
                                      {"graph": "g", "cypher": "RETURN 1"}, False)
        (ma, mu), (tma, tmu) = _tool_pair(a1, u1, *_t_pairs(tr)[:2], "toolu_MIDMATCHED0001", "Read",
                                          {"file_path": p("kit-src/README.md")}, False)
        s1 += [ma, mu]
        t_mid = [xa, xu, tma, tmu]
    tail = line[:cut]
    if variant == "midmissing-sameid":
        # the cut-off line carries X's OWN message id, yet a stream-matched message (M) follows X
        tail = tail.replace(a2["message"]["id"], "msg_MIDMISSING0001")
    attempt1 = _dump(s1) + tail         # no newline: timeout cut the line
    # attempt 2: a fresh init (same flags), one legitimate read, one result
    init2 = copy.deepcopy(init)
    init2["uuid"] = "init-attempt-2"
    extra = [{"inp": {"file_path": p("kit-src/README.md")}, "name": "Read", "tid": "toolu_ATTEMPT2LEGIT0001"}]
    if variant == "planted":
        extra.append({"inp": {"file_path": REPO_AGENTS}, "name": "Read", "tid": "toolu_ATTEMPT2PLANT0001"})
    s2 = [init2]
    t_extra = []
    t_pairs = _t_pairs(tr)
    for e in extra:
        (sa, su), (ta, tu) = _tool_pair(a1, u1, t_pairs[0], t_pairs[1], e["tid"], e["name"], e["inp"], False)
        s2 += [sa, su]
        t_extra += [ta, tu]
    s2.append(_read(CAP_STREAM)[-1])                                  # the result event
    t_idx = [i for i, r in enumerate(tr) if r["type"] == "assistant"][-1]
    tr[t_idx:t_idx] = t_extra
    if t_mid:                                                    # between a1's result and the Write (a2)
        w = next(i for i, r in enumerate(tr) if r["type"] == "assistant"
                 and r["message"]["content"][0].get("id") == a2["message"]["content"][0]["id"])
        tr[w:w] = t_mid
    return {"index": _index_two(), "1.stream": attempt1, "2.stream": _dump(s2), "transcript": tr}


TWO_VARIANTS = ("clean", "cutid", "cutearly", "planted", "planted-truncated", "midmissing", "midmissing-early",
                "midmissing-sameid", "x-before-tail")


def all_derived():
    """name -> bytes of every committed derived fixture."""
    out = {}
    for k in VIOLATIONS:
        st, tr = build_violation(k)
        out[f"derived.{k}.stream.jsonl"] = _dump(st)
        out[f"derived.{k}.transcript.jsonl"] = _dump(tr)
    st, tr = build_init_extra_tool()
    out["derived.init-extra-tool.stream.jsonl"] = _dump(st)
    out["derived.init-extra-tool.transcript.jsonl"] = _dump(tr)
    for v in TWO_VARIANTS:
        d = build_twoattempt(v)
        out[f"twoattempt-{v}.index.jsonl"] = _dump(d["index"])
        out[f"twoattempt-{v}.1.stream.jsonl"] = d["1.stream"]
        out[f"twoattempt-{v}.2.stream.jsonl"] = d["2.stream"]
        out[f"twoattempt-{v}.transcript.jsonl"] = _dump(d["transcript"])
    return out


# =============================================================================================
# materialising a session into a throwaway eval dir
# =============================================================================================

class Env:
    def __init__(self, tmp: Path):
        self.eval_dir = tmp / "ccg-eval" / "2fbff4d1102c"
        self.projects = tmp / "projects"
        self.roles = self.eval_dir / "roles"
        self.root = self.roles / "capture-dummy"
        for d in (self.root / "out", self.root / "kit-src", self.eval_dir / "runs", self.projects / "-proj"):
            d.mkdir(parents=True, exist_ok=True)
        (self.root / "kit-src" / "README.md").write_text("# dummy kit\n")

    def sub(self, text: str) -> str:
        return text.replace(ROLES_DIR, str(self.roles))

    def put(self, name: str, text: str, where: str):
        dest = {"runs": self.eval_dir / "runs", "projects": self.projects / "-proj"}[where] / name
        dest.write_text(self.sub(text))

    def single(self, stream_text, transcript_text, index_text=None):
        self.put(f"{SESSION}.1.stream.jsonl", stream_text, "runs")
        if transcript_text is not None:
            self.put(f"{SESSION}.jsonl", transcript_text, "projects")
        self.put("index.jsonl", index_text or CAP_INDEX.read_text(), "runs")
        return self

    def two(self, variant, index_text=None):
        self.put(f"{SESSION}.1.stream.jsonl", (FIX / f"twoattempt-{variant}.1.stream.jsonl").read_text(), "runs")
        self.put(f"{SESSION}.2.stream.jsonl", (FIX / f"twoattempt-{variant}.2.stream.jsonl").read_text(), "runs")
        self.put(f"{SESSION}.jsonl", (FIX / f"twoattempt-{variant}.transcript.jsonl").read_text(), "projects")
        self.put("index.jsonl", index_text or (FIX / f"twoattempt-{variant}.index.jsonl").read_text(), "runs")
        return self

    def run(self, **kw):
        return audit.audit_run(self.eval_dir, self.projects, **kw)

    def session(self, **kw):
        rep, details = self.run(**kw)
        assert len(rep["sessions"]) == 1
        return rep["sessions"][0], rep, details


def derived(env: Env, kind: str):
    return env.single((FIX / f"derived.{kind}.stream.jsonl").read_text(),
                      (FIX / f"derived.{kind}.transcript.jsonl").read_text())


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# =============================================================================================
# fixtures are what the builders say
# =============================================================================================

def test_derived_fixtures_match_builders():
    for name, text in all_derived().items():
        assert (FIX / name).read_text() == text, name


def test_capture_fixture_is_the_real_shape():
    # the audit is built against these facts; if S55's capture changes, this must be re-read
    st = _read(CAP_STREAM)
    assert st[0]["subtype"] == "init" and st[0]["tools"] == ["Read", "Grep", "Glob", "Write", "Edit"]
    assert st[0]["mcp_servers"] == [] and st[0]["skills"] == [] and st[0]["slash_commands"] == []
    assert st[-1]["type"] == "result"


# =============================================================================================
# audit.py
# =============================================================================================

def test_audit_real_capture_is_clean(env):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    s, rep, _ = env.session()
    assert s["verdict"] == "clean" and s["void"] == {} and s["refused"] == {}
    assert s["attempts"] == 1 and s["toolCalls"] == 2 and rep["verdict"] == "clean"


@pytest.mark.parametrize("kind", ["inroot-read", "inroot-relative-read"])
def test_audit_legit_in_root_read_not_flagged(env, kind):
    derived(env, kind)
    s, _, d = env.session()
    assert s["verdict"] == "clean" and s["void"] == {} and s["toolCalls"] == 3 and d == []


@pytest.mark.parametrize("kind,vtype", [
    ("abs-repo-read", "path-outside-root"),
    ("relative-dotdot-read", "path-outside-root"),
    ("dotdot-grep", "path-outside-root"),
    ("sibling-root-read", "path-outside-root"),
    ("mcp-call", "mcp-call"),
    ("disallowed-bash", "tool-not-allowed"),
    ("write-outside-out", "write-outside-out"),
])
def test_audit_flags_violation(env, kind, vtype):
    derived(env, kind)
    s, rep, d = env.session()
    assert s["verdict"] == "void" and s["void"] == {vtype: 1} and s["refused"] == {}
    assert rep["verdict"] == "void" and rep["totals"]["voidViolations"] == {vtype: 1}
    assert [x["type"] for x in d] == [vtype] and d[0]["refused"] is False


def test_audit_refused_attempt_counted_not_voiding(env):
    derived(env, "refused-abs-read")
    s, _, d = env.session()
    assert s["verdict"] == "clean" and s["void"] == {} and s["refused"] == {"path-outside-root": 1}
    assert d[0]["refused"] is True


def _is_result_of(r, tid):
    c = r["message"]["content"] if r["type"] == "user" else None
    return isinstance(c, list) and c[0].get("tool_use_id") == tid


def _retag(stream_recs, transcript_recs, tid, *, decision, is_error, content):
    """Change how the call `tid`'s result reads, in both logs."""
    for r in stream_recs:
        if _is_result_of(r, tid):
            blk = r["message"]["content"][0]
            blk["content"] = content
            blk.pop("is_error", None)
            if is_error:
                blk["is_error"] = True
            r["tool_result_meta"] = [{"id": tid, "permission_decision": {"decision": decision}}]
    for r in transcript_recs:
        if _is_result_of(r, tid):
            blk = r["message"]["content"][0]
            blk["content"] = content
            blk.pop("is_error", None)
            if is_error:
                blk["is_error"] = True
            r["permissionDecision"] = {"decision": decision}


@pytest.mark.parametrize("decision,is_error,content,refused", [
    ("deny", False, "ok", True),                                        # a deny decision alone is a refusal
    ("accept", True, "guard-paths: denied, outside the session root", True),   # a denial-shaped error alone
    ("accept", True, "File does not exist.", False),                    # an ordinary tool error is NOT a refusal
    ("accept", False, "ok", False),
])
def test_audit_refusal_evidence(env, decision, is_error, content, refused):
    st, tr = build_violation("abs-repo-read")
    tid = "toolu_PLANTEDabsrepo" + "read0001"
    _retag(st, tr, tid, decision=decision, is_error=is_error, content=content)
    env.single(_dump(st), _dump(tr))
    s, _, _ = env.session()
    if refused:
        assert s["refused"] == {"path-outside-root": 1} and s["void"] == {} and s["verdict"] == "clean"
    else:
        assert s["refused"] == {} and s["void"] == {"path-outside-root": 1}


def test_audit_other_sessions_transcript_is_not_this_sessions(env):
    env.single(CAP_STREAM.read_text(), None)
    (env.projects / "-proj" / "11111111-2222-3333-4444-555555555555.jsonl").write_text(env.sub(CAP_TRANSCRIPT.read_text()))
    assert env.session()[0]["void"] == {"transcript-missing": 1}


def test_audit_every_tool_use_block_of_a_message_is_checked(env):
    st, tr = build_violation("inroot-read")                     # has a legit planted call (own message)
    st2, tr2 = build_violation("abs-repo-read")
    extra_id = "toolu_PLANTEDabsrepo" + "read0001"
    for recs, recs2 in ((st, st2), (tr, tr2)):                  # move the violating block into the SAME message
        blk = next(r for r in recs2 if r["type"] == "assistant" and r["message"]["content"][0].get("id") == extra_id)
        res = next(r for r in recs2 if _is_result_of(r, extra_id))
        host = next(r for r in recs if r["type"] == "assistant" and r["message"]["content"][0].get("id") == "toolu_PLANTEDinrootread0001")
        host["message"]["content"].append(blk["message"]["content"][0])
        i = max(j for j, r in enumerate(recs) if _is_result_of(r, "toolu_PLANTEDinrootread0001"))
        recs.insert(i + 1, res)
    env.single(_dump(st), _dump(tr))
    s, _, d = env.session()
    assert s["void"] == {"path-outside-root": 1} and d[0]["id"] == extra_id


def test_audit_init_tool_set_unequal_to_tools(env):
    derived(env, "init-extra-tool")
    s, _, _ = env.session()
    assert s["verdict"] == "void" and s["void"] == {"init-tools": 1}


def test_audit_tools_argument_is_what_init_is_compared_to(env):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    s, _, _ = env.session(tools="Read,Grep,Glob,Edit")         # launcher said 4 tools (no Write), init shows 5
    assert s["void"] == {"init-tools": 1, "tool-not-allowed": 1}   # ...and the capture's Write is now disallowed


def test_audit_init_other_fields(env):
    base = _read(CAP_STREAM)
    for field, val, vt in [("mcp_servers", [{"name": "cypher"}], "init-mcp"),
                           ("skills", ["x"], "init-skills"), ("slash_commands", ["x"], "init-slash-commands"),
                           ("plugins", [{"name": "p", "path": "/home/x", "source": "s"}], "init-plugins"),
                           ("claude_code_version", "2.1.314", "init-version"),
                           ("model", "claude-opus-5-5", "init-model"),
                           ("cwd", "/home/mauricio", "init-cwd"),
                           ("session_id", "other", "init-session")]:
        st = copy.deepcopy(base)
        st[0][field] = val
        e = Env(env.eval_dir.parent.parent / field)
        e.single(_dump(st), CAP_TRANSCRIPT.read_text())
        s, _, _ = e.session()
        assert s["void"] == {vt: 1}, (field, s["void"])


def test_audit_builtin_plugins_accepted_agents_checked_only_when_given(env):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    assert env.session()[0]["verdict"] == "clean"
    s, _, _ = env.session(accepted_agents=["ccg-annotator"])
    assert s["void"] == {"init-agents": 1}
    init_agents = _INIT["agents"]
    assert env.session(accepted_agents=init_agents)[0]["verdict"] == "clean"


def test_audit_stream_call_missing_from_transcript(env):
    st, tr = build_violation("inroot-read")
    tr = [r for r in tr if "PLANTEDinrootread0001" not in json.dumps(r)]
    env.single(_dump(st), _dump(tr))
    s, _, _ = env.session()
    assert s["void"] == {"stream-transcript-mismatch": 1}


def test_audit_transcript_call_missing_from_stream(env):
    st, tr = build_violation("inroot-read")
    st = [r for r in st if "PLANTEDinrootread0001" not in json.dumps(r)]
    env.single(_dump(st), _dump(tr))
    s, _, _ = env.session()
    assert s["void"] == {"stream-transcript-mismatch": 1}


def test_audit_name_mismatch_between_logs(env):
    st, tr = build_violation("inroot-read")
    for r in tr:
        if r["type"] == "assistant" and r["message"]["content"][0].get("id") == "toolu_PLANTEDinrootread0001":
            r["message"]["content"][0]["name"] = "Grep"
    env.single(_dump(st), _dump(tr))
    assert env.session()[0]["void"] == {"stream-transcript-mismatch": 1}


def test_audit_missing_transcript(env):
    env.single(CAP_STREAM.read_text(), None)
    s, _, _ = env.session()
    assert s["verdict"] == "void" and s["void"] == {"transcript-missing": 1}


def test_audit_missing_attempt_log(env):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    (env.eval_dir / "runs" / f"{SESSION}.1.stream.jsonl").unlink()
    s, _, _ = env.session()
    assert s["void"].get("attempt-log-missing") == 1 and s["verdict"] == "void"


def test_audit_transcript_found_by_session_id_under_any_dir(env):
    env.single(CAP_STREAM.read_text(), None)
    d = env.projects / "-some-other-dir-name"
    d.mkdir()
    (d / f"{SESSION}.jsonl").write_text(env.sub(CAP_TRANSCRIPT.read_text()))
    assert env.session()[0]["verdict"] == "clean"


def test_audit_stream_corrupt_middle_line(env):
    lines = CAP_STREAM.read_text().splitlines()
    lines.insert(3, "{not json")
    env.single("\n".join(lines) + "\n", CAP_TRANSCRIPT.read_text())
    assert env.session()[0]["void"] == {"stream-corrupt": 1}


def test_audit_symlink_in_root_resolved(env):
    outside = env.eval_dir.parent / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    os.symlink(outside, env.root / "kit-src" / "link")
    st, tr = with_call("Read", {"file_path": p("kit-src/link/secret.txt")}, "toolu_PLANTEDsymlink0001")
    env.single(_dump(st), _dump(tr))
    assert env.session()[0]["void"] == {"path-outside-root": 1}


def test_audit_symlinked_out_dir_resolves_outside_root(env):
    shutil.rmtree(env.root / "out")
    other = env.eval_dir.parent / "else-out"
    other.mkdir()
    os.symlink(other, env.root / "out")
    st, tr = with_call("Write", {"file_path": p("out/x.jsonl"), "content": "x"}, "toolu_PLANTEDoutlink0001")
    env.single(_dump(st), _dump(tr))
    # the planted write AND the capture's own out/done.txt write both resolve outside the root now
    assert env.session()[0]["void"] == {"path-outside-root": 2}


def test_audit_symlink_inside_out_to_elsewhere_in_root_is_not_out(env):
    os.symlink(env.root / "kit-src", env.root / "out" / "link")
    st, tr = with_call("Write", {"file_path": p("out/link/x.md"), "content": "x"}, "toolu_PLANTEDinlink0001")
    env.single(_dump(st), _dump(tr))
    assert env.session()[0]["void"] == {"write-outside-out": 1}


@pytest.mark.parametrize("name,inp,expect", [
    ("Read", {"file_path": "kit-src/x.md"}, []),
    ("Read", {"file_path": "/etc/passwd"}, ["path-outside-root"]),
    ("Read", {"file_path": "~/x"}, ["path-outside-root"]),
    ("Read", {"file_path": "out/../../x"}, ["path-outside-root"]),
    ("Read", {}, ["path-malformed"]),
    ("Read", {"file_path": ""}, ["path-malformed"]),
    ("Glob", {"pattern": "**/*.py"}, []),
    ("Glob", {"pattern": "kit-src/**/*.md", "path": "."}, []),
    ("Glob", {"pattern": "../*"}, ["path-outside-root"]),
    ("Glob", {"pattern": "**/../x"}, ["path-outside-root"]),
    ("Glob", {"pattern": "/etc/*"}, ["path-outside-root"]),
    ("Glob", {"pattern": "*.md", "path": "/etc"}, ["path-outside-root"]),
    ("Grep", {"pattern": "x"}, []),
    ("Grep", {"pattern": "x", "path": "kit-src"}, []),
    ("Grep", {"pattern": "x", "glob": "../*.py"}, ["path-outside-root"]),
    ("Grep", {"pattern": "x", "path": "/home"}, ["path-outside-root"]),
    ("Write", {"file_path": "out/KA.jsonl"}, []),
    ("Write", {"file_path": "out"}, ["write-outside-out"]),
    ("Write", {"file_path": "BRIEF.md"}, ["write-outside-out"]),
    ("Write", {"file_path": "out/../BRIEF.md"}, ["write-outside-out"]),
    ("Edit", {"file_path": "/tmp/x"}, ["path-outside-root"]),
    ("Bash", {"command": "ls"}, ["tool-not-allowed"]),
    ("WebFetch", {"url": "http://x"}, ["tool-not-allowed"]),
    ("Agent", {}, ["tool-not-allowed"]),
    ("mcp__falkor-chat-agent-team__ingest_document", {}, ["mcp-call"]),
])
def test_classify_call(tmp_path, name, inp, expect):
    root = tmp_path / "role"
    (root / "out").mkdir(parents=True)
    got = audit.classify_call(name, inp, str(root), set(audit.DEFAULT_TOOLS.split(",")))
    assert got == expect


def test_audit_role_root_comes_from_the_index_not_the_log(env):
    # the log says cwd = the tmp root; the index says another role dir. Resolution uses the index's.
    idx = _read(CAP_INDEX)
    idx[0]["stratum"] = "other"
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text(), _dump(idx))
    s, _, _ = env.session()
    assert "init-cwd" in s["void"]                                  # init cwd != the indexed role root
    assert s["void"].get("path-outside-root", 0) >= 1               # the same reads are outside THAT root


def test_audit_hash_drift_between_attempts(env):
    idx = _read(FIX / "twoattempt-clean.index.jsonl")
    idx[2]["settings_sha256"] = "0" * 64
    env.two("clean", _dump(idx))
    assert env.session()[0]["void"] == {"hash-drift": 1}
    for key in ("agents_sha256", "bin_sha256", "cli_version"):
        idx = _read(FIX / "twoattempt-clean.index.jsonl")
        idx[2][key] = "9" * 8
        e = Env(env.eval_dir.parent.parent / key)
        e.two("clean", _dump(idx))
        assert "hash-drift" in e.session()[0]["void"], key


def test_audit_settings_hash_against_current_file(env, tmp_path):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    f = tmp_path / "settings.json"
    f.write_text("{}")
    assert env.session(settings=f)[0]["void"] == {"hash-drift": 1}
    ix = _read(CAP_INDEX)
    ix[0]["settings_sha256"] = hashlib.sha256(b"{}").hexdigest()
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text(), _dump(ix))
    assert env.session(settings=f)[0]["verdict"] == "clean"


def test_audit_index_orphan_log_and_unknown_session(env):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    (env.eval_dir / "runs" / "deadbeef-0000.1.stream.jsonl").write_text("")
    rep, _ = env.run()
    assert rep["verdict"] == "void" and rep["totals"]["voidViolations"] == {"index-orphan-log": 1}
    rep, _ = env.run(sessions=["no-such-session"])
    assert rep["totals"]["voidViolations"] == {"session-not-in-index": 1}


# --- two-attempt sessions -----------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["clean", "cutid"])
def test_audit_two_attempts_no_false_violation(env, variant):
    env.two(variant)
    s, rep, d = env.session()
    assert s["attempts"] == 2 and s["attemptsTruncated"] == 1
    assert s["streamTruncatedCalls"] == 1                         # reported as stream-truncated
    assert s["void"] == {} and s["refused"] == {} and s["verdict"] == "clean" and rep["verdict"] == "clean"
    assert s["toolCalls"] == 3                                    # attempt 1 Read, the cut-off Write, attempt 2 Read


def test_audit_two_attempts_truncated_line_unparseable_is_not_corruption(env):
    env.two("clean")
    log = (env.eval_dir / "runs" / f"{SESSION}.1.stream.jsonl").read_text()
    assert not log.endswith("\n")                                 # the fixture really is cut mid-line
    with pytest.raises(ValueError):
        json.loads(log.splitlines()[-1])
    assert env.session()[0]["void"] == {}


def test_audit_two_attempts_planted_read_in_attempt_2_flagged(env):
    env.two("planted")
    s, _, d = env.session()
    assert s["verdict"] == "void" and s["void"] == {"path-outside-root": 1}
    assert [(x["attempt"], x["type"]) for x in d] == [(2, "path-outside-root")]


def test_audit_two_attempts_violation_only_in_cut_off_line_is_still_caught(env):
    env.two("planted-truncated")
    s, _, d = env.session()
    assert s["void"] == {"path-outside-root": 1} and s["streamTruncatedCalls"] == 1
    assert d[0]["tool"] == "Read" and d[0]["attempt"] is None      # judged from the transcript alone


def test_audit_two_attempts_one_log_only_is_flagged(env):
    env.two("clean")
    (env.eval_dir / "runs" / f"{SESSION}.2.stream.jsonl").unlink()
    s, _, _ = env.session()
    assert s["void"].get("attempt-log-missing") == 1
    assert s["void"].get("stream-transcript-mismatch", 0) >= 1    # attempt 2's calls have no stream record


def test_audit_unmatched_transcript_call_without_truncation_is_a_mismatch(env):
    # same two-attempt data, but attempt 1's log is complete: the Write has no stream record
    # and no truncated line to excuse it
    env.two("clean")
    log = env.eval_dir / "runs" / f"{SESSION}.1.stream.jsonl"
    log.write_text("\n".join(log.read_text().splitlines()[:-1]) + "\n")
    s, _, _ = env.session()
    assert s["void"] == {"stream-transcript-mismatch": 1} and s["attemptsTruncated"] == 0


# --- review Pass 4 fixes -----------------------------------------------------------------------

ROLE_SET = set(audit.DEFAULT_TOOLS.split(","))

# the coverage probe: (pattern form) x (what it reaches) x (Glob `pattern` | Grep `glob`)
ESCAPES = {
    "plain": {"abs": "/etc/*", "home": "~/x/*", "dotdot": "../x/*"},
    "brace": {"abs": "{/etc,kit-src}/*", "home": "{~/.ssh,x}/*", "dotdot": "{..,x}/*"},
    "nested-brace": {"abs": "{a,{/etc,b}}/*", "home": "{a,{~,b}}/x", "dotdot": "k/{a,{..,b}}/x"},
    "class": {"abs": "[/]etc/*", "home": "[~]/x/*", "dotdot": "[.][.]/x/*"},
    "range-class": {"abs": "[+-0]etc/*", "home": "[!a]/x/*", "dotdot": "[--/][-.]/x/*"},
    "question": {"abs": "?etc/*", "home": "?/x/*", "dotdot": ".?/x/*"},
    "doublestar": {"abs": "/**/x", "home": "~/**/x", "dotdot": "**/../x"},
}
BENIGN = ["kit-src/**/*.md", "{kit-src,out}/*", "[ab]*/x", "kit-src/?.md", "**/*.py", "kit-src/[a-c]?/*.{md,txt}",
          "out/*", "*", "k[", "[]x]/y"]


def _tool_input(tool, pattern):
    return {"pattern": pattern} if tool == "Glob" else {"pattern": "x", "glob": pattern}


@pytest.mark.parametrize("tool", ["Glob", "Grep"])
@pytest.mark.parametrize("form", sorted(ESCAPES))
@pytest.mark.parametrize("reach", ["abs", "home", "dotdot"])
def test_classify_pattern_escape_probe(tmp_path, tool, form, reach):
    root = tmp_path / "role"
    (root / "out").mkdir(parents=True)
    got = audit.classify_call(tool, _tool_input(tool, ESCAPES[form][reach]), str(root), ROLE_SET)
    assert got == ["path-outside-root"], (tool, form, reach)


@pytest.mark.parametrize("tool", ["Glob", "Grep"])
@pytest.mark.parametrize("pattern", BENIGN)
def test_classify_pattern_benign_control(tmp_path, tool, pattern):
    root = tmp_path / "role"
    (root / "out").mkdir(parents=True)
    assert audit.classify_call(tool, _tool_input(tool, pattern), str(root), ROLE_SET) == []


def test_classify_pattern_too_many_readings_is_an_escape(tmp_path):
    root = tmp_path / "role"
    root.mkdir()
    assert audit.classify_call("Glob", {"pattern": "?" * 6 + "/x"}, str(root), ROLE_SET) == ["path-outside-root"]


@pytest.mark.parametrize("tool", ["Glob", "Grep"])
@pytest.mark.parametrize("pattern", [r"kit-src/\.\./\.\./\.\./\.\./etc/*", r"\.\./x/*", r"kit-src/\.\./\.\./x"])
def test_classify_backslash_escaped_dots_are_read_both_ways(tmp_path, tool, pattern):
    # N-1: `\x` may mean a literal x; the pattern is judged as written AND with `\x` read as `x`
    root = tmp_path / "role"
    (root / "kit-src").mkdir(parents=True)
    assert audit.classify_call(tool, _tool_input(tool, pattern), str(root), ROLE_SET) == ["path-outside-root"]


@pytest.mark.parametrize("pattern", [r"kit-src/\*.md", r"kit-src/a\ b/*", r"kit-src/\.hidden/*"])
def test_classify_backslash_benign_control(tmp_path, pattern):
    root = tmp_path / "role"
    (root / "kit-src").mkdir(parents=True)
    assert audit.classify_call("Glob", {"pattern": pattern}, str(root), ROLE_SET) == []


def test_pattern_escapes_brace_groups_are_bounded_in_time(tmp_path):
    # N-2: 30 `{a,b}` groups are 2**30 readings; the audit must give up (escape) without expanding them
    import time
    root = tmp_path / "role"
    root.mkdir()
    t0 = time.monotonic()
    assert audit.pattern_escapes(str(root), str(root), "kit-src/" + "{a,b}" * 30) is True
    assert time.monotonic() - t0 < 1.0


def test_pattern_escapes_reading_cap_is_pinned(tmp_path):
    # MAX_PATTERN_VARIANTS = 512: nine two-reading classes (512 readings) are audited; ten (1024) escape
    root = tmp_path / "role"
    root.mkdir()
    r = str(root)
    assert audit.MAX_PATTERN_VARIANTS == 512
    assert audit.pattern_escapes(r, r, "kit-src/" + "[.a]" * 9) is False
    assert audit.pattern_escapes(r, r, "kit-src/" + "[.a]" * 10) is True


def test_pattern_escapes_brace_cap_is_pinned(tmp_path):
    # the brace alternatives share the same budget: nine groups (512 alternatives) pass, ten escape
    root = tmp_path / "role"
    root.mkdir()
    r = str(root)
    assert audit.pattern_escapes(r, r, "kit-src/" + "{a,b}" * 9) is False
    assert audit.pattern_escapes(r, r, "kit-src/" + "{a,b}" * 10) is True


def test_pattern_escapes_one_wide_brace_group_is_capped_at_exactly_512(tmp_path):
    root = tmp_path / "role"
    root.mkdir()
    r = str(root)
    wide = lambda n: "kit-src/{" + ",".join(f"f{i}" for i in range(n)) + "}"
    assert audit.pattern_escapes(r, r, wide(512)) is False
    assert audit.pattern_escapes(r, r, wide(513)) is True


def test_pattern_escapes_budget_is_shared_between_braces_and_wildcards(tmp_path):
    # 2 alternatives x 512 class readings = 1024 > 512: the budget is for the whole pattern
    root = tmp_path / "role"
    root.mkdir()
    r = str(root)
    assert audit.pattern_escapes(r, r, "kit-src/{a,b}" + "[.a]" * 9) is True


def test_classify_glob_path_is_the_base_for_the_pattern(tmp_path):
    # pins `base_dir = resolved.get("path", root)`: `link/*` under path=kit-src goes through an
    # in-root symlink that points out; against the root instead, `link` does not exist and looks clean
    root = tmp_path / "role"
    (root / "kit-src").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", root / "kit-src" / "link")
    assert audit.classify_call("Glob", {"pattern": "link/*", "path": "kit-src"}, str(root), ROLE_SET) == ["path-outside-root"]
    assert audit.classify_call("Glob", {"pattern": "link/*"}, str(root), ROLE_SET) == []     # control: not under kit-src


@pytest.mark.parametrize("kind", ["abs", "home"])
def test_audit_brace_pattern_end_to_end(env, kind):
    pat = ESCAPES["brace"][kind]
    st, tr = with_call("Glob", {"pattern": pat}, "toolu_PLANTEDbrace0001")
    env.single(_dump(st), _dump(tr))
    assert env.session()[0]["void"] == {"path-outside-root": 1}


def test_audit_cutearly_tail_unrecoverable_id_clean(env):
    env.two("cutearly")
    s, _, _ = env.session()
    assert s["verdict"] == "clean" and s["streamTruncatedCalls"] == 1 and s["void"] == {}


@pytest.mark.parametrize("variant", ["midmissing", "midmissing-early"])
def test_audit_mid_session_missing_call_beside_cut_off_tail_voids(env, variant):
    env.two(variant)
    s, _, d = env.session()
    # X (an MCP call only the transcript holds, before a stream-matched message) is NOT excused ...
    assert s["void"] == {"mcp-call": 1, "stream-transcript-mismatch": 1}
    # ... and the cut-off call itself still is
    assert s["streamTruncatedCalls"] == 1
    assert {x["id"] for x in d} == {"toolu_MIDMISSING0001"}


def test_audit_excuse_never_reaches_a_call_with_a_later_matched_message(env):
    env.two("midmissing-sameid")
    s, _, d = env.session()
    # X (before the matched message M) is not excused although the tail names X's message id; the
    # Write, whose message is not the tail's any more, has no excuse either
    ids = sorted(x["id"] for x in d if x["type"] == "stream-transcript-mismatch")
    assert s["void"].get("stream-transcript-mismatch") == 2 and "toolu_MIDMISSING0001" in ids
    assert s["streamTruncatedCalls"] == 0


def test_audit_recovered_message_id_beats_position(env):
    # X sits between the last matched message and the tail; the id says the tail is the NEXT message
    env.two("x-before-tail")
    s, _, d = env.session()
    mism = [x["id"] for x in d if x["type"] == "stream-transcript-mismatch"]
    assert mism == ["toolu_MIDMISSING0001"] and s["streamTruncatedCalls"] == 1
    assert s["void"] == {"mcp-call": 1, "stream-transcript-mismatch": 1}


def test_audit_excuse_does_not_reach_other_messages_of_the_attempt(env):
    # recoverable id: the excuse is for the cut-off line's own message only, even though the tail
    # is the last thing in the log and the missing call sits right before it
    env.two("clean")
    st, tr = with_call("Read", {"file_path": p("kit-src/README.md")}, "toolu_PLANTEDextra0001")
    log = env.eval_dir / "runs" / f"{SESSION}.1.stream.jsonl"
    tr_path = env.projects / "-proj" / f"{SESSION}.jsonl"
    recs = [json.loads(l) for l in tr_path.read_text().splitlines()]
    extra = [r for r in tr if "toolu_PLANTEDextra0001" in json.dumps(r)]
    w = next(i for i, r in enumerate(recs) if r["type"] == "assistant"
             and r["message"]["content"][0].get("name") == "Write")
    recs[w:w] = [json.loads(env.sub(json.dumps(x))) for x in extra]
    tr_path.write_text(_dump(recs))
    s, _, _ = env.session()
    assert s["void"] == {"stream-transcript-mismatch": 1} and s["streamTruncatedCalls"] == 1


def test_audit_input_differing_between_stream_and_transcript_voids(env):
    st, tr = build_violation("inroot-read")
    for r in tr:
        if r["type"] == "assistant" and r["message"]["content"][0].get("id") == "toolu_PLANTEDinrootread0001":
            r["message"]["content"][0]["input"] = {"file_path": "/etc/passwd"}
    env.single(_dump(st), _dump(tr))
    s, _, _ = env.session()
    assert s["void"] == {"path-outside-root": 1, "stream-transcript-mismatch": 1}


def test_audit_refusal_needs_a_denial_message_not_a_word(env):
    for n, (content, refused) in enumerate([
            ("File does not exist: /etc/permission-x", False),            # existence oracle: NOT a refusal
            ("denied", False),
            ("guard-paths: Read path '/etc/permission-x' resolves outside the session root", True)]):
        e = Env(env.eval_dir.parent.parent / f"r{n}")
        st, tr = with_call("Read", {"file_path": "/etc/permission-x"}, "toolu_PLANTEDperm0001")
        _retag(st, tr, "toolu_PLANTEDperm0001", decision="accept", is_error=True, content=content)
        e.single(_dump(st), _dump(tr))
        s, _, _ = e.session()
        assert (s["refused"] == {"path-outside-root": 1}) is refused, content
        assert (s["void"] == {}) is refused


def test_audit_plugins_pinned_and_unpinned(env, tmp_path):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    rep, _ = env.run()
    assert rep["verdict"] == "clean" and rep["unpinned"] == ["plugins", "agents"]
    names = [p["name"] for p in _INIT["plugins"]]
    rep, _ = env.run(accepted_plugins=names, accepted_agents=_INIT["agents"])
    assert rep["verdict"] == "clean" and rep["unpinned"] == []
    for bad in (names[:-1], names + ["extra"], ["x"] + names[1:]):
        rep, _ = env.run(accepted_plugins=bad)
        assert rep["sessions"][0]["void"] == {"init-plugins": 1}, bad
    st = _read(CAP_STREAM)
    st[0]["plugins"][0]["source"] = "evil@market"                  # path says builtin, source does not
    env.single(_dump(st), CAP_TRANSCRIPT.read_text())
    assert env.run(accepted_plugins=names)[0]["sessions"][0]["void"] == {"init-plugins": 1}
    assert env.run()[0]["sessions"][0]["void"] == {}              # unpinned: only the path is judged


def test_audit_empty_runs_is_not_clean(env):
    rep, _ = env.run()
    assert rep["verdict"] == "void" and rep["totals"]["voidViolations"] == {"no-sessions": 1}
    cmd = [PY, str(TOOLS / "audit.py"), "--eval-dir", str(env.eval_dir), "--projects-dir", str(env.projects)]
    assert subprocess.run(cmd, capture_output=True).returncode == 1


def test_audit_malformed_index_start_line_is_a_usage_error(env):
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    idx = _read(CAP_INDEX)
    del idx[0]["n"]
    (env.eval_dir / "runs" / "index.jsonl").write_text(_dump(idx))
    cmd = [PY, str(TOOLS / "audit.py"), "--eval-dir", str(env.eval_dir), "--projects-dir", str(env.projects)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 2 and "Traceback" not in r.stderr and "start line" in r.stderr


# --- output shape --------------------------------------------------------------------------------

def test_audit_report_has_counts_and_types_only(env):
    derived(env, "abs-repo-read")
    rep, details = env.run()
    text = json.dumps(rep)
    assert REPO_AGENTS not in text and "/home/" not in text and str(env.root) not in text
    assert "file_path" not in text
    assert all(set(x) == {"session", "attempt", "type", "tool", "id", "refused"} for x in details)
    assert REPO_AGENTS not in json.dumps(details)


def test_audit_cli_exit_codes_and_rerun_identical(env, tmp_path):
    derived(env, "mcp-call")
    cmd = [PY, str(TOOLS / "audit.py"), "--eval-dir", str(env.eval_dir), "--projects-dir", str(env.projects),
           "--details", str(tmp_path / "d.jsonl")]
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r1.returncode == 1 and r1.stdout == r2.stdout
    assert json.loads(r1.stdout)["verdict"] == "void"
    assert json.loads((tmp_path / "d.jsonl").read_text().splitlines()[0])["type"] == "mcp-call"
    env.single(CAP_STREAM.read_text(), CAP_TRANSCRIPT.read_text())
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    bad = subprocess.run([PY, str(TOOLS / "audit.py"), "--eval-dir", str(tmp_path / "nope")],
                         capture_output=True)
    assert bad.returncode == 2


# =============================================================================================
# export_kit.py
# =============================================================================================

def _git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)


REPO_FILES = {
    "falkor-chat/server/app.py": "def f():\n    return 1\n",
    "falkor-chat/server/pyproject.toml": "[project]\nname='x'\n",
    "cypher-mcp/server.py": ('from mcp.server.fastmcp import FastMCP\nmcp = FastMCP(name="cypher")\n\n'
                             '@mcp.tool(\n    name="query",\n)\ndef query_impl(graph, cypher):\n    return 1\n'),
    "cypher-mcp/requirements.txt": "mcp\n",
    "cpg/ccg/ids.py": "x = 1\n",
    "cpg/pyproject.toml": "[project]\n",
    "deprecated/old.py": "y = 2\n",
    "docs/BACKLOG.md": "backlog\n",
    "docs/README.md": "plain doc\n",
    "docs/plans/connected-code-graph-p0-2.md": "plan\n",
    "docs/plans/sub/connected-code-graph/notes.md": "notes\n",
    "docs/guide.md": "This is about the Connected Code Graph work.\n",     # content hit (case-insens.)
    "falkor-chat/server/hit_code.py": "# see C-1001\nz = 1\n",             # content hit, code
    "falkor-chat/server/hit_cfg.yaml": "name: ccg-thing\n",               # content hit, config
    "falkor-chat/AGENTS.md": "uses CCG_ENV here\n",                       # content hit, prompt-md
    "falkor-chat/server/tests/test_hit.py": "import ccg_ids\n",           # content hit, test
    "falkor-chat/server/near_miss.py": "accg_x = 1\nname = 'ccgfoo'\n",   # NOT hits: \b and separator
    "README.md": "hello\n",
    "web/package.json": "{}\n",
    "web/node_modules/dep/package.json": "{}\n",
    "web/tests/fixtures/package.json": "{}\n",
}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    for rel, text in REPO_FILES.items():
        f = r / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "c")
    return r, _git(r, "rev-parse", "HEAD").stdout.strip()


def _export(repo_, sha, out):
    return subprocess.run([PY, str(TOOLS / "export_kit.py"), "--repo", str(repo_), "--eval-commit", sha,
                           "--out-dir", str(out)], capture_output=True, text=True)


def _kit_files(kit):
    return sorted(q.relative_to(kit).as_posix() for q in kit.rglob("*") if q.is_file())


def test_export_excludes_paths_and_content_hits(repo, tmp_path):
    r, sha = repo
    out = tmp_path / "out"
    res = _export(r, sha, out)
    assert res.returncode == 0, res.stderr
    kit = out / "kit-src"
    assert _kit_files(kit) == sorted([
        "README.md", "cypher-mcp/requirements.txt", "cypher-mcp/server.py", "docs/README.md",
        "falkor-chat/server/app.py", "falkor-chat/server/near_miss.py", "falkor-chat/server/pyproject.toml",
        "web/node_modules/dep/package.json", "web/package.json", "web/tests/fixtures/package.json"])
    assert not (kit / ".git").exists()
    m = json.loads((out / "kit-manifest.json").read_text())
    hits = {e["path"]: e for e in m["exclusions"]["contentCheck"]["files"]}
    assert set(hits) == {"docs/guide.md", "falkor-chat/server/hit_code.py", "falkor-chat/server/hit_cfg.yaml",
                         "falkor-chat/AGENTS.md", "falkor-chat/server/tests/test_hit.py"}
    assert all(e["reason"] == "content-check" for e in hits.values())
    assert {k: e["class"] for k, e in hits.items()} == {
        "docs/guide.md": "doc", "falkor-chat/server/hit_code.py": "code", "falkor-chat/server/hit_cfg.yaml": "config",
        "falkor-chat/AGENTS.md": "prompt-md", "falkor-chat/server/tests/test_hit.py": "test"}
    assert m["excludedScope"] == {"byClass": {"code": 1, "config": 1, "doc": 1, "prompt-md": 1, "test": 1}, "count": 3}
    assert m["counts"]["pathExcluded"] == 6                  # cpg x2, deprecated, BACKLOG, 2 connected-code-graph docs
    assert m["evalCommit"] == sha


def test_export_manifest_sorted_hashed_and_complete(repo, tmp_path):
    r, sha = repo
    out = tmp_path / "out"
    assert _export(r, sha, out).returncode == 0
    m = json.loads((out / "kit-manifest.json").read_text())
    paths = [f["path"] for f in m["files"]]
    assert paths == sorted(paths) and paths == _kit_files(out / "kit-src")
    for f in m["files"]:
        assert f["sha256"] == hashlib.sha256((out / "kit-src" / f["path"]).read_bytes()).hexdigest()
    assert "docs/**/connected-code-graph*" in m["exclusions"]["paths"]


def test_export_regrep_zero_and_readonly(repo, tmp_path):
    r, sha = repo
    out = tmp_path / "out"
    assert _export(r, sha, out).returncode == 0
    kit = out / "kit-src"
    rx = re.compile(r"connected-code-graph|connected code graph|C-1001|\bccg[_ -]", re.I)
    assert [f for f in _kit_files(kit) if rx.search((kit / f).read_text())] == []
    assert not os.access(kit / "README.md", os.W_OK) and not os.access(kit, os.W_OK)
    assert not os.access(kit / "falkor-chat", os.W_OK)
    with pytest.raises(PermissionError):
        (kit / "new.txt").write_text("x")
    with pytest.raises(PermissionError):
        (kit / "README.md").write_text("x")
    for dp, dns, fns in os.walk(kit):                         # leave it deletable for tmp cleanup
        os.chmod(dp, 0o755)


def test_export_rerun_byte_identical_and_refuses_existing(repo, tmp_path):
    r, sha = repo
    assert _export(r, sha, tmp_path / "a").returncode == 0
    assert _export(r, sha, tmp_path / "b").returncode == 0
    assert (tmp_path / "a" / "kit-manifest.json").read_bytes() == (tmp_path / "b" / "kit-manifest.json").read_bytes()
    again = _export(r, sha, tmp_path / "a")
    assert again.returncode == 3 and "refusing" in again.stderr
    for t in ("a", "b"):
        for dp, dns, fns in os.walk(tmp_path / t):
            os.chmod(dp, 0o755)


def test_export_stdout_has_counts_not_content(repo, tmp_path):
    r, sha = repo
    res = _export(r, sha, tmp_path / "o")
    assert "content-check-excluded: 5" in res.stdout
    assert "Connected Code Graph" not in res.stdout and "ccg_ids" not in res.stdout
    for dp, dns, fns in os.walk(tmp_path / "o"):
        os.chmod(dp, 0o755)


def test_export_refuses_a_symlink_in_the_archive(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    (r / "a.txt").write_text("x")
    os.symlink("a.txt", r / "link")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "c")
    res = _export(r, _git(r, "rev-parse", "HEAD").stdout.strip(), tmp_path / "o")
    assert res.returncode == 3 and "link" in res.stderr
    # nothing half-extracted is left behind: no kit-src, no staging dir, no manifest
    assert sorted(q.name for q in (tmp_path / "o").iterdir()) == []


@pytest.fixture
def history(tmp_path):
    """Three commits; the middle and the last change what the first exported."""
    r = tmp_path / "hist"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    shas = []
    steps = [{"a.txt": "alpha-1\n", "b.txt": "beta-1\n", "sub/c.py": "def c():\n    return 1\n"},
             {"a.txt": "alpha-2\n", "d.txt": "delta-2\n"},
             {"b.txt": None, "sub/c.py": "def c():\n    return 3\n", "e.txt": "eps-3\n"}]
    for files in steps:
        for rel, text in files.items():
            f = r / rel
            if text is None:
                f.unlink()
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        _git(r, "add", "-A")
        _git(r, "commit", "-q", "-m", "c")
        shas.append(_git(r, "rev-parse", "HEAD").stdout.strip())
    return r, shas


def test_export_non_tip_commit_with_short_sha(history, tmp_path):
    r, shas = history
    first = shas[0]
    out = tmp_path / "o"
    res = _export(r, first[:9], out)                       # an ABBREVIATED, non-tip commit
    assert res.returncode == 0, res.stderr
    kit = out / "kit-src"
    want = _git(r, "ls-tree", "-r", "--name-only", first).stdout.split()
    assert _kit_files(kit) == sorted(want) == ["a.txt", "b.txt", "sub/c.py"]
    for rel in want:
        assert (kit / rel).read_bytes() == subprocess.run(["git", "-C", str(r), "show", f"{first}:{rel}"],
                                                          capture_output=True).stdout
    m = json.loads((out / "kit-manifest.json").read_text())
    assert m["evalCommit"] == first and len(m["evalCommit"]) == 40 and m["evalCommit"] != first[:9]
    assert [f["sha256"] for f in m["files"]] == [hashlib.sha256((kit / f["path"]).read_bytes()).hexdigest()
                                                 for f in m["files"]]
    for dp, dns, fns in os.walk(out):
        os.chmod(dp, 0o755)


def test_draw_normalizes_a_short_commit_to_40_hex(flow_kit, history, tmp_path):
    r, shas = history
    full = shas[1]
    # (a) through --repo
    a = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--repo", str(r), "--eval-commit", full[:10],
             "--out", str(tmp_path / "a.jsonl"))
    b = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--eval-commit", full, "--out", str(tmp_path / "b.jsonl"))
    assert a.returncode == 0 and b.returncode == 0 and f"eval-commit: {full}" in a.stdout
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    # (b) through the manifest next to the kit
    (flow_kit.parent / "kit-manifest.json").write_text(json.dumps({"evalCommit": full}))
    c = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--eval-commit", full[:12], "--out", str(tmp_path / "c.jsonl"))
    assert c.returncode == 0 and (tmp_path / "c.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    # (c) a commit that is not the manifest's: refused
    d = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--eval-commit", shas[0], "--out", str(tmp_path / "d.jsonl"))
    assert d.returncode == 1 and "differs" in d.stderr and not (tmp_path / "d.jsonl").exists()


def test_draw_short_commit_without_repo_or_manifest_is_a_usage_error(flow_kit, tmp_path):
    res = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--eval-commit", COMMIT[:10], "--out", str(tmp_path / "o"))
    assert res.returncode == 2 and "40-hex" in res.stderr


# --- the frame rulings (method owner, Pass 4) ---------------------------------------------------

@pytest.fixture
def frame2(tmp_path):
    kit = tmp_path / "k2" / "kit-src"
    (kit / "svc").mkdir(parents=True)
    (kit / "svc" / "pyproject.toml").write_text("")
    body = "{\n  const a = 1;\n  return a;\n}\n"
    (kit / "svc" / "d1.js").write_text("export default function (x) " + body)
    (kit / "svc" / "d2.js").write_text("export default class {\n  m() " + body + "}\n")
    (kit / "svc" / "d3.js").write_text("export default (x) => " + body)
    (kit / "svc" / "d4.js").write_text("export default function named(x) " + body)
    (kit / "svc" / "anon.js").write_text(
        "function host() {\n  cb(() => {\n    function helper() " + body.replace("\n}", "\n  }") + "  });\n"
        "  function helper() " + body.replace("\n}", "\n  }") + "}\n")
    (kit / "svc" / "dupclass.js").write_text(
        "class Dup {\n  m() " + body.replace("\n}", "\n  }") + "}\nclass Dup {\n  n() " + body.replace("\n}", "\n  }") + "}\n")
    (kit / "svc" / "dup.py").write_text(
        "class C:\n    def m(self):\n        a = 1\n        return a\n\n"
        "class C:\n    def n(self):\n        a = 1\n        return a\n")
    (kit / "svc" / "keys.js").write_text(
        "const o = {\n  ok() " + body.replace("\n}", "\n  },") + "  \"a.b\"() " + body.replace("\n}", "\n  },")
        + "  1() " + body.replace("\n}", "\n  },") + "  \"quoted\"() " + body.replace("\n}", "\n  }") + "};\n")
    (kit / "svc" / "app.py").write_text("def kept(a):\n    b = a\n    return b\n")
    for d in ("vendored", "_vendor", "third-party", "node_modules", "vendor"):
        (kit / "svc" / d).mkdir()
        (kit / "svc" / d / "lib.py").write_text("def hidden(a):\n    b = a\n    return b\n")
    (kit / "svc" / "third-party" / "x.js").write_text("export function hiddenJs(a) " + body)
    (kit / "svc" / "broken.py").write_text("def broken(:\n")
    (kit / "svc" / "broken.js").write_text("function broken( {{{ \n")
    return kit


def _q(frame2, lang="python"):
    return {c.qname for c in export_kit.flow_frame(frame2)[lang]}


def test_frame_default_exports(frame2):
    js = _q(frame2, "javascript")
    assert "svc/d1.js::default" in js and "svc/d2.js::default.m" in js and "svc/d3.js::default" in js
    assert "svc/d4.js::named" in js and "svc/d4.js::default" not in js


def test_frame_anonymous_subtree_does_not_pollute_duplicate_counting(frame2):
    js = _q(frame2, "javascript")
    assert "svc/anon.js::host.helper" in js                      # the real sibling keeps the bare name
    assert not any("helper#" in q for q in js)
    assert len([q for q in js if q.startswith("svc/anon.js::")]) == 2     # host and host.helper; the callback's is not


def test_frame_children_of_a_duplicate_are_excluded(frame2):
    js, py = _q(frame2, "javascript"), _q(frame2, "python")
    assert "svc/dupclass.js::Dup.m" in js and not any("#" in q for q in js)
    assert "svc/dupclass.js::Dup#2.n" not in js
    assert "svc/dup.py::C.m" in py and not any("#" in q for q in py)


def test_frame_non_identifier_object_keys_are_anonymous(frame2):
    js = _q(frame2, "javascript")
    assert "svc/keys.js::ok" in js
    assert not any(q.endswith(("::a.b", "::1")) for q in js)
    assert "svc/keys.js::quoted" in js                           # a quoted key that IS an identifier is a name


def test_frame_vendored_names_below_the_unit_root(frame2):
    py, js = _q(frame2), _q(frame2, "javascript")
    assert "svc/app.py::kept" in py
    assert not any("hidden" in q for q in py | js)


def test_frame_counts_files_it_could_not_parse(frame2, tmp_path):
    stats: dict = {}
    export_kit.flow_frame(frame2, stats)
    assert stats["skippedFiles"] == 1 and stats["errorTrees"] == 1
    assert "broken" not in "".join(_q(frame2))
    # the CLI prints the count (the frame is short here, so use a kit that can draw)
    kit = tmp_path / "w" / "kit-src"
    (kit / "x").mkdir(parents=True)
    (kit / "x" / "m.py").write_text("".join(f"def f{i}(a):\n    b = a\n    return b\n\n" for i in range(9)))
    (kit / "x" / "n.js").write_text("".join(f"function g{i}(a) {{\n  const b = a;\n  return b;\n}}\n" for i in range(9)))
    (kit / "x" / "bad.py").write_text("def (:\n")
    res = _cli("--draw-flow-anchors", "--kit-src", str(kit), "--eval-commit", COMMIT, "--out", str(tmp_path / "o.jsonl"))
    assert res.returncode == 0 and "skipped-files: 1" in res.stdout


@pytest.mark.parametrize("path", ["a/b.pyi", "a/x.c++", "a/x.hh", "a/x.hxx", "a/x.ino", "a/Dockerfile", "a/Dockerfile.dev"])
def test_path_class_code_extensions(path):
    assert export_kit.path_class(path) == "code"


def test_export_bad_commit(repo, tmp_path):
    r, _ = repo
    assert _export(r, "deadbeef", tmp_path / "o").returncode == 4


def test_export_update_briefs(repo, tmp_path):
    r, sha = repo
    out = tmp_path / "o"
    assert _export(r, sha, out).returncode == 0
    (out / "roles" / "KA-flow").mkdir(parents=True)
    (out / "roles" / "KA-flow" / "BRIEF.md").write_text("brief")
    res = subprocess.run([PY, str(TOOLS / "export_kit.py"), "--update-briefs", "--out-dir", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0
    m = json.loads((out / "kit-manifest.json").read_text())
    assert m["briefs"] == {"roles/KA-flow/BRIEF.md": hashlib.sha256(b"brief").hexdigest()}
    for dp, dns, fns in os.walk(out):
        os.chmod(dp, 0o755)


def _cli(*a):
    return subprocess.run([PY, str(TOOLS / "export_kit.py"), *a], capture_output=True, text=True)


def test_list_cypher_tools_name_kw_function_name_and_failure(tmp_path):
    kit = tmp_path / "kit"
    (kit / "cypher-mcp").mkdir(parents=True)
    (kit / "cypher-mcp" / "server.py").write_text(REPO_FILES["cypher-mcp/server.py"])
    res = _cli("--list-cypher-tools", "--kit-src", str(kit))
    assert res.returncode == 0 and [json.loads(l) for l in res.stdout.splitlines()] == [{"name": "query"}]
    (kit / "cypher-mcp" / "server.py").write_text(
        "@mcp.tool()\ndef alpha(): pass\n\n@mcp.tool\ndef beta(): pass\n\n@mcp.tool('gamma')\ndef g(): pass\n\n"
        "@other.thing(name='no')\ndef nope(): pass\n")
    res = _cli("--list-cypher-tools", "--kit-src", str(kit))
    assert [json.loads(l)["name"] for l in res.stdout.splitlines()] == ["alpha", "beta", "gamma"]
    (kit / "cypher-mcp" / "server.py").write_text("def nothing(): pass\n")
    res = _cli("--list-cypher-tools", "--kit-src", str(kit))
    assert res.returncode == 4 and res.stdout == "" and "no @<x>.tool" in res.stderr
    (kit / "cypher-mcp" / "server.py").unlink()
    assert _cli("--list-cypher-tools", "--kit-src", str(kit)).returncode == 4


def test_list_cypher_tools_on_the_real_server():
    # the real registration (plan: name="query"), read from the working tree
    res = _cli("--list-cypher-tools", "--kit-src", str(CPG.parent))
    assert [json.loads(l)["name"] for l in res.stdout.splitlines()] == ["query"]


def test_list_units(tmp_path, repo):
    r, sha = repo
    out = tmp_path / "o"
    assert _export(r, sha, out).returncode == 0
    res = _cli("--list-units", "--kit-src", str(out / "kit-src"))
    units = [json.loads(l) for l in res.stdout.splitlines()]
    assert units == [{"lang": "python", "ports": [], "rootPath": "cypher-mcp"},
                     {"lang": "python", "ports": [], "rootPath": "falkor-chat/server"},
                     {"lang": "javascript", "ports": [], "rootPath": "web"}]
    for dp, dns, fns in os.walk(out):
        os.chmod(dp, 0o755)


def test_list_units_both_ecosystems_one_dir(tmp_path):
    kit = tmp_path / "kit"
    (kit / "m").mkdir(parents=True)
    for f in ("pyproject.toml", "requirements.txt", "package.json"):
        (kit / "m" / f).write_text("")
    (kit / "requirements.txt").write_text("")
    units = [json.loads(l) for l in _cli("--list-units", "--kit-src", str(kit)).stdout.splitlines()]
    assert units == [{"lang": "python", "ports": [], "rootPath": "."},
                     {"lang": "javascript", "ports": [], "rootPath": "m"},
                     {"lang": "python", "ports": [], "rootPath": "m"}]


# --- --draw-flow-anchors -----------------------------------------------------------------------

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _py_kit_source():
    fns = "".join(f"def py_fn_{i:02d}(a):\n    b = a + {i}\n    return b\n\n" for i in range(12))
    return fns + (
        "def too_short(a):\n    return a\n\n"                                # 2 lines: lineEnd-lineStart = 1
        "class Svc:\n"
        "    def method_a(self):\n        x = 1\n        return x\n\n"
        "    def dup(self):\n        x = 1\n        return x\n\n"
        "    def dup(self):\n        x = 2\n        return x\n\n"             # Svc.dup#2: never a candidate
        "    @property\n    def prop(self):\n        x = 1\n        return x\n\n"
        "def outer():\n    def inner():\n        x = 1\n        return x\n    return inner\n\n"
        "def deco(f):\n    return f\n\n"
        "@deco\ndef decorated(a):\n    x = 1\n    return x\n\n"
        "def uses_lambda(xs):\n    return sorted(xs,\n        key=lambda v: v)\n\n"
        "from typing import overload\n"
        "@overload\ndef ov(a: int) -> int: ...\n"
        "@overload\ndef ov(a: str) -> str: ...\n"
        "def ov(a):\n    b = a\n    return b\n\n"
        "class P:\n    def stub(self):\n        ...\n")


JS_SOURCE = (
    "".join(f"export function jsFn{i:02d}(a) {{\n  const b = a + {i};\n  return b;\n}}\n\n" for i in range(12)) +
    "const named = (a) => {\n  const b = a;\n  return b;\n};\n\n"
    "class Box {\n  constructor(a) {\n    this.a = a;\n    this.b = 1;\n  }\n"
    "  get size() {\n    const s = 1;\n    return s;\n  }\n}\n\n"
    "function host() {\n  return [1].map((v) => {\n    function insideAnon() {\n      const q = 1;\n      return q;\n    }\n"
    "    return v;\n  });\n}\n\n"
    "const tiny = (a) => a;\n")


@pytest.fixture
def flow_kit(tmp_path):
    kit = tmp_path / "w" / "kit-src"
    (kit / "pkg").mkdir(parents=True)
    (kit / "pkg" / "mod.py").write_text(_py_kit_source())
    (kit / "web").mkdir()
    (kit / "web" / "lib.ts").write_text(JS_SOURCE)
    (kit / "web" / "view.tsx").write_text("export function View(p: any) {\n  const x = p;\n  return x;\n}\n")
    # excluded from the frame: test code, d.ts, pyi
    (kit / "pkg" / "tests").mkdir()
    (kit / "pkg" / "tests" / "helper.py").write_text("def only_in_tests(a):\n    b = 1\n    return b\n")
    (kit / "pkg" / "test_mod.py").write_text("def test_x():\n    a = 1\n    assert a\n")
    (kit / "pkg" / "conftest.py").write_text("def fixt():\n    a = 1\n    return a\n")
    (kit / "web" / "lib.test.ts").write_text("export function inTest(a) {\n  const b = a;\n  return b;\n}\n")
    (kit / "web" / "types.d.ts").write_text("export function decl(a: number): number;\n")
    return kit


def _expected_order(lang, qnames, commit=COMMIT):
    seed = f"ccg-p0f-flow-v1|{lang}|{commit}"
    return sorted(qnames, key=lambda q: hashlib.sha256(f"{seed}|{q}".encode()).hexdigest())


def test_draw_frame_membership(flow_kit):
    frame = export_kit.flow_frame(flow_kit)
    py = {c.qname for c in frame["python"]}
    js = {c.qname for c in frame["javascript"]}
    pre = "pkg/mod.py::"
    assert {pre + f"py_fn_{i:02d}" for i in range(12)} <= py
    assert pre + "Svc.method_a" in py and pre + "Svc.dup" in py and pre + "outer.inner" in py
    assert pre + "Svc.prop.<get>" in py and pre + "ov" in py and pre + "uses_lambda" in py
    for absent in ("too_short", "Svc.dup#2", "P.stub", "ov#2", "ov#3"):
        assert pre + absent not in py, absent                  # <3 lines, #k duplicate, stub, overload stubs not counted
    assert not any("test" in q or "conftest" in q for q in py)
    pj = "web/lib.ts::"
    assert {pj + f"jsFn{i:02d}" for i in range(12)} <= js
    assert pj + "named" in js and pj + "Box.<init>" in js and pj + "Box.size.<get>" in js
    assert pj + "tiny" not in js and not any("insideAnon" in q for q in js)
    assert "web/view.tsx::View" in js
    assert not any("lib.test" in q or "types.d" in q for q in js)


def test_draw_matches_independent_seeded_ordering(flow_kit):
    rows = export_kit.draw_flow_anchors(flow_kit, COMMIT)
    for lang in ("python", "javascript"):
        got = [r["qname"] for r in rows if r["lang"] == lang]
        frame = [c.qname for c in export_kit.flow_frame(flow_kit)[lang]]
        assert len(got) == 8 and got == _expected_order(lang, frame)[:8]
        assert [r["rank"] for r in rows if r["lang"] == lang] == list(range(1, 9))


def test_draw_depends_on_seed_commit_and_lang(flow_kit):
    a = export_kit.draw_flow_anchors(flow_kit, COMMIT)
    b = export_kit.draw_flow_anchors(flow_kit, "f" * 40)
    assert [r["qname"] for r in a if r["lang"] == "python"] != [r["qname"] for r in b if r["lang"] == "python"]
    assert [r["qname"] for r in a if r["lang"] == "javascript"] != [r["qname"] for r in b if r["lang"] == "javascript"]


def test_draw_cli_writes_jsonl_byte_identical_and_prints_no_content(flow_kit, tmp_path):
    out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    r1 = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--eval-commit", COMMIT, "--out", str(out1))
    r2 = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--eval-commit", COMMIT, "--out", str(out2))
    assert r1.returncode == 0 and out1.read_bytes() == out2.read_bytes()
    rows = [json.loads(l) for l in out1.read_text().splitlines()]
    assert len(rows) == 16 and all(set(r) == {"lang", "rank", "qname", "path", "lineStart", "lineEnd"} for r in rows)
    assert [r["lang"] for r in rows] == ["python"] * 8 + ["javascript"] * 8
    assert all(r["lineEnd"] - r["lineStart"] >= 2 and r["qname"].startswith(r["path"] + "::") for r in rows)
    assert "pkg/mod.py::" not in r1.stdout and hashlib.sha256(out1.read_bytes()).hexdigest() in r1.stdout
    # expected through the CLI too: the seed is the --eval-commit argument
    exp = _expected_order("python", [c.qname for c in export_kit.flow_frame(flow_kit)["python"]])[:8]
    assert [r["qname"] for r in rows[:8]] == exp
    r3 = _cli("--draw-flow-anchors", "--kit-src", str(flow_kit), "--eval-commit", "f" * 40, "--out", str(tmp_path / "c.jsonl"))
    assert (tmp_path / "c.jsonl").read_bytes() != out1.read_bytes()


def test_draw_python_span_includes_decorators(flow_kit):
    frame = {c.qname: c for c in export_kit.flow_frame(flow_kit)["python"]}
    c = frame["pkg/mod.py::decorated"]
    src = (flow_kit / "pkg/mod.py").read_text().splitlines()
    assert src[c.lineStart - 1] == "@deco" and src[c.lineEnd - 1].strip() == "return x"


def test_draw_line_spans_match_source(flow_kit):
    rows = {r["qname"]: r for r in export_kit.draw_flow_anchors(flow_kit, COMMIT)}
    frame = {c.qname: c for lang in export_kit.flow_frame(flow_kit).values() for c in lang}
    c = frame["pkg/mod.py::Svc.method_a"]
    src = (flow_kit / "pkg/mod.py").read_text().splitlines()
    assert src[c.lineStart - 1].strip() == "def method_a(self):" and c.lineEnd - c.lineStart == 2
    c = frame["web/lib.ts::named"]
    src = (flow_kit / "web/lib.ts").read_text().splitlines()
    assert src[c.lineStart - 1].startswith("const named") or "(a) => {" in src[c.lineStart - 1]


def test_draw_short_frame_fails(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "m.py").write_text("def a():\n    x = 1\n    return x\n")
    res = _cli("--draw-flow-anchors", "--kit-src", str(kit), "--eval-commit", COMMIT, "--out", str(tmp_path / "o.jsonl"))
    assert res.returncode == 1 and not (tmp_path / "o.jsonl").exists()


def test_draw_uses_spike_tree_sitter_not_regex(flow_kit):
    import tree_sitter  # noqa: F401  (spike extra)
    assert "ts" in export_kit._ts_parsers() and "tsx" in export_kit._ts_parsers()


# =============================================================================================
# strip_gold.py
# =============================================================================================

STRIP_EC = "2fbff4d1102c9999"


def _qw(n=7):
    rows = []
    for i in range(n):
        rows.append({"qid": f"q-{i:03d}", "lang": "en" if i % 2 else "pt", "text": f"question number {i}?",
                     "gold": [{"type": "symbol", "path": "a.py", "qname": "a.py::f"}],
                     "goldOptions": [1, 2], "intent": "impact", "group": f"g{i % 3}",
                     "ambiguityKind": "none" if i != 5 else "scope"})
    return rows


def _write_qw(path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows))


def _strip(tmp_path, rows=None, ec=STRIP_EC, extra=()):
    qw = tmp_path / "QW.jsonl"
    _write_qw(qw, rows or _qw())
    out, amap = tmp_path / "QW-nogold.jsonl", tmp_path / "work" / "aid-map.jsonl"
    res = subprocess.run([PY, str(TOOLS / "strip_gold.py"), "--eval-commit", ec, "--in", str(qw), "--out", str(out),
                          "--aid-map", str(amap), *extra], capture_output=True, text=True)
    return res, out, amap


def _aid(ec, qid):
    return hashlib.sha256(f"ccg-p0f-an-v1|{ec}|{qid}".encode()).hexdigest()[:12]


def test_strip_whitelist_aid_sorted_and_map(tmp_path):
    res, out, amap = _strip(tmp_path)
    assert res.returncode == 0, res.stderr
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 7 and all(set(l) == {"aid", "lang", "text"} for l in lines)
    assert [l["aid"] for l in lines] == sorted(l["aid"] for l in lines)
    m = {json.loads(l)["aid"]: json.loads(l)["qid"] for l in amap.read_text().splitlines()}
    assert m == {_aid(STRIP_EC, f"q-{i:03d}"): f"q-{i:03d}" for i in range(7)}
    by_aid = {l["aid"]: l for l in lines}
    for i in range(7):
        assert by_aid[_aid(STRIP_EC, f"q-{i:03d}")]["text"] == f"question number {i}?"
    for bad in ("gold", "goldOptions", "intent", "group", "ambiguityKind", "q-00"):
        assert bad not in out.read_text()


def test_strip_uses_12_hex_from_the_given_commit(tmp_path):
    (tmp_path / "x").mkdir()
    res, out, _ = _strip(tmp_path)
    assert all(re.fullmatch(r"[0-9a-f]{12}", json.loads(l)["aid"]) for l in out.read_text().splitlines())
    res2, out2, _ = _strip(tmp_path / "x", ec="0" * 16)
    a = {json.loads(l)["aid"] for l in out.read_text().splitlines()}
    b = {json.loads(l)["aid"] for l in out2.read_text().splitlines()}
    assert a != b and a.isdisjoint(b)


def test_strip_order_independent_of_input_order_and_rerun_identical(tmp_path):
    d1, d2 = tmp_path / "1", tmp_path / "2"
    d1.mkdir(); d2.mkdir()
    r1, o1, m1 = _strip(d1, _qw())
    r2, o2, m2 = _strip(d2, list(reversed(_qw())))
    assert o1.read_bytes() == o2.read_bytes() and m1.read_bytes() == m2.read_bytes()
    r3, o3, _ = _strip(d1, _qw())
    assert o3.read_bytes() == o1.read_bytes()


def test_strip_stdout_has_no_item_content(tmp_path):
    res, out, _ = _strip(tmp_path)
    assert "question number" not in res.stdout and "q-00" not in res.stdout
    assert "7" in res.stdout


def test_strip_whitelist_drops_unknown_keys_even_when_they_look_harmless(tmp_path):
    rows = _qw(3)
    for r in rows:
        r["notes"] = "secret-note"
    res, out, _ = _strip(tmp_path, rows)
    assert "secret-note" not in out.read_text()


def test_strip_rejects_duplicate_or_missing_qid_and_bad_lang(tmp_path):
    rows = _qw(3)
    rows[1]["qid"] = rows[0]["qid"]
    res, out, _ = _strip(tmp_path, rows)
    assert res.returncode == 1 and not out.exists() and "duplicate qid" in res.stderr
    rows = _qw(3)
    del rows[0]["text"]
    res, out, _ = _strip(tmp_path, rows)
    assert res.returncode == 1


def test_strip_verify_passes_and_catches_planted_leaks(tmp_path):
    res, out, amap = _strip(tmp_path)
    qw = tmp_path / "QW.jsonl"
    v = lambda o, m=amap: subprocess.run([PY, str(TOOLS / "strip_gold.py"), "--verify", "--eval-commit", STRIP_EC,
                                           "--in", str(qw), "--out", str(o), "--aid-map", str(m)],
                                          capture_output=True, text=True)
    assert v(out).returncode == 0
    lines = out.read_text().splitlines()
    # (1) an extra, innocuously named key (only the exact-keys check can see it)
    bad = tmp_path / "bad1.jsonl"
    d = json.loads(lines[0]); d["note"] = "n"
    bad.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    assert v(bad).returncode == 1
    # (1b) an edited text, and an edited lang (byte-identity with the whitelist re-derived from QW)
    d = json.loads(lines[0]); d["text"] = d["text"] + " "
    bad.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    r = v(bad)
    assert r.returncode == 1 and d["aid"] in r.stderr and "question number" not in r.stderr
    d = json.loads(lines[0]); d["lang"] = "xx"
    bad.write_text("\n".join([json.dumps(d, sort_keys=True, separators=(",", ":"))] + lines[1:]) + "\n")
    assert v(bad).returncode == 1
    # (1b') a re-serialized but identical line is NOT byte-identical either
    d = json.loads(lines[0])
    bad.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    assert v(bad).returncode == 1
    # (1c) a tampered aid map
    badmap = tmp_path / "badmap.jsonl"
    ml = amap.read_text().splitlines()
    mm = json.loads(ml[0]); mm["qid"] = "q-999"
    badmap.write_text("\n".join([json.dumps(mm)] + ml[1:]) + "\n")
    assert v(out, badmap).returncode == 1
    # (2) a gold substring inside the text value
    d = json.loads(lines[0]); d["text"] = "what about goldOptions here"
    bad.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    assert v(bad).returncode == 1
    # (3) a dropped line (a qid no longer maps to exactly one line)
    bad.write_text("\n".join(lines[1:]) + "\n")
    assert v(bad).returncode == 1
    # (4) a duplicated line
    bad.write_text("\n".join(lines + lines[:1]) + "\n")
    assert v(bad).returncode == 1
    # (5) a wrong aid for a qid (map says one thing, output another)
    d = json.loads(lines[0]); d["aid"] = "0" * 12
    bad.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    assert v(bad).returncode == 1
    # (6) unsorted
    bad.write_text("\n".join(reversed(lines)) + "\n")
    assert v(bad).returncode == 1
    # (7) the map checked against a different EVAL_COMMIT
    rr = subprocess.run([PY, str(TOOLS / "strip_gold.py"), "--verify", "--eval-commit", "1" * 16, "--in", str(qw),
                         "--out", str(out), "--aid-map", str(amap)], capture_output=True, text=True)
    assert rr.returncode == 1


def _verify_rows(tmp_path, texts, groups=None):
    rows = _qw(len(texts))
    for r, tx in zip(rows, texts):
        r["text"] = tx
    for r, g in zip(rows, groups or []):
        r["group"] = g
    res, out, amap = _strip(tmp_path, rows)
    assert res.returncode == 0
    return subprocess.run([PY, str(TOOLS / "strip_gold.py"), "--verify", "--eval-commit", STRIP_EC,
                           "--in", str(tmp_path / "QW.jsonl"), "--out", str(out), "--aid-map", str(amap)],
                          capture_output=True, text=True)


@pytest.mark.parametrize("text", [
    "What is the intent of the Composer component?", "Do the groups of tools differ?",
    "Is the grouping intentional?", "Is this the gold standard for goldfish?",
    "Which ambiguity kinds exist, and what is a group?", "Does q-0001 call it?",       # longer token, not the qid q-000
    "What does the key 'intent' mean?",                                                 # quoted word but no field shape
])
def test_strip_verify_honest_questions_pass(tmp_path, text):
    r = _verify_rows(tmp_path, [text])
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("text,shape", [
    ('the "gold": [1] list', "quoted-field"),
    ("{'gold': 1}", "quoted-key"),
    ("{'intent': 'x'}", "quoted-key"),
    ('and "goldOptions" here', "quoted-field"),
    ('see "intent" and "group"', "quoted-field"),
    ("goldOptions: x", "colon-field"),
    ("gold: x", "colon-field"),
    ("intent=impact", "assign-field"),
    ("intent\n= impact", "assign-field"),                       # a JSON-escaped newline: only parsed text shows it
    ("group = g", "assign-field"),
    ("ambiguityKind: scope", "colon-field"),
    ("ambiguityKind'x", "ambiguityKind-quote"),
    ('double-escaped \\"gold\\" form', "quoted-field"),        # literal backslashes around the quotes
    ("it was written for q-000", "own-qid"),
    ("(q-000)", "own-qid"),
    ("about g0 please", "own-group"),
])
def test_strip_verify_leak_shapes_fail_listing_aid_only(tmp_path, text, shape):
    r = _verify_rows(tmp_path, [text])
    assert r.returncode == 1, text
    assert f"aid {_aid(STRIP_EC, 'q-000')}: text leak ({shape})" in r.stderr
    assert text not in r.stderr and "text leak" in r.stderr


def test_strip_verify_qid_and_group_are_the_items_own_only(tmp_path):
    # item 0 mentions item 1's qid and group: not a leak of item 0's own
    r = _verify_rows(tmp_path, ["compare with q-001 in g1", "plain"], ["g0", "g1"])
    assert r.returncode == 0, r.stderr


def test_strip_verify_leak_in_one_item_names_that_aid_only(tmp_path):
    r = _verify_rows(tmp_path, ["fine", "intent=x", "fine too"])
    assert r.returncode == 1
    assert _aid(STRIP_EC, "q-001") in r.stderr
    assert _aid(STRIP_EC, "q-000") not in r.stderr and _aid(STRIP_EC, "q-002") not in r.stderr


# =============================================================================================
if __name__ == "__main__":                                     # pragma: no cover
    if "--write-fixtures" in sys.argv:
        for n, t in all_derived().items():
            (FIX / n).write_text(t)
        print(f"wrote {len(all_derived())} fixtures")
