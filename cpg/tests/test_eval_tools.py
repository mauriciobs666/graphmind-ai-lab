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
mk = _load("match_keys")

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


# =============================================================================================
# match_keys.py (S49 / P0-42a): EP2 §2 oracle K1-K48, EP §13.4 oracle cases for the library,
# validate / scope / sample / leak check
# =============================================================================================

def _nk(kind, key, form="provider"):
    return mk.normalize_contract({"type": "contract", "kind": kind, "keyNorm": key}, form)["keyNorm"]


# (id, kind, input, expected). Every http output in K1-K33 and K42-K47 must also be a fixed point.
K_UNIT = [
    ("K1", "http", "GET /shop/api/health", "GET /shop/api/health"),
    ("K2", "http", "POST /hooks/x7k2q9w4m1p8z3v6", "POST /hooks/{#}"),
    ("K3", "http", "POST /hooks/a1b2c3d4e5f6g7h8", "POST /hooks/{#}"),
    ("K4", "http", "POST /hooks/{}", "POST /hooks/{}"),
    ("K5", "http", "POST /notify/x7k2-q9w4-m1p8-z3v6", "POST /notify/{#}"),
    ("K6", "http", "POST /notify/x7k2+q9w4+m1p8+z3v6", "POST /notify/{#}"),
    ("K7", "http", "GET /builds/3f2a9c1e5b7d", "GET /builds/{#}"),
    ("K8", "http", "DELETE /items/550e8400-e29b-41d4-a716-446655440000", "DELETE /items/{#}"),
    ("K9", "http", "POST /cb/sk-abcdefghijklmnopqrs", "POST /cb/{#}"),
    ("K10", "http", "POST /services/T024BE7LD/B01ABCDEFGH/abcdefghijklmnopqrstuvwx",
     "POST /services/T024BE7LD/B01ABCDEFGH/{#}"),
    ("K11", "http", "GET /t/3f2a9c1e5b7d/x7k2q9w4m1p8z3v6/info", "GET /t/{#}/{#}/info"),
    ("K12", "http", "post hooks//x7k2q9w4m1p8z3v6/", "POST /hooks/{#}/"),
    ("K13", "http", "GET /hooks/x7k2q9w4%6d1p8z3v6", "GET /hooks/{#}"),
    ("K14", "http", "GET /a/x7k2q9w4%2fm1p8z3v6", "GET /a/x7k2q9w4%2Fm1p8z3v6"),
    ("K15", "http", "GET /x/{x7k2q9w4m1p8z3v6}", "GET /x/{}"),
    ("K16", "http", "GET /static/{p:path}", "GET /static/{**}"),
    ("K17", "http", "GET /ntfy/kxqzmwvptryhbnjdlc", "GET /ntfy/kxqzmwvptryhbnjdlc"),
    ("K18", "http", "GET /wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "GET /wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ("K19", "http", "GET /release-2024-notes/s3-bucket-v2-us-east-1", "GET /release-2024-notes/s3-bucket-v2-us-east-1"),
    ("K20", "http", "POST /hooks/{#}", "POST /hooks/{#}"),
    ("K21", "http", "GET /x/{?}", "GET /x/{?}"),
    ("K22", "http", "GET /x7k2q9w4m1p8z3v6", "GET /{#}"),
    ("K23", "mcp_tool", "mcp__falkor__x7k2q9w4m1p8z3v6", "x7k2q9w4m1p8z3v6"),
    ("K30", "http", "GET /f/{x7k2q9w4m1p8z3v6}.json", "GET /f/{#}"),
    ("K31", "http", "GET /a/x7k2q9w4m1p8z3v6/../b", "GET /a/b"),
    ("K32", "http", "GET /a/./x7k2q9w4m1p8z3v6", "GET /a/{#}"),
    ("K33", "http", "POST /hooks/x7k2q9w4m1p8", "POST /hooks/x7k2q9w4m1p8"),
    ("K42", "http", "GET /a/%2E%2E/b", "GET /b"),
    ("K43", "http", "GET /a/b/..", "GET /a/"),
    ("K44", "http", "GET /a/b/.", "GET /a/b/"),
    ("K45", "http", "GET /a/b/%2E%2E", "GET /a/"),
    ("K46", "http", "GET /a/..", "GET /"),
    ("K47", "http", "GET /..", "GET /"),
    ("K48", "http", "GET /a/%2%45%2%45/b", "GET /a/%2E%2E/b"),
]


@pytest.mark.parametrize("kid,kind,inp,want", K_UNIT, ids=[r[0] for r in K_UNIT])
def test_oracle_k_unit(kid, kind, inp, want):
    assert _nk(kind, inp) == want
    if kid != "K48":                                           # K48's output is NOT a fixed point (EP2 §1.1)
        assert _nk(kind, want) == want, "fixed point"


def test_oracle_k48_second_pass_changes_it():
    assert _nk("http", "GET /a/%2E%2E/b") == "GET /b"          # i.e. a second pass of K48's output decodes again


def test_oracle_count_is_48():
    unit_ids = {r[0] for r in K_UNIT} | {"K48"}
    integration = {f"K{n}" for n in (24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 40, 41)}
    assert len(unit_ids) == 34 and len(unit_ids | integration) == 48


def test_http_bad_verb_and_form_are_refused():
    with pytest.raises(ValueError):
        mk.normalize_http_key("FETCH /x")
    with pytest.raises(ValueError):
        mk.normalize_http_key("GET /x", form="both")
    assert mk.normalize_http_key("/bare/path") == "* /bare/path"


def test_consumer_form_only_changes_the_placeholder():
    assert _nk("http", "POST /hooks/x7k2q9w4m1p8z3v6", "consumer") == "POST /hooks/{?}"
    assert _nk("mcp_tool", "mcp__falkor__query", "consumer") == "query"          # form has no effect


def test_pre_8a_form_keeps_the_literal_and_drops_parameter_syntax():
    assert mk.normalize_http_key("post /hooks/x7k2q9w4m1p8z3v6", mask=False) == "POST /hooks/x7k2q9w4m1p8z3v6"
    assert mk.normalize_http_key("get /api/items/{item_id}", mask=False) == "GET /api/items/{}"


def _cid(key, kind="http", form="provider"):
    return mk.contract_id({"type": "contract", "kind": kind, "keyNorm": key}, form)


def test_oracle_k24_k34_merge_and_placeholder_identity():
    cp = _ids_cp("http", "POST /hooks/{#}")
    assert _cid("POST /hooks/x7k2q9w4m1p8z3v6") == _cid("POST /hooks/a1b2c3d4e5f6g7h8") == cp   # K24
    assert _cid("POST /hooks/x7k2q9w4m1p8z3v6") == _cid("POST /hooks/{#}") == cp                  # K34


def _ids_cp(kind, key):
    from ccg import ids
    return ids.contract_point_id(kind, key)


def test_oracle_k28_gold_stored_normalized():
    gold = mk.normalize_contract({"type": "contract", "kind": "http", "keyNorm": "POST /hooks/x7k2q9w4m1p8z3v6"})
    assert _ids_cp(gold["kind"], gold["keyNorm"]) == _ids_cp("http", "POST /hooks/{#}")


def test_oracle_k39_unserved_gold():
    k10 = {"type": "contract", "kind": "http", "keyNorm": "POST /services/T024BE7LD/B01ABCDEFGH/abcdefghijklmnopqrstuvwx"}
    k1 = {"type": "contract", "kind": "http", "keyNorm": "GET /shop/api/health"}
    served = {mk.contract_id(k1)}
    assert mk.unserved_gold_candidate(k10, served) is True
    assert mk.normalize_contract(k10, "consumer")["keyNorm"] == "POST /services/T024BE7LD/B01ABCDEFGH/{?}"
    assert mk.unserved_gold_candidate(k1, set()) is False              # no masked segment: never applied
    assert mk.unserved_gold_candidate(k10, {mk.contract_id(k10)}) is False   # it IS served


def test_oracle_k27_contact_shares_the_anchor_id(tmp_path):
    kit, units = _mk_kit(tmp_path)
    obj = _kitem(1, {"type": "contract", "kind": "http", "keyNorm": "POST /hooks/a1b2c3d4e5f6g7h8"},
                 _sym("a.py::f"), provider=_sym("a.py::f"))
    rec, _ = mk.normalize_item(obj, "KA", units, kit)
    c_raw = {"type": "contract", "kind": "http", "keyNorm": "POST /hooks/x7k2q9w4m1p8z3v6"}
    assert mk.contract_id(c_raw) == rec["anchorId"]              # same anchorId -> the item is contacted


# --- the library: E, H1, §5.4 (P2-2a-d) -----------------------------------------------------

def _P22():
    """Class C in a.py: getter C.x (10-12), setter C.x#2 (14-16), nested C.x#2.check (line 15)."""
    return [{"qname": "a.py::C.x", "lineStart": 10, "lineEnd": 12},
            {"qname": "a.py::C.x#2", "lineStart": 14, "lineEnd": 16},
            {"qname": "a.py::C.x#2.check", "lineStart": 15, "lineEnd": 15}]


@pytest.mark.parametrize("q,want", [
    ("a.py::C.x#2", "a.py::C.x"), ("a.py::C.x#2.check", "a.py::C.x.check"), ("a.py::f.<anon>#1.g", "a.py::f"),
    ("a.py::<anon>#1", "a.py::<module>"), ("s.sh::<anon>#1", "s.sh::<script>"), ("d.md::<anon>#2", "d.md::<document>"),
    ("a.py::C.x", "a.py::C.x"), ("a.py::f#12.g#3", "a.py::f.g")])
def test_E(q, want):
    assert mk.E(q) == want


def test_p22a_duplicate_getter_setter():
    getter, setter, _ = _P22()
    e = {"qname": "a.py::C.x"}
    assert mk.h1_symbol_credit(getter, e, [15]) is False         # the getter must NOT be credited
    assert mk.h1_symbol_credit(setter, e, [15]) is True          # the setter is


def test_p22b_plain_element_still_hits():
    getter = _P22()[0]
    assert mk.h1_symbol_credit(getter, {"qname": "a.py::C.x"}, [11]) is True


def test_p22b_owned_region_counts_and_both_conditions_are_required():
    x = {"qname": "a.py::C", "lineStart": 1, "lineEnd": 3, "ownedRegions": [[20, 22]]}
    assert mk.h1_symbol_credit(x, {"qname": "a.py::C"}, [21]) is True
    assert mk.h1_symbol_credit(x, {"qname": "a.py::C"}, [10]) is False
    assert mk.h1_symbol_credit(dict(x, qname="a.py::D"), {"qname": "a.py::C"}, [2]) is False


def test_p22c_no_correction_back_to_an_ineligible_id():
    e = {"qname": "a.py::C.x.check"}
    assert mk.source_check_needed(e, [15], _P22()) is False      # C.x#2.check qualifies under E
    assert mk.source_check_needed(e, [15], _P22()[:2]) is True   # nothing in the snapshot has that E


def test_p22d_line_outside_every_qualifying_span():
    assert mk.source_check_needed({"qname": "a.py::C.x"}, [30], _P22()) is True


# --- synthetic kit + item helpers ------------------------------------------------------------

KIT_PY = '''import os


class C:
    def x(self):
        return 1

    @property
    def p(self):
        return lambda: 2


def f(a):
    b = a
    return b


def f(a):
    return a
'''


def _mk_kit(tmp_path):
    kit = tmp_path / "kit"
    (kit / "pkg").mkdir(parents=True, exist_ok=True)
    (kit / "pkg" / "a.py").write_text(KIT_PY)
    (kit / "pkg" / "run.py").write_text("print('x')\nif __name__ == '__main__':\n    pass\n")
    (kit / "s.sh").write_text("#!/bin/sh\necho hi\n")
    (kit / ".mcp.json").write_text(json.dumps({"mcpServers": {"cypher": {"command": "x", "args": ["a", "b"]}}}))
    (kit / "compose.yaml").write_text("services:\n  db:\n    ports:\n      - '6379:6379'\n")
    (kit / "mcp-monitor").mkdir(exist_ok=True)
    (kit / "mcp-monitor" / "c.toml").write_text("[server.f]\nurl = 'x'\n\n[[watch]]\nname = 'w'\n")
    (kit / ".env.example").write_text("SERVER_URL=http://localhost:8080\nOTHER=1\n")
    (kit / "AGENTS.md").write_text("# agents\nrun it\n")
    (kit / "web").mkdir(exist_ok=True)
    (kit / "web" / "ui.ts").write_text("export function g() {\n  return 1;\n}\n")
    units = [{"rootPath": "pkg", "lang": "python", "ports": [8080]}, {"rootPath": "web", "lang": "javascript", "ports": []},
             {"rootPath": "cypher-mcp", "lang": "python", "ports": []}]
    return kit, units


def _sym(qname, **kw):
    path = qname.split("::")[0]
    d = {"type": "symbol", "repoId": "graphmind-ai-lab", "path": path, "qname": qname}
    d.update(kw)
    return d


def _kitem(seq, anchor, impacted, provider=None, serves=(), lines=(1,), chain=(), stratum="http"):
    return {"schema": "ccg-key-v1", "seq": seq, "stratum": stratum, "anchor": anchor, "provider": provider,
            "serves": list(serves), "impacted": impacted, "lines": list(lines), "chain": list(chain),
            "hops": len(chain), "why": "w", "deployment": "default"}


HTTP = lambda k: {"type": "contract", "kind": "http", "keyNorm": k}


# --- normalize (EP §13.4) --------------------------------------------------------------------

def test_normalize_item_basics(tmp_path):
    kit, units = _mk_kit(tmp_path)
    o = _kitem(3, HTTP("post /hooks/x7k2q9w4m1p8z3v6"), {"type": "symbol", "path": "./pkg//a.py", "qname": "pkg/a.py::C.x#2.<anon>#1"},
               provider={"type": "symbol", "path": "pkg/a.py", "qname": "pkg/a.py::f"}, lines=(3, 2, 3))
    o["why"] = "  café  "                                 # NFC + strip
    rec, log = mk.normalize_item(o, "KA", units, kit)
    assert rec["impacted"] == {"type": "symbol", "repoId": "graphmind-ai-lab", "path": "pkg/a.py", "qname": "pkg/a.py::C.x"} \
        and rec["anchor"]["keyNorm"] == "POST /hooks/{#}" and rec["why"] == "café" and rec["lines"] == [2, 3]
    assert {"seq": 3, "rule": "8a", "masked": 1} in log and {"seq": 3, "rule": "E"} in log
    assert rec["anchorId"] == _ids_cp("http", "POST /hooks/{#}") and rec["strategy"] == "provider-first"
    from ccg import ids
    assert rec["itemId"] == ids.tuple_id(["evalItem", rec["anchorId"], rec["impactedId"]])
    assert mk.normalize_item(o, "KB", units, kit)[0]["strategy"] == "consumer-first"


def test_normalize_repoid_default_and_mismatch(tmp_path):
    kit, units = _mk_kit(tmp_path)
    o = _kitem(1, HTTP("GET /a"), _sym("pkg/a.py::f"))
    assert mk.normalize_item(o, "KA", units, kit)[0]["impacted"]["repoId"] == "graphmind-ai-lab"
    o["impacted"]["repoId"] = "other"
    with pytest.raises(mk.Invalid, match="repoId"):
        mk.normalize_item(o, "KA", units, kit)


def test_normalize_mcp_namespace_and_log_without_literals(tmp_path):
    kit, units = _mk_kit(tmp_path)
    o = _kitem(1, {"type": "contract", "kind": "mcp_tool", "keyNorm": "mcp__cypher__query"}, _sym("pkg/a.py::f"), stratum="mcp")
    rec, log = mk.normalize_item(o, "KA", units, kit)
    assert rec["anchor"]["keyNorm"] == "query" and log == [{"seq": 1, "rule": "V3"}]


def test_oracle_k26_v3_log_and_stdout_has_no_literal(tmp_path):
    kit, units = _mk_kit(tmp_path)
    o = _kitem(7, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), _sym("pkg/a.py::f"), provider=_sym("pkg/a.py::f"))
    (tmp_path / "KA.jsonl").write_text(json.dumps(o) + "\n")
    ctx = tmp_path / "ctx.json"
    ctx.write_text(json.dumps({"units": units}))
    res = subprocess.run([PY, str(TOOLS / "match_keys.py"), "normalize", "--writer", "KA", "--in", str(tmp_path / "KA.jsonl"),
                          "--out", str(tmp_path / "n.jsonl"), "--log", str(tmp_path / "log.jsonl"), "--kit-src", str(kit),
                          "--context", str(ctx)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    rec = json.loads((tmp_path / "n.jsonl").read_text())
    assert rec["anchor"]["keyNorm"] == "POST /hooks/{#}"                    # valid, not invalidated
    assert json.loads((tmp_path / "log.jsonl").read_text()) == {"masked": 1, "rule": "8a", "seq": 7}
    assert "x7k2" not in res.stdout and "hooks" not in res.stdout and "rewrites" in res.stdout


def test_normalize_drops_unparseable_final_line_and_is_deterministic(tmp_path):
    kit, units = _mk_kit(tmp_path)
    good = json.dumps(_kitem(1, HTTP("GET /a"), _sym("pkg/a.py::f")))
    (tmp_path / "KA.jsonl").write_text(good + "\n" + '{"schema": "ccg-key-v1", "seq": 2, "anc')
    r1, inv, log, dropped = mk.run_normalize("KA", tmp_path / "KA.jsonl", {"units": units}, kit)
    assert len(r1) == 1 and dropped == 1 and inv == []
    r2 = mk.run_normalize("KA", tmp_path / "KA.jsonl", {"units": units}, kit)[0]
    assert [mk.canon(x) for x in r1] == [mk.canon(x) for x in r2]
    (tmp_path / "KA.jsonl").write_text("{bad\n" + good + "\n")                 # a bad NON-final line is an error
    with pytest.raises(ValueError):
        mk.run_normalize("KA", tmp_path / "KA.jsonl", {"units": units}, kit)


def test_normalize_merges_duplicates_within_a_writer(tmp_path):
    kit, units = _mk_kit(tmp_path)
    a = _kitem(2, HTTP("GET /a"), _sym("pkg/a.py::f"), lines=(13,), chain=[{"kind": "link", "from": "x:1", "to": "y:2"}])
    b = _kitem(1, HTTP("get a/"), _sym("pkg/a.py::f#2"), lines=(17,), chain=[{"kind": "caller", "from": "p:1", "to": "q:2"}])
    ra, _ = mk.normalize_item(a, "KA", units, kit)
    rb, _ = mk.normalize_item(b, "KA", units, kit)
    assert ra["itemId"] != rb["itemId"]                                       # "GET /a/" keeps its trailing slash
    b2 = _kitem(1, HTTP("get /a"), _sym("pkg/a.py::f#2"), lines=(17,), chain=[{"kind": "caller", "from": "p:1", "to": "q:2"}])
    rb2, _ = mk.normalize_item(b2, "KA", units, kit)
    assert ra["itemId"] == rb2["itemId"]
    (m,) = mk.merge_within_writer([ra, rb2])
    assert m["lines"] == [13, 17] and m["chain"][0]["kind"] == "caller"      # first chain BY SEQ (seq 1)


def test_normalize_config_grain_and_reanchor(tmp_path):
    kit, units = _mk_kit(tmp_path)
    imp = {"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.cypher.args[1]"}
    o = _kitem(1, {"type": "contract", "kind": "mcp_tool", "keyNorm": "query"}, imp, provider=_sym("cypher-mcp/server.py::query"),
               stratum="wiring")
    rec, log = mk.normalize_item(o, "KB", units, kit)
    assert rec["impacted"]["keyPath"] == "mcpServers.cypher"
    assert rec["anchor"] == {"type": "unit", "repoId": "graphmind-ai-lab", "rootPath": "cypher-mcp", "lang": "python"}
    assert rec["serves"] == [{"type": "contract", "kind": "mcp_tool", "keyNorm": "query"}]
    assert {"seq": 1, "rule": "reanchor"} in log and {"seq": 1, "rule": "config-grain"} in log
    o["provider"] = None
    with pytest.raises(mk.Invalid, match="reanchor"):
        mk.normalize_item(o, "KB", units, kit)
    o["provider"] = _sym("elsewhere/x.py::f")
    with pytest.raises(mk.Invalid, match="reanchor"):
        mk.normalize_item(o, "KB", units, kit)


def test_normalize_unit_anchor_rules(tmp_path):
    kit, units = _mk_kit(tmp_path)
    unit = {"type": "unit", "repoId": "graphmind-ai-lab", "rootPath": "cypher-mcp", "lang": "python"}
    ok = _kitem(1, unit, {"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.cypher"}, serves=[{"type": "contract", "kind": "mcp_tool", "keyNorm": "query"}])
    assert mk.normalize_item(ok, "KA", units, kit)[0]["anchor"]["rootPath"] == "cypher-mcp"
    with pytest.raises(mk.Invalid, match="unit-not-listed"):
        mk.normalize_item(dict(ok, anchor=dict(unit, rootPath="nope")), "KA", units, kit)
    with pytest.raises(mk.Invalid, match="class-mismatch"):                    # a contract-class entry under a unit anchor
        mk.normalize_item(dict(ok, impacted={"type": "config", "path": ".mcp.json", "keyPath": "other.thing"}), "KA", units, kit)
    with pytest.raises(mk.Invalid, match="unit-anchor-needs-config"):
        mk.normalize_item(dict(ok, impacted=_sym("pkg/a.py::f")), "KA", units, kit)


@pytest.mark.parametrize("extra", [{"within": "pkg/a.py::f", "line": 3, "nth": 1}, {"within": "pkg/a.py::f"}, {"nth": 1}])
def test_normalize_refuses_ancestor_plus_line_form(tmp_path, extra):
    kit, units = _mk_kit(tmp_path)                   # keyed on `within` OR `nth` each alone (n-1); the form has no qname
    o = _kitem(1, HTTP("GET /a"), dict({"type": "symbol", "path": "pkg/a.py", "qname": "pkg/a.py::f"}, **extra))
    with pytest.raises(mk.Invalid, match="ancestor-plus-line"):
        mk.normalize_item(o, "KA", units, kit)


@pytest.mark.parametrize("path,kp,env,want", [
    (".mcp.json", "mcpServers.cypher.args[0]", None, ("mcpServers.cypher", "server")),
    ("k/.kiro/agents/a.json", "mcpServers.x.cmd", None, ("mcpServers.x", "server")),
    ("k/.kiro/agents/a.json", "tools[3]", None, ("tools[3]", "contract")),
    ("k/.kiro/agents/a.json", "allowedTools[0].x", None, ("allowedTools[0]", "contract")),
    ("k/.kiro/agents/a.json", "toolAliases.q.z", None, ("toolAliases.q", "contract")),
    ("k/.kiro/agents/a.json", "includeMcpJson", None, ("includeMcpJson", "server")),
    ("k/.kiro/agents/a.json", "name.first.x", None, ("name.first", "contract")),
    (".claude/settings.json", "enabledMcpjsonServers[2]", None, ("enabledMcpjsonServers[2]", "server")),
    (".claude/settings.local.json", "permissions.allow[1].deep", None, ("permissions.allow[1]", "contract")),
    (".claude/settings.json", "hooks.PreToolUse[0].hooks[0]", None, ("hooks.PreToolUse[0]", "contract")),
    ("mcp-monitor/c.toml", "server.f.url", None, ("server.f", "server")),
    ("mcp-monitor/c.toml", "watch[0].name", None, ("watch[0]", "contract")),
    ("falkor-chat/compose.yaml", "services.db.ports[0]", None, ("services.db", "server")),
    ("docker-compose.dev.yml", "services.db", None, ("services.db", "server")),
    ("a/settings.yaml", "a.b.c.d", None, ("a.b", "contract")),
    ("a/x.json", 'a["b.c"][0].d', None, ('a["b.c"]', "contract")),
    ("a/x.toml", "[0].k", None, ("[0].k", "contract")),
])
def test_config_grain_table(path, kp, env, want):
    assert mk.config_grain(path, kp, env) == want


def test_config_grain_env_class_from_value_and_ports():
    units = [{"rootPath": "p", "lang": "python", "ports": [8080]}]
    g = lambda v: mk.config_grain(".env.example", "SERVER_URL", v, units)
    assert g("http://localhost:8080") == ("SERVER_URL", "server")
    assert g("http://localhost:9999") == ("SERVER_URL", "contract")
    assert mk.config_grain(".env", "PORT", "8080", units)[1] == "server"
    assert mk.config_grain(".env", "PORT", "81", units)[1] == "contract"
    assert g("http://h/x") == ("SERVER_URL", "contract")                       # default port 80 is nobody's
    assert mk.config_grain(".env", "X", "http://h:8080/", units)[1] == "server"


# --- validate (V1/V2/V3) ---------------------------------------------------------------------

def _val(tmp_path, impacted, lines=(1,), anchor=None, provider=None, decisions=()):
    kit, units = _mk_kit(tmp_path)
    o = _kitem(1, anchor or HTTP("GET /a"), impacted, provider=provider or _sym("pkg/a.py::f"), lines=lines)
    rec, _ = mk.normalize_item(o, "KA", units, kit)
    return mk.validate_records([rec], kit, decisions), rec


def test_validate_valid_python_symbol_and_lines(tmp_path):
    (valid, invalid, _), _ = _val(tmp_path, _sym("pkg/a.py::f"), lines=(13, 14))
    assert len(valid) == 1 and invalid == []
    (valid, invalid, _), _ = _val(tmp_path, _sym("pkg/a.py::C.p.<get>"), lines=(9, 10))   # decorator line is in the span
    assert len(valid) == 1, invalid


@pytest.mark.parametrize("impacted,lines,check", [
    (_sym("pkg/a.py::nope"), (1,), "V1:name-not-in-file"),
    (_sym("pkg/a.py::C.os"), (1,), "V2:qname-not-derived"),             # every token exists in the file; the chain does not
    (_sym("pkg/missing.py::f"), (1,), "V1:path-not-in-kit"),
    (_sym("pkg/a.py::f"), (999,), "V1:line-out-of-range"),
    (_sym("pkg/a.py::f"), (5,), "V2:line-outside-span"),                  # a line of C.x, not of f
    (_sym("pkg/a.py::f"), (0,), "V1:line-out-of-range"),
    (_sym("tests/test_a.py::f"), (1,), "V1:path-scope"),
    (_sym("cpg/ccg/ids.py::f"), (1,), "V1:path-scope"),
    (_sym("AGENTS.md::x"), (1,), "V1:name-not-in-file"),
    ({"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.nope"}, (1,), "V2:keypath-unresolved"),
    ({"type": "config", "path": "mcp-monitor/c.toml", "keyPath": "server.zzz"}, (1,), "V2:keypath-unresolved"),
    ({"type": "config", "path": "compose.yaml", "keyPath": "services.web"}, (1,), "V2:keypath-unresolved"),
])
def test_validate_failures(tmp_path, impacted, lines, check):
    (valid, invalid, _), rec = _val(tmp_path, impacted, lines)
    assert valid == [] and check in invalid[0]["checks"] and invalid[0]["itemId"] == rec["itemId"]


def test_validate_config_files_that_resolve(tmp_path):
    for imp in ({"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.cypher"},
                {"type": "config", "path": "mcp-monitor/c.toml", "keyPath": "server.f"},
                {"type": "config", "path": "mcp-monitor/c.toml", "keyPath": "watch[0]"},
                {"type": "config", "path": "compose.yaml", "keyPath": "services.db"},
                {"type": "config", "path": ".env.example", "keyPath": "OTHER"}):
        (valid, invalid, _), _ = _val(tmp_path / imp["path"].replace("/", "_"), imp)
        assert len(valid) == 1, (imp, invalid)


def test_validate_decisions_by_item_id(tmp_path):
    (valid, invalid, dropped), rec = _val(tmp_path, _sym("pkg/a.py::nope"))
    assert len(invalid) == 1
    (valid, invalid, dropped), rec = _val(tmp_path / "d", _sym("pkg/a.py::nope"), decisions=[{"itemId": _val(tmp_path / "e", _sym("pkg/a.py::nope"))[1]["itemId"], "action": "drop"}])
    assert valid == [] and invalid == [] and dropped["KA|V1:name-not-in-file"] == 1


def test_py_symbols_spans_and_names():
    syms = {s["chain"]: (s["lineStart"], s["lineEnd"]) for s in mk.py_symbols(KIT_PY)}
    assert syms["<module>"] == (1, 19) and syms["C"] == (4, 10) and syms["C.x"] == (5, 6)
    assert syms["C.p.<get>"] == (8, 10) and syms["C.p.<get>.<anon>#1"] == (10, 10)      # decorator line in the span
    assert syms["f"] == (13, 15) and syms["f#2"] == (18, 19)
    over = "from typing import overload\n@overload\ndef g(a: int): ...\n@overload\ndef g(a: str): ...\ndef g(a):\n    return a\n"
    assert [s["chain"] for s in mk.py_symbols(over)] == ["<module>", "g"]                # overloads are not symbols
    deco = "def d(f):\n    return f\n\n@d\ndef h(a=lambda: 1):\n    return lambda: 2\n"
    ch = {s["chain"] for s in mk.py_symbols(deco)}
    assert "<anon>#1" in ch and "h.<anon>#1" in ch                                       # a default's lambda is the ENCLOSING scope's


# --- stratum, match, near-miss, scope, sample ------------------------------------------------

CTX = {"flowAnchors": ["web/ui.ts::g"], "cypherMcpTools": ["query"],
       "units": [{"rootPath": "pkg", "lang": "python", "ports": []}, {"rootPath": "web", "lang": "javascript", "ports": []},
                 {"rootPath": "cypher-mcp", "lang": "python", "ports": []}]}


def _rec(tmp_path, anchor, impacted, writer="KA", stratum="http", **kw):
    kit, units = _mk_kit(tmp_path)
    r, _ = mk.normalize_item(_kitem(kw.pop("seq", 1), anchor, impacted, stratum=stratum, **kw), writer, CTX["units"], kit)
    r["derivedStratum"] = mk.derive_stratum(r, CTX, kit)
    return r


def test_derive_stratum_rules_in_order(tmp_path):
    kit, units = _mk_kit(tmp_path)
    d = lambda anchor, imp, **kw: _rec(tmp_path, anchor, imp, **kw)["derivedStratum"]
    assert d(HTTP("GET /a"), _sym("pkg/a.py::f")) == "http"
    assert d({"type": "contract", "kind": "mcp_tool", "keyNorm": "other"}, _sym("pkg/a.py::f")) == "mcp"
    assert d({"type": "contract", "kind": "mcp_tool", "keyNorm": "query"}, _sym("pkg/a.py::f")) == "scripts"      # cypherMcpTools
    assert d(HTTP("GET /a"), _sym("s.sh::<script>")) == "scripts"                                                   # shell impacted
    assert d(_sym("s.sh::<script>"), _sym("pkg/a.py::f")) == "scripts"
    assert d(_sym("pkg/run.py::<module>"), _sym("pkg/a.py::f")) == "scripts"                                        # __main__ guard
    assert d(_sym("pkg/a.py::<module>"), _sym("pkg/a.py::f")) is None                                               # unclassified
    assert d(_sym("web/ui.ts::g"), _sym("pkg/a.py::f")) == "flow"                                                   # rule 1 first
    assert d(_sym("web/ui.ts::g"), {"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.cypher"}) == "flow"
    assert d(HTTP("GET /a"), {"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.cypher.x"},
             provider=_sym("pkg/a.py::f")) == "wiring"


def _grp(tmp_path, seed_anchor, imp_q, writer, **kw):
    return _rec(tmp_path, seed_anchor, _sym(imp_q), writer=writer, **kw)


def test_match_on_item_id_k24(tmp_path):
    a = _rec(tmp_path, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), _sym("pkg/a.py::f"), writer="KA", lines=(13,))
    b = _rec(tmp_path, HTTP("POST /hooks/a1b2c3d4e5f6g7h8"), _sym("pkg/a.py::f"), writer="KB", lines=(15,))
    m = mk.match_records([a, b])
    (g,) = m["records"]
    assert g["sessions"] == ["KA", "KB"] and g["strategies"] == ["consumer-first", "provider-first"] and g["history"] == "101"
    assert nearmiss_of(m) == []


def nearmiss_of(m):
    return mk.nearmiss_pairs(m)


def test_k25_no_false_match_and_an_n2_pair(tmp_path):
    a = _rec(tmp_path, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), _sym("pkg/a.py::f"), writer="KA")
    b = _rec(tmp_path, HTTP("POST /hooks/{}"), _sym("pkg/a.py::f"), writer="KB")
    m = mk.match_records([a, b])
    assert len(m["records"]) == 2                                          # no mechanical match
    (p,) = mk.nearmiss_pairs(m)
    assert p["rule"] == "N2" and p["stratum"] == "http" and len(p["pairId"]) == 12 and p["x"]["writer"] == "KA"
    from hashlib import sha256
    assert p["pairId"] == sha256(f"ccg-p0f-pair-v1|{a['itemId']}|{b['itemId']}".encode()).hexdigest()[:12]


def test_nearmiss_n1_n2provider_n3_and_exclusions(tmp_path):
    f1, f2 = _sym("pkg/a.py::f"), _sym("pkg/a.py::C.x")
    # N1: same anchor, same path, different impacted
    m = mk.match_records([_rec(tmp_path, HTTP("GET /a"), f1, writer="KA"), _rec(tmp_path, HTTP("GET /a"), f2, writer="KB")])
    assert [p["rule"] for p in mk.nearmiss_pairs(m)] == ["N1"]
    # N1 across paths: no
    m = mk.match_records([_rec(tmp_path, HTTP("GET /a"), f1, writer="KA"), _rec(tmp_path, HTTP("GET /a"), _sym("web/ui.ts::g"), writer="KB")])
    assert mk.nearmiss_pairs(m) == []
    # N2 through equal provider ids even when the keys differ
    prov = _sym("pkg/a.py::f")
    m = mk.match_records([_rec(tmp_path, HTTP("GET /a/b"), f2, writer="KA", provider=prov),
                          _rec(tmp_path, HTTP("GET /zz/other"), f2, writer="KB", provider=prov)])
    assert [p["rule"] for p in mk.nearmiss_pairs(m)] == ["N2"]
    # N2 casefold + trailing slash + {…} -> {}
    m = mk.match_records([_rec(tmp_path, HTTP("GET /A/{id}/"), f2, writer="KA"), _rec(tmp_path, HTTP("GET /a/{n}"), f2, writer="KB")])
    assert [p["rule"] for p in mk.nearmiss_pairs(m)] == ["N2"]
    # same strategy never pairs (KA vs KA2)
    m = mk.match_records([_rec(tmp_path, HTTP("GET /a"), f1, writer="KA"), _rec(tmp_path, HTTP("GET /a"), f2, writer="KA2")])
    assert mk.nearmiss_pairs(m) == []
    # N3: two unit anchors, one impacted
    cfg = {"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.cypher"}
    u1 = {"type": "unit", "repoId": "graphmind-ai-lab", "rootPath": "cypher-mcp", "lang": "python"}
    u2 = dict(u1, rootPath="pkg")
    sv = [{"type": "contract", "kind": "mcp_tool", "keyNorm": "query"}]
    m = mk.match_records([_rec(tmp_path, u1, cfg, writer="KA", serves=sv, stratum="wiring"), _rec(tmp_path, u2, cfg, writer="KB", serves=sv, stratum="wiring")])
    assert [p["rule"] for p in mk.nearmiss_pairs(m)] == ["N3"]


def test_match_counts_unclassified_and_stratum_disagreements(tmp_path):
    a = _rec(tmp_path, _sym("pkg/a.py::<module>"), _sym("pkg/a.py::f"), writer="KA")
    b = _rec(tmp_path, HTTP("GET /a"), _sym("pkg/a.py::f"), writer="KA", stratum="mcp")
    m = mk.match_records([a, b])
    assert m["unclassified"] == 1 and m["stratumDisagreements"] == 1 and len(m["records"]) == 1


def _flow_group(tmp_path, anchor_q, imp_q, chain, writer="KA"):
    return _rec(tmp_path, _sym(anchor_q), _sym(imp_q), writer=writer, stratum="flow", chain=chain)


FLOW = lambda *k: [{"kind": kk, "from": "a:1", "to": "b:2"} for kk in k]


def test_scope_v4_per_item_and_counting_m6(tmp_path):
    ctx = dict(CTX, flowAnchors=["web/ui.ts::g", "pkg/a.py::f"])
    kit, units = _mk_kit(tmp_path)
    # in scope: hops <= 2 with a flow hop; cross-unit counts, same-unit does not
    same = _rec(tmp_path, _sym("pkg/a.py::f"), _sym("pkg/a.py::C.x"), stratum="flow", chain=FLOW("return-use"))
    cross = _rec(tmp_path, _sym("web/ui.ts::g"), _sym("pkg/a.py::f"), stratum="flow", chain=FLOW("callee-arg", "caller"))
    callonly = _rec(tmp_path, _sym("web/ui.ts::g"), _sym("pkg/a.py::C.x"), stratum="flow", chain=FLOW("caller", "caller"))
    toolong = _rec(tmp_path, _sym("pkg/a.py::f"), _sym("pkg/a.py::C.p.<get>"), stratum="flow", chain=FLOW("control", "caller", "caller"))
    for r in (same, cross, callonly, toolong):
        r["derivedStratum"] = mk.derive_stratum(r, ctx, kit)
    http = _rec(tmp_path, HTTP("GET /a"), _sym("pkg/a.py::f"))
    m = mk.match_records([same, cross, callonly, toolong, http])
    m, oos, summary = mk.scope_records(m, ctx["units"])
    assert sorted(oos) == sorted([callonly["itemId"], toolong["itemId"]])
    by = {g["itemId"]: g for g in m["records"]}
    assert by[same["itemId"]]["counting"] is False and by[cross["itemId"]]["counting"] is True and by[http["itemId"]]["counting"] is True
    assert summary["H_all"] == 3 and summary["H"] == 2 and summary["outOfScope"] == 2


def test_scope_one_in_scope_member_keeps_the_item_m1(tmp_path):
    ctx = dict(CTX, flowAnchors=["web/ui.ts::g"])
    kit, units = _mk_kit(tmp_path)
    a = _rec(tmp_path, _sym("web/ui.ts::g"), _sym("pkg/a.py::f"), writer="KA", stratum="flow", chain=FLOW("caller", "caller", "caller"))
    b = _rec(tmp_path, _sym("web/ui.ts::g"), _sym("pkg/a.py::f"), writer="KB", stratum="flow", chain=FLOW("return-use"))
    for r in (a, b):
        r["derivedStratum"] = mk.derive_stratum(r, ctx, kit)
    m, oos, s = mk.scope_records(mk.match_records([a, b]), ctx["units"])
    assert oos == [] and s["H_all"] == 1


def test_oracle_m6_counting_below_75_while_all_items_at_least_75():
    recs = [{"itemId": f"i{i:03d}", "derivedStratum": "flow" if i < 20 else "http", "strategies": ["provider-first"],
             "sessions": ["KA"], "members": [{"anchor": {"type": "symbol", "path": "pkg/a.py", "repoId": "graphmind-ai-lab"},
                                              "impacted": {"type": "symbol", "path": "pkg/b.py", "repoId": "graphmind-ai-lab"},
                                              "chain": FLOW("return-use")}]} for i in range(80)]
    m, oos, s = mk.scope_records({"records": recs}, CTX["units"])
    assert s["H_all"] == 80 >= 75 and s["H"] == 60 < 75           # same-unit flow items do not count


def _spot_match(counts, tag="s"):
    """counts: {(stratum, class): n} -> a match dict of synthetic records."""
    recs = []
    for (s, c), n in counts.items():
        for i in range(n):
            recs.append({"itemId": f"{tag}-{s}-{c}-{i:02d}", "derivedStratum": s,
                         "strategies": ["provider-first"] if c == "singleton" else ["consumer-first", "provider-first"]})
    return {"records": recs}


def _full():
    return {(s, c): 6 for s in mk.STRATA for c in ("singleton", "doubleton")}


def test_sample_spot_is_4_plus_4_per_stratum_by_hash_sort():
    m = _spot_match(_full())
    spot = mk.spot_sample(m, "E" * 40)
    assert len(spot) == 40
    for s in mk.STRATA:
        for c in ("singleton", "doubleton"):
            got = sorted(r["itemId"] for r in spot if r["cell"] == f"{s}/{c}")
            ranked = sorted((r["itemId"] for r in m["records"] if r["derivedStratum"] == s and (len(r["strategies"]) == 1) == (c == "singleton")),
                            key=lambda i: hashlib.sha256(f"ccg-p0f-spot-v1|{'E' * 40}|{i}".encode()).hexdigest())
            assert got == sorted(ranked[:4])


def test_sample_spot_seed_depends_on_eval_commit():
    m = _spot_match(_full())
    assert [r["itemId"] for r in mk.spot_sample(m, "A" * 40)] != [r["itemId"] for r in mk.spot_sample(m, "B" * 40)]


def test_sample_spot_shortfall_goes_to_next_stratum_same_class_wrapping():
    counts = _full()
    counts[("flow", "singleton")] = 1            # short by 3 -> wraps to http's singleton surplus (6 - 4 = 2) then mcp's
    m = _spot_match(counts)
    spot = mk.spot_sample(m, "E" * 40)
    assert len(spot) == 40
    flow_s = [r for r in spot if r["cell"] == "flow/singleton"]
    assert len(flow_s) == 4 and {r["stratum"] for r in flow_s} == {"flow", "http", "mcp"}
    assert sum(1 for r in flow_s if r["stratum"] == "http") == 2 and sum(1 for r in flow_s if r["stratum"] == "mcp") == 1
    assert len({r["itemId"] for r in spot}) == 40                                  # nobody twice


def test_sample_spot_shortfall_falls_back_to_the_other_class():
    counts = _full()
    for s in mk.STRATA:
        counts[(s, "singleton")] = 4                                             # no surplus of singletons anywhere
    counts[("mcp", "singleton")] = 2                                              # short by 2
    spot = mk.spot_sample(_spot_match(counts), "E" * 40)
    assert len(spot) == 40
    taken = [r for r in spot if r["cell"] == "mcp/singleton"]
    assert len(taken) == 4 and sum(1 for r in taken if r["class"] == "doubleton") == 2


def test_sample_spot_takes_all_when_the_whole_union_is_small():
    spot = mk.spot_sample(_spot_match({("http", "singleton"): 2, ("flow", "doubleton"): 1}), "E" * 40)
    assert len(spot) == 3


def _qw_sample(n_pt_una=9, n_pt_amb=3, n_en=10):
    rows = []
    for i in range(n_pt_una):
        rows.append({"qid": f"QW-pt-u{i}", "lang": "pt", "intent": "unambiguous", "text": "t"})
    for i in range(n_pt_amb):
        rows.append({"qid": f"QW-pt-a{i}", "lang": "pt", "intent": "ambiguous", "text": "t"})
    for i in range(n_en):
        rows.append({"qid": f"QW-en-{i}", "lang": "en", "intent": "unambiguous" if i % 3 else "ambiguous", "text": "t"})
    return rows


def test_sample_lang_gold_6_plus_2_by_hash_sort():
    c = "C" * 40
    got = mk.lang_gold_sample(_qw_sample(), "pt", "ccg-p0f-ptgold-v1", c)
    assert len(got) == 8 and sum(1 for q in got if q["intent"] == "ambiguous") == 2 and all(q["lang"] == "pt" for q in got)
    una = sorted((q["qid"] for q in _qw_sample() if q["lang"] == "pt" and q["intent"] == "unambiguous"),
                 key=lambda i: hashlib.sha256(f"ccg-p0f-ptgold-v1|{c}|{i}".encode()).hexdigest())
    assert {q["qid"] for q in got if q["intent"] == "unambiguous"} == set(una[:6])
    en = mk.lang_gold_sample(_qw_sample(), "en", "ccg-p0f-engold-v1", c)
    assert len(en) == 8 and all(q["lang"] == "en" for q in en)
    assert {q["qid"] for q in en} != {q["qid"] for q in mk.lang_gold_sample(_qw_sample(), "en", "ccg-p0f-ptgold-v1", c)}   # the tag is part of the seed
    short = mk.lang_gold_sample(_qw_sample(n_pt_amb=1), "pt", "ccg-p0f-ptgold-v1", c)
    assert len(short) == 8 and sum(1 for q in short if q["intent"] == "ambiguous") == 1       # a short class is filled from the other


def test_sample_cli_writes_files_and_prints_counts_only(tmp_path):
    m = _spot_match(_full())
    (tmp_path / "match.json").write_text(json.dumps(m))
    (tmp_path / "QW.jsonl").write_text("".join(json.dumps(q) + "\n" for q in _qw_sample()))
    cmd = [PY, str(TOOLS / "match_keys.py"), "sample", "--match", str(tmp_path / "match.json"), "--qw", str(tmp_path / "QW.jsonl"),
           "--eval-commit", "E" * 40, "--out-dir", str(tmp_path / "work")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "spot: 40" in r.stdout and "ptgold: 8" in r.stdout and "QW-pt" not in r.stdout and "s-http" not in r.stdout
    first = (tmp_path / "work" / "spot.jsonl").read_bytes()
    subprocess.run(cmd, capture_output=True)
    assert (tmp_path / "work" / "spot.jsonl").read_bytes() == first                  # deterministic rerun
    assert len((tmp_path / "work" / "engold.jsonl").read_text().splitlines()) == 8


# --- renormalize-check: the drift guard (K35, K40) -----------------------------------------------

def _stored(loc, form="provider"):
    return {"id": mk.contract_id(loc, form), "locator": loc, "form": form}


K33 = {"type": "contract", "kind": "http", "keyNorm": "POST /hooks/x7k2q9w4m1p8"}
K10L = {"type": "contract", "kind": "http", "keyNorm": "POST /services/T024BE7LD/B01ABCDEFGH/abcdefghijklmnopqrstuvwx"}


def _strict_screen(monkeypatch):
    from ccg import keynorm
    orig = keynorm.screen_fails
    monkeypatch.setattr(keynorm, "screen_fails", lambda t: orig(t) or len(t) >= 12)       # "T1 lowered to 12" (a test double)


def test_oracle_k35_drift_guard(monkeypatch):
    entries = [_stored(K33), _stored({"type": "contract", "kind": "http", "keyNorm": "POST /hooks/x7k2q9w4m1p8z3v6"}),
               _stored({"type": "contract", "kind": "mcp_tool", "keyNorm": "mcp__s__query"})]
    assert mk.renormalize_check(entries) == {"checked": 3, "differences": 0, "diffs": []}
    _strict_screen(monkeypatch)
    res = mk.renormalize_check(entries)
    assert res["differences"] == 1 and res["diffs"][0]["stored"] == entries[0]["id"]      # only K33's id changed


def test_oracle_k40_consumer_form_gold_under_the_drift_guard(monkeypatch):
    stored = _stored(K10L, "consumer")
    assert stored["id"] != _stored(K10L, "provider")["id"]
    assert mk.renormalize_check([stored])["differences"] == 0
    assert mk.renormalize_check([dict(stored, form="provider")])["differences"] == 1      # why the form is an input
    _strict_screen(monkeypatch)
    assert mk.renormalize_check([stored])["differences"] == 0                              # K10's segment is masked either way


def test_renormalize_check_cli_exit_codes_and_output(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text("".join(json.dumps(e) + "\n" for e in [_stored(K10L, "consumer")]))
    cmd = [PY, str(TOOLS / "match_keys.py"), "renormalize-check", "--in", str(f)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0 and "differences: 0" in r.stdout and "T024BE7LD" not in r.stdout
    f.write_text(json.dumps(dict(_stored(K10L, "consumer"), form="provider")) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 1 and "differences: 1" in r.stdout and "T024BE7LD" not in r.stdout


# --- the report leak check (EP §9.4, EP2 §1.4): K29, K36-K38, K41, m4, P2-3 ------------------------

def _h(n):
    return hashlib.sha256(str(n).encode()).hexdigest()[:32]


def _ac5(n, anchor, impacted, lines=(5,), provider=None, serves=(), anchor_id=None, path=None):
    o = _kitem(n, anchor, impacted, provider=provider, serves=serves, lines=lines)
    return {"itemId": _h(f"i{n}"), "anchorId": anchor_id or _h(f"a{n}"), "impactedId": _h(f"p{n}"), "original": o}


def _ac24(n, text, anchors):
    return {"qid": f"QW-{n}", "text": text, "original": {"text": text, "gold": {"anchors": anchors}, "goldOptions": None}}


def _leak(report, ho, dev=(), contact=()):
    hits, n = mk.leak_check(report, list(ho), list(dev), list(contact))
    return hits


SYM = lambda q, **kw: _sym(q, **kw)


def test_k29_masked_twin_both_probes():
    ho = _ac5(1, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), SYM("pkg/a.py::f"))
    dev = _ac5(2, HTTP("GET /other/route"), SYM("pkg/a.py::g"), serves=[HTTP("POST /hooks/a1b2c3d4e5f6g7h8")])
    assert _leak("missed post /hooks/x7k2q9w4m1p8z3v6 here", [ho], [dev])                    # (a) the pre-8a probe survives
    assert not _leak("missed post /hooks/{#} here", [ho], [dev])                              # (b) removed by dev equality
    # without the dev twin the normalized probe is live
    assert _leak("missed post /hooks/{#} here", [ho], [])


def test_k29_a_dev_ac24_gold_twin_works_too():
    ho = _ac5(1, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), SYM("pkg/a.py::f"))
    dev = _ac24(1, "something unrelated about the weather today", [HTTP("POST /hooks/a1b2c3d4e5f6g7h8")])
    assert not _leak("missed post /hooks/{#} here", [ho], [dev]) and _leak("missed post /hooks/x7k2q9w4m1p8z3v6 here", [ho], [dev])


def test_k36_writer_spelling_is_not_a_false_positive():
    ho = _ac5(1, HTTP("get /shop/api/products"), SYM("pkg/a.py::f"))
    dev = _ac5(2, HTTP("GET /shop/api/products"), SYM("pkg/a.py::g"))
    assert not _leak("the report names GET /shop/api/products openly", [ho], [dev])


def test_k37_no_new_probe_class_for_parameter_names():
    ho = _ac5(1, HTTP("GET /api/items/{item_id}"), SYM("pkg/a.py::f"))
    assert not _leak("a report containing GET /api/items/{item_id} here", [ho])
    assert _leak("a report containing GET /api/items/{} here", [ho])                          # the normalized probe IS live


def test_k38_normalized_probe_without_a_twin_fails_under_trailing_punctuation():
    ho = _ac5(1, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), SYM("pkg/a.py::f"))
    assert _leak("missed post /hooks/{#}.", [ho])


def test_k41_gold_probes_come_from_the_writer_original():
    gold = HTTP("post /hooks/x7k2q9w4m1p8z3v6")
    ho = _ac24(12, "a held-out question about something quite different here", [gold])
    assert _leak("q12 gold post /hooks/x7k2q9w4m1p8z3v6 not understood", [ho])                 # (a)
    dev = _ac24(3, "a dev question about the very weather in the north", [gold])
    assert not _leak("dev q3 gold post /hooks/x7k2q9w4m1p8z3v6", [ho], [dev])                 # (b)
    # a STORED normalized gold carries no literal: the probe must not be derived from it
    stored = {"qid": "QW-12", "text": "x", "original": {"gold": {"anchors": [HTTP("post /hooks/x7k2q9w4m1p8z3v6")]}},
              "gold": {"anchors": [HTTP("POST /hooks/{#}")]}}
    assert _leak("q12 gold post /hooks/x7k2q9w4m1p8z3v6 not understood", [stored])


def test_k41_dev_with_a_different_literal_does_not_remove_it():
    ho = _ac24(12, "some held-out question that goes on for a while here", [HTTP("post /hooks/x7k2q9w4m1p8z3v6")])
    dev = _ac24(3, "some dev question that goes on for a while there", [HTTP("post /hooks/a1b2c3d4e5f6g7h8")])
    assert _leak("gold post /hooks/x7k2q9w4m1p8z3v6", [ho], [dev])


def _qrec(n, qname, line=12, path="pkg/a.py"):
    return _ac5(n, HTTP(f"GET /route/{n}"), SYM(qname, path=path), lines=(line,))


def test_m4_planted_qname_fails_contacted_and_dev_prefix_do_not():
    ho = _qrec(1, "a.py::f", path="a.py")                     # the oracle's own literal (7 chars)
    assert _leak("held-out item a.py::f in the report", [ho]) and _leak("held-out item a.py::f.", [ho])
    assert _leak("a.py::f: missed", [ho])
    assert not _leak("held-out item a.py::f in the report", [ho], contact=[SYM("a.py::f")])
    dev0 = _qrec(2, "a.py::f.g", path="a.py")
    assert _leak("a.py::f here", [ho], [dev0]) and not _leak("only a.py::f.g here", [ho], [dev0])
    ho = _qrec(1, "pkg/a.py::f")
    assert _leak("held-out item pkg/a.py::f in the report", [ho])
    assert not _leak("held-out item pkg/a.py::f in the report", [ho], contact=[SYM("pkg/a.py::f")])     # contacted: public
    dev = _qrec(2, "pkg/a.py::f.g")
    assert _leak("pkg/a.py::f here", [ho], [dev])                                              # dev a.py::f.g does NOT remove a.py::f
    assert not _leak("only pkg/a.py::f.g here", [ho], [dev])                                   # ... nor does the probe hit inside it


@pytest.mark.parametrize("text", ["held-out item pkg/a.py::f.", "pkg/a.py::f: missed", "pkg/a.py::f..", "./pkg/a.py::f",
                                  "pkg/a.py:12.", "get /api/items/{}: missed", "(pkg/a.py::f)", "see pkg/a.py::f, then", "PKG/A.PY::F"])
def test_p23_trailing_punctuation_hits(text):
    ho = _ac5(1, HTTP("GET /api/items/{id}"), SYM("pkg/a.py::f"), lines=(12,))
    assert _leak(text, [ho]), text


@pytest.mark.parametrize("text", ["held-out item a.py::f.", "a.py::f: missed", "a.py::f..", "./a.py::f", "(a.py::f)", "A.PY::F"])
def test_p23_oracle_literal_a_py_f_hits(text):
    ho = _ac5(1, HTTP("GET /api/items/{id}"), SYM("a.py::f"), lines=(12,))
    assert _leak(text, [ho]), text


@pytest.mark.parametrize("text", ["a.py::f.g", "x/a.py::f", "a.py::fg", "a.py::f::g"])
def test_p23_oracle_literal_a_py_f_no_hit(text):
    ho = _ac5(1, HTTP("GET /api/items/{id}"), SYM("a.py::f"), lines=(12,))
    assert not _leak(text, [ho]), text


@pytest.mark.parametrize("text", ["pkg/a.py::f.g", "x/pkg/a.py::f", "pkg/a.py::fg", "pkg/a.py::f::g", "pkg/a.py::f.:g",
                                  "pkg/a.py:123", "get /api/items/{}/x", "pkg/a.py::f_x", "pkg/a.py::f-x"])
def test_p23_no_hit(text):
    ho = _ac5(1, HTTP("GET /api/items/{id}"), SYM("pkg/a.py::f"), lines=(12,))
    assert not _leak(text, [ho]), text


def test_leak_check_floor_drops_only_bare_short_probes_ep_9_4():
    ho = _ac5(1, HTTP("GET /q"), SYM("a.py::f"))                 # the oracle literal: 7 chars but NOT bare ('::')
    assert _leak("held-out item a.py::f.", [ho]) and _leak("a.py::f: missed", [ho])
    tool = _ac5(3, {"type": "contract", "kind": "mcp_tool", "keyNorm": "mcp__cypher__query"}, SYM("pkg/a.py::f"))
    assert not _leak("the query tool and the cypher query", [tool])       # bare 'query' (5 chars) stays dropped
    for bare_ok in ("a/b", "a#b", "get /q"):                                 # each non-bare marker keeps a short probe
        assert not mk._bare_short(bare_ok), bare_ok
    assert mk._bare_short("query") and mk._bare_short("a.py:12") and not mk._bare_short("abcdefgh")
    assert not mk._bare_short("a b")
    ho2 = _ac5(2, HTTP("GET /q"), SYM("pkg/a.py::f"))
    for v in (ho2["itemId"], ho2["itemId"][:12], ho2["anchorId"], ho2["impactedId"][:12]):
        assert _leak(f"xx{v}yy", [ho2]), v
    assert not _leak("nothing here at all", [ho2])


def test_leak_check_text_shingles_are_substrings_and_removed_by_dev_equality():
    q = _ac24(1, "What breaks in the storefront if I change the products route today?", [HTTP("GET /shop/api/products")])
    assert _leak("... storefront IF I change   the products ...", [q])
    dev = _ac24(2, "What breaks in the storefront if I change the products route today?", [HTTP("GET /other/one")])
    assert not _leak("... storefront if I change the products ...", [q], [dev])


def test_leak_check_output_names_probes_by_hash_only(tmp_path):
    ho = _ac5(1, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), SYM("pkg/a.py::f"))
    (tmp_path / "ho.jsonl").write_text(json.dumps(ho) + "\n")
    (tmp_path / "report.md").write_text("missed post /hooks/x7k2q9w4m1p8z3v6 and pkg/a.py::f\n")
    r = subprocess.run([PY, str(TOOLS / "match_keys.py"), "check-report", "--report", str(tmp_path / "report.md"),
                        "--heldout", str(tmp_path / "ho.jsonl")], capture_output=True, text=True)
    assert r.returncode == 1 and "hits: 2" in r.stdout and "x7k2" not in r.stdout + r.stderr and "a.py" not in r.stdout
    assert re.findall(r"hit: ([0-9a-f]{12})", r.stdout)
    (tmp_path / "report.md").write_text("a clean report with counts only\n")
    assert subprocess.run([PY, str(TOOLS / "match_keys.py"), "check-report", "--report", str(tmp_path / "report.md"),
                           "--heldout", str(tmp_path / "ho.jsonl")], capture_output=True).returncode == 0


# --- CLI pipeline -----------------------------------------------------------------------------

def test_cli_pipeline_normalize_validate_match_nearmiss_scope(tmp_path):
    kit, units = _mk_kit(tmp_path)
    ctx = tmp_path / "ctx.json"
    ctx.write_text(json.dumps(CTX))
    ka = [_kitem(1, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), _sym("pkg/a.py::f"), provider=_sym("pkg/a.py::f"), lines=(13,))]
    kb = [_kitem(1, HTTP("POST /hooks/{}"), _sym("pkg/a.py::f"), lines=(14,)),
          _kitem(2, HTTP("GET /bad"), _sym("pkg/a.py::nope"))]
    for name, rows in (("KA", ka), ("KB", kb)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = lambda *a: subprocess.run([PY, str(TOOLS / "match_keys.py"), *map(str, a)], capture_output=True, text=True)
    w = tmp_path / "work"
    for name in ("KA", "KB"):
        assert run("normalize", "--writer", name, "--in", tmp_path / f"{name}.jsonl", "--out", w / f"n{name}.jsonl",
                   "--kit-src", kit, "--context", ctx).returncode == 0
    v = run("validate", "--in", w / "nKA.jsonl", w / "nKB.jsonl", "--kit-src", kit, "--out-dir", w)
    assert v.returncode == 1 and "valid: 2" in v.stdout and "invalid: 1" in v.stdout
    assert "V1:name-not-in-file" in json.loads((w / "invalid.jsonl").read_text())["checks"]
    assert run("match", "--in", w / "valid.jsonl", "--out", w / "match.json").returncode == 0
    nm = run("nearmiss", "--match", w / "match.json", "--out", w / "nearmiss.jsonl")
    assert nm.returncode == 0 and "N2: 1" in nm.stdout
    sc = run("scope", "--match", w / "match.json", "--context", ctx, "--out", w / "scoped.json", "--out-of-scope", w / "oos.jsonl")
    assert sc.returncode == 0 and '"H":2' in sc.stdout
    first = (w / "nearmiss.jsonl").read_bytes()
    run("nearmiss", "--match", w / "match.json", "--out", w / "nearmiss.jsonl")
    assert (w / "nearmiss.jsonl").read_bytes() == first


def test_normalize_contract_strips_nfc_and_keeps_only_the_namespace_prefix():
    assert _nk("mcp_tool", "  mcp__srv__tool__x  ") == "tool__x"                 # only the first namespace is stripped
    assert _nk("mcp_tool", "plain_tool") == "plain_tool"
    assert _nk("http", "  get   /café  ") == "GET /café"
    assert mk.normalize_contract({"type": "contract", "kind": "http", "keyNorm": "GET /a", "extra": "kept"})["extra"] == "kept"
    src = {"type": "contract", "kind": "http", "keyNorm": "get /a"}
    mk.normalize_contract(src)
    assert src["keyNorm"] == "get /a"                                              # a pure function: the input is untouched


def test_py_symbols_decorated_class_span_starts_at_its_decorator():
    syms = {s["chain"]: (s["lineStart"], s["lineEnd"]) for s in mk.py_symbols("def d(c):\n    return c\n\n@d\nclass K:\n    pass\n")}
    assert syms["K"] == (4, 6)


def test_validate_contract_kind_must_be_http_or_mcp_tool(tmp_path):
    (valid, invalid, _), _ = _val(tmp_path, _sym("pkg/a.py::f"), anchor={"type": "contract", "kind": "rpc", "keyNorm": "/svc.A/B"})
    assert valid == [] and "V3:kind" in invalid[0]["checks"]
    (valid, invalid, _), _ = _val(tmp_path / "ok", _sym("pkg/a.py::f"), lines=(13,), anchor={"type": "contract", "kind": "mcp_tool", "keyNorm": "query"})
    assert len(valid) == 1


def test_nearmiss_never_pairs_records_found_by_both_strategies(tmp_path):
    f1, f2 = _sym("pkg/a.py::f"), _sym("pkg/a.py::C.x")
    both = [_rec(tmp_path, HTTP("GET /a"), f1, writer=w) for w in ("KA", "KB")]       # one item, found by both
    other = _rec(tmp_path, HTTP("GET /a"), f2, writer="KB")                             # an N1 candidate against it
    m = mk.match_records(both + [other])
    assert mk.nearmiss_pairs(m) == []                                                   # the matched record is not "unmatched"


# --- fix round (Pass 7): M-2, m-1, M-3, m-2, m-3, m-4, adjudicate ---------------------------------

def _mkcli(*a):
    return subprocess.run([PY, str(TOOLS / "match_keys.py"), *map(str, a)], capture_output=True, text=True)


def _jl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


KIRO = "kiro/.kiro/agents/x.json"
KIRO_LOC = {"type": "config", "repoId": "graphmind-ai-lab", "path": KIRO, "keyPath": "includeMcpJson"}


@pytest.mark.parametrize("val,ok", [(True, True), (False, False), ("true", False), (1, False), (None, False)])
def test_v2_kiro_include_mcp_json_must_be_true(tmp_path, val, ok):
    kit = tmp_path / "kit"
    (kit / KIRO).parent.mkdir(parents=True)
    (kit / KIRO).write_text(json.dumps({"name": "x", "includeMcpJson": val}))
    assert mk.check_locator(KIRO_LOC, kit) == ([] if ok else ["V2:includeMcpJson-not-true"])


def test_v2_include_mcp_json_rule_is_for_that_key_in_kiro_agents_only(tmp_path):
    kit = tmp_path / "kit"
    (kit / KIRO).parent.mkdir(parents=True)
    (kit / KIRO).write_text(json.dumps({"mcpServers": {"s": {}}, "includeMcpJson": False, "tools": [False]}))
    assert mk.check_locator(dict(KIRO_LOC, keyPath="mcpServers.s"), kit) == []
    assert mk.check_locator(dict(KIRO_LOC, keyPath="tools[0]"), kit) == []
    (kit / "other.json").write_text(json.dumps({"includeMcpJson": False}))
    assert mk.check_locator(dict(KIRO_LOC, path="other.json"), kit) == []


def test_config_grain_include_mcp_json_false_is_not_a_server_entry(tmp_path):
    g = lambda **kw: mk.config_grain(KIRO, "includeMcpJson", **kw)
    assert g(value=True) == ("includeMcpJson", "server") and g() == ("includeMcpJson", "server")
    for v in (False, "true", 1, None):
        assert g(value=v) == ("includeMcpJson", "contract"), v
    kit = tmp_path / "kit"
    (kit / KIRO).parent.mkdir(parents=True)
    for val, cls in ((True, "server"), (False, "contract")):
        (kit / KIRO).write_text(json.dumps({"includeMcpJson": val}))
        assert mk._norm_locator(dict(KIRO_LOC), 1, [], [], kit)["_class"] == cls


ENV_IMPACTED = {"type": "config", "path": ".env.example", "keyPath": "SERVER_URL"}
PKG_UNIT = {"type": "unit", "repoId": "graphmind-ai-lab", "rootPath": "pkg", "lang": "python"}


def test_env_locator_without_a_kit_is_refused_not_defaulted_to_contract(tmp_path):
    kit, units = _mk_kit(tmp_path)
    o = _kitem(1, PKG_UNIT, ENV_IMPACTED)
    with pytest.raises(mk.NeedsKit, match="--kit-src"):
        mk.normalize_item(o, "KA", units, None)
    rec, _ = mk.normalize_item(o, "KA", units, kit)                         # with the kit: the value is read -> server
    assert rec["anchor"]["rootPath"] == "pkg" and "_class" not in rec["impacted"]


def test_normalize_cli_env_locator_needs_kit_src_exit_2(tmp_path):
    kit, units = _mk_kit(tmp_path)
    _jl(tmp_path / "KA.jsonl", [_kitem(1, PKG_UNIT, ENV_IMPACTED)])
    _jl(tmp_path / "ctx.json", [])
    (tmp_path / "ctx.json").write_text(json.dumps(CTX | {"units": units}))
    base = ["normalize", "--writer", "KA", "--in", tmp_path / "KA.jsonl", "--out", tmp_path / "out.jsonl", "--context", tmp_path / "ctx.json"]
    r = _mkcli(*base)
    assert r.returncode == 2 and "--kit-src" in r.stderr and not (tmp_path / "out.jsonl").exists()
    r = _mkcli(*base, "--kit-src", kit)
    assert r.returncode == 0 and "records: 1" in r.stdout


def test_normalize_non_env_config_still_works_without_a_kit(tmp_path):
    _, units = _mk_kit(tmp_path)
    rec, _ = mk.normalize_item(_kitem(1, PKG_UNIT, {"type": "config", "path": ".mcp.json", "keyPath": "mcpServers.cypher.args"}),
                               "KA", units + [{"rootPath": "pkg", "lang": "python", "ports": []}], None)
    assert rec["impacted"]["keyPath"] == "mcpServers.cypher"


# --- M-3: the plumbing of sample (CLI) and normalize ----------------------------------------------

def _ids_of(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def test_sample_cli_draws_use_their_own_tags_languages_and_the_eval_commit(tmp_path):
    commit = "F" * 40
    m, qw = _spot_match(_full()), _qw_sample()
    (tmp_path / "match.json").write_text(json.dumps(m))
    _jl(tmp_path / "QW.jsonl", qw)
    r = _mkcli("sample", "--match", tmp_path / "match.json", "--qw", tmp_path / "QW.jsonl", "--eval-commit", commit, "--out-dir", tmp_path / "w")
    assert r.returncode == 0
    spot = [x for x in _ids_of(tmp_path / "w" / "spot.jsonl") if "itemId" in x]
    pt, en = _ids_of(tmp_path / "w" / "ptgold.jsonl"), _ids_of(tmp_path / "w" / "engold.jsonl")
    # the library at the SAME commit, with each file's own language and tag
    assert [x["itemId"] for x in spot] == [x["itemId"] for x in mk.spot_sample(m, commit)]
    assert pt == mk.lang_gold_sample(qw, "pt", "ccg-p0f-ptgold-v1", commit)
    assert en == mk.lang_gold_sample(qw, "en", "ccg-p0f-engold-v1", commit)
    # ... and these fixtures can tell the wrong plumbing apart: another commit, tag or language gives other rows
    assert [x["itemId"] for x in spot] != [x["itemId"] for x in mk.spot_sample(m, "0" * 40)]
    assert pt != mk.lang_gold_sample(qw, "pt", "ccg-p0f-engold-v1", commit) and pt != mk.lang_gold_sample(qw, "pt", "ccg-p0f-ptgold-v1", "0" * 40)
    assert en != mk.lang_gold_sample(qw, "en", "ccg-p0f-ptgold-v1", commit) and en != mk.lang_gold_sample(qw, "en", "ccg-p0f-engold-v1", "0" * 40)
    assert all(q["lang"] == "pt" for q in pt) and all(q["lang"] == "en" for q in en) and pt != en


def test_normalize_serves_are_normalized_masked_deduped_and_kept(tmp_path):
    kit, units = _mk_kit(tmp_path)
    serves = [HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), HTTP("post /hooks//a1b2c3d4e5f6g7h8/"),
              {"type": "contract", "kind": "mcp_tool", "keyNorm": "mcp__cypher__query"}, HTTP("POST /hooks/x7k2q9w4m1p8z3v6")]
    o = _kitem(1, HTTP("GET /a"), _sym("pkg/a.py::f"), provider=_sym("pkg/a.py::f"), serves=serves)
    rec, log = mk.normalize_item(o, "KA", units, kit)
    assert [(s["kind"], s["keyNorm"]) for s in rec["serves"]] == [("http", "POST /hooks/{#}"), ("http", "POST /hooks/{#}/"), ("mcp_tool", "query")]
    assert len(rec["original"]["serves"]) == 4                                     # the writer's own list is untouched
    dup = _kitem(1, HTTP("GET /a"), _sym("pkg/a.py::f"), serves=[HTTP("GET /s/{id}"), HTTP("GET /s/:x")])
    assert [s["keyNorm"] for s in mk.normalize_item(dup, "KA", units, kit)[0]["serves"]] == ["GET /s/{}"]


# --- m-2: the shortfall marker --------------------------------------------------------------------

def test_spot_shortfall_record_only_when_short():
    assert mk.spot_shortfall_record(mk.spot_sample(_spot_match(_full()), "E" * 40)) is None
    small = mk.spot_sample(_spot_match({("http", "singleton"): 2, ("flow", "doubleton"): 1}), "E" * 40)
    assert mk.spot_shortfall_record(small) == {"record": "shortfall", "expected": 40, "got": 3, "short": 37}


def test_sample_cli_says_so_in_spot_jsonl_and_stdout_when_short(tmp_path):
    (tmp_path / "match.json").write_text(json.dumps(_spot_match({("http", "singleton"): 2, ("flow", "doubleton"): 1})))
    _jl(tmp_path / "QW.jsonl", _qw_sample())
    run = lambda d: _mkcli("sample", "--match", tmp_path / "match.json", "--qw", tmp_path / "QW.jsonl", "--eval-commit", "E" * 40, "--out-dir", tmp_path / d)
    r = run("w")
    rows = _ids_of(tmp_path / "w" / "spot.jsonl")
    assert r.returncode == 0 and "spot: 3\n" in r.stdout and "spot-shortfall: 37" in r.stdout
    assert rows[-1] == {"record": "shortfall", "expected": 40, "got": 3, "short": 37} and sum(1 for x in rows if "itemId" in x) == 3
    (tmp_path / "match.json").write_text(json.dumps(_spot_match(_full())))
    r = run("w2")
    assert "spot-shortfall: 0" in r.stdout and all("itemId" in x for x in _ids_of(tmp_path / "w2" / "spot.jsonl"))


# --- m-3: nothing from a stored entry on stderr ---------------------------------------------------

def test_renormalize_check_names_a_failing_entry_by_stored_id_only(tmp_path):
    bad = {"id": "a" * 32, "locator": {"type": "contract", "kind": "http", "keyNorm": "SECRETVERB /x"}, "form": "provider"}
    good = _stored(K33)
    _jl(tmp_path / "s.jsonl", [good, bad, "not-a-dict", {"locator": None}])
    r = _mkcli("renormalize-check", "--in", tmp_path / "s.jsonl")
    assert r.returncode == 1 and "SECRETVERB" not in r.stdout + r.stderr
    assert "checked: 4" in r.stdout and f"changed: {'a' * 32} -> (does not normalize)" in r.stdout
    assert mk.renormalize_check([bad])["diffs"] == [{"stored": "a" * 32, "now": None}]


def test_cli_error_paths_print_no_stored_text(tmp_path):
    (tmp_path / "bad.jsonl").write_text("SECRETTEXT\n{}\n")
    r = _mkcli("renormalize-check", "--in", tmp_path / "bad.jsonl")
    assert r.returncode == 4 and "SECRETTEXT" not in r.stderr and "line 1" in r.stderr
    kit, units = _mk_kit(tmp_path)
    _jl(tmp_path / "KA.jsonl", [_kitem(1, PKG_UNIT, ENV_IMPACTED)])
    (tmp_path / "ctx.json").write_text(json.dumps({"units": [{"rootPath": "pkg", "lang": "python", "ports": ["SECRETPORT"]}]}))
    r = _mkcli("normalize", "--writer", "KA", "--in", tmp_path / "KA.jsonl", "--out", tmp_path / "o.jsonl", "--kit-src", kit, "--context", tmp_path / "ctx.json")
    assert r.returncode == 4 and "SECRETPORT" not in r.stderr and "ValueError" in r.stderr


# --- m-4: the match.json contract for capture.py --------------------------------------------------

def test_match_json_carries_t_per_stratum_and_the_contract_is_documented(tmp_path):
    a = _rec(tmp_path, HTTP("GET /a"), _sym("pkg/a.py::f"), writer="KA")
    k2 = _rec(tmp_path, HTTP("GET /b"), _sym("pkg/a.py::f"), writer="KA2", seq=2)
    m = mk.match_records([a])
    assert m["t"] == {s: 2 for s in mk.STRATA}
    m = mk.match_records([a, k2])
    assert m["t"]["http"] == 3 and all(m["t"][s] == 2 for s in mk.STRATA if s != "http")
    assert mk.match_records([a], ka2_strata=["flow"])["t"]["flow"] == 3
    assert [g["history"] for g in m["records"]] == ["100", "010"]                  # (KA, KA2, KB) order
    for needle in ("match.json", "inScope", "scope", "capture.py", "``t``", "history", "KA2", "adjudicate"):
        assert needle in mk.__doc__, needle


def test_match_cli_ka2_strata_and_scope_keeps_t_and_marks_every_record(tmp_path):
    _, units = _mk_kit(tmp_path)
    r = _rec(tmp_path, HTTP("GET /a"), _sym("pkg/a.py::f"), writer="KA")
    _jl(tmp_path / "v.jsonl", [r])
    assert _mkcli("match", "--in", tmp_path / "v.jsonl", "--out", tmp_path / "m.json", "--ka2-strata", "scripts", "wiring").returncode == 0
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["t"] == {"http": 2, "mcp": 2, "scripts": 3, "wiring": 3, "flow": 2}
    (tmp_path / "ctx.json").write_text(json.dumps(CTX))
    assert _mkcli("scope", "--match", tmp_path / "m.json", "--context", tmp_path / "ctx.json", "--out", tmp_path / "s.json", "--out-of-scope", tmp_path / "o.jsonl").returncode == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["t"] == m["t"] and all(g["inScope"] is True for g in s["records"])


# --- adjudicate (EP §6.3, §13.4) ------------------------------------------------------------------

def _adj_fixture(tmp_path, ka_writer="KA"):
    a = _rec(tmp_path, HTTP("POST /hooks/x7k2q9w4m1p8z3v6"), _sym("pkg/a.py::f"), writer=ka_writer)
    b = _rec(tmp_path, HTTP("POST /hooks/{}"), _sym("pkg/a.py::f"), writer="KB")
    m = mk.match_records([a, b])
    return a, b, m, mk.nearmiss_pairs(m)


def _vd(pairs, verdict="same", keep="x"):
    return [{"pairId": p["pairId"], "verdict": verdict, "keep": keep} for p in pairs]


def test_adjudicate_same_makes_one_matched_record_and_keeps_the_keep_side(tmp_path):
    a, b, m, pairs = _adj_fixture(tmp_path)
    assert len(pairs) == 1 and pairs[0]["x"]["itemId"] == a["itemId"]
    for keep, survivor, gone in (("x", a, b), ("y", b, a)):
        out, log, summary = mk.apply_verdicts(m, pairs, _vd(pairs, keep=keep))
        (g,) = out["records"]
        assert g["itemId"] == survivor["itemId"] and g["anchorId"] == survivor["anchorId"] and g["absorbed"] == [gone["itemId"]]
        assert g["sessions"] == ["KA", "KB"] and g["history"] == "101" and g["strategies"] == ["consumer-first", "provider-first"]
        assert len(g["members"]) == 2
        assert log == [{"pairId": pairs[0]["pairId"], "verdict": "same", "keep": keep}]
        assert summary == {"pairs": 1, "verdicts": 1, "same": 1, "different": 0, "unadjudicated": 0}
    assert len(m["records"]) == 2                                              # the input match was not changed in place
    assert mk.nearmiss_pairs(out) == []                                        # a merged record is found by both strategies


def test_adjudicate_ka2_pair_history_is_011(tmp_path):
    _, _, m, pairs = _adj_fixture(tmp_path, ka_writer="KA2")
    (g,) = mk.apply_verdicts(m, pairs, _vd(pairs))[0]["records"]
    assert g["history"] == "011" and g["sessions"] == ["KA2", "KB"]


def test_adjudicate_different_and_unadjudicated_pairs_count_as_different(tmp_path):
    _, _, m, pairs = _adj_fixture(tmp_path)
    for verdicts, want in (([], {"different": 0, "unadjudicated": 1}), (_vd(pairs, "different", "x"), {"different": 1, "unadjudicated": 0})):
        out, log, summary = mk.apply_verdicts(m, pairs, verdicts)
        assert len(out["records"]) == 2 and out["records"] == m["records"]
        assert summary["same"] == 0 and {k: summary[k] for k in want} == want
        assert [e["verdict"] for e in log] == [v["verdict"] for v in verdicts]


def _fan_in(tmp_path):
    x = _rec(tmp_path, HTTP("GET /a"), _sym("pkg/a.py::f"), writer="KA")
    y1 = _rec(tmp_path, HTTP("GET /a"), _sym("pkg/a.py::C.x"), writer="KB")
    y2 = _rec(tmp_path, HTTP("GET /a"), _sym("pkg/a.py::C.p.<get>"), writer="KB", seq=2)
    m = mk.match_records([x, y1, y2])
    return m, mk.nearmiss_pairs(m)


def test_adjudicate_rejects_a_record_that_is_same_in_two_pairs_and_applies_nothing(tmp_path):
    m, pairs = _fan_in(tmp_path)
    assert len(pairs) == 2 and {p["rule"] for p in pairs} == {"N1"}
    with pytest.raises(mk.Inconsistent) as e:
        mk.apply_verdicts(m, pairs, _vd(pairs))
    assert sorted(e.value.args[0]) == sorted(("record-same-twice", p["pairId"]) for p in pairs)
    out, _, s = mk.apply_verdicts(m, pairs, [_vd(pairs)[0], dict(_vd(pairs)[1], verdict="different")])    # one same + one different is fine
    assert s["same"] == 1 and len(out["records"]) == 2


@pytest.mark.parametrize("mut,code", [
    (lambda v, p: [dict(v[0], pairId="ffffffffffff")], "unknown-pair"),
    (lambda v, p: [v[0], v[0]], "repeated-pair"),
    (lambda v, p: [dict(v[0], verdict="maybe")], "bad-verdict"),
    (lambda v, p: [dict(v[0], keep="z")], "bad-keep"),
    (lambda v, p: [{k: x for k, x in v[0].items() if k != "keep"}], "bad-keep"),
])
def test_adjudicate_rejects_malformed_verdicts(tmp_path, mut, code):
    _, _, m, pairs = _adj_fixture(tmp_path)
    with pytest.raises(mk.Inconsistent) as e:
        mk.apply_verdicts(m, pairs, mut(_vd(pairs), pairs))
    assert code in {c for c, _ in e.value.args[0]}


def test_adjudicate_rejects_a_pair_whose_items_are_not_in_the_match(tmp_path):
    _, _, m, pairs = _adj_fixture(tmp_path)
    stale = dict(m, records=m["records"][:1])
    with pytest.raises(mk.Inconsistent) as e:
        mk.apply_verdicts(stale, pairs, _vd(pairs))
    assert ("item-not-in-match", pairs[0]["pairId"]) in e.value.args[0]


def test_adjudicate_cli_applies_logs_and_rejects(tmp_path):
    a, b, m, pairs = _adj_fixture(tmp_path)
    (tmp_path / "match.json").write_text(mk.canon(m) + "\n")
    _jl(tmp_path / "nm.jsonl", pairs)
    _jl(tmp_path / "v.jsonl", _vd(pairs, keep="y"))
    r = _mkcli("adjudicate", "--match", tmp_path / "match.json", "--nearmiss", tmp_path / "nm.jsonl", "--verdicts", tmp_path / "v.jsonl",
             "--out", tmp_path / "adj.json", "--log", tmp_path / "work" / "adjudication.jsonl")
    assert r.returncode == 0 and "same: 1" in r.stdout and "unadjudicated: 0" in r.stdout and "records: 1" in r.stdout
    assert "POST" not in r.stdout + r.stderr
    (g,) = json.loads((tmp_path / "adj.json").read_text())["records"]
    assert g["itemId"] == b["itemId"] and g["absorbed"] == [a["itemId"]]
    assert _ids_of(tmp_path / "work" / "adjudication.jsonl") == [{"pairId": pairs[0]["pairId"], "verdict": "same", "keep": "y"}]
    # a rejected list: exit 1, ids only on stderr, nothing written
    m2, pairs2 = _fan_in(tmp_path / "fan")
    (tmp_path / "m2.json").write_text(mk.canon(m2) + "\n")
    _jl(tmp_path / "nm2.jsonl", pairs2)
    _jl(tmp_path / "v2.jsonl", _vd(pairs2))
    r = _mkcli("adjudicate", "--match", tmp_path / "m2.json", "--nearmiss", tmp_path / "nm2.jsonl", "--verdicts", tmp_path / "v2.jsonl",
             "--out", tmp_path / "adj2.json", "--log", tmp_path / "adj2.log")
    assert r.returncode == 1 and "record-same-twice" in r.stderr and pairs2[0]["pairId"] in r.stderr
    assert not (tmp_path / "adj2.json").exists() and not (tmp_path / "adj2.log").exists()
    assert subprocess.run([PY, str(TOOLS / "match_keys.py"), "adjudicate", "--help"], capture_output=True).returncode == 0


def test_adjudicate_then_scope_keeps_the_merged_record(tmp_path):
    _, _, m, pairs = _adj_fixture(tmp_path)
    out, _, _ = mk.apply_verdicts(m, pairs, _vd(pairs))
    out, oos, summary = mk.scope_records(out, CTX["units"])
    assert oos == [] and summary["H_all"] == 1 and out["records"][0]["inScope"] is True
