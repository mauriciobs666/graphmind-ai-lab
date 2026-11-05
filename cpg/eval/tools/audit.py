#!/usr/bin/env python3
"""audit.py — the writer-session audit (S48, P0-41; plan §6.5 is the specification, eval-protocol §13.3).

    audit.py --eval-dir ~/.local/share/ccg-eval/<E> [--projects-dir ~/.claude/projects]
             [--tools Read,Grep,Glob,Write,Edit] [--session ID ...] [--accepted-agents FILE]
             [--settings FILE] [--agents FILE] [--details FILE]

Audits every session in ``<eval-dir>/runs/index.jsonl`` (or the ``--session`` ones), all attempts
together. Sources: the per-attempt ``runs/<session>.<n>.stream.jsonl`` logs (primary) and the
session transcript ``<projects-dir>/*/<session>.jsonl`` (secondary, found by session id). The role
root is ``<eval-dir>/roles/<role>-<stratum>`` (from the index), never read from the log under audit.

Checks (§6.5): the init event of every attempt; every ``tool_use`` (name allowed, every path
resolved against the root with ``..``/symlinks normalised, writes strictly inside ``out/``, no
``mcp__*``); refused calls are counted but do not void, any other violating call voids; the
stream/transcript id cross-check over the union of attempts (a line cut off by ``timeout`` at the
end of an attempt's log is "stream-truncated": the call is checked from the transcript alone);
a missing transcript or attempt log; index <-> logs.

Output (stdout, JSON): counts and violation TYPES only, no paths and no tool inputs (§13.3).
``--details FILE`` additionally writes (session, attempt, tool, tool_use id, type, refused) lines
for GC's own investigation. Exit: 0 every session clean · 1 any session void · 2 usage.

Assumptions not yet observed (S54 confirms): a call is "refused" only on positive evidence: a
``permission_decision``/``permissionDecision`` whose decision is not accept/allow, or ``is_error``
with a result text that STARTS with a denial message (``guard-paths:``, ``ccg path guard:``, ...;
never a word elsewhere in it). Anything else about a violating call counts as successful (voids).
Plugins: every one must have ``path == "builtin"``; with ``--accepted-plugins FILE`` the names must
equal that list and each source end ``@builtin``; without it the report says ``unpinned`` (as for
agents, checked only with ``--accepted-agents``).
Glob/Grep patterns: braces are expanded, and ``[class]``/``?`` are read as ``/``, ``~`` or ``.`` where
they could be; any reading that leaves the root is an escape. A line cut off by ``timeout`` excuses
the id mismatch only for the transcript message that line was (message id recovered from its prefix;
else the first transcript message after the attempt's last stream-matched one), never for a message
a later stream-matched message of the attempt follows. Stream and transcript inputs must be equal.
No sessions to audit is void (``no-sessions``); a malformed index start line is exit 2.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_TOOLS = "Read,Grep,Glob,Write,Edit"
WRITE_TOOLS = {"Write", "Edit", "NotebookEdit", "MultiEdit"}
PATH_KEYS = ("file_path", "notebook_path", "path")
# A denial MESSAGE, anchored at the start of the result text (never a word anywhere in it: a path such
# as /etc/permission-x in a "File does not exist" error must not read as a refusal). guard-paths.sh
# messages start "guard-paths:"; settings.json's fail-closed hook text starts "ccg path guard:".
DENIAL_RE = re.compile(r"\s*(guard-paths:|ccg path guard:|Permission to use \S+ has been denied|"
                       r"PreToolUse:\S+ hook (blocking error|denied))", re.IGNORECASE)
MSG_ID_RE = re.compile(r'"id"\s*:\s*"(msg_[A-Za-z0-9]+)"')
ASSISTANT_HEAD_RE = re.compile(r'^\s*\{\s*"type"\s*:\s*"assistant"')
MAX_PATTERN_VARIANTS = 512
SPECIALS = ("/", "~", ".")


# ----------------------------------------------------------------------------- reading logs

def read_jsonl(path: Path):
    """(records, truncated_tail, corrupt_lines). The last non-empty line, if unparseable, is the
    ``timeout``-truncated tail; an unparseable earlier line is corruption."""
    recs, corrupt, tail = [], 0, None
    lines = [l for l in path.read_bytes().decode("utf-8", "replace").split("\n") if l.strip()]
    for i, line in enumerate(lines):
        try:
            recs.append(json.loads(line))
        except ValueError:
            if i == len(lines) - 1:
                tail = line
            else:
                corrupt += 1
    return recs, tail, corrupt


def canon_json(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- path rules

def _inside(p: str, base: str) -> bool:
    return p == base or p.startswith(base.rstrip(os.sep) + os.sep)


def _resolve(root: str, p) -> str | None:
    """Absolute, symlink-resolved path of ``p`` against ``root``; None for a malformed path or a
    home-relative one (both are outside by definition)."""
    if not isinstance(p, str) or not p or "\0" in p or p.startswith("~"):
        return None
    return os.path.realpath(os.path.join(root, p))


def _expand_braces(pat: str, limit: int = MAX_PATTERN_VARIANTS) -> list[str] | None:
    """Brace-free alternatives of ``pat``; None as soon as there would be more than ``limit`` of them
    (counted while expanding, so a hostile pattern costs O(limit), never 2**groups)."""
    out: list[str] = []
    stack = [pat]
    while stack:
        cur = stack.pop()
        m = re.search(r"\{([^{}]*)\}", cur)
        if not m:
            out.append(cur)
            if len(out) > limit:
                return None
            continue
        alts = m.group(1).split(",")
        stack.extend(cur[:m.start()] + alt + cur[m.end():] for alt in alts)
    return out


def _class_may_match(body: str, ch: str) -> bool:
    if body[:1] in ("!", "^"):
        return True                                  # a negated class can match anything: be conservative
    chars, i = set(), 0
    while i < len(body):
        if i + 2 < len(body) and body[i + 1] == "-":
            if body[i] <= ch <= body[i + 2]:
                return True
            i += 3
        else:
            chars.add(body[i])
            i += 1
    return ch in chars


def _wildcard_variants(pat: str) -> list[str] | None:
    """Concrete readings of a brace-free pattern in which every ``[class]`` and ``?`` that could stand
    for ``/``, ``~`` or ``.`` does (plus a neutral reading). ``*`` stays, as it cannot cross a ``/``.
    None if there are too many to enumerate (the caller treats that as an escape)."""
    parts: list[list[str]] = []
    i = 0
    while i < len(pat):
        c = pat[i]
        j = -1
        if c == "[":
            k = i + 1 + (1 if pat[i + 1:i + 2] in ("!", "^") else 0)
            k += 1 if pat[k:k + 1] == "]" else 0
            j = pat.find("]", k)
        if c == "[" and j != -1:
            body = pat[i + 1:j]
            parts.append(["x"] + [ch for ch in SPECIALS if _class_may_match(body, ch)])
            i = j + 1
        elif c == "?":
            parts.append(["x", *SPECIALS])
            i += 1
        else:
            parts.append([c])
            i += 1
    n = 1
    for p in parts:
        n *= len(p)
        if n > MAX_PATTERN_VARIANTS:
            return None
    return ["".join(v) for v in itertools.product(*parts)]


def _literal_base(base_dir: str, pat: str) -> str | None:
    """Resolved literal prefix (components before the first wildcard) of one concrete pattern."""
    if pat.startswith("~"):
        return None
    comps = pat.replace("\\", "/").split("/")
    if ".." in comps:
        return None
    lit = []
    for c in comps:
        if any(ch in c for ch in "*?[{"):
            break
        lit.append(c)
    prefix = "/".join(lit) if lit else "."
    if pat.startswith("/"):
        prefix = "/" + prefix.lstrip("/") if lit else "/"
    return os.path.realpath(os.path.join(base_dir, prefix))


def _pattern_escapes_one(root: str, base_dir: str, pat: str) -> bool:
    budget = MAX_PATTERN_VARIANTS            # readings for the whole pattern: braces x wildcards
    alts = _expand_braces(pat, budget)
    if alts is None:
        return True
    for alt in alts:
        variants = _wildcard_variants(alt)
        if variants is None:
            return True
        budget -= len(variants)
        if budget < 0:
            return True
        for v in variants:
            b = _literal_base(base_dir, v)
            if b is None or not _inside(b, root):
                return True
    return False


def pattern_escapes(root: str, base_dir: str, pat) -> bool:
    """True if any reading of the glob ``pat`` (braces expanded; ``[class]`` and ``?`` read as ``/``,
    ``~`` or ``.`` where they could be; and with ``\\x`` read as ``x``, since a backslash may escape a
    dot) can name something outside ``root``. Braces and wildcard readings share one budget of
    ``MAX_PATTERN_VARIANTS``; exceeding it is an escape (fail closed). Known limit: the readings are
    enumerated over the whole pattern, so an honest glob with many ``?``/classes (``kit-src/**/????-??-??.json``)
    exceeds the budget and is voided; narrowing that needs path-depth reasoning, not done."""
    if not isinstance(pat, str) or "\0" in pat:
        return True
    unescaped = re.sub(r"\\(.)", r"\1", pat, flags=re.DOTALL)
    return any(_pattern_escapes_one(root, base_dir, r) for r in dict.fromkeys((pat, unescaped)))


def classify_call(name: str, inp, root: str, allowed: set[str]) -> list[str]:
    """Violation types of one call, independent of its result."""
    if not isinstance(name, str):
        return ["tool-not-allowed"]
    if name.startswith("mcp__"):
        return ["mcp-call"]
    if name not in allowed:
        return ["tool-not-allowed"]
    inp = inp if isinstance(inp, dict) else {}
    root = os.path.realpath(root)
    types: list[str] = []
    present = [k for k in PATH_KEYS if k in inp and inp[k] is not None]
    if name in WRITE_TOOLS | {"Read"} and not present:
        return ["path-malformed"]
    resolved: dict[str, str] = {}
    for k in present:
        r = _resolve(root, inp[k])
        if r is None:
            types.append("path-outside-root" if isinstance(inp[k], str) and inp[k] else "path-malformed")
        elif not _inside(r, root):
            types.append("path-outside-root")
        else:
            resolved[k] = r
    base_dir = resolved.get("path", root)
    pk = {"Glob": "pattern", "Grep": "glob"}.get(name)
    if pk and inp.get(pk) is not None:
        if pattern_escapes(root, base_dir, inp[pk]):
            types.append("path-outside-root")
    if name in WRITE_TOOLS and not types:
        out = os.path.realpath(os.path.join(root, "out"))
        for r in resolved.values():
            if not _inside(out, root) or r == out or not _inside(r, out):
                types.append("write-outside-out")
                break
    return sorted(set(types))


# ----------------------------------------------------------------------------- call extraction

def _is_refused(decision, is_error, content) -> bool:
    if isinstance(decision, dict):
        d = decision.get("decision")
        if d is not None and d not in ("accept", "allow"):
            return True
    if is_error:
        return isinstance(content, str) and bool(DENIAL_RE.match(content))
    return False


def extract_stream(recs):
    """calls {id: (name, input)}, results {id: refused_bool}, inits [init events], mids {assistant message ids}."""
    calls, results, inits, mids = {}, {}, [], set()
    for r in recs:
        t = r.get("type")
        if t == "system" and r.get("subtype") == "init":
            inits.append(r)
        elif t == "assistant":
            mids.add((r.get("message") or {}).get("id") or r.get("uuid"))
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls[b.get("id")] = (b.get("name"), b.get("input"))
        elif t == "user":
            meta = {m.get("id"): m.get("permission_decision") for m in r.get("tool_result_meta") or []
                    if isinstance(m, dict)}
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    i = b.get("tool_use_id")
                    results[i] = _is_refused(meta.get(i), b.get("is_error"), b.get("content"))
    return calls, results, inits, mids


def extract_transcript(recs):
    """calls, results, msgs {assistant message id: [tool_use ids]} in order of first appearance."""
    calls, results, msgs = {}, {}, {}
    for r in recs:
        t = r.get("type")
        if t == "assistant":
            ids = msgs.setdefault((r.get("message") or {}).get("id") or r.get("uuid"), [])
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls[b.get("id")] = (b.get("name"), b.get("input"))
                    ids.append(b.get("id"))
        elif t == "user" and isinstance(r.get("message"), dict):
            c = r["message"].get("content")
            for b in c if isinstance(c, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = _is_refused(r.get("permissionDecision"),
                                                                b.get("is_error"), b.get("content"))
    return calls, results, msgs


# ----------------------------------------------------------------------------- init checks

def check_init(init, *, session, root, allowed, index_rec, accepted_agents, accepted_plugins=None) -> list[str]:
    bad = []
    if init.get("mcp_servers") != []:
        bad.append("init-mcp")
    if sorted(init.get("tools") or []) != sorted(allowed) or len(init.get("tools") or []) != len(allowed):
        bad.append("init-tools")
    plugins = init.get("plugins") or []
    if any(not (isinstance(p, dict) and p.get("path") == "builtin"
                and (accepted_plugins is None or str(p.get("source", "")).endswith("@builtin")))
           for p in plugins):
        bad.append("init-plugins")
    elif accepted_plugins is not None and sorted(p.get("name") for p in plugins) != sorted(accepted_plugins):
        bad.append("init-plugins")
    if init.get("skills"):
        bad.append("init-skills")
    if init.get("slash_commands"):
        bad.append("init-slash-commands")
    if accepted_agents is not None and sorted(init.get("agents") or []) != sorted(accepted_agents):
        bad.append("init-agents")
    alias = index_rec.get("model")
    model = init.get("model") or ""
    if alias and not (model == alias or alias in model):
        bad.append("init-model")
    if index_rec.get("cli_version") and init.get("claude_code_version") != index_rec["cli_version"]:
        bad.append("init-version")
    if init.get("session_id") != session:
        bad.append("init-session")
    cwd = init.get("cwd")
    if not isinstance(cwd, str) or os.path.realpath(cwd) != os.path.realpath(root):
        bad.append("init-cwd")
    return bad


# ----------------------------------------------------------------------------- one session

def _bump(d: dict, k: str, n: int = 1):
    d[k] = d.get(k, 0) + n


def audit_session(session: str, starts: list[dict], *, eval_dir: Path, projects_dir: Path,
                  allowed: set[str], accepted_agents, settings_sha, agents_sha, details: list,
                  accepted_plugins=None) -> dict:
    runs = eval_dir / "runs"
    rec0 = starts[0]
    root = str(eval_dir / "roles" / f"{rec0['role']}-{rec0['stratum']}")
    void: dict = {}
    refused: dict = {}

    def violate(t, attempt=None, tool=None, tid=None):
        _bump(void, t)
        details.append({"session": session, "attempt": attempt, "type": t, "tool": tool, "id": tid, "refused": False})

    # index consistency across attempts
    for key in ("cli_version", "bin_sha256", "settings_sha256", "agents_sha256", "model"):
        if len({s.get(key) for s in starts}) > 1:
            violate("hash-drift" if key.endswith("sha256") or key == "cli_version" else "init-model")
    if settings_sha and any(s.get("settings_sha256") != settings_sha for s in starts):
        violate("hash-drift")
    if agents_sha and any(s.get("agents_sha256") != agents_sha for s in starts):
        violate("hash-drift")

    s_calls: dict = {}
    s_results: dict = {}
    tails: list[tuple[str, set]] = []          # (cut-off tail text, message ids the attempt's stream did parse)
    all_matched: set = set()
    n_trunc = 0
    for s in starts:
        n = s["n"]
        log = runs / s.get("stream", f"{session}.{n}.stream.jsonl")
        if not log.is_file():
            violate("attempt-log-missing", n)
            continue
        recs, tail, corrupt = read_jsonl(log)
        if corrupt:
            violate("stream-corrupt", n)
        calls, results, inits, mids = extract_stream(recs)
        all_matched |= mids
        if tail is not None:
            tails.append((tail, mids))
            n_trunc += 1
        s_calls.update({k: (v, n) for k, v in calls.items()})
        s_results.update(results)
        if not inits:
            violate("init-missing", n)
        for init in inits:
            for t in check_init(init, session=session, root=root, allowed=allowed, index_rec=s,
                                accepted_agents=accepted_agents, accepted_plugins=accepted_plugins):
                violate(t, n)

    # transcript, found by session id
    t_files = sorted(glob.glob(str(projects_dir / "*" / f"{session}.jsonl")))
    t_calls: dict = {}
    t_results: dict = {}
    t_msgs: dict = {}
    if not t_files:
        violate("transcript-missing")
    for f in t_files:
        recs, _tail, _c = read_jsonl(Path(f))
        c, r, m = extract_transcript(recs)
        t_calls.update(c)
        t_results.update(r)
        for k, v in m.items():
            t_msgs.setdefault(k, []).extend(x for x in v if x not in t_msgs.get(k, []))

    # cross-check over the union of attempts
    # A call whose only stream-side record is the line `timeout` cut off may be excused from the id
    # mismatch, and only if it belongs to the transcript assistant message that line was: found by
    # the message id recovered from the line's prefix; if the prefix lost it, the first transcript
    # message after the attempt's last stream-matched message. Never a message that a later
    # stream-matched message of the same attempt follows, and never a stream-matched message.
    excusable: set = set()
    order = list(t_msgs)
    for tail, mids in tails:
        if not ASSISTANT_HEAD_RE.match(tail):
            continue
        pos_last = max((order.index(m) for m in mids if m in t_msgs), default=-1)
        found = MSG_ID_RE.search(tail)
        if found:
            cand = found.group(1) if found.group(1) in t_msgs and order.index(found.group(1)) > pos_last else None
        else:
            cand = order[pos_last + 1] if pos_last + 1 < len(order) else None
        if cand is not None and cand not in all_matched:
            excusable.update(t_msgs[cand])
    truncated_calls = 0
    final_calls: dict = {}
    for i, ((name, inp), n) in s_calls.items():
        final_calls[i] = (name, inp, n, None)
    if t_files:
        for i, (name, inp) in t_calls.items():
            if i in s_calls:
                if s_calls[i][0][0] != name:
                    violate("stream-transcript-mismatch", s_calls[i][1], name, i)
                elif canon_json(s_calls[i][0][1]) != canon_json(inp):
                    violate("stream-transcript-mismatch", s_calls[i][1], name, i)
                    final_calls[i] = final_calls[i][:3] + (inp,)     # judge BOTH inputs
                continue
            if i in excusable:
                truncated_calls += 1
            else:
                violate("stream-transcript-mismatch", None, name, i)
            final_calls[i] = (name, inp, None, None)
        for i in s_calls:
            if i not in t_calls:
                violate("stream-transcript-mismatch", s_calls[i][1], s_calls[i][0][0], i)

    # per-call checks (stream record where there is one, transcript otherwise)
    for i, (name, inp, n, alt) in final_calls.items():
        types = classify_call(name, inp, root, allowed)
        if alt is not None:
            types = sorted(set(types) | set(classify_call(name, alt, root, allowed)))
        if not types:
            continue
        ref = s_results[i] if i in s_results else t_results.get(i, False)
        for t in types:
            if ref:
                _bump(refused, t)
            else:
                _bump(void, t)
            details.append({"session": session, "attempt": n, "type": t, "tool": name, "id": i, "refused": ref})

    return {"session": session, "attempts": len(starts), "attemptsTruncated": n_trunc,
            "toolCalls": len(final_calls), "streamTruncatedCalls": truncated_calls,
            "refused": dict(sorted(refused.items())), "void": dict(sorted(void.items())),
            "verdict": "void" if void else "clean"}


# ----------------------------------------------------------------------------- the run

def audit_run(eval_dir: Path, projects_dir: Path, tools: str = DEFAULT_TOOLS, sessions=None,
              accepted_agents=None, settings: Path | None = None, agents: Path | None = None,
              accepted_plugins=None):
    """Raises ValueError for a malformed index start line (the CLI maps it to exit 2)."""
    allowed = {t for t in tools.split(",") if t}
    runs = eval_dir / "runs"
    details: list = []
    by_session: dict[str, list] = {}
    idx = runs / "index.jsonl"
    torn = 0
    if idx.is_file():
        for line in idx.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                torn += 1
                continue
            if r.get("event") == "start":
                if not (isinstance(r.get("session"), str) and isinstance(r.get("n"), int)
                        and not isinstance(r.get("n"), bool) and isinstance(r.get("role"), str)
                        and isinstance(r.get("stratum"), str)):
                    raise ValueError("runs/index.jsonl: a start line lacks session, n, role or stratum")
                by_session.setdefault(r["session"], []).append(r)
    reports = []
    for sid in sorted(by_session):
        if sessions and sid not in sessions:
            continue
        starts = sorted(by_session[sid], key=lambda s: s["n"])
        reports.append(audit_session(
            sid, starts, eval_dir=eval_dir, projects_dir=projects_dir, allowed=allowed,
            accepted_agents=accepted_agents, accepted_plugins=accepted_plugins,
            settings_sha=sha256_file(settings) if settings else None,
            agents_sha=sha256_file(agents) if agents else None, details=details))
    run_void: dict = {}
    if not reports and not sessions:
        _bump(run_void, "no-sessions")             # nothing was audited: that is not a clean audit
    for sid in (sessions or []):
        if sid not in by_session:
            _bump(run_void, "session-not-in-index")
    # every stream log on disk must belong to an indexed (session, n)
    indexed = {(s["session"], s["n"]) for ss in by_session.values() for s in ss}
    for f in sorted(runs.glob("*.stream.jsonl")) if runs.is_dir() else []:
        m = re.fullmatch(r"(.+)\.(\d+)\.stream\.jsonl", f.name)
        if m and (m.group(1), int(m.group(2))) not in indexed and (not sessions or m.group(1) in sessions):
            _bump(run_void, "index-orphan-log")
    totals = {"sessions": len(reports), "clean": sum(r["verdict"] == "clean" for r in reports),
              "void": sum(r["verdict"] == "void" for r in reports),
              "refused": {}, "voidViolations": dict(run_void)}
    for r in reports:
        for k, v in r["refused"].items():
            _bump(totals["refused"], k, v)
        for k, v in r["void"].items():
            _bump(totals["voidViolations"], k, v)
    totals["refused"] = dict(sorted(totals["refused"].items()))
    totals["voidViolations"] = dict(sorted(totals["voidViolations"].items()))
    ok = totals["void"] == 0 and not run_void
    unpinned = (["plugins"] if accepted_plugins is None else []) + (["agents"] if accepted_agents is None else [])
    return {"sessions": reports, "totals": totals, "indexTornLines": torn, "unpinned": unpinned,
            "verdict": "clean" if ok else "void"}, details


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--projects-dir", type=Path, default=Path.home() / ".claude" / "projects")
    ap.add_argument("--tools", default=DEFAULT_TOOLS, help="the launcher's --tools value")
    ap.add_argument("--session", action="append", help="audit only this session id (repeatable)")
    ap.add_argument("--accepted-agents", type=Path, help="JSON list: the agent list S54 accepted")
    ap.add_argument("--accepted-plugins", type=Path,
                    help="JSON list of plugin names S54 accepted (set equality; each builtin, source @builtin). "
                         "Without it any builtin plugin passes and the report says 'unpinned'")
    ap.add_argument("--settings", type=Path, help="settings.json to hash against the index")
    ap.add_argument("--agents", type=Path, help="agents.json to hash against the index")
    ap.add_argument("--details", type=Path, help="write per-violation lines (no inputs) here")
    a = ap.parse_args(argv)
    if not (a.eval_dir / "runs").is_dir():
        print(f"audit: {a.eval_dir}/runs not found", file=sys.stderr)
        return 2
    accepted = json.loads(a.accepted_agents.read_text()) if a.accepted_agents else None
    plugins = json.loads(a.accepted_plugins.read_text()) if a.accepted_plugins else None
    try:
        report, details = audit_run(a.eval_dir, a.projects_dir, a.tools, a.session, accepted, a.settings,
                                    a.agents, plugins)
    except ValueError as e:
        print(f"audit: {e}", file=sys.stderr)
        return 2
    if a.details:
        a.details.parent.mkdir(parents=True, exist_ok=True)
        a.details.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in details), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "clean" else 1


if __name__ == "__main__":
    sys.exit(main())
