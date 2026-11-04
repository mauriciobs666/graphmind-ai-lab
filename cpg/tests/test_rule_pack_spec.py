"""Checks of ``docs/plans/connected-code-graph-rule-pack-spec.md`` (S30).

The spec is the oracle's *author*, so this file makes its claims falsifiable instead of trusting
the prose: it extracts the document's YAML blocks and tables, closes the pack format over them
(``check_pack``), runs every matcher of every pack against real ``s14a`` record text, passes the
paper launch records through the frozen argv gate, runs every infrastructure signature, sample and
negative, and evaluates the severity formula on its vectors and on mutants of itself.

The ``s14a`` bundles are gitignored (``cpg/.ccg-work``); tests that need them skip when absent.
"""

from __future__ import annotations

import collections
import itertools
import json
import re
import string
from pathlib import Path

import pytest
import yaml

from ccg.bundle import bind_value_allowed, gate_argv

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs/plans/connected-code-graph-rule-pack-spec.md"
SPEC3 = ROOT / "docs/plans/connected-code-graph-bundle-spec3.md"
S14A = ROOT / "cpg/.ccg-work/vertical-slice/s14a"
CAPTURES = Path(__file__).parent / "data/infra_captures.tsv"
TEXT = SPEC.read_text()


# ---------------------------------------------------------------- extraction

def yaml_blocks() -> list[tuple[int, str]]:
    out = []
    for m in re.finditer(r"```yaml\n(.*?)```", TEXT, re.S):
        out.append((TEXT[: m.start()].count("\n") + 1, m.group(1)))
    return out


def load_blocks() -> list[tuple[int, object]]:
    res = []
    for line, body in yaml_blocks():
        if "<subject>" in body or "<runtime>" in body:
            continue  # the §1.2 shape sketch, not data
        res.append((line, yaml.safe_load(body)))
    return res


BLOCKS = load_blocks()
PACKS = {b["id"]: b for _, b in BLOCKS if isinstance(b, dict) and "packFormat" in b}


def table_rows(header_start: str) -> list[list[str]]:
    """Rows of the first markdown table whose header line starts with ``header_start``."""
    lines = TEXT.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header_start):
            rows = []
            for r in lines[i + 2 :]:
                if not r.startswith("|"):
                    break
                rows.append([c.strip() for c in r.strip().strip("|").split("|")])
            return rows
    raise AssertionError(header_start)


# ---------------------------------------------------------------- matchers (§1.3)

BAD_RE = re.compile(r"\(\?[=!<]|\\[1-9]|\(\?>|[*+?}]\+")


def check_regex(r: str) -> None:
    assert len(r) <= 256, f"regex longer than 256: {r[:40]}"
    assert not BAD_RE.search(r), f"outside the RE2 subset: {r}"
    re.compile(r)


def glob_re(g: str) -> re.Pattern:
    out, i = "", 0
    while i < len(g):
        if g.startswith("**", i):
            out += ".*"; i += 2
        elif g[i] == "*":
            out += "[^/]*"; i += 1
        elif g[i] == "?":
            out += "."; i += 1
        else:
            out += re.escape(g[i]); i += 1
    return re.compile(out)


def matches(m, s: str) -> bool:
    if isinstance(m, list):
        return any(matches(x, s) for x in m)
    if not isinstance(m, dict):
        return s == m
    ((k, v),) = [(k, v) for k, v in m.items() if k != "ignoreCase"]
    if k == "exact":
        return s == v
    if k == "glob":
        return glob_re(v).fullmatch(s) is not None
    if k == "basename":
        return s.rsplit("/", 1)[-1] == v
    if k == "regex":
        flags = re.I if m.get("ignoreCase") else 0
        check_regex(v)
        return re.fullmatch(v, s, flags) is not None
    if k == "prefix":
        return s == v or s.startswith(v + ("/" if v.startswith("/") else "."))
    if k == "template":
        return template_re(v).fullmatch(s) is not None
    raise AssertionError(k)


def template_re(t: str) -> re.Pattern:
    """§1.3 `template`: `{}` is one non-empty segment, `{**}` the one or more remaining."""
    out, i = "", 0
    while i < len(t):
        if t.startswith("{**}", i):
            out += ".+"; i += 4
        elif t.startswith("{}", i):
            out += "[^/]+"; i += 2
        else:
            out += re.escape(t[i]); i += 1
    return re.compile(out)


UNRESERVED = set(string.ascii_letters + string.digits + "-._~")


def normalize_path(p: str) -> str:
    """§1.3 `normalize_path` = spec §6 `http` steps 4-6, one pass over the original text."""
    if not p.startswith("/"):
        p = "/" + p
    p = re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)) if chr(int(m[1], 16)) in UNRESERVED else "%" + m[1].upper(), p)
    out, trailing = [], False
    segs = p.split("/")[1:]
    for i, seg in enumerate(segs):
        last = i == len(segs) - 1
        if seg in ("", ".", ".."):
            if seg == ".." and out:
                out.pop()
            trailing = last  # a trailing `.`/`..`/empty segment leaves the directory form
            continue
        out.append(seg)
        trailing = False
    return "/" + "/".join(out) + ("/" if trailing and out else "")


def path_matches(m: dict, s: str) -> bool:
    """`matches` for a path-like operand: operand and subject both normalized once, from the original text."""
    ((k, v),) = m.items()
    return matches({k: normalize_path(v)}, normalize_path(s))


# ---------------------------------------------------------------- closure (the B-2 check)

MATCHER_KEYS = {"exact", "template", "glob", "prefix", "regex", "basename"}
KINDS = {"framework", "protocol", "data-store", "test-framework", "security", "infra-signatures", "severity"}
TOP = {"packFormat", "id", "version", "kind", "vocabVersion", "status", "rules", "mounts", "liftings",
       "embedded", "launch", "urlBindings", "vocab", "link", "queries", "runtimes", "signatures",
       "formula", "rows"}
SUBJECT_FIELDS = {
    "decorator": {"form", "name", "args", "target"},
    "callsite": {"callee", "callKind", "resolution", "externalTarget", "target", "args"},
    "config": {"format", "keyPath", "bindValue"},
    "launch": {"kind", "command", "args", "port", "envVar", "mountPath"},
    "symbol": {"kind", "name", "qname", "parentKind"},
    "marker": {"framework", "kindHint", "evidence"},
    "lifted": {"lifting"},
    "query": None,
    "evidence": None,
}
EXPR = {"arg", "literal", "group", "target", "config", "field", "yield", "lifted", "firstOf"}
EMITS = {
    "provides": {"kind", "key", "keyForm", "status", "reason"},
    "consumes": {"kind", "key", "keyForm", "status", "reason"},
    "reads": {"kind", "key", "keyForm", "status", "reason"},
    "writes": {"kind", "key", "keyForm", "status", "reason"},
    "test": {"kind"}, "mark": {"as", "class"},
    "finding": {"finding", "class", "target"}, "launches": {"target"},
}
ENUMS = {
    ("emit", "keyForm"): {"literal", "glob", "regex", "template"},
    ("onUnknown",): {"drop", "partial", "runtime-built", "config-derived"},
    ("link", "matcher"): {"exact", "path-template", "glob", "prefix", "regex"},
    ("targetArg", "form"): {"module-attr"},
    ("envPorts", "adoptedBy"): {"same-file-launch"},
    ("pathMatch",): {"exact", "prefix"},
    ("scope",): {"file", "unit"},
    ("sampleSource",): {"captured", "runtime-source", "authored"},
    ("class",): {"connection-refused", "dns-failure", "timeout", "oom-kill", "tls-error", "third-party-exception"},
    ("runtime",): {"python", "node", "jvm", "dotnet", "go", "any"},
    ("level",): {"critical", "high", "medium", "low"},
    ("finding",): {"injection", "authorization", "configuration", "secrets"},
    ("status",): {"paper"},                       # §1.1
    ("how",): {"app"},                            # §1.6 roots
    ("resolves",): {"factory-call"},              # §1.6 mounts.child
    ("mountKind",): {"class-decorator"},          # §1.6
    ("language",): {"sql", "cypher"},             # §1.6 embedded
    ("companion",): {"verb"},                     # §1.6 liftings.companions.role
    ("fileRole",): {"primary", "example", "doc"},  # spec §5.2 File.role
    ("emitStatus",): {"confirmed", "possible", "none"},
    ("target-expr",): {"name", "qname"},
    ("config-expr",): {"keyPath", "bindValue"},   # N-20
    ("lifted-expr",): {"url", "verb"},
}
# §9.1 `when` vocabulary: the features the formula block uses (the spec names them in §9.1-§9.3 prose and
# vectors, not in one table; the checker closes over exactly that set).
FEATURES = {
    "source": {"network", "local"}, "sink": {"exec", "data"}, "sanitizer": {"none", "partial"},
    "entry": {"network", "internal"}, "operation": {"write", "read"}, "authCheck": {"none", "partial"},
    "state": {"head", "removed"}, "fileRole": ENUMS[("fileRole",)],
}
BOOL_FEATURES = {"chainConfirmed", "flow"}
FILE_FEATURES = {"fileRole", "path", "basename"}
KIND_FEATURES = {
    "injection": {"source", "sink", "chainConfirmed", "sanitizer"} | FILE_FEATURES,
    "authorization": {"entry", "operation", "chainConfirmed", "authCheck"} | FILE_FEATURES,
    "configuration": {"class"} | FILE_FEATURES,
    "secrets": {"state", "class", "flow"} | FILE_FEATURES,
}
CLASSES = {"configuration": {"host-compromise", "exposed-service", "exposed-app", "hygiene"},
           "secrets": {"K", "P", "G"}}
SEV_PREFIX = {"injection": "INJ", "authorization": "AUTHZ", "configuration": "RCFG", "secrets": "SEC"}
ARGPATH = r"(a\d+|kw\.[A-Za-z_]\w*|recv)(\.[A-Za-z_]\w*)*"


def only(obj, allowed, where):
    assert isinstance(obj, dict), f"{where}: not an object"
    extra = set(obj) - set(allowed)
    assert not extra, f"{where}: unknown keys {sorted(extra)}"


def enum(value, key, where):
    assert isinstance(value, str) and value in ENUMS[key], f"{where}: {value!r} not in {sorted(ENUMS[key])}"


def small_int(v, lo, hi, where):
    assert isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi, f"{where}: {v!r} not an int in {lo}..{hi}"


def check_argpath(a, where):
    assert isinstance(a, str) and re.fullmatch(ARGPATH, a), f"{where}: bad argument path {a!r}"


def check_matcher(m, where):
    if isinstance(m, list):
        assert m, f"{where}: empty any-of"
        for x in m:
            check_matcher(x, where)
        return
    if not isinstance(m, dict):
        assert isinstance(m, str), f"{where}: a bare matcher is a string, got {m!r}"
        return  # bare scalar = exact
    keys = set(m) - {"ignoreCase"}
    assert len(keys) == 1 and keys <= MATCHER_KEYS, f"{where}: one matcher per field, got {sorted(m)}"
    ((k, v),) = [(k, v) for k, v in m.items() if k != "ignoreCase"]
    assert isinstance(v, str), f"{where}: matcher operand must be a string"
    if k == "regex":
        check_regex(v)
    if "ignoreCase" in m:
        assert k == "regex" and isinstance(m["ignoreCase"], bool), f"{where}: ignoreCase belongs to regex"


def check_expr(e, where):
    only(e, EXPR | {"upper", "onUnknown", "default"}, where)
    keys = set(e) & EXPR
    assert len(keys) == 1, f"{where}: exactly one expression, got {sorted(keys)}"
    if "onUnknown" in e:
        enum(e["onUnknown"], ("onUnknown",), where)
    if "default" in e:
        assert isinstance(e["default"], str)
    if "upper" in e:
        assert "group" in e and e["upper"] is True, f"{where}: upper belongs to group"
    if "firstOf" in e:
        assert e["firstOf"], f"{where}: empty firstOf"
        for x in e["firstOf"]:
            check_expr(x, where + ".firstOf")
    if "arg" in e:
        check_argpath(e["arg"], where)
    if "group" in e:
        small_int(e["group"], 0, 9, where)
    if "literal" in e:
        assert isinstance(e["literal"], str)
    if "target" in e:
        enum(e["target"], ("target-expr",), where)
    if "config" in e:
        enum(e["config"], ("config-expr",), where)
    if "lifted" in e:
        enum(e["lifted"], ("lifted-expr",), where)
    for k in ("field", "yield"):
        if k in e:
            assert isinstance(e[k], str) and e[k], where


def check_args(args, where):
    assert isinstance(args, list), where
    for a in args:
        only(a, {"arg", "value"}, where + ".args")
        assert "arg" in a, where
        check_argpath(a["arg"], where)
        if "value" in a:
            check_matcher(a["value"], where)


def check_clause(subject, clause, where):
    fields = SUBJECT_FIELDS[subject]
    if subject == "query":
        assert isinstance(clause, str)
        return
    assert isinstance(clause, dict), f"{where}: clause is not an object"
    if fields:
        only(clause, fields, where)
    for k, v in clause.items():
        if k == "args":
            check_args(v, where)
        elif k == "target" and isinstance(v, dict):
            only(v, {"kind", "name", "parentKind"}, where)
            for tk, tv in v.items():
                check_matcher(tv, f"{where}.target.{tk}")
        else:
            check_matcher(v, f"{where}.{k}")


def check_match(match, where):
    assert isinstance(match, dict), where
    if "anyOf" in match:
        assert len(match) == 1 and match["anyOf"], f"{where}: empty or mixed anyOf"
        for c in match["anyOf"]:
            check_match(c, where + ".anyOf")
        return
    assert len(match) == 1, f"{where}: exactly one subject"
    ((subject, clause),) = match.items()
    assert subject in SUBJECT_FIELDS, f"{where}: subject {subject}"
    check_clause(subject, clause, where)


def check_call(c, where):
    only(c, {"callee", "args", "resolution", "externalTarget", "callKind"}, where)
    for k, v in c.items():
        if k == "args":
            check_args(v, where)
        else:
            check_matcher(v, where)


def check_scope(s, where):
    only(s, {"imports", "path", "notPath", "role"}, where)
    for g in s.get("imports", []):
        only(g, {"module", "names"}, where)
        check_matcher(g["module"], where)
        assert all(isinstance(n, str) for n in g.get("names", []))
    for k in ("path", "notPath"):
        if k in s:
            check_matcher(s[k], where)
    for r in s.get("role", []):
        enum(r, ("fileRole",), where)


def check_rule(r, queries, where):
    only(r, {"id", "match", "scope", "extract", "emit"}, where)
    assert re.fullmatch(r"[a-z][a-z0-9-]*", r["id"])
    check_match(r["match"], where)
    m = r["match"]
    if "query" in m:
        q = next((q for q in queries if q["id"] == m["query"]), None)
        assert q and "handler" in q["yield"], f"{where}: query without a handler column"
    if "scope" in r:
        check_scope(r["scope"], where)
    for slot, e in r.get("extract", {}).items():
        check_expr(e, f"{where}.extract.{slot}")
        if e.get("onUnknown") == "config-derived":
            assert any(k in ("consumes", "provides") and v["kind"] == "topic" for em in r["emit"] for k, v in em.items()), "N-27"
    assert r["emit"], f"{where}: emit []"
    for em in r["emit"]:
        assert len(em) == 1
        ((k, v),) = em.items()
        only(v, EMITS[k], f"{where}.emit.{k}")
        if "keyForm" in v:
            enum(v["keyForm"], ("emit", "keyForm"), where)
        if "status" in v:
            enum(v["status"], ("emitStatus",), where)


JOIN_FIELDS = {"id", "symbolId", "parentId", "callSiteId", "unitId", "targetId", "fileId", "point", "cond", "use", "def"}


def check_query(q, where):
    only(q, {"id", "from", "yield"}, where)
    assert 1 <= len(q["from"]) <= 4
    seen = set()
    for i, b in enumerate(q["from"]):
        only(b, {"as", "record", "where"}, where)
        assert b["record"] != "secretSighting"
        joined = False
        for f, v in b["where"].items():
            assert re.fullmatch(r"[A-Za-z_]\w*", f), f"{where}: field {f!r}"
            if isinstance(v, str) and v.startswith("$"):
                alias, fld = v[1:].split(".", 1)
                if not fld.startswith("args."):
                    assert alias in seen and f in JOIN_FIELDS and fld in JOIN_FIELDS, f"{where}: join {f}={v}"
                    joined = True
            else:
                check_matcher(v, where)
        assert i == 0 or joined, f"{where}: binding {b['as']} has no join equality"
        seen.add(b["as"])
    assert q["yield"], f"{where}: no yield columns"
    for col, ref in q["yield"].items():
        assert re.fullmatch(r"[a-z]\w*", col)
        assert isinstance(ref, str) and re.fullmatch(r"\$(\w+)\.(\w+)(\.\w+)*", ref) and ref[1:].split(".")[0] in seen, ref


def check_when(when, kind, where):
    """`when` maps a feature to a value (equality) or, for path/basename, to a §1.3 matcher (§9.1)."""
    assert isinstance(when, dict) and when, f"{where}: empty when"
    allowed = KIND_FEATURES[kind] if kind else set().union(*KIND_FEATURES.values())
    for k, v in when.items():
        assert k in allowed, f"{where}: feature {k!r} not in {sorted(allowed)}"
        if k in ("path", "basename"):
            check_matcher(v, where)
        elif k in BOOL_FEATURES:
            assert isinstance(v, bool), f"{where}: {k} is a boolean"
        elif k == "class":
            domain = CLASSES[kind] if kind else set().union(*CLASSES.values())
            assert isinstance(v, str) and v in domain, f"{where}: class {v!r}"
        else:
            assert isinstance(v, str) and v in FEATURES[k], f"{where}: {k}={v!r} not in {sorted(FEATURES[k])}"


def check_formula(f, where="formula"):
    only(f, {"caps", "injection", "authorization", "configuration", "secrets"}, where)
    assert set(SEV_PREFIX) <= set(f), f"{where}: a severity pack carries all four kinds"
    for c in f.get("caps", []):
        only(c, {"when", "max"}, "caps")
        check_when(c["when"], None, "caps")
        small_int(c["max"], 0, 3, "caps.max")
    for k in SEV_PREFIX:
        only(f[k], {"base", "adjust", "penalties"}, k)
        assert f[k]["base"], f"{k}: no base"
        assert "adjust" not in f[k] or k == "secrets", f"{k}: adjust is secrets-only (§9.1 step 2)"
        for section in ("base", "adjust", "penalties"):
            for e in f[k].get(section, []):
                only(e, {"when", "value"}, f"{k}.{section}")
                check_when(e["when"], k, f"{k}.{section}")
                small_int(e["value"], 0, 3, f"{k}.{section}.value")


def check_pack(p: dict) -> None:
    only(p, TOP, "pack")
    assert p["packFormat"] == 1 and re.fullmatch(r"[a-z][a-z0-9-]{0,39}", p["id"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(p["version"])), "semver"
    assert p["kind"] in KINDS and p["vocabVersion"] == "1.0"
    if "status" in p:
        enum(p["status"], ("status",), "pack")
    # §1.1: each body belongs to its kind; severity and infra-signatures packs carry theirs instead of `rules`
    for body, kind in (("formula", "severity"), ("rows", "severity"),
                       ("runtimes", "infra-signatures"), ("signatures", "infra-signatures")):
        assert body not in p or p["kind"] == kind, f"{body} belongs to kind {kind}"
    if p["kind"] in ("severity", "infra-signatures"):
        assert "rules" not in p, "a severity or infra-signatures pack carries no rules"
    if p["kind"] == "severity":
        assert "formula" in p and "rows" in p
    if p["kind"] == "infra-signatures":
        assert "runtimes" in p and "signatures" in p
    queries = p.get("queries", [])
    for q in queries:
        check_query(q, "queries")
    ids = [r["id"] for r in p.get("rules", [])]
    assert len(ids) == len(set(ids)), "rule ids unique"
    for r in p.get("rules", []):
        check_rule(r, queries, f"{p['id']}.rules.{r['id']}")
    for m in p.get("mounts", []):
        only(m, {"id", "call", "scope", "child", "prefix", "roots", "constructorPrefix", "unmounted",
                 "recordConditions", "kind", "decorator"}, "mounts")
        assert re.fullmatch(r"[a-z][a-z0-9-]*", m["id"])
        check_call(m["call"], "mounts.call")
        if "scope" in m:
            check_scope(m["scope"], "mounts.scope")
        if "child" in m:
            only(m["child"], {"arg", "resolves"}, "mounts.child")
            check_argpath(m["child"]["arg"], "mounts.child")
            enum(m["child"]["resolves"], ("resolves",), "mounts.child")
        if "prefix" in m:
            assert "child" in m, "N-16"
            only(m["prefix"], {"arg", "default"}, "mounts.prefix")
            check_argpath(m["prefix"]["arg"], "mounts.prefix")
            assert isinstance(m["prefix"].get("default", ""), str)
        for rt in m.get("roots", []):
            only(rt, {"call", "how", "notWith"}, "roots")
            check_call(rt["call"], "roots.call")
            enum(rt["how"], ("how",), "roots")
            for nw in rt.get("notWith", []):
                check_call(nw, "roots.notWith")
        for cp in m.get("constructorPrefix", []):
            only(cp, {"call", "arg"}, "constructorPrefix")
            check_call(cp["call"], "constructorPrefix.call")
            check_argpath(cp["arg"], "constructorPrefix")
        if "unmounted" in m:
            only(m["unmounted"], {"status", "reason"}, "mounts.unmounted")
            enum(m["unmounted"]["status"], ("emitStatus",), "mounts.unmounted")
            assert isinstance(m["unmounted"]["reason"], str)
        if "recordConditions" in m:
            assert isinstance(m["recordConditions"], bool)
        if "kind" in m:
            enum(m["kind"], ("mountKind",), "mounts")
        if "decorator" in m:
            check_clause("decorator", m["decorator"], "mounts.decorator")
    for l in p.get("liftings", []):
        only(l, {"id", "role", "seeds", "companions"}, "liftings")
        only(l["role"], {"kind", "slot"}, "liftings.role")
        for s in l["seeds"]:
            only(s, {"call", "arg"}, "seeds")
            assert "arg" in s, "N-15"
            check_argpath(s["arg"], "seeds")
            check_call(s["call"], "seeds")
        for c in l.get("companions", []):
            only(c, {"role", "arg", "bareDefault", "unknown"}, "companions")
            enum(c["role"], ("companion",), "companions")
            check_argpath(c["arg"], "companions")
    for e in p.get("embedded", []):
        only(e, {"call", "arg", "language", "scope"}, "embedded")
        check_call(e["call"], "embedded.call")
        check_argpath(e["arg"], "embedded")
        enum(e["language"], ("language",), "embedded")
        if "scope" in e:
            check_scope(e["scope"], "embedded.scope")
    if "launch" in p:
        only(p["launch"], {"frameworkDefaults", "portFlags", "targetArg", "envPorts", "mountPaths"}, "launch")
        for e in p["launch"].get("frameworkDefaults", []):
            only(e, {"command", "port", "protocol"}, "frameworkDefaults"); check_matcher(e["command"], "fd")
            small_int(e["port"], 1, 65535, "fd.port"); assert isinstance(e["protocol"], str)
        for e in p["launch"].get("portFlags", []):
            only(e, {"command", "flags"}, "portFlags"); check_matcher(e["command"], "pf")
            assert e["flags"] and all(isinstance(x, str) for x in e["flags"])
        for e in p["launch"].get("targetArg", []):
            only(e, {"command", "index", "form"}, "targetArg"); enum(e["form"], ("targetArg", "form"), "targetArg")
            small_int(e["index"], 0, 99, "targetArg.index")
            if "command" in e:
                check_matcher(e["command"], "targetArg")
        for e in p["launch"].get("envPorts", []):
            only(e, {"envVar", "excludeVars", "adoptedBy"}, "envPorts")
            enum(e["adoptedBy"], ("envPorts", "adoptedBy"), "envPorts")
            check_matcher(e["envVar"], "envPorts")
            if "excludeVars" in e:
                check_matcher(e["excludeVars"], "envPorts")
        for e in p["launch"].get("mountPaths", []):
            only(e, {"call", "path", "app", "pathMatch"}, "mountPaths")
            check_call(e["call"], "mountPaths.call")
            check_expr(e["path"], "mountPaths.path")
            only(e["app"], {"arg", "fromCall"}, "mountPaths.app")
            check_argpath(e["app"]["arg"], "mountPaths.app")
            check_call(e["app"]["fromCall"], "fromCall")
            enum(e["pathMatch"], ("pathMatch",), "mountPaths")
    for ub in p.get("urlBindings", []):
        only(ub, {"id", "kinds", "urlSources"}, "urlBindings")
        assert ub["kinds"] and all(isinstance(k, str) for k in ub["kinds"])
        for s in ub["urlSources"]:
            if "config" in s:
                only(s, {"config"}, "urlSource")
                check_clause("config", s["config"], "urlSource.config")
            else:
                only(s, {"callsite", "arg", "scope"}, "urlSource")
                check_clause("callsite", s["callsite"], "urlSource.callsite")
                check_argpath(s["arg"], "urlSource")
                if "scope" in s:
                    enum(s["scope"], ("scope",), "urlSource")
    for v in p.get("vocab", []):
        only(v, {"kind", "keyNorm", "example", "transport"}, "vocab")
        assert "keyNorm" in v, "N-11"
        if "transport" in v:
            assert re.fullmatch(r"[a-z][a-z0-9-]*", v["transport"]), "N-19"
            if v["transport"].startswith("inhouse-"):
                assert v["transport"] == f"inhouse-{p['id']}", "N-19"
    for l in p.get("link", []):
        only(l, {"kind", "matcher"}, "link"); enum(l["matcher"], ("link", "matcher"), "link")
    for rt in p.get("runtimes", []):
        only(rt, {"runtime", "regex"}, "runtimes"); check_regex(rt["regex"])
        enum(rt["runtime"], ("runtime",), "runtimes")
    for sg in p.get("signatures", []):
        only(sg, {"id", "class", "runtime", "regex", "sample", "sampleSource", "library"}, "signatures")
        enum(sg["class"], ("class",), sg["id"]); enum(sg["runtime"], ("runtime",), sg["id"])
        check_regex(sg["regex"])
        assert isinstance(sg["sample"], str)
        if "sampleSource" in sg:
            enum(sg["sampleSource"], ("sampleSource",), sg["id"])
    if "formula" in p:
        check_formula(p["formula"])
    seen_rows = set()
    for r in p.get("rows", []):
        only(r, {"id", "finding", "rank", "level"}, "rows")
        enum(r["level"], ("level",), r["id"]); enum(r["finding"], ("finding",), r["id"])
        assert r["rank"] in (0, 1, 2, 3)
        assert r["level"] == ["low", "medium", "high", "critical"][r["rank"]], "N-18/N-24"
        assert r["id"] == f"{SEV_PREFIX[r['finding']]}-{4 - r['rank']}", "N-24: the derived id"
        assert (r["finding"], r["rank"]) not in seen_rows, "N-24: one row per (finding, rank)"
        seen_rows.add((r["finding"], r["rank"]))



def assembled_infra_pack() -> dict:
    head = PACKS["infra-signatures"]
    sig_block = next(b for _, b in BLOCKS if isinstance(b, list) and b and "class" in b[0])
    return {**head, "signatures": sig_block}


def all_packs() -> dict[str, dict]:
    packs = dict(PACKS)
    packs["infra-signatures"] = assembled_infra_pack()
    return packs


# ---------------------------------------------------------------- reason registry (§1.9)

def registry() -> dict[str, dict]:
    reg: dict[str, dict] = {}
    t3 = SPEC3.read_text()
    start = t3.index("### 7.2 Status reasons")
    for ln in t3[start : t3.index("### 7.3")].splitlines():
        if ln.startswith("| `"):
            c = [x.strip() for x in ln.strip("|").split("|")]
            code = c[0].strip("`")
            reg[code] = {"status": c[1], "on": c[2]}
    for r in table_rows("| Code | Status | On (added"):
        code = r[0].strip("`")
        reg.setdefault(code, {"status": r[1], "on": ""})
        reg[code]["on"] += ", " + r[2]
    return reg


def test_registry_covers_every_reason_a_pack_emits():
    reg = registry()
    assert "severity-rule:<id>" in reg and "key-match" in reg
    for pid, p in all_packs().items():
        for m in p.get("mounts", []):
            u = m.get("unmounted")
            if u:
                assert u["reason"] in reg and "PROVIDES" in reg[u["reason"]]["on"], (pid, u)
                assert reg[u["reason"]]["status"] == u["status"], (pid, u)
        for r in p.get("rules", []):
            for em in r["emit"]:
                for k, v in em.items():
                    if "reason" in v:
                        assert v["reason"] in reg
    # the extension rows the packs rely on
    assert "PROVIDES" in reg["key-match"]["on"] and "PROVIDES" in reg["partial-key"]["on"]
    assert "CONSUMES" in reg["partial-key"]["on"]


# ---------------------------------------------------------------- the format is closed

@pytest.mark.parametrize("pid", ["fastapi", "fetch", "fastmcp", "kafka", "paper-inhouse-bus",
                                 "infra-signatures", "severity-rules"])
def test_every_pack_validates_against_the_closed_format(pid):
    packs = all_packs()
    assert pid in packs
    check_pack(packs[pid])


def test_illustrative_fragments_use_only_defined_keys():
    for line, b in BLOCKS:
        if isinstance(b, dict) and "packFormat" not in b and set(b) <= {"mounts", "liftings", "launch"}:
            check_pack({"packFormat": 1, "id": "frag", "version": "0.0.0", "kind": "framework", "vocabVersion": "1.0", **b})


NEGATIVES = [
    ("N-1 packFormat", lambda p: p.update(packFormat=2)),
    ("N-3 kind", lambda p: p.update(kind="plugin")),
    ("N-21 unknown top key", lambda p: p.update(repoId="x")),
    ("N-4 emit []", lambda p: p["rules"][0].update(emit=[])),
    ("N-5 two subjects", lambda p: p["rules"][0].update(match={"decorator": {}, "callsite": {}})),
    ("N-8 two matchers", lambda p: p["rules"][0]["match"]["decorator"].update(name={"exact": "x", "glob": "y"})),
    ("N-6 lookahead", lambda p: p["rules"][0]["match"]["decorator"].update(name={"regex": "(?=x)abc"})),
    ("N-6 backreference", lambda p: p["rules"][0]["match"]["decorator"].update(name={"regex": r"(a)\1"})),
    ("N-7 length", lambda p: p["rules"][0]["match"]["decorator"].update(name={"regex": "a" * 257})),
    ("N-12 arg grammar", lambda p: p["rules"][0]["extract"].update(t={"arg": "arg0"})),
    ("N-26 planted key", lambda p: p["rules"][0]["emit"][0]["provides"].update(planted=1)),
    ("N-23 link matcher", lambda p: p.update(link=[{"kind": "http", "matcher": "fuzzy"}])),
    ("N-25 form enum", lambda p: p.update(launch={"targetArg": [{"index": 0, "form": "dotted"}]})),
    ("N-27 config-derived on a provides of kind http", lambda p: p["rules"][0]["extract"].update(route={"arg": "a0", "onUnknown": "config-derived"})),
    ("N-28 empty firstOf", lambda p: p["rules"][0]["extract"].update(route={"firstOf": []})),
]


@pytest.mark.parametrize("name,mutate", NEGATIVES, ids=[n for n, _ in NEGATIVES])
def test_negative_examples_are_rejected_by_the_checker(name, mutate):
    import copy
    p = copy.deepcopy(PACKS["fastapi"])
    mutate(p)
    with pytest.raises(AssertionError):
        check_pack(p)


def test_n27_config_derived_on_a_consumes_of_http_is_rejected_by_the_checker():
    import copy
    p = copy.deepcopy(PACKS["fetch"])
    p["rules"][0]["extract"].update(url={"lifted": "url", "onUnknown": "config-derived"})
    with pytest.raises(AssertionError, match="N-27"):
        check_pack(p)


def test_config_derived_on_a_provides_of_kind_topic_is_accepted_by_the_checker():
    """Spec 1.4 (v1.4, O-16): `config-derived` is valid on a `consumes` or a `provides` of kind `topic`."""
    import copy
    p = copy.deepcopy(PACKS["kafka"])
    p["rules"][0]["extract"]["topic"]["onUnknown"] = "config-derived"
    check_pack(p)


def test_envports_without_excludevars_is_accepted_by_the_checker():
    """Spec 1.6 (v1.4): `excludeVars` is optional."""
    import copy
    p = copy.deepcopy(PACKS["fastmcp"])
    del p["launch"]["envPorts"][0]["excludeVars"]
    check_pack(p)


# N-26: one out-of-range value per enum / typed leaf / body-kind rule, on the pack that carries it.
RANGE_NEGATIVES = [
    ("status live", "fastapi", lambda p: p.update(status="live")),
    ("base value 9", "severity-rules", lambda p: p["formula"]["injection"]["base"][0].update(value=9)),
    ("penalty value -1", "severity-rules", lambda p: p["formula"]["injection"]["penalties"][0].update(value=-1)),
    ("cap max 4", "severity-rules", lambda p: p["formula"]["caps"][0].update(max=4)),
    ("when source nowhere", "severity-rules", lambda p: p["formula"]["injection"]["base"][0]["when"].update(source="nowhere")),
    ("when feature of another kind", "severity-rules", lambda p: p["formula"]["injection"]["base"][0]["when"].update(state="head")),
    ("when chainConfirmed not boolean", "severity-rules", lambda p: p["formula"]["injection"]["penalties"][0]["when"].update(chainConfirmed="no")),
    ("when class outside the kind", "severity-rules", lambda p: p["formula"]["secrets"]["base"][0]["when"].update(**{"class": "hygiene"})),
    ("kind framework carrying a formula", "severity-rules", lambda p: p.update(kind="framework")),
    ("kind framework carrying rows only", "severity-rules", lambda p: (p.update(kind="framework"), p.pop("formula"))),
    ("severity pack with rules", "severity-rules", lambda p: p.update(rules=[])),
    ("severity pack without the secrets formula", "severity-rules", lambda p: p["formula"].pop("secrets")),
    ("adjust outside secrets", "severity-rules", lambda p: p["formula"]["injection"].update(adjust=[{"when": {"chainConfirmed": True}, "value": 1}])),
    ("row id not the derived one", "severity-rules", lambda p: p["rows"][0].update(id="INJ-9")),
    ("second row for one (finding, rank)", "severity-rules", lambda p: p["rows"].append(dict(p["rows"][0]))),
    ("infra pack declared framework", "infra-signatures", lambda p: p.update(kind="framework")),
    ("infra pack with rules", "infra-signatures", lambda p: p.update(rules=[])),
    ("notWith with two matchers", "fastapi", lambda p: p["mounts"][0]["roots"][0]["notWith"][0].update(callKind="x", callee={"exact": "a", "glob": "b"})),
    ("constructorPrefix arg p0", "fastapi", lambda p: p["mounts"][0]["constructorPrefix"][0].update(arg="p0")),
    ("roots how enum", "fastapi", lambda p: p["mounts"][0]["roots"][0].update(how="router")),
    ("child resolves enum", "fastapi", lambda p: p["mounts"][0]["child"].update(resolves="variable")),
    ("N-20 config value", "fastapi", lambda p: p["rules"][0]["extract"].update(v={"config": "value"})),
    ("slot expression not a dict", "fastapi", lambda p: p["rules"][0]["extract"].update(v=1)),
    ("query yield not a reference", "paper-inhouse-bus", lambda p: p["queries"][0]["yield"].update(x="plain")),
    ("urlBindings config clause unknown field", "fastmcp", lambda p: p["urlBindings"][0]["urlSources"][0]["config"].update(path="x")),
    ("mountPaths path not an expression", "fastmcp", lambda p: p["launch"]["mountPaths"][0].update(path={"arg": "a0", "x": 1})),
    ("signature sample not a string", "infra-signatures", lambda p: p["signatures"][0].update(sample=1)),
]


@pytest.mark.parametrize("name,pid,mutate", RANGE_NEGATIVES, ids=[n for n, *_ in RANGE_NEGATIVES])
def test_out_of_range_values_and_body_kind_rules_are_rejected(name, pid, mutate):
    import copy
    p = copy.deepcopy(all_packs()[pid])
    mutate(p)
    with pytest.raises(AssertionError):
        check_pack(p)


PLANTED = "zz_planted"


def dict_positions(node, path=()):
    """Every dict reachable in a pack, as a path of keys and list indices."""
    if isinstance(node, dict):
        yield path
        for k, v in node.items():
            yield from dict_positions(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from dict_positions(v, path + (i,))


def plant_everywhere(packs: dict[str, dict]) -> tuple[int, list]:
    """N-26: plant one unknown key at every dict of every pack; return (positions tried, accepted ones)."""
    import copy
    tried, accepted = 0, []
    for pid, pack in packs.items():
        for path in dict_positions(pack):
            mutant = copy.deepcopy(pack)
            target = mutant
            for step in path:
                target = target[step]
            target[PLANTED] = 1
            tried += 1
            try:
                check_pack(mutant)
            except AssertionError:
                continue
            accepted.append((pid, path))
    return tried, accepted


def yaml_mappings_in_pack_blocks() -> int:
    """Count mappings with the YAML composer (a walker independent of `dict_positions`) over the pack blocks."""
    def count(node) -> int:
        if isinstance(node, yaml.MappingNode):
            return 1 + sum(count(k) + count(v) for k, v in node.value)
        if isinstance(node, yaml.SequenceNode):
            return sum(count(x) for x in node.value)
        return 0
    total = 0
    for (_, body), (_, data) in zip([b for b in yaml_blocks() if "<subject>" not in b[1] and "<runtime>" not in b[1]], BLOCKS):
        is_pack = isinstance(data, dict) and "packFormat" in data
        is_signature_list = isinstance(data, list) and data and "class" in data[0]
        if is_pack or is_signature_list:
            total += count(yaml.compose(body))
    return total


def test_a_key_planted_at_every_position_of_every_pack_is_rejected():
    tried, accepted = plant_everywhere(all_packs())
    # a probe that walked fewer dicts than the packs hold would pass vacuously: its walk must equal an
    # independent count of the YAML mappings in the same blocks (268 pack-block mappings + the 45 signatures)
    assert tried == yaml_mappings_in_pack_blocks(), tried
    assert tried > 250
    assert not accepted, f"{len(accepted)} of {tried} positions accept a planted key: {accepted[:10]}"


def test_the_planted_key_probe_catches_an_open_checker():
    """The probe must be able to fail: against a checker that ignores unknown keys under `formula`, it does."""
    global only
    real_only = only
    def lax(obj, allowed, where):
        if where in ("formula", "caps") or where.startswith(("injection", "authorization", "configuration", "secrets")):
            return
        real_only(obj, allowed, where)
    only = lax
    try:
        _, accepted = plant_everywhere({"severity-rules": PACKS["severity-rules"]})
    finally:
        only = real_only
    assert accepted


def test_urlbindings_scope_outside_a_callsite_source_is_rejected():
    import copy
    p = copy.deepcopy(PACKS["fastmcp"])
    p["urlBindings"][0]["urlSources"][0]["scope"] = "file"
    with pytest.raises(AssertionError):
        check_pack(p)
    p = copy.deepcopy(PACKS["fastmcp"])
    p["urlBindings"][0]["scope"] = "file"
    with pytest.raises(AssertionError):
        check_pack(p)


def test_query_join_rules_reject_unjoined_and_non_id_joins():
    import copy
    q = copy.deepcopy(PACKS["paper-inhouse-bus"]["queries"][0])
    q["from"][1]["where"] = {"name": {"regex": "x"}}  # no join equality
    with pytest.raises(AssertionError):
        check_query(q, "q")
    q = copy.deepcopy(PACKS["paper-inhouse-bus"]["queries"][0])
    q["from"][1]["where"] = {"id": "$dec.name"}  # non-id join field
    with pytest.raises(AssertionError):
        check_query(q, "q")
    q = copy.deepcopy(PACKS["paper-inhouse-bus"]["queries"][0])
    q["from"] += [q["from"][0]] * 2  # five bindings
    with pytest.raises(AssertionError):
        check_query(q, "q")


def test_severity_row_inconsistency_is_rejected():
    import copy
    p = copy.deepcopy(PACKS["severity-rules"])
    p["rows"][0]["level"] = "low"
    with pytest.raises(AssertionError):
        check_pack(p)
    p = copy.deepcopy(PACKS["severity-rules"])
    p["rows"][0]["level"] = "severe"
    with pytest.raises(AssertionError):
        check_pack(p)


def test_a_variation_of_each_walked_pack_with_one_rule_removed_is_valid():
    import copy
    for pid in ("fastapi", "fetch", "fastmcp"):
        p = copy.deepcopy(PACKS[pid])
        p["rules"] = p["rules"][1:]
        check_pack(p)


# ---------------------------------------------------------------- the minimal bundle query evaluator

def eval_query(q, records):
    rows = [{}]
    for b in q["from"]:
        nxt = []
        for env in rows:
            for rec in records:
                if rec["type"] != b["record"]:
                    continue
                ok = True
                for f, v in b["where"].items():
                    if isinstance(v, str) and v.startswith("$"):
                        alias, fld = v[1:].split(".", 1)
                        if fld.startswith("args."):
                            continue
                        ok &= rec.get(f) == env[alias].get(fld)
                    else:
                        ok &= f in rec and matches(v, rec[f])
                if ok:
                    nxt.append({**env, b["as"]: rec})
        rows = nxt
        assert len(rows) <= 20000
    return rows


def test_handler_ns_query_yields_one_row_on_the_invented_bundle():
    q = PACKS["paper-inhouse-bus"]["queries"][0]
    recs = [
        {"type": "symbol", "id": "T", "name": "InvoiceHandlers", "kind": "type"},
        {"type": "symbol", "id": "H", "name": "handle", "kind": "method", "parentId": "T"},
        {"type": "decorator", "id": "D1", "name": "on", "targetId": "H", "args": [{"arg": "a0", "values": [{"const": "invoice.*"}]}]},
        {"type": "decorator", "id": "D2", "name": "component", "targetId": "T", "args": [{"arg": "a0", "values": [{"const": "billing"}]}]},
        {"type": "decorator", "id": "D3", "name": "on", "targetId": "H2", "args": []},  # handler of an undecorated class
        {"type": "symbol", "id": "H2", "name": "other", "kind": "method", "parentId": "T2"},
        {"type": "symbol", "id": "T2", "name": "Plain", "kind": "type"},
    ]
    rows = eval_query(q, recs)
    assert [r["h"]["id"] for r in rows] == ["H"]  # inner join: the class without @component drops out
    # the `on` regex rejects `component` itself and unrelated names
    assert not matches(q["from"][0]["where"]["name"], "component")
    assert matches(q["from"][0]["where"]["name"], "bus.on")


# ---------------------------------------------------------------- real records (s14a)

def load_bundle(pred):
    for f in sorted(S14A.glob("*.bundle.jsonl")):
        recs = [json.loads(l) for l in f.open()]
        if pred(recs):
            return recs
    pytest.skip("s14a bundles absent (gitignored cpg/.ccg-work)")


@pytest.fixture(scope="module")
def server():
    return Corpus(load_bundle(lambda r: any(x.get("path", "").startswith("falkor-chat/server") for x in r if x["type"] == "file")))


@pytest.fixture(scope="module")
def sales():
    return Corpus(load_bundle(lambda r: any(x.get("path", "").startswith("salesperson") for x in r if x["type"] == "file")))


class Corpus:
    def __init__(self, recs):
        self.by = collections.defaultdict(list)
        for r in recs:
            self.by[r["type"]].append(r)
        self.sym = {r["id"]: r for r in self.by["symbol"]}
        self.file = {r["id"]: r for r in self.by["file"]}
        self.vs = collections.defaultdict(dict)
        for v in self.by["valueSet"]:
            self.vs[v["callSiteId"]][v["arg"]] = v["values"]
        self.imports = collections.defaultdict(list)
        for i in self.by["import"]:
            self.imports[i["fileId"]].append(i["module"])
        self.in_symbol = collections.defaultdict(set)
        for c in self.by["callsite"]:
            self.in_symbol[c["symbolId"]].add(c["calleeName"])
        self.cs = {c["id"]: c for c in self.by["callsite"]}
        self.data = collections.defaultdict(list)
        for d in self.by["data"]:
            self.data[d["use"]].append(d)

    def path_of(self, sym_id):
        return self.file[self.sym[sym_id]["fileId"]]["path"]

    def file_imports(self, sym_id, scope):
        mods = self.imports[self.sym[sym_id]["fileId"]]
        return any(matches({"prefix": s}, m) for s in scope for m in mods)

    def short(self, text):
        return [r["id"] for rs in self.by.values() for r in rs if r["id"].startswith(text)]


FASTAPI_RULE = re.compile(PACKS["fastapi"]["rules"][0]["match"]["decorator"]["name"]["regex"])


def test_fastapi_route_regex_selects_the_62_decorators_all_behind_the_import_gate(server):
    pack = PACKS["fastapi"]
    rx = [r["match"]["decorator"]["name"]["regex"] for r in pack["rules"]]
    sel = [d for d in server.by["decorator"] if any(re.fullmatch(x, d["name"]) for x in rx)]
    assert len(sel) == 62
    assert all(server.file_imports(d["targetId"], ["fastapi"]) for d in sel)
    incl = [c for c in server.by["callsite"] if c["calleeName"] == "include_router"]
    assert len(incl) == 7 and all(server.file_imports(c["symbolId"], ["fastapi"]) for c in incl)


def test_roots_notwith_changes_exactly_the_two_variable_router_handlers(server):
    factories = set()
    for c in server.by["callsite"]:
        if c["calleeName"] != "include_router":
            continue
        for d in server.data[c["id"]]:
            if d["toPath"] == "a0" and d["def"] in server.cs:
                factories |= {t["symbolId"] for t in server.cs[d["def"]]["targets"]}
    rx = [r["match"]["decorator"]["name"]["regex"] for r in PACKS["fastapi"]["rules"]]
    root = PACKS["fastapi"]["mounts"][0]["roots"][0]  # the declaration under test, not a copy of its names
    root_call, not_with = root["call"]["callee"], {n["callee"] for n in root["notWith"]}
    assert root_call and not_with
    old_unmounted, new_unmounted = set(), set()
    for d in server.by["decorator"]:
        if not any(re.fullmatch(x, d["name"]) for x in rx):
            continue
        h = server.sym[d["targetId"]]
        parent = h.get("parentId")
        if parent in factories:
            continue
        names = server.in_symbol.get(parent, set())
        is_root_old = root_call in names
        is_root_new = root_call in names and not names & not_with
        if not is_root_old:
            old_unmounted.add(h["id"])
        if not is_root_new:
            new_unmounted.add(h["id"])
    diff = {server.sym[i]["name"] + "@" + server.path_of(i).rsplit("/", 1)[-1] for i in new_unmounted - old_unmounted}
    assert diff == {"join@test_app.py", "probe@test_input_bound_parity.py"}
    # no S14 APIRouter call has a prefix argument, so constructorPrefix changes nothing on S14
    cp = PACKS["fastapi"]["mounts"][0]["constructorPrefix"][0]
    assert not any(cp["arg"] in server.vs[c["id"]] for c in server.by["callsite"] if c["calleeName"] == cp["call"]["callee"])


def test_mcp_tool_rule_gate_and_args(server):
    pack = PACKS["fastmcp"]
    rx = pack["rules"][0]["match"]["decorator"]["name"]["regex"]
    sel = [d for d in server.by["decorator"] if re.fullmatch(rx, d["name"])]
    mcp_py = [d for d in sel if server.path_of(d["targetId"]).endswith("falkorchat/mcp.py")]
    assert len(mcp_py) == 25 and len(sel) == 28
    assert all(server.file_imports(d["targetId"], ["mcp.server.fastmcp", "fastmcp"]) for d in sel)
    assert all(d["args"] == [] for d in mcp_py)  # firstOf falls through to the function name
    send = next(d for d in mcp_py if server.sym[d["targetId"]]["name"] == "send_message")
    assert send["id"].startswith("de368fdf")


# ---------------------------------------------------------------- planted records (no S14 shape reaches these)

def alternatives(expr, rec, target_name, name_match):
    """The value alternatives of one `extract` expression over a decorator record (§1.2, §1.4: consts only)."""
    if "firstOf" in expr:
        for x in expr["firstOf"]:
            got = alternatives(x, rec, target_name, name_match)
            if got:
                return got
        return []
    if "arg" in expr:
        for a in rec["args"]:
            if a["arg"] == expr["arg"]:
                return [v["const"] for v in a["values"] if "const" in v]
        return []
    if "target" in expr:
        return [target_name]
    if "literal" in expr:
        return [expr["literal"]]
    if "group" in expr:
        g = name_match.group(expr["group"])
        return [g.upper() if expr.get("upper") else g]
    raise AssertionError(expr)


def decorator_keys(rule, rec, target_name):
    """Run one decorator `provides` rule over a record with `mount` empty: the keys it emits."""
    m = re.fullmatch(rule["match"]["decorator"]["name"]["regex"], rec["name"])
    if not m:
        return []
    slots = {k: alternatives(e, rec, target_name, m) for k, e in rule["extract"].items()}
    key = rule["emit"][0]["provides"]["key"]
    return [key.format(mount="", **dict(zip(slots, combo))) for combo in itertools.product(*slots.values())]


def dec(name, **args):
    """A decorator record whose arguments are consts: `dec("mcp.tool", a0="x")`, `dec("r.get", **{"kw.path": "/x"})`."""
    return {"type": "decorator", "form": "decorator", "name": name,
            "args": [{"arg": a, "values": [{"const": v}]} for a, v in args.items()]}


def rule_of(pid, rid):
    return next(r for r in PACKS[pid]["rules"] if r["id"] == rid)


def test_mcp_tool_name_comes_from_kw_name_then_a0_then_the_function_name():
    rule = rule_of("fastmcp", "tool")
    assert decorator_keys(rule, dec("mcp.tool", **{"kw.name": "lookup"}), "fn") == ["lookup"]
    assert decorator_keys(rule, dec("mcp.tool", a0="lookup"), "fn") == ["lookup"]  # FastMCP.tool's first positional is `name`
    assert decorator_keys(rule, dec("mcp.tool"), "send_message") == ["send_message"]
    assert decorator_keys(rule, dec("mcp.other", a0="x"), "fn") == []


def test_route_path_comes_from_a0_then_kw_path_and_the_verb_from_the_decorator_name():
    rule = rule_of("fastapi", "route")
    assert decorator_keys(rule, dec("router.get", a0="/x"), "h") == ["GET /x"]
    assert decorator_keys(rule, dec("router.post", **{"kw.path": "/y"}), "h") == ["POST /y"]
    assert decorator_keys(rule, dec("router.get"), "h") == []  # no path: nothing to drop onto a key
    assert decorator_keys(rule, dec("router.api_route", a0="/z"), "h") == []  # the any-method rule owns it
    assert decorator_keys(rule_of("fastapi", "route-any-method"), dec("router.api_route", **{"kw.path": "/z"}), "h") == ["* /z"]


def test_constructor_prefix_marks_a_factory_whose_router_carries_its_own_prefix():
    """The declaration is applied to planted records: an `APIRouter(prefix="/v1")` call, then one without."""
    decl = PACKS["fastapi"]["mounts"][0]["constructorPrefix"][0]
    recs = [
        {"type": "callsite", "id": "c1", "symbolId": "F", "calleeName": "APIRouter"},
        {"type": "valueSet", "callSiteId": "c1", "arg": "kw.prefix", "values": [{"const": "/v1"}]},  # what `prefix="/v1"` writes (spec §5.1)
        {"type": "callsite", "id": "c2", "symbolId": "G", "calleeName": "APIRouter"},
        {"type": "callsite", "id": "c3", "symbolId": "H", "calleeName": "Other"},
        {"type": "valueSet", "callSiteId": "c3", "arg": "kw.prefix", "values": [{"const": "/v1"}]},
    ]
    corpus = Corpus(recs)

    def carries_own_prefix(parent):
        return any(c["symbolId"] == parent and matches(decl["call"]["callee"], c["calleeName"])
                   and decl["arg"] in corpus.vs[c["id"]] for c in corpus.by["callsite"])

    assert carries_own_prefix("F")
    assert not carries_own_prefix("G")  # APIRouter() with no prefix argument
    assert not carries_own_prefix("H")  # a prefix argument on some other call


def test_mount_path_glob_matches_the_real_record_and_the_old_glob_did_not(server):
    glob = PACKS["fastmcp"]["launch"]["mountPaths"][0]["app"]["fromCall"]["externalTarget"]
    mounts = [c for c in server.by["callsite"] if c["calleeName"] == "mount"
              and server.path_of(c["symbolId"]).endswith("falkorchat/app.py")]
    hit = []
    for c in mounts:
        for d in server.data[c["id"]]:
            if d["toPath"] == "a1" and d["def"] in server.cs:
                target = server.cs[d["def"]].get("externalTarget", "")
                if matches(glob, target):
                    hit.append((c["line"], server.vs[c["id"]]["a0"]))
    assert hit == [(543, [{"const": "/mcp"}])]
    real = "mcp/server/fastmcp.py:<module>.FastMCP.streamable_http_app"
    assert not matches({"glob": "*FastMCP.streamable_http_app"}, real)
    assert matches({"glob": "**FastMCP.streamable_http_app"}, real)
    assert {c["line"] for c in mounts} == {528, 543, 551}


def test_fetch_seed_requires_external_or_unresolved_and_s14_has_one(sales):
    seeds = [c for c in sales.by["callsite"] if c["calleeName"] == "fetch"]
    assert len(seeds) == 1 and seeds[0]["resolution"] == "external" and seeds[0]["id"].startswith("41d8c78a")
    assert matches(PACKS["fetch"]["liftings"][0]["seeds"][0]["call"]["resolution"], seeds[0]["resolution"])


def test_every_record_id_cited_in_the_walkthroughs_exists(server, sales):
    ids = {r["id"][:8] for c in (server, sales) for rs in c.by.values() for r in rs if "id" in r}
    sec = TEXT[TEXT.index("## 2. Paper pack 1"): TEXT.index("## 5. Paper pack 4")]
    cited = set(re.findall(r"`(?:\w+ )?([0-9a-f]{8})`", sec)) | set(re.findall(r"(?<![\w.])([0-9a-f]{8})(?![\w])", sec))
    cited = {c for c in cited if c != "defca60f" and not c.isdigit() and re.search(r"[a-f]", c) and re.search(r"\d", c)}
    missing = sorted(cited - ids)
    assert not missing, missing


def test_target_arg_resolves_to_one_in_repo_file(server, sales):
    ends = [r["path"] for c in (server, sales) for r in c.by["file"] if r["path"].endswith("falkorchat/app.py")]
    assert ends == ["falkor-chat/server/falkorchat/app.py"]


# ---------------------------------------------------------------- the paper launch records (B-1)

def launch_rows() -> dict[str, str]:
    """The §4.3 table: record text by row id (L1.., C1)."""
    return {r[0]: r[2] for r in table_rows("| # | Source | Record")}


def stated_command_and_args(text: str) -> tuple[str, list[str]]:
    """`command X` and `args [a, b, ...]` as §4.3 writes a launch record."""
    cmd = re.search(r"`command ([^`]+)`", text).group(1)
    args = re.search(r"`args \[([^\]]*)\]`", text).group(1).split(", ")
    return cmd, args


# argv words as the shell tokenizer yields them from the source lines §4.3 cites (the spec abbreviates the last one)
L2_ARGV = ['$VENV_DIR/bin/uvicorn', "falkorchat.app:app", "$UVICORN_ARGS", '${LOG_CONFIG_ARGS[@]+"${LOG_CONFIG_ARGS[@]}"}']


def test_paper_launch_records_pass_through_the_frozen_argv_gate():
    rows = launch_rows()
    # L1: the row's own argv is left unchanged by the gate
    cmd1, args1 = stated_command_and_args(rows["L1"])
    assert gate_argv([cmd1, *args1]) == [cmd1, *args1] and cmd1 == "uvicorn"
    # L2: the gate's output is what the row states (command AND args), word for word
    cmd2, args2 = stated_command_and_args(rows["L2"])
    assert gate_argv(L2_ARGV) == [cmd2, *args2], (cmd2, args2)
    assert cmd2 == "<redacted>" and args2[0] == "falkorchat.app:app"
    # L4 is stated to be the same shape as L2
    assert "same shape as L2" in rows["L4"]
    # C1: the config record's keyPath/bindValue pass the gate
    kp = re.search(r"`keyPath ([^`]+)`", rows["C1"]).group(1)
    bv = re.search(r'`bindValue "([^"]+)"`', rows["C1"]).group(1)
    assert bind_value_allowed(kp, bv)
    # consequences the walkthrough states: basename uvicorn cannot match the redacted command
    assert matches({"basename": "uvicorn"}, cmd1)
    assert not matches({"basename": "uvicorn"}, cmd2)
    tf = PACKS["fastmcp"]["launch"]["targetArg"][0]
    assert "command" not in tf and re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", args2[0])


def test_env_port_heuristic_claims():
    ep = PACKS["fastmcp"]["launch"]["envPorts"][0]
    assert matches(ep["envVar"], "AGENT_TEAM_PORT") and not matches(ep["excludeVars"], "AGENT_TEAM_PORT")
    assert matches(ep["envVar"], "FALKORDB_PORT") and matches(ep["excludeVars"], "FALKORDB_PORT")
    # documented soundness limit: exclusion is anchored at the start of the name
    assert matches(ep["envVar"], "AGENT_TEAM_FALKORDB_PORT") and not matches(ep["excludeVars"], "AGENT_TEAM_FALKORDB_PORT")


def test_prefix_matcher_boundaries():
    assert matches({"prefix": "/mcp"}, "/mcp") and matches({"prefix": "/mcp"}, "/mcp/x")
    assert not matches({"prefix": "/mcp"}, "/mcpx")
    assert matches({"prefix": "fastapi"}, "fastapi.responses") and not matches({"prefix": "fastapi"}, "fastapix")


def test_normalize_path_reproduces_the_vectors_the_spec_states():
    para = TEXT[TEXT.index("**Path normalization (testable).**"): TEXT.index("Why `regex` is restricted")]
    stated = [(a or b, c) for a, b, c in re.findall(r'`(?:normalize_path\("([^"]*)"\)|([^`"(]+))` is `([^`]+)`', para)]
    assert stated == [("/mcp/%2E%2E/admin", "/admin"), ("/mcp/./x", "/mcp/x"), ("/a//b", "/a/b")]
    for raw, want in stated:
        assert normalize_path(raw) == want, raw
    # ... and the consequences it draws
    assert not path_matches({"prefix": "/mcp"}, "/mcp/%2E%2E/admin")
    assert path_matches({"prefix": "/mcp"}, "/mcp/./x")
    assert "so `{prefix: /mcp}` does not match it" in para.replace("\n", " ")


def test_normalize_path_matches_spec3_step_4_to_6_examples():
    for raw, want in {
        "/a/%2E%2E/b": "/b", "/a/%7Euser/%2f": "/a/~user/%2F",
        "/a/b/..": "/a/", "/a/b/.": "/a/b/", "/a/..": "/", "/a/b/": "/a/b/", "": "/", "x/y": "/x/y",
        "/../a": "/a",
        "/a/%2%45%2%45/b": "/a/%2E%2E/b",  # a `%` that begins no escape: one pass, not two
    }.items():
        assert normalize_path(raw) == want, raw


def test_both_operands_are_normalized_once_before_exact_prefix_and_template():
    assert path_matches({"exact": "/mcp/../shop/x"}, "/shop//x")
    assert path_matches({"prefix": "/a/./mcp"}, "/a/mcp/x")
    assert not path_matches({"prefix": "/mcp"}, "/mcpx")
    assert path_matches({"template": "/shop/{}/health"}, "/shop/%7Eme/health")
    assert not path_matches({"exact": "/a/b"}, "/a/b/")  # a trailing `/` is kept (spec §6 step 8a note)


def test_template_matcher_segments():
    t = lambda tpl, s: matches({"template": tpl}, s)
    assert t("/users/{}", "/users/42") and not t("/users/{}", "/users/") and not t("/users/{}", "/users/4/2")
    assert t("/files/{**}", "/files/a") and t("/files/{**}", "/files/a/b/c") and not t("/files/{**}", "/files/")
    assert t("/a/{}/b/{}", "/a/x/b/y") and not t("/a/{}/b/{}", "/a//b/y")
    assert not t("/files/{**}", "/file/a")


def test_topic_packs_regexes_and_transport_rule():
    kafka = PACKS["kafka"]
    assert matches(kafka["rules"][0]["match"]["callsite"]["callee"], "send")
    assert not matches(kafka["rules"][0]["match"]["callsite"]["callee"], "sender")
    assert PACKS["paper-inhouse-bus"]["vocab"][0]["transport"] == "inhouse-paper-inhouse-bus"
    g = glob_re("inhouse-paper-inhouse-bus:billing.invoice.*")
    assert g.fullmatch("inhouse-paper-inhouse-bus:billing.invoice.paid")
    assert not g.fullmatch("inhouse-paper-inhouse-bus:billing.invoice.a/b")  # `*` stops at `/`
    # §5.2 step 3: the subscribe pattern of the §5.1 snippet is the regex key, read as the RE2 subset
    pattern = re.search(r'c\.subscribe\(pattern=r"([^"]+)"\)', TEXT).group(1)
    rule = next(r for r in kafka["rules"] if r["id"] == "consume-pattern")
    assert rule["emit"][0]["consumes"]["keyForm"] == "regex" and rule["emit"][0]["consumes"]["key"] == "kafka:{pattern}"
    check_regex(pattern)
    assert re.fullmatch(pattern, "orders.created") and not re.fullmatch(pattern, "xorders.created")
    assert f"kafka:{pattern}" in TEXT


def test_the_topic_direction_markers_are_gone_from_the_spec():
    """Option A (ruled): a publisher PROVIDES and a subscriber CONSUMES; the interim direction marker is deleted."""
    assert not re.search(r"provisio[n]al", TEXT)
    for p in (PACKS["kafka"], PACKS["paper-inhouse-bus"]):
        for r in p["rules"]:
            assert {k for e in r["emit"] for k in e} <= {"provides", "consumes"}


# ---------------------------------------------------------------- infrastructure signatures (§8)

def sigs() -> list[dict]:
    return assembled_infra_pack()["signatures"]


def negatives() -> dict:
    return next(b for _, b in BLOCKS if isinstance(b, dict) and "near-miss" in b)


RUNTIMES = {r["runtime"]: r["regex"] for r in PACKS["infra-signatures"]["runtimes"]}
COMPOSE = re.compile(r"^([\w.-]+)\s+\| ")


def blocks_of(evidence: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    for ln in evidence.splitlines():
        m = COMPOSE.match(ln)
        out[m.group(1) if m else ""].append(ln[m.end():] if m else ln)
    return out


def detect(lines: list[str]) -> str | None:
    for rt, rx in RUNTIMES.items():
        if any(re.search(rx, ln) for ln in lines):
            return rt
    return None


def signatures_for(evidence: str, entries=None) -> set[str]:
    hits = set()
    for _, lines in blocks_of(evidence).items():
        rt = detect(lines)
        for e in entries or sigs():
            if rt is not None and e["runtime"] not in (rt, "any"):
                continue
            if any(re.search(e["regex"], ln) for ln in lines):
                hits.add(e["id"])
    return hits


def test_signature_inventory_and_grammar():
    s = sigs()
    assert len(s) == 45
    assert collections.Counter(e["runtime"] for e in s) == {"python": 13, "jvm": 9, "node": 6, "dotnet": 6, "go": 6, "any": 5}
    assert len({e["id"] for e in s}) == 45
    for e in s:
        pat = rf"{e['runtime']}-(?:{e['class']}|third-party)(?:-[a-z0-9-]+)?"
        assert re.fullmatch(pat, e["id"]), e["id"]
        assert ("library" in e) == (e["class"] == "third-party-exception" and e["runtime"] in {"python", "jvm"}) or e["runtime"] in {"node", "dotnet", "go"}
    captured = [e for e in s if e.get("sampleSource") == "captured"]
    authored = [e for e in s if e.get("sampleSource", "authored") == "authored"]
    assert (len(captured), len(authored)) == (11, 34)
    # §8.2 states the split in prose: "11 carry a `captured` sample ... 34 are `authored`"
    said = re.search(r"Of the (\d+)\s+entries,\s+\*\*(\d+) carry a `captured` sample\*\*.*?\*\*(\d+) are `authored`", TEXT, re.S)
    assert said and tuple(map(int, said.groups())) == (len(s), len(captured), len(authored)), said and said.groups()
    # ... "each for one alternative; only the 3 single-alternative ones are wholly supported"
    whole = sorted(e["id"] for e in captured if len(alternatives_of(e["regex"])) == 1)
    assert whole == ["jvm-dns-failure", "jvm-oom-kill", "python-oom-kill"]
    assert re.search(rf"only the {len(whole)} single-alternative ones\s+\(`python-oom-kill`, `jvm-dns-failure`, `jvm-oom-kill`\)", TEXT)


def alternatives_of(rx: str, any_depth: bool = True) -> list[str]:
    """Split a regex at every unescaped `|` outside a character class, at any group depth.

    §8.1 defines "supported" over *top-level* alternatives, but §8.2's "only the 3 single-alternative ones"
    only holds when alternation inside a group counts too (`jvm-connection-refused` is one top-level
    alternative with three inside its group); this counts the stricter way, which is what a sample proves.
    """
    out, cls, esc, cur, depth = [], False, False, "", 0
    for ch in rx:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif cls:
            cls = ch != "]"
        elif ch == "[":
            cls = True
        elif ch in "()":
            depth += 1 if ch == "(" else -1
        elif ch == "|" and (any_depth or depth == 0):
            out.append(cur)
            cur = ""
            continue
        cur += ch
    return out + [cur]


def test_every_sample_matches_its_own_regex_and_no_negative_matches_any_entry():
    for e in sigs():
        assert re.search(e["regex"], e["sample"]), e["id"]
    neg = negatives()
    lines = list(neg["shared"]) + [l for ls in neg["near-miss"].values() for l in ls]
    for l in lines:
        hit = [e["id"] for e in sigs() if re.search(e["regex"], l)]
        assert not hit, (l, hit)
    for cls, ls in neg["near-miss"].items():
        assert len(ls) >= 3, cls
    assert set(neg["near-miss"]) == {e["class"] for e in sigs()}


def test_captured_samples_are_the_recorded_lines():
    rows = [l.rstrip("\n").split("\t", 1) for l in CAPTURES.read_text().splitlines() if l and not l.startswith("#")]
    cap = dict(rows)
    claimed = {e["id"]: e["sample"] for e in sigs() if e.get("sampleSource") == "captured"}
    assert set(cap) == set(claimed)
    for k, v in claimed.items():
        assert cap[k] == v, k


def test_cross_runtime_and_mixed_blob_cases():
    neg = negatives()
    for c in neg["cross-runtime"]:
        got = {e["id"] for e in sigs() if re.search(e["regex"], c["line"])}
        assert got == set(c["ids"]), (c["line"], got)
    blob = neg["mixed-blob"]
    assert signatures_for(blob["evidence"]) == set(blob["ids"])
    # without per-block narrowing (whole evidence as one block) the node signature is lost
    whole = "\n".join(COMPOSE.sub("", l) for l in blob["evidence"].splitlines())
    assert "node-connection-refused" not in signatures_for(whole)


def test_runtime_detection_picks_the_right_runtime_per_sample_line():
    samples = {
        "python": 'Traceback (most recent call last):', "node": "    at Socket.<anonymous> (/app/x.js:10:5)",
        "jvm": "    at com.acme.Foo.bar(Foo.java:12)", "dotnet": "   at Acme.Foo.Bar() in /src/Foo.cs:line 12",
        "go": "goroutine 1 [running]:",
    }
    for rt, line in samples.items():
        assert detect([line]) == rt, rt


def test_regex_edits_that_broaden_an_entry_fail_the_negatives():
    neg = negatives()
    lines = list(neg["shared"]) + [l for ls in neg["near-miss"].values() for l in ls]
    by_id = {e["id"]: e for e in sigs()}
    mutants = {
        "python-timeout": by_id["python-timeout"]["regex"] + "|timed out",
        "any-oom-kill": by_id["any-oom-kill"]["regex"].replace(r"137\b", "137"),
        "any-oom-kill/bare": by_id["any-oom-kill"]["regex"] + "|oom-kill",
        "python-oom-kill": by_id["python-oom-kill"]["regex"] + "|memory",
        "any-connection-refused": by_id["any-connection-refused"]["regex"] + r"|refused",
        "go-timeout": ".*",
    }
    for name, rx in mutants.items():
        assert any(re.search(rx, l) for l in lines), f"mutant {name} survived the negatives"


def test_known_recall_cases_now_match_a_connection_refused_class():
    cases = {
        "psycopg2.OperationalError: connection to server at \"db\" port 5432 failed: Connection refused": "connection-refused",
        "OSError: [Errno 113] No route to host": "connection-refused",
        "ConnectionResetError: [Errno 104] Connection reset by peer": "connection-refused",
        "redis.exceptions.ConnectionError: Error 111 connecting to redis:6379. Connection refused.": "connection-refused",
        "ERROR: container exit code 137": None,
    }
    for line, cls in cases.items():
        got = {e["class"] for e in sigs() if re.search(e["regex"], line)}
        if cls:
            assert cls in got, line
    assert not [e for e in sigs() if re.search(e["regex"], "exit code 1370")]
    assert [e["id"] for e in sigs() if re.search(e["regex"], "container exited with code 137")] == ["any-oom-kill"]


# ---------------------------------------------------------------- §8.1 deduplication, exactly

TP = "third-party-exception"


def pairs_of(evidence: str, entries=None) -> list[dict]:
    """§8.1 "Dedupe, exactly": per block and line, one pair per class; third-party entries add their
    `library` to the other classes' pairs and form a pair of their own only when no other class matched."""
    out = []
    for block, lines in blocks_of(evidence).items():
        rt = detect(lines)
        for line in lines:
            hit = [e for e in (entries or sigs())
                   if (rt is None or e["runtime"] in (rt, "any")) and re.search(e["regex"], line)]
            plain = [e for e in hit if e["class"] != TP]
            libs = {e["library"] for e in hit if e["class"] == TP and "library" in e}
            classes = {e["class"] for e in plain} or ({TP} if hit else set())
            for cls in sorted(classes):
                members = [e for e in hit if e["class"] == cls]
                out.append({"block": block, "line": line, "class": cls, "ids": sorted(e["id"] for e in members),
                            "libraries": sorted(libs if plain else {e["library"] for e in members if "library" in e})})
    return out


def test_the_psycopg_line_is_one_connection_refused_pair_carrying_its_library():
    for case in negatives()["dedupe"]:
        got = pairs_of(case["line"])
        assert [(p["class"], p["libraries"]) for p in got] == [(w["class"], w["libraries"]) for w in case["pairs"]], case["line"]
    # the spec's example, spelled out: the third-party entry and the generic one are two *entries*, one pair
    (pair,) = pairs_of(negatives()["dedupe"][0]["line"])
    assert pair["ids"] == ["any-connection-refused"] and pair["libraries"] == ["psycopg"]
    assert any(e["id"] == "python-third-party-psycopg" and re.search(e["regex"], pair["line"]) for e in sigs())


def test_a_third_party_match_is_its_own_pair_only_when_no_other_class_matched():
    (pair,) = pairs_of("pymongo.errors.ServerSelectionTimeoutError: db:27017")
    assert (pair["class"], pair["ids"], pair["libraries"]) == (TP, ["python-third-party-pymongo"], ["pymongo"])
    # the cross-runtime httpx line: dns-failure matches too, so the library folds into that pair
    (pair,) = pairs_of("httpx.ConnectError: [Errno -2] Name or service not known")
    assert (pair["class"], pair["ids"], pair["libraries"]) == ("dns-failure", ["python-dns-failure"], ["httpx"])


def test_the_library_is_added_to_every_other_class_pair_of_the_line():
    pairs = pairs_of("httpx.ConnectError: [Errno -2] Name or service not known [Errno 110]")
    assert [(p["class"], p["libraries"]) for p in pairs] == [("dns-failure", ["httpx"]), ("timeout", ["httpx"])]


def test_one_failure_is_one_pair_per_class_it_matches():
    # two entries of one class on one line: one pair, ids sorted
    (pair,) = pairs_of("ConnectionRefusedError: [Errno 111] Connection refused")
    assert pair["class"] == "connection-refused" and pair["ids"] == ["any-connection-refused", "python-connection-refused"]
    # §8.1's own example: a Go `i/o timeout` inside a DNS lookup is two pairs
    pairs = pairs_of("dial tcp: lookup db on 127.0.0.11:53: server misbehaving: i/o timeout")
    assert [p["class"] for p in pairs] == ["dns-failure", "timeout"]
    assert [p["ids"] for p in pairs] == [["go-dns-failure"], ["go-timeout"]]


def test_the_block_is_part_of_the_pair_key():
    ev = "api-1  | ECONNREFUSED 1.2.3.4:5\nweb-1  | ECONNREFUSED 1.2.3.4:5"
    pairs = pairs_of(ev)
    assert sorted((p["block"], p["class"]) for p in pairs) == [("api-1", "connection-refused"), ("web-1", "connection-refused")]
    # the same line twice in one block is two lines, hence two pairs (the key includes the line, not its text)
    assert len(pairs_of("ECONNREFUSED a\nECONNREFUSED b")) == 2


def test_the_dedupe_block_is_what_the_spec_example_says():
    text = TEXT[TEXT.index("- **Dedupe, exactly.**"): TEXT.index("### 8.2 Patterns")]
    assert "one `connection-refused` pair with\n  `libraries [psycopg]`" in text
    cases = negatives()["dedupe"]
    assert cases
    for case in cases:
        assert [{"class": p["class"], "libraries": p["libraries"]} for p in pairs_of(case["line"])] == case["pairs"]
    assert cases[0]["pairs"] == [{"class": "connection-refused", "libraries": ["psycopg"]}]


def test_two_third_party_entries_on_one_line_are_one_pair_naming_both_libraries():
    """§8.1's prose example (not in the YAML dedupe block): read the line from the spec text."""
    text = re.sub(r"\s+", " ", TEXT[TEXT.index("- **Dedupe, exactly.**"): TEXT.index("### 8.2 Patterns")])
    line = re.search(r"`(sqlalchemy\.exc\.OperationalError: [^`]+)` is one `third-party-exception` pair with `libraries \[psycopg, sqlalchemy\]`", text).group(1)
    (pair,) = pairs_of(line)
    assert pair["class"] == TP and pair["libraries"] == ["psycopg", "sqlalchemy"]
    assert pair["ids"] == ["python-third-party-psycopg", "python-third-party-sqlalchemy"]


def test_the_when_vocabulary_table_equals_the_checkers_structures():
    sec = TEXT[TEXT.index("The **`when` vocabulary**"): TEXT.index("The pack (S32 encodes it")]
    rows = {}
    for ln in sec.splitlines():
        m = re.match(r"\| (.+?) \| (.+) \|$", ln)
        if m and not m.group(1).startswith(("Body", "---")):
            rows[m.group(1).strip("`")] = {}
            for part in m.group(2).replace("\\|", ",").split(";"):
                feat, _, vals = part.strip().partition(" ")
                rows[m.group(1).strip("`")][feat.strip("`")] = [v.strip() for v in vals.split(",")] if vals else []
    kinds = {k: v for k, v in rows.items() if k in SEV_PREFIX}
    assert set(kinds) == set(SEV_PREFIX) and len(rows) == 6
    enums, bools, classes = {}, set(), {}
    for kind, feats in kinds.items():
        for feat, vals in feats.items():
            if vals == ["true", "false"]:
                bools.add(feat)
            elif feat == "class":
                classes[kind] = set(vals)
            else:
                enums[feat] = set(vals)
    caps_row = next(v for k, v in rows.items() if k.startswith("caps"))
    file_line = next(l for l in sec.splitlines() if l.startswith("| file features"))
    assert all(f"`{f}`" in file_line for f in FILE_FEATURES)
    enums["fileRole"] = set(caps_row["fileRole"])
    assert enums == FEATURES and bools == BOOL_FEATURES and classes == CLASSES
    for kind, feats in kinds.items():
        assert KIND_FEATURES[kind] == set(feats) | FILE_FEATURES, kind


def test_supported_counts_alternatives_at_any_group_depth_and_agrees_with_the_8_2_count():
    s81 = re.sub(r"\s+", " ", TEXT[TEXT.index("- **Dedupe") - 6000: TEXT.index("- **Dedupe")])
    assert "every alternative of its regex, at any group depth (not only top-level `|`)" in s81
    by_id = {e["id"]: e for e in sigs()}
    rx = by_id["jvm-connection-refused"]["regex"]
    assert len(alternatives_of(rx, any_depth=False)) == 1 and len(alternatives_of(rx)) == 3
    whole = sorted(e["id"] for e in sigs() if e.get("sampleSource") == "captured" and len(alternatives_of(e["regex"])) == 1)
    n = int(re.search(r"only the (\d+) single-alternative ones", TEXT).group(1))
    assert len(whole) == n == 3
    assert all(f"`{i}`" in TEXT[TEXT.index("only the 3 single"): TEXT.index("only the 3 single") + 200] for i in whole)


# ---------------------------------------------------------------- known false matches, holes (§8.2)

def test_known_false_matches_still_match_exactly_the_ids_the_spec_lists():
    kfm = negatives()["known-false-matches"]
    assert len(kfm) == 2
    for case in kfm:
        got = {e["id"] for e in sigs() if re.search(e["regex"], case["line"])}
        assert got == set(case["ids"]), (case["line"], got)
    # they are imprecision the spec documents, so they are in no negative list
    neg = negatives()
    clean = set(neg["shared"]) | {l for ls in neg["near-miss"].values() for l in ls}
    assert not clean & {c["line"] for c in kfm}
    assert "S32 asserts they DO match" in TEXT


def test_bare_no_route_and_network_unreachable_match_nothing_but_the_specific_forms_do():
    for bare in ("No route to host", "Network is unreachable", "network is unreachable", "no route to host",
                 "Network is unreachable (cached)"):
        assert not [e["id"] for e in sigs() if re.search(e["regex"], bare)], bare
    # the near-miss the spec lists for the class is exactly this line
    assert "Network is unreachable (cached)" in negatives()["near-miss"]["connection-refused"]
    # the forms the Known holes paragraph says still match
    forms = {
        "OSError: [Errno 113] No route to host": "python-connection-refused",
        "Error: connect EHOSTUNREACH 10.0.0.1:80": "node-connection-refused",
        "Error: connect ENETUNREACH ::1:80": "node-connection-refused",
        "java.net.NoRouteToHostException: No route to host": "jvm-connection-refused",
        "dial tcp 10.0.0.1:80: connect: no route to host": "go-connection-refused",
        "dial tcp 10.0.0.1:80: connect: network is unreachable": "go-connection-refused",
    }
    for line, sid in forms.items():
        assert sid in {e["id"] for e in sigs() if re.search(e["regex"], line)}, line
    holes = TEXT[TEXT.index("**Known holes"): TEXT.index("**P5 validation")].replace("\n", " ")
    assert "as bare text are no longer matched outside the Python errno, Node, JVM and Go forms" in holes


def test_go_timeout_alternatives_each_match_their_own_line():
    rx = next(e["regex"] for e in sigs() if e["id"] == "go-timeout")
    assert len(alternatives_of(rx)) == 3
    for line in ('Get "http://x": context deadline exceeded', "read tcp 1.2.3.4:5->6.7.8.9:10: i/o timeout",
                 "Get x: net/http: request canceled (Client.Timeout exceeded while awaiting headers)"):
        assert re.search(rx, line), line


# ---------------------------------------------------------------- §1.9 / §9.1 consistency

def test_example_config_is_in_the_registry_table_with_finding_in_its_on_set():
    rows = {r[0].strip("`"): r for r in table_rows("| Code | Status | On (added")}
    assert "example-config" in rows
    code, status, on, used = rows["example-config"][:4]
    assert status == "possible" and "`Finding`" in [x.strip() for x in on.split(",")]
    assert "§9.1" in used
    # the merged registry (spec3 §7.2 plus this table) therefore lists it on Finding as well as Link/BOUND_BY
    on_all = registry()["example-config"]["on"]
    assert "Finding" in on_all and "Link" in on_all
    # §9.1 relies on it: an example-role file ranks 0 (cap) and carries `example-config` beside `severity-rule`
    sec = TEXT[TEXT.index("### 9.1 Method"): TEXT.index("`when` maps a feature")]
    assert "reason `example-config`" in sec and "`Finding` is added to that code's `On` set in §1.9" in sec
    assert sev_pack()["formula"]["caps"] == [{"when": {"fileRole": "example"}, "max": 0}]
    # the other §1.9 rows keep their On sets
    assert "Finding" in rows["severity-rule:<id>"][2] and "PROVIDES" in rows["key-match"][2]


# ---------------------------------------------------------------- severity (§9)

def sev_pack() -> dict:
    return PACKS["severity-rules"]


def feature_match(when: dict, f: dict) -> bool:
    for k, want in when.items():
        if k in ("path", "basename"):
            have = f.get("file", "")
            have = have.rsplit("/", 1)[-1] if k == "basename" else have
            if not matches(want, have):
                return False
        elif f.get(k) != want:
            return False
    return True


def rank(formula: dict, kind: str, f: dict, order: str = "spec") -> int:
    spec = formula[kind]
    bases = [e["value"] for e in spec["base"] if feature_match(e["when"], f)]
    assert len(bases) == 1, (kind, f, bases)
    adj = sum(e["value"] for e in spec.get("adjust", []) if feature_match(e["when"], f))
    pen = sum(e["value"] for e in spec.get("penalties", []) if feature_match(e["when"], f))
    if order == "spec":
        r = min(3, bases[0] + adj) - pen
    elif order == "cap-last":
        r = min(3, bases[0] + adj - pen)
    elif order == "nocap":
        r = bases[0] + adj - pen
    else:
        raise AssertionError(order)
    r = max(0, r)
    for c in formula.get("caps", []):
        if feature_match(c["when"], f):
            r = min(r, c["max"])
    return r


PREFIX = {"injection": "INJ", "authorization": "AUTHZ", "configuration": "RCFG", "secrets": "SEC"}


def vector_rows():
    rows = table_rows("| # | Kind | Features | Rank | Row |")
    out = []
    for r in rows:
        f = {}
        for tok in r[2].split():
            k, v = tok.split("=", 1)
            f[k] = {"true": True, "false": False}.get(v, v)
        f.setdefault("chainConfirmed", True)
        f.setdefault("fileRole", "primary")
        out.append((r[0], r[1], f, int(r[3]), r[4]))
    return out


# Test-owned vectors beside §9.3's 44 (the spec's table is not edited here): every §9.3 vector uses
# `tests/test_x.py` for the test path and `dev`/`override` for the dev variant, so a narrower test-path
# regex, a lost `e2e`/`__tests__` alternative or a lost `local`/`test` token survived all 44.
# (id, kind, features, rank, why); the base and penalty are the spec's: head P = 2, test path -1, dev variant -1.
EXTRA_VECTORS = [
    ("X1", "secrets", "state=head class=P file=src/tests/unit/test_x.py", 1, "tests/ below the top level"),
    ("X2", "secrets", "state=head class=P file=web/__tests__/a.test.ts", 1, "__tests__ directory"),
    ("X3", "secrets", "state=head class=P file=e2e/login.spec.ts", 1, "e2e directory"),
    ("X4", "secrets", "state=head class=P file=test/helpers.py", 1, "singular test/ directory"),
    ("X5", "secrets", "state=head class=P file=src/attests/a.py", 2, "`tests` as a substring of a directory"),
    ("X6", "secrets", "state=head class=P file=src/e2e.py", 2, "e2e as a file name, not a directory"),
    ("X7", "configuration", "class=exposed-service file=docker-compose.local.yml", 1, "local token"),
    ("X8", "configuration", "class=exposed-service file=.env.local", 1, "local token, dotfile"),
    ("X9", "configuration", "class=exposed-service file=docker-compose.test.yml", 1, "test token"),
    ("X10", "configuration", "class=exposed-service file=deploy/override.yml", 1, "override token as the whole stem"),
    ("X11", "configuration", "class=exposed-service file=docker-compose.localhost.yml", 2, "local as a substring of a token"),
]


def extra_vector_rows():
    out = []
    for vid, kind, feats, want, _why in EXTRA_VECTORS:
        f = {k: {"true": True, "false": False}.get(v, v) for k, v in (t.split("=", 1) for t in feats.split())}
        f.setdefault("chainConfirmed", True)
        f.setdefault("fileRole", "primary")
        out.append((vid, kind, f, want, f"{PREFIX[kind]}-{4 - want}"))
    return out


def run_vectors(formula: dict, order="spec") -> list[str]:
    bad = []
    for vid, kind, f, want, row in vector_rows() + extra_vector_rows():
        try:
            got = rank(formula, kind, f, order)
        except AssertionError:
            bad.append(vid)
            continue
        if got != want:
            bad.append(vid)
    return bad


def test_vectors_reproduce_and_name_the_derived_row():
    vs = vector_rows()
    assert len(vs) == 44
    assert run_vectors(sev_pack()["formula"]) == []
    for vid, kind, f, want, row in vs:
        assert row == f"{PREFIX[kind]}-{4 - want}", vid


def test_rows_are_the_derivation_of_the_formula():
    rows = {r["id"]: r for r in sev_pack()["rows"]}
    assert len(rows) == 16
    for k, pre in PREFIX.items():
        for rk in range(4):
            assert rows[f"{pre}-{4 - rk}"]["rank"] == rk and rows[f"{pre}-{4 - rk}"]["finding"] == k
    used = {row for *_, row in vector_rows()}
    assert used == set(rows), f"rows never exercised: {sorted(set(rows) - used)}"


def test_the_bases_are_exhaustive_and_ranks_stay_in_range():
    f_ = sev_pack()["formula"]
    import itertools
    spaces = {
        "injection": dict(source=["network", "local"], sink=["exec", "data"], chainConfirmed=[True, False], sanitizer=["none", "partial"]),
        "authorization": dict(entry=["network", "internal"], operation=["write", "read"], chainConfirmed=[True, False], authCheck=["none", "partial"]),
        "configuration": dict(**{"class": ["host-compromise", "exposed-service", "exposed-app", "hygiene"]}, file=["a.yml", "a.dev.yml"]),
        "secrets": dict(state=["head", "removed"], **{"class": ["K", "P", "G"]}, flow=[True, False], file=["src/a.py", "tests/a.py"]),
    }
    for kind, sp in spaces.items():
        for combo in itertools.product(*sp.values()):
            f = dict(zip(sp, combo), fileRole="primary")
            assert 0 <= rank(f_, kind, f) <= 3


def mutants(formula: dict):
    import copy
    names = []
    for kind in ("injection", "authorization", "configuration", "secrets"):
        for section in ("base", "adjust", "penalties"):
            for i, e in enumerate(formula[kind].get(section, [])):
                for delta in (-1, 1) if section == "base" else (-1, 1, 2):
                    if e["value"] + delta < 0 or (section == "base" and e["value"] + delta > 3):
                        continue
                    m = copy.deepcopy(formula)
                    m[kind][section][i]["value"] += delta
                    names.append((f"{kind}.{section}[{i}]{delta:+d}", m, "spec"))
        for section in ("adjust", "penalties"):
            for i, e in enumerate(formula[kind].get(section, [])):
                m = copy.deepcopy(formula)
                del m[kind][section][i]
                names.append((f"{kind}.{section}[{i}] removed", m, "spec"))
    m = copy.deepcopy(formula); m["caps"] = []
    names.append(("no example cap", m, "spec"))
    names.append(("cap after penalty", formula, "cap-last"))
    names.append(("no ceiling", formula, "nocap"))
    # two-level collapse: network x exec / clean else low, and the same for authorization
    for kind, key in (("injection", "source"), ("authorization", "entry")):
        m = copy.deepcopy(formula)
        for e in m[kind]["base"]:
            e["value"] = 3 if e["value"] == 3 else 1
        names.append((f"{kind} two-level base", m, "spec"))
    # dev variant by substring instead of token
    m = copy.deepcopy(formula)
    m["configuration"]["penalties"][0]["when"] = {"basename": {"regex": ".*(?:dev|local|test|override).*"}}
    names.append(("substring dev variant", m, "spec"))
    # flow adjustment applies to removed secrets
    m = copy.deepcopy(formula)
    m["secrets"]["adjust"][0]["when"] = {"flow": True}
    names.append(("flow on removed", m, "spec"))
    # test-path penalty by substring
    m = copy.deepcopy(formula)
    m["secrets"]["penalties"][0]["when"] = {"path": {"regex": ".*test.*"}}
    names.append(("substring test path", m, "spec"))
    # test-path penalty narrowed: top level only, without e2e, without __tests__, without the singular
    for label, rx in (("top-level tests/ only", r"tests?/.*"),
                      ("no e2e", r"(?:.*/)?(?:tests?|__tests__)/.*"),
                      ("no __tests__", r"(?:.*/)?(?:tests?|e2e)/.*"),
                      ("plural tests only", r"(?:.*/)?(?:tests|__tests__|e2e)/.*")):
        m = copy.deepcopy(formula)
        m["secrets"]["penalties"][0]["when"] = {"path": {"regex": rx}}
        names.append((f"test path: {label}", m, "spec"))
    # dev variant loses one token at a time
    for token in ("dev", "local", "test", "override"):
        m = copy.deepcopy(formula)
        rx = m["configuration"]["penalties"][0]["when"]["basename"]["regex"]
        assert token in rx
        kept = "|".join(t for t in ("dev", "local", "test", "override") if t != token)
        m["configuration"]["penalties"][0]["when"]["basename"]["regex"] = rx.replace("dev|local|test|override", kept)
        names.append((f"dev variant without {token}", m, "spec"))
    return names


def test_extra_vectors_name_the_row_their_rank_derives():
    assert len(EXTRA_VECTORS) == 11
    assert run_vectors(sev_pack()["formula"]) == []
    rows = {r["id"] for r in sev_pack()["rows"]}
    assert all(row in rows for *_, row in extra_vector_rows())
    # the vectors' own premises are the spec's: head P is base 2, the penalties weigh 1
    base = next(e["value"] for e in sev_pack()["formula"]["secrets"]["base"] if e["when"] == {"state": "head", "class": "P"})
    assert base == 2 and sev_pack()["formula"]["secrets"]["penalties"][0]["value"] == 1
    assert sev_pack()["formula"]["configuration"]["penalties"][0]["value"] == 1


def test_the_narrowed_test_path_and_missing_dev_tokens_are_among_the_mutants():
    names = [n for n, *_ in mutants(sev_pack()["formula"])]
    for want in ("test path: top-level tests/ only", "test path: no e2e", "test path: no __tests__", "dev variant without local"):
        assert want in names


def test_every_mutant_of_the_formula_fails_at_least_one_vector():
    survivors = []
    for name, m, order in mutants(sev_pack()["formula"]):
        if not run_vectors(m, order):
            survivors.append(name)
    assert not survivors, survivors


def test_dev_variant_basename_regex_on_real_names():
    rx = sev_pack()["formula"]["configuration"]["penalties"][0]["when"]["basename"]
    yes = ["docker-compose.dev.yml", "docker-compose.override.yml", ".env.local", "compose.test.yml", "Dockerfile.dev", "dev.yml"]
    no = ["docker-compose.yml", "docker-compose.latest.yml", "contest.yml", "device.yml", "attestation.yml", "developers.yml", "localization.yml"]
    assert all(matches(rx, n) for n in yes) and not any(matches(rx, n) for n in no)
