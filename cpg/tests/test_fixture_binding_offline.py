"""P1-53a: the offline pin of defect D3 (`docs/reviews/connected-code-graph-p1-fixtures-binding-rca.md`, R1-R6).

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-53 (53a). The fixture tree must satisfy the P1 binding contract the Binder
implements: a server binds through its `command` entry and the `Dockerfile` beside the script (the script hop is lexical and stored
nowhere), a Dockerfile `CMD` resolves to a unit by path suffix, a prompt mention is never `confirmed`, a loopback port two units claim
is `port-shared`, a shell consumer is bound by no path, and an `mcp_tool` consumer whose URL names loopback 8200 binds to unit `svc`.

**No Joern, no store.** The real build (`ccg.build.build_unit`) runs over the config and shell units of `fx-svc` and `fx-job` of the
materialized fixture repositories; every Python and TypeScript unit is a header plus `file` records of its tracked files, and the
code-level facts a Joern build would supply (providers, the prompt mention, the loopback consumer, the `:8200/mcp` consumers) are
synthetic, in the style of `linker_bind_helpers.py`, placed at the locators of the manifest rows. They go through the real
`run_packs`, `Binder` and `link`.

**Expected values do not come from the Binder.** The launch units are a table of Dockerfile -> unit, checked against an independent
parse of each `CMD` and the path-suffix rule; the `BOUND_BY` entries are resolved through the bundles' records at the locators
(path + keyPath) of the manifest `aux` rows, never by id; every (status, reason) is the manifest row's.

`test_the_fixtures_as_committed_meet_the_binding_contract` is red until P1-53b edits the fixtures (P1-52 is not needed offline: the
fastmcp `streamablehttp_client` URL binding stands in for the TS pack's `new URL(...)` binding, `scope: file` rather than `scope: unit`;
the `scope` mutant is P1-52's, the Link-level rows of the real TS pack are in `test_fixture_binding_live.py`). The other tests run on the fixtures plus
the edits the plan prescribes for 53b (`CONFORMING_EDITS`) and are green now: the pin accepts a conforming tree, and each mutant of it
(each named with its file) turns it red with the message asserted.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from ccg import atrest, build, bundle, units as U
from ccg.linker import Binder, link
from ccg.manifest import load_manifest
from ccg.packrt import run_packs
from ccg.rulepack import load_pack
from ccg.staleness import local_head
from linker_helpers import cons, mention, prov
from packrt_helpers import call, data, sym, vs

CPG = Path(__file__).resolve().parents[1]
SYSTEM_YAML = CPG / "fixtures" / "system1" / "system.yaml"
SYSTEM = "fixture-binding-pin"


def _load_materialize():
    spec = importlib.util.spec_from_file_location("ccg_fixture_materialize_pin", CPG / "fixtures" / "materialize.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_materialize()

# ---- the contract, as data ---------------------------------------------------------------------------------------------------------
#: unit name -> (repoId, rootPath, language): the three code units the rows name.
CODE_UNITS = {"svc": ("fx-svc", ".", "python"), "job": ("fx-job", ".", "python"), "web": ("fx-web", "app", "javascript")}
#: Dockerfile -> the unit its `CMD` launches (R2, R5; the real `cypher-mcp/Dockerfile` ends `python server.py`, the fixture's `scripts/Dockerfile`
#: CMD is inert: resolution is by path suffix only).
DOCKERFILES = {("fx-svc", "scripts/Dockerfile"): "svc", ("fx-svc", "deploy/Dockerfile"): "svc", ("fx-job", "Dockerfile"): "job"}
#: LNK-45's chain after the `.mcp.json` entry (the `.mcp.json` locator is the manifest's `aux` row; this one cannot be a manifest locator
#: until the file exists, because fixtures `TP-002` resolves every locator at the head).
DOCKERFILE_ENTRY = ("fx-svc", "scripts/Dockerfile", "CMD")

#: The fixture edits P1-53b makes (plan P1-53 "The contract the fixtures conform to"), as {repo-relative path: new text | None (delete)}.
SVC_SCRIPTS_DOCKERFILE = ('FROM python:3.12-slim\nWORKDIR /app\nCOPY . /app\nRUN pip install .\n'
                          'CMD ["uvicorn", "mcp.server:app", "--host", "127.0.0.1", "--port", "8200"]\n')
JOB_COMPOSE_DEV = ('services:\n  job-dev:\n    build:\n      context: ..\n      dockerfile: Dockerfile\n    ports:\n      - "8000:8000"\n')
CONFORMING_EDITS = {
    "fx-svc/scripts/Dockerfile": SVC_SCRIPTS_DOCKERFILE,
    "fx-svc/deploy/Dockerfile": ("sub", "fx_svc.api.app:create_app", "api.app:create_app"),
    "fx-job/deploy/compose.dev.yaml": JOB_COMPOSE_DEV,
}
#: the four mutants of the conforming tree, each (file, edits that turn the pin red, a message the red must carry).
MUTANTS = {
    "delete fx-svc/scripts/Dockerfile": ({"fx-svc/scripts/Dockerfile": None}, "missing Dockerfile fx-svc:scripts/Dockerfile", None),
    # red only through the per-Dockerfile launchUnit multiset: no link row changes (R5 has no link effect)
    "change fx-svc/deploy/Dockerfile CMD module to fx_svc.api.app": (
        {"fx-svc/deploy/Dockerfile": ("sub", "api.app:create_app", "fx_svc.api.app:create_app")}, "launchUnit multiset of fx-svc:deploy/Dockerfile",
        "fx-svc:deploy/Dockerfile"),
    "delete fx-job/deploy/compose.dev.yaml": ({"fx-job/deploy/compose.dev.yaml": None}, "LNK-16: no link from the loopback :8000 consumer", None),
}

REPO_ROWS = ("LNK-45", "LNK-16", "LNK-19", "LNK-12", "LNK-02", "LNK-06", "AC6-EX-4")


# ---- worlds: a root of the four repositories, edited and committed when a mutant needs it ------------------------------------------------
@pytest.fixture(scope="module")
def base_root(tmp_path_factory):
    out = tmp_path_factory.mktemp("fx-pin-base")
    M.materialize(out)
    return out


@pytest.fixture(scope="module")
def manifests():
    return M.load_all()


_GIT_ENV = {"GIT_AUTHOR_DATE": "2026-02-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-02-01T00:00:00Z", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "pin", "GIT_AUTHOR_EMAIL": "pin@example.invalid",
            "GIT_COMMITTER_NAME": "pin", "GIT_COMMITTER_EMAIL": "pin@example.invalid"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env={**os.environ, **_GIT_ENV}).stdout


def edited_root(base: Path, dest: Path, edits: dict, *, idempotent: bool = False) -> Path:
    """A copy of ``base`` with ``edits`` committed on each repository's current branch: ``path -> text``, ``None`` (delete), or
    ``("sub", old, new)`` (replace ``old`` once; ``old`` must occur, or, only with ``idempotent=True`` (``CONFORMING_EDITS``, which a
    committed fixture may already carry), ``new``)."""
    shutil.copytree(base, dest, symlinks=True)
    touched = set()
    for rel, new in edits.items():
        repo, inner = rel.split("/", 1)
        f = dest / repo / inner
        if new is None:
            f.unlink(missing_ok=True)
        elif isinstance(new, tuple):
            text = f.read_text()
            # idempotent: once P1-53b has edited the committed fixtures, the conforming edit is already in the tree; a mutant's `old` must occur
            assert new[1] in text or (idempotent and new[2] in text), f"{rel}: {new[1]!r} not found"
            f.write_text(text.replace(new[1], new[2], 1))
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(new)
        touched.add(repo)
    for repo in touched:
        _git(dest / repo, "add", "-A")
        _git(dest / repo, "commit", "-q", "--allow-empty", "-m", "pin edit")
    return dest


@dataclass
class World:
    root: Path
    units: list
    links: list
    binder: Binder
    facts: object
    records: dict          # record id -> (record, repoId) over every bundle
    sym_of: dict           # "repo:path::qname" -> symbol id
    loc_of: dict           # symbol id -> "repo:path::qname"
    unit_id: dict          # unit name -> unit id
    unit_name: dict        # unit id -> unit name
    tracked: dict          # repo -> set of tracked paths
    texts: dict            # (repo, path) -> text of a file at the head, absent when not tracked


def _loc(text: str) -> dict:
    return M.parse_locator(text)


def _short(repo: str) -> str:
    return repo.removeprefix("fx-")


def locator_str(repo: str, path: str, qname: str) -> str:
    return f"{_short(repo)}:{path}::{qname}"


def merged_row(manifests, rid: str) -> dict:
    """The row's comparable core, merged across the manifests that hold it (they must agree: fixtures `TP-004`)."""
    held = M.holders(manifests)[rid]
    views = [M.row_view(r) for r in held.values()]
    out = {"status": {v["status"] for v in views}, "reason": {v["reason"] for v in views}}
    assert all(len(v) == 1 for v in out.values()), (rid, out)
    return {k: next(iter(v)) for k, v in out.items()}


def row_locators(manifests, rid: str, field: str) -> list[dict]:
    seen, out = set(), []
    for r in M.holders(manifests)[rid].values():
        for f, loc in M.row_locators(r):
            key = (loc["repo"], loc["path"], loc.get("qname"), loc.get("keyPath"))
            if f in {"provider": ("P", "provider"), "consumer": ("C", "consumer"), "aux": ("aux",)}[field] and key not in seen:
                seen.add(key)
                out.append(loc)
    return out


def build_world(root: Path, work: Path, manifests) -> World:
    mf = load_manifest(SYSTEM_YAML)
    repos = {r.repo_id: root / r.path for r in mf.repos}
    heads = {k: local_head(v) for k, v in repos.items()}
    units = U.discover_units(root, mf, commit=heads)
    keys, area = work / "keys", work / "area"
    for d in (keys, area):
        d.mkdir(parents=True, mode=0o700, exist_ok=True)
    ctx = build.BuildContext(units=tuple(units), root=area, keys_root=keys, manifest=SYSTEM_YAML)
    bundles: list[list[dict]] = []
    for u in units:                                                       # the real extractors, host units only (no Joern)
        if u.repoId in ("fx-svc", "fx-job") and u.language in ("config", "shell"):
            res = build.build_unit(u, repos[u.repoId], heads[u.repoId], SYSTEM, ctx)
            assert (res.ok, res.fail_reason) == (True, None), u
            bundles.append(bundle.loads(atrest.read_bundle(SYSTEM, u.unitId, heads[u.repoId], area)))
    key = atrest.hmac_key(SYSTEM, keys)

    by_root = {(u.repoId, u.rootPath, u.language): u for u in units}
    unit_id = {n: by_root[k].unitId for n, k in CODE_UNITS.items()}
    unit_name = {v: k for k, v in unit_id.items()}
    tracked = {repo: set(_git(repos[repo], "ls-tree", "-r", "--name-only", heads[repo]).splitlines()) for repo in repos}
    texts = {}
    for repo in ("fx-svc", "fx-job"):
        for rel in tracked[repo]:
            if rel.endswith("Dockerfile") or rel.endswith(".sh") or rel in ("clients/peer.py", "deploy/compose.yaml", "deploy/compose.example.yaml"):
                texts[(repo, rel)] = _git(repos[repo], "show", f"{heads[repo]}:{rel}")

    # code units: header + the file records of their tracked files; the symbols the rows name are added below
    def owner(repo: str, path: str, lang: str) -> str | None:
        cands = [u for u in units if u.repoId == repo and u.language == lang and (u.rootPath == "." or path.startswith(u.rootPath + "/"))]
        return max(cands, key=lambda u: len(u.rootPath)).unitId if cands else None

    code: dict[str, list[dict]] = {}
    for name, (repo, root_path, lang) in CODE_UNITS.items():
        uid = unit_id[name]
        code[uid] = [{"type": "header", "unitId": uid, "unit": {"repoId": repo, "rootPath": root_path, "lang": lang}}]
    exts = {"python": (".py",), "javascript": (".ts", ".tsx")}
    for repo, paths in tracked.items():
        for path in sorted(paths):
            for lang, ext in exts.items():
                if path.endswith(ext) and (uid := owner(repo, path, lang)) in code:
                    code[uid].append({"type": "file", "id": f"file:{repo}:{path}", "repoId": repo, "unitId": uid, "path": path, "lang": lang,
                                      "kind": "code", "role": "primary"})
    md_unit = next(u for u in units if (u.repoId, u.rootPath) == ("fx-svc", ".claude/agents/exec.md"))
    prompt = [{"type": "header", "unitId": md_unit.unitId, "unit": {"repoId": "fx-svc", "rootPath": md_unit.rootPath, "lang": "markdown"}},
              {"type": "file", "id": "file:fx-svc:exec.md", "repoId": "fx-svc", "unitId": md_unit.unitId, "path": md_unit.rootPath,
               "lang": "markdown", "kind": "doc", "role": "primary"}]

    sym_of, loc_of = {}, {}

    def symbol(unit_records: list[dict], repo: str, path: str, qname: str, fid: str | None = None) -> str:
        key_ = locator_str(repo, path, qname)
        if key_ not in sym_of:
            sid = f"S-{len(sym_of)}"
            fid = fid or f"file:{repo}:{path}"
            assert any(r.get("id") == fid for r in unit_records), f"{key_}: no file record {fid}"
            unit_records.append(sym(sid, qname.split(".")[-1], fid=fid, path=path, qname=f"{path}::{qname}"))
            sym_of[key_], loc_of[sid] = sid, key_
        return sym_of[key_]

    def at(loc: dict) -> str:
        return symbol(code[unit_id[next(n for n, (r, _, _) in CODE_UNITS.items() if r == loc["repo"])]], loc["repo"], loc["path"], loc["qname"])

    providers, consumers, lifted, extra = [], [], [], []
    for rid, kind, key_ in (("LNK-12", "mcp_tool", "reindex"), ("LNK-02", "mcp_tool", "get_item"), ("LNK-06", "mcp_tool", "list_orders"),
                            ("LNK-16", "http", "GET /job/status"), ("AC6-EX-4", "http", "GET /api/health")):
        for loc in row_locators(manifests, rid, "provider"):
            providers.append(prov(at(loc), kind, key_))
    for loc in row_locators(manifests, "LNK-20", "provider"):              # every mcp_tool provider of the system, the second `reindex` among them (NC-5)
        providers.append(prov(at(loc), "mcp_tool", loc["qname"]))
    seen_p = set()
    providers = [p for p in providers if (p["anchorId"], p["key"]) not in seen_p and not seen_p.add((p["anchorId"], p["key"]))]
    # the loopback consumer of LNK-16
    c16 = row_locators(manifests, "LNK-16", "consumer")[0]
    cs16 = at(c16)
    consumers.append(cons(cs16, "http", "GET /job/status", "site-lnk16"))
    lifted.append({"packId": "fetch", "lifting": "fetch-url", "siteId": "site-lnk16", "symbolId": cs16, "path": c16["path"], "line": 1,
                   "urlPath": "a0", "alternatives": ["http://localhost:8000/job/status"], "resolved": True, "leaf": True, "hops": [], "verbs": None,
                   "partialAlternatives": False, "weakenedBy": None, "status": "confirmed", "reason": "lifted-role"})
    # AC6-EX-4 (P1-53c): the peer consumer reads its base from `os.environ.get("FX_PEER_BASE", "")`; only the example Compose file sets it
    c_ex4 = row_locators(manifests, "AC6-EX-4", "consumer")[0]
    s_ex4 = at(c_ex4)
    consumers.append(cons(s_ex4, "http", "GET /api/health", "site-ex4"))
    lifted.append({"packId": "requests", "lifting": "fetch-url", "siteId": "site-ex4", "symbolId": s_ex4, "path": c_ex4["path"], "line": 1,
                   "urlPath": "a0", "alternatives": ["\x00/api/health"], "resolved": True, "leaf": True, "hops": [], "verbs": None,
                   "partialAlternatives": False, "weakenedBy": None, "status": "confirmed", "reason": "lifted-role"})
    # the real extractor's shape of `os.environ.get(...)`: bare member as calleeName, the table's externalTarget (tests/data/p1_54_environ_reads.json)
    code[unit_id["svc"]] += [call("env-ex4", s_ex4, "get", resolution="external", external="os.py:<module>.<member>(environ).get"), vs("env-ex4", "a0", "FX_PEER_BASE"), vs("env-ex4", "a1", ""),
                             data("site-ex4", "env-ex4", "a0", "r")]
    # the `:8200/mcp` mcp_tool consumers of LNK-02, 06, 12 (stand-in for the TS pack: a `streamablehttp_client` callsite in the consumer's file)
    for rid, key_ in (("LNK-02", "get_item"), ("LNK-06", "list_orders"), ("LNK-12", "reindex")):
        loc = row_locators(manifests, rid, "consumer")[0]
        s = at(loc)
        consumers.append(cons(s, "mcp_tool", key_, f"site-{rid}"))
        u = code[unit_id["web"]]
        if not any(r.get("id") == f"uc-{loc['path']}" for r in u):
            u += [call(f"uc-{loc['path']}", s, "streamablehttp_client"), vs(f"uc-{loc['path']}", "a0", "http://localhost:8200/mcp")]
    # LNK-45: the prompt mention, in fx-svc's own repository
    d = symbol(prompt, "fx-svc", ".claude/agents/exec.md", "<document>", fid="file:fx-svc:exec.md")
    prompt.append(mention("m-lnk45", d, "mcp__fx-svc-exec__list_orders"))
    # controls whose binding is Binder-level (LNK-22, LNK-15, LNK-14): sites with a lifted URL; no link is asserted
    for site, url in (("site-lnk22", "http://api:8000/api/health"), ("site-lnk15", "http://localhost:8300/job/trigger")):
        lifted.append({"packId": "fetch", "lifting": "fetch-url", "siteId": site, "symbolId": cs16, "path": c16["path"], "line": 1, "urlPath": "a0",
                       "alternatives": [url], "resolved": True, "leaf": True, "hops": [], "verbs": None, "partialAlternatives": False,
                       "weakenedBy": None, "status": "confirmed", "reason": "lifted-role"})

    all_bundles = [*bundles, *code.values(), prompt]
    packs = [load_pack(CPG / "packs" / n / "pack.yaml") for n in ("fastmcp", "shell-mcp-call")]
    facts = run_packs(all_bundles, packs)
    facts.providers.extend(providers)
    facts.consumers.extend(consumers)
    facts.lifted.extend(lifted)
    binder = Binder(facts, all_bundles, key=key, repos={r.repo_id: r.path for r in mf.repos}, packs=packs, units=units)
    links = link(facts, all_bundles, packs=packs, bindings=binder)
    records = {}
    for b in all_bundles:
        repo = b[0]["unit"]["repoId"]
        for r in b:
            if isinstance(r.get("id"), str):
                records[r["id"]] = (r, r.get("repoId", repo))
    # the real shell symbols (`scripts/refresh.sh::<script>`) are located by their bundle records too
    for r, repo in list(records.values()):
        if r.get("type") == "symbol" and r["id"] not in loc_of:
            loc_of[r["id"]] = locator_str(repo, r["path"], r["qname"].split("::", 1)[-1])
    return World(root, units, links.links, binder, facts, records, sym_of, loc_of, unit_id, unit_name, tracked, texts)


# ---- the independent expectations ------------------------------------------------------------------------------------------------------
_CMD = re.compile(r"^\s*(?:CMD|ENTRYPOINT)\s+(\[.*\])\s*$", re.M)
_MODULE_ATTR = re.compile(r"^([A-Za-z_][\w.]*):[A-Za-z_]\w*$")


def cmd_module_unit(w: World, repo: str, text: str) -> tuple[str | None, str | None]:
    """(unit name, module:attr) the Dockerfile's `CMD` names by the path-suffix rule: independent of the pack runtime, parsed here."""
    import json
    m = _CMD.search(text)
    if not m:
        return None, None
    args = json.loads(m.group(1))
    target = next((a for a in args if _MODULE_ATTR.match(a)), None)
    if target is None:
        return None, None
    suffix = _MODULE_ATTR.match(target).group(1).replace(".", "/") + ".py"
    hits = [p for p in w.tracked[repo] if p.endswith(".py") and (p == suffix or p.endswith("/" + suffix))]
    if len(hits) != 1:
        return None, target
    owners = [n for n, (r, root, _) in CODE_UNITS.items() if r == repo and (root == "." or hits[0].startswith(root + "/"))]
    return (owners[0] if len(owners) == 1 else None), target


def launch_unit_multiset(w: World, repo: str, path: str) -> Counter:
    ids = {r["id"] for r, rp in w.records.values() if r.get("type") == "launchFact" and r["path"] == path and rp == repo}
    return Counter(w.unit_name.get(f["unitId"], f["unitId"]) for f in w.facts.launch_facts if f["fact"] == "launchUnit" and f["launchId"] in ids)


def entry_matches(rec: dict, repo: str, path: str, key_path: str | None, rec_repo: str) -> bool:
    """A bundle record at a manifest locator (path + keyPath): a `configEntry` by its keyPath, a `launchFact` of a `.mcp.json` server by its
    `service` (`mcpServers.<server>`) or of a Dockerfile by `CMD`."""
    if rec_repo != repo or rec.get("path") != path:
        return False
    if rec["type"] == "configEntry":
        return key_path is not None and (rec["keyPath"] == key_path or rec["keyPath"].startswith(key_path + "."))
    if rec["type"] != "launchFact" or rec.get("kind") != "command":
        return False
    if key_path == "CMD":
        return path.endswith("Dockerfile") and "service" not in rec
    return key_path is not None and key_path.startswith("mcpServers.") and rec.get("service") == key_path.split(".", 1)[1]


def chain_of(w: World, link_) -> list[tuple[int, dict, str]]:
    return [(e["ord"], *w.records[e["entryId"]]) for e in link_.evidence if e["edge"] == "BOUND_BY"]


def check_chain(w: World, chain, want: list[tuple[str, str, str | None]], label: str) -> list[str]:
    """``chain``: [(ord, record, repo)] in order; ``want``: the (repo, path, keyPath) of each expected entry, ord 0 first."""
    got = [(o, r, rp) for o, r, rp in chain]
    if [o for o, _, _ in got] != list(range(len(want))):
        return [f"{label}: BOUND_BY ords {[o for o, _, _ in got]}, expected {list(range(len(want)))} ({len(got)} entries, expected {len(want)})"]
    return [f"{label}: BOUND_BY ord {o} is {rp}:{r.get('path')} {r.get('service') or r.get('keyPath')}, expected {exp[0]}:{exp[1]} @{exp[2]}"
            for (o, r, rp), exp in zip(got, want) if not entry_matches(r, exp[0], exp[1], exp[2], rp)]


def links_from(w: World, consumer_loc: str) -> list:
    """The links whose consumer symbol is the row's ``C`` or a nested arrow of it (``C.<anon>#n``: a ``fetch`` inside a ``useEffect``)."""
    return [l for l in w.links if (c := w.loc_of.get(l.consumer)) is not None and (c == consumer_loc or c.startswith(consumer_loc + ".<anon>#"))]


def provider_locs(w: World, ls) -> list[str]:
    return sorted(w.loc_of.get(l.provider, f"unit:{l.to_unit}") for l in ls)


def loc_of_row(manifests, rid: str, field: str) -> list[str]:
    return [locator_str(l["repo"], l["path"], l["qname"]) for l in row_locators(manifests, rid, field)]


def launch_violations(w: World) -> list[str]:
    bad: list[str] = []
    # (1) the launchUnit multiset per Dockerfile (R2, R5): the Dockerfile exists, its CMD names the unit by an independent parse, the packs agree
    for (repo, path), unit in DOCKERFILES.items():
        text = w.texts.get((repo, path))
        if text is None:
            bad.append(f"missing Dockerfile {repo}:{path} (the launch chain and the launchUnit of unit {unit} need it)")
            continue
        named, target = cmd_module_unit(w, repo, text)
        if named != unit:
            bad.append(f"CMD of {repo}:{path} ({target}) resolves by path suffix to unit {named}, expected {unit}")
        got = launch_unit_multiset(w, repo, path)
        if got != Counter({unit: 1}):
            bad.append(f"launchUnit multiset of {repo}:{path} is {dict(got)}, expected {{{unit!r}: 1}}")
    return bad


def chain_violations(w: World) -> list[str]:
    bad: list[str] = []
    # (2) LNK-21 and LNK-45 launch chains as Binder outcomes (R2): units and BOUND_BY entries, script hop lexical
    svc = w.unit_id["svc"]
    for rid, server in (("LNK-21", "fx-svc"), ("LNK-45", "fx-svc-exec")):
        site = "site-chain"
        b = w.binder.bind({"siteId": f"{site}-{server}", "kind": "mcp_tool", "anchorId": w.sym_of["svc:.claude/agents/exec.md::<document>"],
                           "key": "list_orders", "keyForm": "literal", "line": 1, "status": "confirmed", "reason": "key-match", "hops": [],
                           "server": server})
        if b is None:
            bad.append(f"{rid}: server {server} binds to no unit (chain broken)")
            continue
        if b.units != (svc,):
            bad.append(f"{rid}: server {server} binds to units {[w.unit_name.get(u, u) for u in b.units]}, expected ['svc']")
        want = [("fx-svc", ".mcp.json", f"mcpServers.{server}"), DOCKERFILE_ENTRY]
        recs = [(i, *w.records[e.entry_id]) for i, e in enumerate(b.entries)]
        bad += check_chain(w, recs, want, f"{rid} chain of {server}")
    return bad


def _head_text(w: World, repo: str, path: str) -> str:
    """A tracked file's text: from ``w.texts`` (offline tier), else read from the materialized checkout (the live tier carries no ``texts``)."""
    return w.texts[(repo, path)] if (repo, path) in w.texts else (w.root / repo / path).read_text()


def ex4_chain_violations(w: World, manifests, link_) -> list[str]:
    """AC6-EX-4's chain from the Compose structure, not from the Binder: the example file's `FX_PEER_BASE` entry (the manifest aux locator), then
    the facts of the primary service its URL host names: that service's published port (the URL's port) and its `build.context` mount."""
    import yaml
    from urllib.parse import urlparse
    aux = next(a for a in row_locators(manifests, "AC6-EX-4", "aux") if a.get("keyPath"))
    example = yaml.safe_load(_head_text(w, "fx-svc", aux["path"]))
    url = urlparse(example["services"]["svc"]["environment"]["FX_PEER_BASE"])
    primary = yaml.safe_load(_head_text(w, "fx-svc", "deploy/compose.yaml"))["services"]
    if url.hostname not in primary:
        return [f"AC6-EX-4: {aux['path']} FX_PEER_BASE host {url.hostname!r} is not a service of the primary deploy/compose.yaml {sorted(primary)} (it would bind nothing)"]
    svc_ = primary[url.hostname]
    ports = [int(str(p).split(":")[0]) for p in svc_.get("ports", [])]
    if url.port not in ports:
        return [f"AC6-EX-4: service {url.hostname} publishes {ports}, not the example URL's port {url.port}"]
    want = [(aux["repo"], aux["path"], aux["keyPath"], None), ("fx-svc", "deploy/compose.yaml", None, ("port", url.hostname, url.port)),
            ("fx-svc", "deploy/compose.yaml", None, ("mount", url.hostname, svc_["build"]["context"]))]
    got = chain_of(w, link_)
    if [o for o, _, _ in got] != list(range(len(want))):
        return [f"AC6-EX-4 link: BOUND_BY ords {[o for o, _, _ in got]}, expected {list(range(len(want)))}"]
    bad = []
    for (o, r, rp), (repo, path, kp, fact) in zip(got, want):
        if fact is None:
            ok = entry_matches(r, repo, path, kp, rp)
        else:
            kind, service, val = fact
            ok = (rp, r.get("path"), r.get("type"), r.get("kind"), r.get("service")) == (repo, path, "launchFact", kind, service) and \
                r.get("port" if kind == "port" else "mountPath") == val
        if not ok:
            bad.append(f"AC6-EX-4 link: BOUND_BY ord {o} is {rp}:{r.get('path')} {r.get('keyPath') or (r.get('kind'), r.get('service'))}, expected {want[o][:3]} {fact}")
    return bad


def link_violations(w: World, manifests) -> list[str]:
    """(3) the Link-level rows LNK-45, 16, 19, 12, 02, 06 and AC6-EX-4: the live tier asserts exactly these over the fixture build."""
    bad: list[str] = []
    form = {rid: merged_row(manifests, rid) for rid in REPO_ROWS}
    # LNK-45: possible / prompt-mention (AC-6: a prompt that names a tool is never confirmed), the chain on the link
    ls = [l for l in links_from(w, "svc:.claude/agents/exec.md::<document>") if w.loc_of.get(l.provider) == loc_of_row(manifests, "LNK-45", "provider")[0]]
    if len(ls) != 1:
        bad.append(f"LNK-45: {len(ls)} links from the mention to {loc_of_row(manifests, 'LNK-45', 'provider')[0]}, expected 1")
    else:
        l = ls[0]
        if (l.status, l.reason) != (form["LNK-45"]["status"], form["LNK-45"]["reason"]):
            bad.append(f"LNK-45: link is {l.status}/{l.reason}, manifest row says {form['LNK-45']['status']}/{form['LNK-45']['reason']}")
        aux = [a for a in row_locators(manifests, "LNK-45", "aux") if a.get("keyPath")]
        want = [(a["repo"], a["path"], a["keyPath"]) for a in aux[:1]] + [DOCKERFILE_ENTRY]
        bad += check_chain(w, chain_of(w, l), want, "LNK-45 link")
    # LNK-16: possible / port-shared
    ls = [l for l in links_from(w, loc_of_row(manifests, "LNK-16", "consumer")[0])]
    want_p = loc_of_row(manifests, "LNK-16", "provider")
    if provider_locs(w, ls) != sorted(want_p):
        bad.append(f"LNK-16: no link from the loopback :8000 consumer to {want_p} (links to {provider_locs(w, ls)})")
    else:
        got = {(l.status, l.reason) for l in ls}
        if got != {(form["LNK-16"]["status"], form["LNK-16"]["reason"])}:
            bad.append(f"LNK-16: link is {sorted(got)}, manifest row says {form['LNK-16']['status']}/{form['LNK-16']['reason']}")
    # LNK-19: bound by no path: every reindex provider, confirmed / key-match, no BOUND_BY
    ls = links_from(w, loc_of_row(manifests, "LNK-19", "consumer")[0])
    want_p = sorted(loc_of_row(manifests, "LNK-19", "provider"))
    if provider_locs(w, ls) != want_p:
        bad.append(f"LNK-19: links to {provider_locs(w, ls)}, expected {want_p}")
    for l in ls:
        if (l.status, l.reason) != (form["LNK-19"]["status"], form["LNK-19"]["reason"]):
            bad.append(f"LNK-19: link to {w.loc_of.get(l.provider)} is {l.status}/{l.reason}, manifest row says "
                       f"{form['LNK-19']['status']}/{form['LNK-19']['reason']}")
        if chain_of(w, l):
            bad.append(f"LNK-19: link to {w.loc_of.get(l.provider)} carries BOUND_BY (a shell consumer is bound by no path at P1)")
    # LNK-12, 02, 06: the consumer binds to exactly the svc provider, binding-confirmed, with a chain
    for rid in ("LNK-12", "LNK-02", "LNK-06"):
        ls = links_from(w, loc_of_row(manifests, rid, "consumer")[0])
        want_p = loc_of_row(manifests, rid, "provider")
        if provider_locs(w, ls) != sorted(want_p):
            bad.append(f"{rid}: consumer links to {provider_locs(w, ls)}, expected exactly {sorted(want_p)}")
        for l in ls:
            if (l.status, l.reason) != (form[rid]["status"], form[rid]["reason"]):
                bad.append(f"{rid}: link to {w.loc_of.get(l.provider)} is {l.status}/{l.reason}, manifest row says "
                           f"{form[rid]['status']}/{form[rid]['reason']}")
    # AC6-EX-4 (P1-53c): bound only through the example Compose file's environment entry: possible / example-config, the chain on the link
    peer = _head_text(w, "fx-svc", "clients/peer.py")
    if not re.search(r"""(?:environ\.get|getenv)\(\s*["']FX_PEER_BASE["']""", peer):
        bad.append("AC6-EX-4: clients/peer.py does not read FX_PEER_BASE through environ.get/getenv (the Binder reads no `environ[...]`)")
    ls = links_from(w, loc_of_row(manifests, "AC6-EX-4", "consumer")[0])
    want_p = loc_of_row(manifests, "AC6-EX-4", "provider")
    if provider_locs(w, ls) != sorted(want_p):
        bad.append(f"AC6-EX-4: consumer links to {provider_locs(w, ls)}, expected exactly {sorted(want_p)}")
    for l in ls:
        if (l.status, l.reason) != (form["AC6-EX-4"]["status"], form["AC6-EX-4"]["reason"]):
            bad.append(f"AC6-EX-4: link to {w.loc_of.get(l.provider)} is {l.status}/{l.reason}, manifest row says "
                       f"{form['AC6-EX-4']['status']}/{form['AC6-EX-4']['reason']}")
        bad += ex4_chain_violations(w, manifests, l)
    return bad


def violations(w: World, manifests) -> list[str]:
    return launch_violations(w) + chain_violations(w) + link_violations(w, manifests)


# ---- the pin -----------------------------------------------------------------------------------------------------------------------------
def test_the_fixtures_as_committed_meet_the_binding_contract(base_root, manifests, tmp_path):
    """Red until P1-53b (and P1-52 for the live rows). The assertion messages name each red row."""
    w = build_world(base_root, tmp_path, manifests)
    assert violations(w, manifests) == []


@pytest.fixture(scope="module")
def conforming_root(base_root, tmp_path_factory):
    return edited_root(base_root, tmp_path_factory.mktemp("fx-pin-conforming") / "root", CONFORMING_EDITS, idempotent=True)


def test_the_pin_accepts_the_conforming_tree_the_plan_prescribes(conforming_root, manifests, tmp_path):
    """The control: the fixtures plus the three 53b edits. Green now; if it is red the pin, not the fixtures, is wrong."""
    assert violations(build_world(conforming_root, tmp_path, manifests), manifests) == []


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_each_mutant_of_the_conforming_tree_turns_the_pin_red_with_its_own_message(name, conforming_root, manifests, tmp_path):
    edits, message, only = MUTANTS[name]
    root = edited_root(conforming_root, tmp_path / "root", edits)
    got = violations(build_world(root, tmp_path / "w", manifests), manifests)
    assert any(message in v for v in got), (name, got)
    if only:
        assert all(only in v for v in got), (name, got)


def test_flipping_lnk_45_to_confirmed_in_the_manifest_turns_the_pin_red(conforming_root, manifests, tmp_path):
    """The fourth mutant: the manifest's LNK-45 `confirmed` (the RCA's derivation error) disagrees with the link a prompt mention makes."""
    bad = copy.deepcopy(manifests)
    for r in bad["fx-svc"]["rows"]:
        if r["id"] == "LNK-45":
            r["status"], r["reason"] = "confirmed", "binding-confirmed"
    got = violations(build_world(conforming_root, tmp_path, bad), bad)
    assert any(v.startswith("LNK-45: link is possible/prompt-mention, manifest row says confirmed/binding-confirmed") for v in got), got



def test_edited_root_sub_is_idempotent_only_for_conforming_edits(tmp_path):
    """M3: a `sub` whose `old` is absent but whose `new` is present is a no-op that only a conforming edit may be; a mutant's mistyped `old` fails loudly."""
    base = tmp_path / "base"
    (base / "r").mkdir(parents=True)
    (base / "r" / "f.txt").write_text("already-new\n")
    _git(base / "r", "init", "-q")
    _git(base / "r", "add", "-A")
    _git(base / "r", "commit", "-q", "-m", "x")
    edits = {"r/f.txt": ("sub", "old", "already-new")}
    with pytest.raises(AssertionError, match="not found"):
        edited_root(base, tmp_path / "strict", edits)
    assert (edited_root(base, tmp_path / "lenient", edits, idempotent=True) / "r" / "f.txt").read_text() == "already-new\n"
