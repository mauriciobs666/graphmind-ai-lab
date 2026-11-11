"""Bundles + links -> graph rows (P1-18; ported from ``spikes/vertical-slice/load.py:build_rows``).

Pure code over stored bundles: **rows only, no database write of any kind** (the loader is P1-19). Contract:
``docs/plans/connected-code-graph-p1-graph.md`` (§1 G1-G13, §2 O-9 and ``BOUND_BY``, §3 identity) over
``docs/plans/connected-code-graph-p0-graph.md`` (v1: D5 ``MOUNTED_BY``, D6 no ``candidateCount``, D8 default omission) and
``-p1-linking.md`` P1-18 (1)-(11). The slice equivalence is ``tests/slice/test_assemble_equivalence.py``.

What this module refuses, never writes:

* a bundle the audit check, the version rule or the header schema rejects (:class:`BundleRefused` with a code of
  ``ccg.staleness.FAIL_REASONS``; the message holds the unit id and the code, never a value or an exception text);
* a ``Unit.failReason`` outside ``FAIL_REASONS``; a system name outside the whitelist (re-checked here);
* a ``Link`` that breaks O-9, a ``BOUND_BY`` whose port evidence is malformed, a provider that is not a ``Symbol`` (the P1 edge
  list is closed: ``-p1-graph.md`` §4).

``Unit.flowAvailable`` is written from ``ccg/registry/flow-availability.yaml`` only (index D2); no bundle field and no other file
decides it. Step 8a of the ``http`` ``keyNorm`` (:func:`ccg.keynorm.mask_http_segments`) is imported, never reimplemented: it screens
``MOUNTED_BY.mountPath``, an ungated path from a pack's mount rule.
"""
from __future__ import annotations

import ast
import copy
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from ccg import bundle, ids
from ccg.assemble import snapshot as SN
from ccg.assemble.snapshot import AssembleError
from ccg.keynorm import PROVIDER, mask_http_segments
from ccg.rulepack import load_yaml
from ccg.sandbox_audit import check_sandbox_ids, default_audit_dir
from ccg.staleness import FAIL_REASONS
from ccg.sysname import check_system_name

__all__ = ["AssembleError", "BundleRefused", "EDGE_GROUPS", "GraphRows", "ID_PROP", "NODE_ORDER", "NotMappedUnit", "build_rows",
           "canonical_rows", "check_edge_groups", "classify_bundle", "find_step_8a_definitions", "load_flow_availability", "load_flow_rows"]

CPG_ROOT = Path(__file__).resolve().parents[2]
PACKS_DIR = CPG_ROOT / "packs"
FLOW_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry" / "flow-availability.yaml"

#: The highest bundle schema this assembler reads (spec3 §8.6: a higher minor makes the unit ``stale-kept``).
SUPPORTED_MAJOR, SUPPORTED_MINOR = 1, 1

#: label -> its id property (graph note §3.4; ``-p1-graph.md`` §4 adds ``ConfigEntry`` and ``Pack``).
ID_PROP = {
    "Repo": "repoId", "Unit": "unitId", "File": "fileId", "Symbol": "symbolId", "CallSite": "callSiteId",
    "Param": "pointId", "Ret": "pointId", "Cond": "pointId", "Name": "nameKey", "ContractPoint": "cpId", "ConfigEntry": "entryId",
    "Link": "linkId", "Dependency": "depId", "Pack": "packId",
}
#: ``SkippedFile`` has no id property (p0-graph §1.3): matched by ``(unitId, path)``.
NODE_ORDER = ("Repo", "Unit", "File", "Symbol", "Param", "Ret", "Cond", "CallSite", "Name", "ContractPoint", "ConfigEntry", "Link",
              "Dependency", "SkippedFile", "Pack")

#: D8: a reader coalesces an absent ``status`` / ``reason`` to these, so only a non-default is stored.
DEFAULT_STATUS = {"CALLS": {"status": "confirmed", "reason": "resolved-call"},
                  "DATA": {"status": "confirmed", "reason": "intra-flow"}}

#: The closed list of edge groups (``-p1-graph.md`` §4): v1's 29 plus the five of 1.1. A row in any other group is refused.
EDGE_GROUPS = (
    ("Unit", "IN_REPO", "Repo"), ("Unit", "HAS_FILE", "File"), ("File", "DEFINES", "Symbol"),
    ("Symbol", "HAS_CALL", "CallSite"), ("Symbol", "HAS_PARAM", "Param"), ("Symbol", "HAS_RETURN", "Ret"),
    ("Symbol", "HAS_COND", "Cond"), ("CallSite", "CALLS", "Symbol"), ("CallSite", "CALLS_NAME", "Name"),
    ("Name", "RESOLVES_TO", "Symbol"),
    ("CallSite", "DATA", "Param"), ("CallSite", "DATA", "CallSite"), ("Ret", "DATA", "Param"),
    ("Ret", "DATA", "CallSite"), ("Cond", "DATA", "Param"), ("Cond", "DATA", "CallSite"),
    ("CallSite", "CONTROLLED_BY", "Cond"), ("Ret", "CONTROLLED_BY", "Cond"), ("Cond", "CONTROLLED_BY", "Cond"),
    ("Param", "CONTROLLED_BY", "Cond"),
    ("Symbol", "PROVIDES", "ContractPoint"), ("CallSite", "CONSUMES", "ContractPoint"),
    ("CallSite", "VIA", "Link"), ("Link", "TO", "Symbol"), ("Link", "ON", "ContractPoint"),
    ("Link", "LIFTED_VIA", "CallSite"), ("Link", "LIFTED_VIA", "Symbol"),
    ("Unit", "DEPENDS_ON", "Dependency"), ("Unit", "SKIPS", "SkippedFile"),
    ("Link", "BOUND_BY", "ConfigEntry"), ("ConfigEntry", "LAUNCHES", "File"), ("ConfigEntry", "LAUNCHES", "Unit"),
    ("Symbol", "MOUNTED_BY", "CallSite"), ("Link", "TO_UNIT", "Unit"),
)

#: ``parentId`` is deliberately absent (D7): it becomes ``depth``.
_SYMBOL_PROPS = ("kind", "name", "nameNorm", "qname", "path", "lineStart", "lineEnd", "fingerprint",
                 "lastChangedAt", "lastChangedCommit", "complexity", "lang", "signature")
_LAUNCH_PROPS = (("protocol", "protocol"), ("port", "port"), ("envVar", "envVar"), ("mountPath", "mountPath"),
                 ("service", "service"), ("command", "argv0"))
_PORT_ORIGINS = ("explicit", "env-default", "compose", "pack-default")
_STRENGTH = {"confirmed": 0, "possible": 1}
_RFC3339_Z = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_VERSION = re.compile(r"(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})")
MASK = "[#…]"                      # a screen-failing keyPath segment, as stored (p0-graph §1.5 P2-8)


class BundleRefused(AssembleError):
    """A bundle the assembler will not load: ``code`` is a member of ``FAIL_REASONS``, the message is value-free."""

    def __init__(self, unit_id: str, code: str):
        super().__init__(f"unit {unit_id}: bundle refused ({code})")
        self.unit_id, self.code = unit_id, code


@dataclass(frozen=True)
class NotMappedUnit:
    """A unit of the manifest that has no bundle (never built): a ``Unit`` row with ``buildStatus`` ``not-mapped`` (G4)."""
    unit_id: str
    repo_id: str
    root_path: str
    language: str
    fail_reason: str = "manifest:never-built"


@dataclass
class GraphRows:
    nodes: dict[str, list[dict]] = field(default_factory=dict)
    #: (source label, type, target label) -> rows ``{"s": id, "t": id, "p": {...}}``
    edges: dict[tuple[str, str, str], list[dict]] = field(default_factory=dict)
    #: why a fact was not written, by reason (a count, never a value)
    dropped: dict[str, int] = field(default_factory=dict)
    units: list[dict] = field(default_factory=list)
    files: dict[str, int] = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)

    def node_count(self) -> int:
        return sum(len(v) for v in self.nodes.values())

    def edge_count(self) -> int:
        return sum(len(v) for v in self.edges.values())


def canonical_rows(rows: GraphRows) -> bytes:
    """The rows as canonical JSON (keys sorted, row order kept): two builds of the same inputs give identical bytes."""
    doc = {"nodes": rows.nodes, "edges": {"|".join(k): v for k, v in rows.edges.items()}, "dropped": rows.dropped,
           "units": rows.units, "files": rows.files, "snapshot": rows.snapshot}
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------------------------------------------------
# flow availability (index D2, G5): the one reader
# ---------------------------------------------------------------------------------------------------------------------
def load_flow_rows(path: Path | None = None) -> list[dict]:
    """The registry rows ``{language, flowAvailable, measurement}``, validated: a bool, a known language, a citation, no duplicate."""
    path = FLOW_REGISTRY_PATH if path is None else Path(path)
    try:
        rows = load_yaml(path.read_text(encoding="utf-8"))
    except Exception as e:                                           # noqa: BLE001 - yaml/OS errors become one typed error
        raise AssembleError(f"flow registry unreadable ({type(e).__name__})") from None
    languages = set(bundle.load_schema()["$defs"]["lang"]["enum"])
    if not isinstance(rows, list):
        raise AssembleError("flow registry is a list of rows")
    seen: set[str] = set()
    for r in rows:
        if not isinstance(r, dict) or set(r) != {"language", "flowAvailable", "measurement"}:
            raise AssembleError("flow registry row: keys are exactly language, flowAvailable, measurement")
        if r["language"] not in languages or r["language"] in seen:
            raise AssembleError("flow registry row: an unknown or duplicate language")
        if not isinstance(r["flowAvailable"], bool) or not isinstance(r["measurement"], str) or not r["measurement"].strip():
            raise AssembleError("flow registry row: flowAvailable is a boolean and measurement cites the report")
        seen.add(r["language"])
    return rows


def load_flow_availability(path: Path | None = None) -> dict[str, bool]:
    return {r["language"]: r["flowAvailable"] for r in load_flow_rows(path)}


# ---------------------------------------------------------------------------------------------------------------------
# bundle refusal (P1-18 (7), (11))
# ---------------------------------------------------------------------------------------------------------------------
def _header_validator() -> jsonschema.protocols.Validator:
    schema = bundle.load_schema()
    defs = copy.deepcopy(schema["$defs"])
    # bundle-v1.schema.json still says minItems 1; the host-language ruling (empty iff HOST_LANGS) is enforced by check_sandbox_ids,
    # which classify_bundle calls next, until the spec6 schema amendment lands.
    defs["header"]["properties"]["sandboxAuditIds"].pop("minItems", None)
    sub = {"$schema": schema["$schema"], "$defs": defs, "$ref": "#/$defs/header"}
    return jsonschema.validators.validator_for(sub)(sub)


_HEADER_VALIDATOR = None


def classify_bundle(records: Sequence[Mapping], audit_dir: Path) -> str | None:
    """``None`` when the assembler accepts the bundle, else the one code of ``FAIL_REASONS`` that names why. Value-free by construction:
    nothing of the bundle or of an exception is put in the result. Checks the header (version rule, schema) and every
    ``sandboxAuditIds`` entry (``ccg.sandbox_audit.check_sandbox_ids``). Input is ``bundle.read_bundle`` output: record bodies are not
    checked here (that validation is ``read_bundle``'s), so a malformed body passes this function and fails later in ``build_rows``."""
    global _HEADER_VALIDATOR
    try:
        if not isinstance(records, (list, tuple)) or not records or not isinstance(records[0], Mapping) \
                or records[0].get("type") != "header":
            return "bundle:invalid"
        h = records[0]
        m = _VERSION.fullmatch(h["bundleSchemaVersion"]) if isinstance(h.get("bundleSchemaVersion"), str) else None
        if m is None or int(m[1]) != SUPPORTED_MAJOR:
            return "bundle:invalid"
        if int(m[2]) > SUPPORTED_MINOR:
            return "bundle:schema-minor"
        _HEADER_VALIDATOR = _HEADER_VALIDATOR or _header_validator()
        if next(iter(_HEADER_VALIDATOR.iter_errors(h)), None) is not None:
            return "bundle:invalid"
        if check_sandbox_ids(h, Path(audit_dir)):
            return "bundle:audit-missing"
        return None
    except Exception:                                                # noqa: BLE001 - any failure is "invalid", with no text
        return "bundle:invalid"


# ---------------------------------------------------------------------------------------------------------------------
# step 8a: defined only in ccg.keynorm (P1-18 (6))
# ---------------------------------------------------------------------------------------------------------------------
_PLACEHOLDER_NAMES = frozenset({"PROVIDER", "CONSUMER", "placeholder"})
_PLACEHOLDER_TEXT = frozenset({"{#}", "{?}"})


def _is_step_8a(fn: ast.AST) -> bool:
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "mask_http_segments":
        return True
    for node in ast.walk(fn):
        if isinstance(node, ast.IfExp):
            calls_screen = any(isinstance(c, ast.Call) and getattr(c.func, "id", getattr(c.func, "attr", None)) == "screen_fails"
                               for c in ast.walk(node.test))
            branches = (node.body, node.orelse)
            if calls_screen and any((isinstance(b, ast.Constant) and b.value in _PLACEHOLDER_TEXT)
                                    or (isinstance(b, ast.Name) and b.id in _PLACEHOLDER_NAMES) for b in branches):
                return True
    return False


def find_step_8a_definitions(root: Path) -> list[Path]:
    """Files under ``root`` (``ccg/``, ``clients/``, ``extractors/``, ``eval/``) that define step 8a of the ``http`` ``keyNorm``: a function
    named ``mask_http_segments``, or any function that replaces a segment by a ``{#}``/``{?}`` placeholder when ``screen_fails`` says so.
    Only ``ccg/keynorm.py`` may be in the answer."""
    root = Path(root)
    out = []
    for sub in ("ccg", "clients", "extractors", "eval"):
        for p in sorted((root / sub).rglob("*.py")) if (root / sub).is_dir() else ():
            tree = ast.parse(p.read_text(encoding="utf-8"))
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_step_8a(n) for n in ast.walk(tree)):
                out.append(p.relative_to(root))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# row helpers (the slice's, ported)
# ---------------------------------------------------------------------------------------------------------------------
def _clean(props: dict[str, Any]) -> dict[str, Any]:
    """Drop absent values and strip strings (an unstripped indexed value is silently missed; graph note §3.4)."""
    return {k: (v.strip() if isinstance(v, str) else v) for k, v in props.items() if v is not None}


def _symbol_depths(parents: dict[str, str | None]) -> dict[str, int]:
    """Nesting depth per symbol id from ``parentId``: a root (no parent, or a parent outside the snapshot) is 0; a cycle terminates."""
    depth: dict[str, int] = {}
    for sid in parents:
        chain, cur = [], sid
        while cur in parents and cur not in depth and cur not in chain:
            chain.append(cur)
            cur = parents[cur]
        base = depth[cur] if cur in depth else -1
        for node in reversed(chain):
            base += 1
            depth[node] = base
    return depth


def _non_default(etype: str, status: str, reason: str) -> dict[str, str]:
    """D8: the ``status`` / ``reason`` of a ``CALLS`` / ``DATA`` edge that differ from the default, each on its own."""
    default = DEFAULT_STATUS[etype]
    return {k: v for k, v in (("status", status), ("reason", reason)) if v != default[k]}


def _flatten_regions(regions: list[list[int]]) -> list[int]:
    return [n for pair in regions for n in pair]


def _mask_key_path(key_path: str) -> str:
    """``keyPath`` with each screen-failing segment replaced by ``[#…]`` (p0-graph §1.5 P2-8). The ``entryId`` is the record's, from the
    raw path. An unparseable path is stored fully masked (fail closed)."""
    out, pos = [], 0
    while pos < len(key_path):
        m = bundle._KEYPATH_PART.match(key_path, pos)            # noqa: SLF001 - the one keyPath grammar (§5.17)
        if not m:
            return MASK
        if m.group("n") is not None:
            out.append(MASK if bundle.screen_fails(m.group("n")) else m.group(0))
        elif m.group("q") is not None:
            try:
                name = json.loads(m.group("q"))
            except ValueError:
                return MASK
            out.append(MASK if bundle.screen_fails(name) else m.group(0))
        else:
            out.append(m.group(0))
        pos = m.end()
    return "".join(out)


def _mount_path(prefix: str) -> str:
    """A mount prefix as a path: the empty prefix is the root, a missing leading ``/`` is added, and step 8a (provider form) screens it."""
    path = prefix if prefix.startswith("/") else "/" + prefix
    return mask_http_segments(path, PROVIDER)


class _Builder:
    def __init__(self) -> None:
        self.rows = GraphRows()
        self.seen_nodes: dict[str, set] = {}
        self.seen_edges: set = set()

    def drop(self, reason: str, n: int = 1) -> None:
        self.rows.dropped[reason] = self.rows.dropped.get(reason, 0) + n

    def node(self, label: str, props: dict) -> bool:
        ident = props[ID_PROP[label]]
        seen = self.seen_nodes.setdefault(label, set())
        if ident in seen:
            return False
        seen.add(ident)
        self.rows.nodes.setdefault(label, []).append(_clean(props))
        return True

    def edge(self, src: tuple[str, str], etype: str, dst: tuple[str, str], props: dict | None = None, key: tuple = ()) -> bool:
        ident = (src, etype, dst, key)
        if ident in self.seen_edges:
            self.drop(f"duplicate-{etype}")
            return False
        self.seen_edges.add(ident)
        row: dict[str, Any] = {"s": src[1], "t": dst[1]}
        if props:
            row["p"] = props
        self.rows.edges.setdefault((src[0], etype, dst[0]), []).append(row)
        return True


# ---------------------------------------------------------------------------------------------------------------------
# the assembly
# ---------------------------------------------------------------------------------------------------------------------
def _check_fail(unit_id: str, failed_at: Any, reason: Any) -> None:
    if not (isinstance(reason, str) and reason in FAIL_REASONS):
        raise AssembleError(f"unit {unit_id}: failReason is not a member of FAIL_REASONS")
    if failed_at is not None and not (isinstance(failed_at, str) and _RFC3339_Z.fullmatch(failed_at)):
        raise AssembleError(f"unit {unit_id}: failedAt is not an RFC 3339 UTC timestamp ending Z")


def build_rows(bundles: Sequence[Sequence[dict]], links: Any, facts: Any, *, system: str, packs: Sequence[Mapping],
               audit_dir: Path | None = None, pack_dirs: Mapping[str, Path] | None = None,
               stale: Mapping[str, tuple[str | None, Any]] | None = None, not_mapped: Iterable[NotMappedUnit] = (),
               flow_registry: Path | None = None, code_root: Path | None = None) -> GraphRows:
    """Graph rows for ``bundles`` (each a bundle's records, header first), the linker's ``links`` (:class:`ccg.linker.Links`) and the
    pack runtime's ``facts`` (:class:`ccg.packrt.PackFacts`).

    ``stale`` maps a ``unitId`` whose bundle is the last good one to ``(failedAt, failReason)`` (``stale-kept``); ``not_mapped`` are units
    with no bundle. ``packs`` are the loaded packs the build ran (identity: ``packSetHash``, ``Pack`` nodes); ``pack_dirs`` maps a pack id
    to its directory (default ``cpg/packs/<id>``). ``audit_dir`` is the directory holding ``sandbox.jsonl`` (default: the runner's).
    Every bundle is checked by :func:`classify_bundle` first; one refusal raises :class:`BundleRefused`.

    ``bundles`` are ``bundle.read_bundle`` output: ``classify_bundle`` checks the header only, so a malformed record body (a record
    without a required key) is not refused with a typed error but raises the raw ``KeyError``/``TypeError`` of the access (its text names
    a key, never a value). ``read_bundle`` is the one validator of record bodies; every caller reads bundles through it."""
    system = check_system_name(system)
    packs = sorted(packs, key=lambda p: p["id"])
    pack_dirs = {p["id"]: PACKS_DIR / p["id"] for p in packs} | dict(pack_dirs or {})
    flow = load_flow_availability(flow_registry)
    audit_dir = default_audit_dir() if audit_dir is None else Path(audit_dir)
    stale = dict(stale or {})
    not_mapped = list(not_mapped)

    # --- refuse before reading anything else -------------------------------------------------------------
    for records in bundles:
        code = classify_bundle(records, audit_dir)
        if code is not None:
            uid = records[0].get("unitId") if records and isinstance(records[0], Mapping) else None
            raise BundleRefused(uid if isinstance(uid, str) and re.fullmatch(r"[0-9a-f]{32}", uid) else "?", code)
    ordered = sorted(bundles, key=lambda r: r[0]["unitId"])
    unit_ids = [r[0]["unitId"] for r in ordered]
    if len(set(unit_ids)) != len(unit_ids):
        raise AssembleError("two bundles carry one unitId")
    if not ordered:
        raise AssembleError("no fresh or stale-kept unit: there is nothing to answer from")
    for uid, (failed_at, reason) in stale.items():
        if uid not in unit_ids:
            raise AssembleError(f"unit {uid}: marked stale-kept but no bundle was given")
        _check_fail(uid, failed_at, reason)
        if failed_at is None:
            raise AssembleError(f"unit {uid}: a stale-kept unit carries failedAt")
    for nm in not_mapped:
        if nm.unit_id in unit_ids or nm.unit_id in stale:
            raise AssembleError(f"unit {nm.unit_id}: both built and not-mapped")
        _check_fail(nm.unit_id, None, nm.fail_reason)
    if len({nm.unit_id for nm in not_mapped}) != len(not_mapped):
        raise AssembleError("two not-mapped units carry one unitId")
    triples = SN.pack_triples(packs, pack_dirs)
    vocabs = {p["vocabVersion"] for p in packs if "vocabVersion" in p}
    if len(vocabs) > 1:
        raise AssembleError("the packs declare more than one vocabVersion")

    b = _Builder()
    rows = b.rows
    point_label: dict[str, str] = {}
    symbol_ids: set[str] = set()
    sym_lang: dict[str, str] = {}
    sym_path: dict[str, str] = {}
    sym_unit: dict[str, str] = {}
    by_name: dict[tuple[str, str], list[str]] = {}
    parents: dict[str, str | None] = {}
    entry_ids: set[str] = set()

    # pass 1: index every point, symbol and config entry, so edges find both ends whichever bundle holds them
    for records in ordered:
        header = records[0]
        for r in records:
            t = r["type"]
            if t == "symbol":
                symbol_ids.add(r["id"])
                sym_lang[r["id"]], sym_path[r["id"]], sym_unit[r["id"]] = r["lang"], r["path"], header["unitId"]
                by_name.setdefault((r["lang"], r["name"]), []).append(r["id"])
                parents[r["id"]] = r.get("parentId")
            elif t in ("param", "ret", "cond", "callsite"):
                point_label[r["id"]] = {"param": "Param", "ret": "Ret", "cond": "Cond", "callsite": "CallSite"}[t]
            elif t in ("configEntry", "launchFact"):
                entry_ids.add(r["id"])
    depths = _symbol_depths(parents)

    # pass 2: nodes and the edges that follow from one record
    repo_head: dict[str, str] = {}
    for fresh_first in (True, False):
        for records in ordered:
            h = records[0]
            if (h["unitId"] not in stale) == fresh_first:
                repo_head.setdefault(h["unit"]["repoId"], h["builtCommit"])
    unit_rows: list[list] = []
    audit_ids: list[str] = []
    max_minor = 0
    for records in ordered:
        header = records[0]
        unit_id, repo_id, lang = header["unitId"], header["unit"]["repoId"], header["unit"]["lang"]
        status = "stale-kept" if unit_id in stale else "fresh"
        minor = int(_VERSION.fullmatch(header["bundleSchemaVersion"])[2])
        max_minor = max(max_minor, minor)
        audit_ids += header["sandboxAuditIds"]
        b.node("Repo", {"repoId": repo_id, "headCommit": repo_head[repo_id]})
        unit = {"unitId": unit_id, "repoId": repo_id, "rootPath": header["unit"]["rootPath"], "language": lang, "buildStatus": status,
                "flowAvailable": flow.get(lang, False), "builtCommit": header["builtCommit"], "builtTree": header["builtTree"],
                "builtAt": header["builtAt"], "bundleSchemaVersion": header["bundleSchemaVersion"]}
        if status == "stale-kept":
            unit["failedAt"], unit["failReason"] = stale[unit_id]
        b.node("Unit", unit)
        unit_rows.append([unit_id, status, SN.bundle_hash(records), unit["flowAvailable"], unit.get("failReason", "")])
        b.edge(("Unit", unit_id), "IN_REPO", ("Repo", repo_id))
        rows.units.append({"unitId": unit_id, "rootPath": header["unit"]["rootPath"], "language": lang})
        rows.files[unit_id] = 0
        for skipped in header.get("skippedFiles", ()):            # a coverage fact with one source: the header (p0-graph §1.3)
            rows.nodes.setdefault("SkippedFile", []).append(
                {"repoId": repo_id, "unitId": unit_id, "path": skipped["path"], "reason": skipped["reason"]})
            b.edge(("Unit", unit_id), "SKIPS", ("SkippedFile", skipped["path"]))
        for r in records:
            _record(b, r, unit_id, repo_id, point_label, symbol_ids, sym_lang, sym_path, depths)
    unit_index = {u["unitId"] for u in rows.nodes["Unit"]}
    for nm in sorted(not_mapped, key=lambda n: n.unit_id):
        b.node("Repo", {"repoId": nm.repo_id})
        b.node("Unit", {"unitId": nm.unit_id, "repoId": nm.repo_id, "rootPath": nm.root_path, "language": nm.language,
                        "buildStatus": "not-mapped", "flowAvailable": flow.get(nm.language, False), "failReason": nm.fail_reason})
        b.edge(("Unit", nm.unit_id), "IN_REPO", ("Repo", nm.repo_id))
        unit_index.add(nm.unit_id)
        unit_rows.append([nm.unit_id, "not-mapped", "", flow.get(nm.language, False), nm.fail_reason])

    # Name -> every symbol of that language and name (graph note §6.1 item 6)
    for n in rows.nodes.get("Name", ()):
        for sym in sorted(by_name.get((n["lang"], n["name"]), ())):
            b.edge(("Name", n["nameKey"]), "RESOLVES_TO", ("Symbol", sym))

    _contracts(b, facts, links, point_label, symbol_ids, entry_ids, unit_index, {p["id"] for p in packs})
    _mounts(b, facts, packs, point_label, symbol_ids, sym_unit)
    _launches(b, facts, entry_ids, unit_index)

    for p_id, version, content in triples:
        b.node("Pack", {"packId": p_id, "version": version, "contentHash": content})
    check_edge_groups(rows)

    fresh_built = [u["builtAt"] for u in rows.nodes["Unit"] if u["buildStatus"] == "fresh"]
    pack_set = SN.pack_set_hash(triples)
    code = SN.code_hash(code_root)
    manifest = SN.manifest_hash(pack_set, code, unit_rows)
    rows.snapshot = _clean({
        "snapshotId": SN.snapshot_id(system, manifest), "systemId": system, "graphModelVersion": SN.GRAPH_MODEL_VERSION,
        "bundleSchemaVersion": f"{SUPPORTED_MAJOR}.{max_minor}", "vocabVersion": next(iter(vocabs), None),
        "manifestHash": manifest, "packSetHash": pack_set, "codeHash": code,
        "auditSetHash": SN.audit_set_hash(audit_ids), "builtAt": max(fresh_built, default=None),
        "verdictLogOffset": SN.VERDICT_LOG_OFFSET, "verdictLogHash": SN.EMPTY_VERDICT_LOG_HASH,
    })
    return rows


def check_edge_groups(rows: GraphRows) -> None:
    """Every edge group is in the closed list of ``-p1-graph.md`` §4; a row in any other group fails the build before any write."""
    for group in rows.edges:
        if group not in EDGE_GROUPS:
            raise AssembleError(f"edge group ({group[0]})-[:{group[1]}]->({group[2]}) is not in the P1 list")


def _record(b: _Builder, r: dict, unit_id: str, repo_id: str, point_label: dict, symbol_ids: set, sym_lang: dict, sym_path: dict,
            depths: dict) -> None:
    t = r["type"]
    if t == "file":
        b.rows.files[unit_id] += 1
        b.node("File", {"fileId": r["id"], "repoId": r["repoId"], "unitId": unit_id, "path": r["path"], "kind": r["kind"],
                        "role": r["role"], "lang": r["lang"], "contentHash": r["contentHash"]})
        b.edge(("Unit", unit_id), "HAS_FILE", ("File", r["id"]))
    elif t == "symbol":
        props = {"symbolId": r["id"], "repoId": r["repoId"], "unitId": unit_id}
        props.update({k: r.get(k) for k in _SYMBOL_PROPS})
        props["depth"] = depths[r["id"]]
        if r.get("ownedRegions"):
            props["ownedRegions"] = _flatten_regions(r["ownedRegions"])
        b.node("Symbol", props)
        b.edge(("File", r["fileId"]), "DEFINES", ("Symbol", r["id"]))
    elif t == "param":
        b.node("Param", {"pointId": r["id"], "index": r["index"], "name": r["name"], "line": r["line"]})
        b.edge(("Symbol", r["symbolId"]), "HAS_PARAM", ("Param", r["id"]))
    elif t == "ret":
        b.node("Ret", {"pointId": r["id"], "line": r["line"]})
        b.edge(("Symbol", r["symbolId"]), "HAS_RETURN", ("Ret", r["id"]))
    elif t == "cond":
        b.node("Cond", {"pointId": r["id"], "line": r["line"], "code": r["code"]})
        b.edge(("Symbol", r["symbolId"]), "HAS_COND", ("Cond", r["id"]))
    elif t == "callsite":
        sid = r["symbolId"]
        b.node("CallSite", {"callSiteId": r["id"], "repoId": repo_id, "unitId": unit_id, "path": sym_path[sid], "line": r["line"],
                            "calleeName": r["calleeName"]})
        b.edge(("Symbol", sid), "HAS_CALL", ("CallSite", r["id"]))
        for target in r["targets"]:
            if target["symbolId"] not in symbol_ids:
                b.drop("calls-target-not-in-snapshot")
                continue
            b.edge(("CallSite", r["id"]), "CALLS", ("Symbol", target["symbolId"]), _non_default("CALLS", r["status"], r["reason"]),
                   key=(r["status"], r["reason"]))
        if r["calleeName"] != "<dynamic>":                          # a <dynamic> site gets no CALLS_NAME (D11)
            nk = ids.name_key(sym_lang[sid], r["calleeName"])
            b.node("Name", {"nameKey": nk, "lang": sym_lang[sid], "name": r["calleeName"]})
            b.edge(("CallSite", r["id"]), "CALLS_NAME", ("Name", nk))
    elif t == "dependency":
        b.node("Dependency", {"depId": r["depId"], "ecosystem": r["ecosystem"], "package": r["package"]})
        b.edge(("Unit", unit_id), "DEPENDS_ON", ("Dependency", r["depId"]),
               {"version": r["version"], "manifestPath": r["manifestPath"], "line": r["line"], "scope": r["scope"]},
               key=(r["manifestPath"], r["line"], r["scope"]))
    elif t == "data":
        use, dfn = r["use"], r["def"]
        ul, dl = point_label.get(use), point_label.get(dfn)
        if ul not in ("CallSite", "Ret", "Cond") or dl not in ("Param", "CallSite"):
            b.drop("data-endpoint-missing")
            return
        props = {"fromPath": r["fromPath"], "toPath": r["toPath"], **_non_default("DATA", r["status"], r["reason"])}
        b.edge((ul, use), "DATA", (dl, dfn), props, key=(r["fromPath"], r["toPath"]))
    elif t == "controlledBy":
        point, cond = r["point"], r["cond"]
        if point == cond:                                           # G-4: a self-edge is dropped
            b.drop("controlled-by-self-edge")
            return
        pl = point_label.get(point)
        if pl is None or point_label.get(cond) != "Cond":
            b.drop("controlled-by-endpoint-missing")
            return
        b.edge((pl, point), "CONTROLLED_BY", ("Cond", cond))
    elif t == "configEntry":                                        # G1: no format, bindValue, example, bindParts
        b.node("ConfigEntry", {"entryId": r["id"], "repoId": r["repoId"], "unitId": unit_id, "path": r["path"], "line": r["line"],
                               "keyPath": _mask_key_path(r["keyPath"]), "valueHash": r["valueHash"]})
    elif t == "launchFact":                                         # G2: exactly one of keyPath and launchKind; no args/default/origin
        props = {"entryId": r["id"], "repoId": repo_id, "unitId": unit_id, "path": r["path"], "line": r["line"], "launchKind": r["kind"]}
        props.update({dst: r[src] for src, dst in _LAUNCH_PROPS if r.get(src) is not None})
        b.node("ConfigEntry", props)


def _cp(b: _Builder, kind: str, key: str) -> str:
    cid = ids.contract_point_id(kind, key)
    b.node("ContractPoint", {"cpId": cid, "kind": kind, "key": key, "keyNorm": key})
    return cid


def _best(items: Iterable[dict]) -> dict:
    """The strongest fact of a group; ties by the lowest ``siteId`` (so the choice does not depend on input order)."""
    return min(items, key=lambda f: (_STRENGTH[f["status"]], f["siteId"]))


def _contracts(b: _Builder, facts: Any, links: Any, point_label: dict, symbol_ids: set, entry_ids: set, unit_index: set,
               pack_ids: set) -> None:
    providers: dict[tuple, list[dict]] = {}
    for p in sorted(facts.providers, key=lambda f: (f["kind"], f["key"] or "", f["anchorId"], f["siteId"])):
        anchor = p["anchorId"]
        if anchor in entry_ids or point_label.get(anchor) == "CallSite":
            raise AssembleError(f"PROVIDES from a {'ConfigEntry' if anchor in entry_ids else 'CallSite'} is not in the P1 edge list "
                                "(-p1-graph.md §4); no P1 pack emits it")
        if anchor not in symbol_ids:
            b.drop("provider-handler-missing")
            continue
        if p["key"] is None:
            b.drop("provider-key-none")
            continue
        providers.setdefault((anchor, p["kind"], p["key"]), []).append(p)
    for (anchor, kind, key), group in providers.items():
        best = _best(group)
        b.edge(("Symbol", anchor), "PROVIDES", ("ContractPoint", _cp(b, kind, key)), {"status": best["status"], "reason": best["reason"]})
    consumers: dict[tuple, list[dict]] = {}
    for c in sorted(facts.consumers, key=lambda f: (f["kind"], f["key"] or "", f["anchorId"], f["siteId"])):
        if point_label.get(c["siteId"]) != "CallSite":
            b.drop("consumer-site-missing")
            continue
        if c["key"] is None:
            b.drop("consumer-key-none")
            continue
        consumers.setdefault((c["siteId"], c["kind"], c["key"]), []).append(c)
    for (site, kind, key), group in consumers.items():
        best = _best(group)
        b.edge(("CallSite", site), "CONSUMES", ("ContractPoint", _cp(b, kind, key)), {"status": best["status"], "reason": best["reason"]})

    seen_links: set[str] = set()
    for lk in sorted(links.links, key=lambda x: x.link_id):
        if lk.link_id in seen_links:
            raise AssembleError(f"link {lk.link_id}: given twice")
        seen_links.add(lk.link_id)
        _check_link(lk, pack_ids)
        if lk.param_map:
            b.drop("link-param-map")                                # G12: a map is not a FalkorDB property value
        if lk.provider is not None and lk.provider not in symbol_ids:
            if lk.provider in entry_ids:
                raise AssembleError(f"link {lk.link_id}: (Link)-[:TO]->(ConfigEntry) is not in the P1 edge list (-p1-graph.md §4)")
            raise AssembleError(f"link {lk.link_id}: a link whose provider is outside the snapshot is a corrupt input")
        if lk.provider is None and lk.to_unit is not None and lk.to_unit not in unit_index:
            raise AssembleError(f"link {lk.link_id}: TO_UNIT names a unit with no Unit row")
        b.node("Link", {"linkId": lk.link_id, "kind": lk.kind, "status": lk.status, "reason": lk.reason, "matcher": lk.matcher})
        # D6: no candidateCount; N is derived per VIA edge at read time
        if lk.provider is not None:
            b.edge(("Link", lk.link_id), "TO", ("Symbol", lk.provider))
        elif lk.to_unit is not None:                                # TO_UNIT iff the never-built unit is known (None: outside every repo)
            b.edge(("Link", lk.link_id), "TO_UNIT", ("Unit", lk.to_unit))
        if lk.key_norm:
            b.edge(("Link", lk.link_id), "ON", ("ContractPoint", _cp(b, lk.kind, lk.key_norm)))
        for v in sorted(lk.sites, key=lambda v: (v.site_id, v.consumer_key or "")):      # F8: row order is the link's, not the linker's
            if point_label.get(v.site_id) != "CallSite":
                b.drop("via-site-missing")
                continue
            props = {"status": v.status, "reason": v.reason}
            if v.consumer_key is not None:
                props["consumerKey"] = v.consumer_key
            b.edge(("CallSite", v.site_id), "VIA", ("Link", lk.link_id), props, key=(v.consumer_key,))
        for e in sorted(lk.evidence, key=lambda e: (e["siteId"], e["ord"], e["edge"], e.get("id") or e.get("entryId") or "")):
            if e["edge"] == "LIFTED_VIA":
                label = "CallSite" if e["kind"] == "callsite" else "Symbol"
                ok = point_label.get(e["id"]) == "CallSite" if label == "CallSite" else e["id"] in symbol_ids
                if not ok:
                    b.drop("lifted-hop-missing")
                    continue
                props = {"ord": e["ord"], "siteId": e["siteId"], "line": e["line"], "urlPath": e["urlPath"]}
                props.update({k: e[k] for k in ("verbPath", "viaSiteId", "viaLine") if e.get(k) is not None})
                b.edge(("Link", lk.link_id), "LIFTED_VIA", (label, e["id"]), props, key=(e["siteId"], e["ord"]))
            else:
                if e["entryId"] not in entry_ids:
                    b.drop("bound-entry-missing")
                    continue
                props = {"siteId": e["siteId"], "ord": e["ord"]}
                props.update({k: e[k] for k in ("port", "portOrigin", "packId") if e.get(k) is not None})
                b.edge(("Link", lk.link_id), "BOUND_BY", ("ConfigEntry", e["entryId"]), props, key=(e["siteId"], e["entryId"]))


def _check_link(lk: Any, pack_ids: set) -> None:
    """O-9 (``-p1-graph.md`` §2) and the BOUND_BY/LIFTED_VIA evidence rules (G7, §8 (c), (d)); ids only in the messages."""
    lid = lk.link_id
    if not lk.sites:
        raise AssembleError(f"link {lid}: no consumer site")
    site_ids = {v.site_id for v in lk.sites}
    bound_sites: set[str] = set()
    seen_bound: set[tuple] = set()
    for e in lk.evidence:
        if e["siteId"] not in site_ids:
            raise AssembleError(f"link {lid}: evidence names a site that is not a VIA site of the link")
        if e["edge"] != "BOUND_BY":
            continue
        if (e["siteId"], e["entryId"]) in seen_bound:
            raise AssembleError(f"link {lid}: two BOUND_BY edges for one (site, entry)")
        seen_bound.add((e["siteId"], e["entryId"]))
        bound_sites.add(e["siteId"])
        port, origin, pack = e.get("port"), e.get("portOrigin"), e.get("packId")
        if (port is None) != (origin is None):
            raise AssembleError(f"link {lid}: BOUND_BY port and portOrigin come together")
        if port is not None and not (isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535):
            raise AssembleError(f"link {lid}: BOUND_BY port outside 1-65535")
        if origin is not None and origin not in _PORT_ORIGINS:
            raise AssembleError(f"link {lid}: BOUND_BY portOrigin outside the closed set")
        if (pack is not None) != (origin == "pack-default"):
            raise AssembleError(f"link {lid}: BOUND_BY packId is present iff portOrigin is pack-default")
        if pack is not None and pack not in pack_ids:
            raise AssembleError(f"link {lid}: BOUND_BY packId names no loaded Pack")
    for v in lk.sites:
        if v.status not in _STRENGTH:
            raise AssembleError(f"link {lid}: a VIA status outside the closed set")
        if (v.reason == "binding-confirmed") != (v.status == "confirmed" and v.site_id in bound_sites):
            raise AssembleError(f"link {lid}: VIA reason violates O-9 (binding-confirmed iff confirmed and bound at that site)")
    if lk.status != ("confirmed" if any(v.status == "confirmed" for v in lk.sites) else "possible"):
        raise AssembleError(f"link {lid}: Link.status is not the best over its sites")
    if (lk.reason == "binding-confirmed") != any(v.status == "confirmed" and v.reason == "binding-confirmed" for v in lk.sites):
        raise AssembleError(f"link {lid}: Link.reason violates O-9 (binding-confirmed iff some confirmed site is)")
    if lk.reason != "binding-confirmed":
        best = min(lk.sites, key=lambda v: (_STRENGTH[v.status], v.site_id, v.consumer_key or ""))
        if lk.reason != best.reason:
            raise AssembleError(f"link {lid}: Link.reason is not the reason of the site that sets the best status")


def _mounts(b: _Builder, facts: Any, packs: Sequence[Mapping], point_label: dict, symbol_ids: set, sym_unit: dict) -> None:
    """``MOUNTED_BY {ord, mountPath}`` (G6): ``ord`` by (mountPath, line), provider ``symbolId`` and mount ``callSiteId`` as ids."""
    mount_facts = {(m["packId"], m["siteId"]): m for m in facts.mounts}
    kinds_of: dict[str, set[str]] = {}
    for p in packs:
        kinds_of.setdefault(p["id"], set()).update(k for u in p.get("urlBindings", ()) for k in u.get("kinds", ()))
    entries: dict[str, set[tuple[str, int, str]]] = {}
    for p in facts.providers:                                       # a handler mounted through router chains (fastapi `include_router`)
        anchor = p["anchorId"]
        if anchor not in symbol_ids:
            continue
        for chain in p.get("evidence", {}).get("mounts", ()):
            for s in chain.get("sites", ()):
                m = mount_facts.get((p["packId"], s["siteId"]))
                if point_label.get(s["siteId"]) != "CallSite":
                    b.drop("mount-site-missing")
                elif m is None:
                    b.drop("mount-fact-missing")
                else:
                    entries.setdefault(anchor, set()).update((_mount_path(x), m["line"], s["siteId"]) for x in m["prefixes"])
    for lf in facts.launch_facts:                                   # a unit-level mount (`mountPaths`): every provider of the unit it serves
        if lf.get("fact") != "mountPath":
            continue
        kinds = kinds_of.get(lf["packId"], set())
        if not kinds:
            b.drop("mount-path-no-kinds")
            continue
        if point_label.get(lf["siteId"]) != "CallSite":
            b.drop("mount-site-missing")
            continue
        for p in facts.providers:
            if p["kind"] in kinds and p["anchorId"] in symbol_ids and sym_unit[p["anchorId"]] == lf["unitId"]:
                entries.setdefault(p["anchorId"], set()).add((_mount_path(lf["path"]), lf["line"], lf["siteId"]))
    for anchor in sorted(entries):
        for ordinal, (path, _line, site) in enumerate(sorted(entries[anchor])):
            b.edge(("Symbol", anchor), "MOUNTED_BY", ("CallSite", site), {"ord": ordinal, "mountPath": path}, key=(site, path))


def _launches(b: _Builder, facts: Any, entry_ids: set, unit_index: set) -> None:
    """``LAUNCHES`` for what the pack runtime resolves (``launchUnit``: a command that names a unit through ``targetArg``)."""
    for lf in sorted((f for f in facts.launch_facts if f.get("fact") == "launchUnit"), key=lambda f: (f["launchId"], f["unitId"])):
        if lf["launchId"] not in entry_ids:
            b.drop("launch-entry-missing")
        elif lf["unitId"] not in unit_index:
            b.drop("launch-unit-missing")
        else:
            b.edge(("ConfigEntry", lf["launchId"]), "LAUNCHES", ("Unit", lf["unitId"]))
