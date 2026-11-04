"""Rule-pack v1 validator (S31): the checks JSON Schema cannot carry.

Contract: ``docs/plans/connected-code-graph-rule-pack-spec.md`` (section 1 and the negative examples of
section 10). ``ccg/schemas/rule-pack-v1.schema.json`` is the structural half (closed objects, enums,
typed leaves); this module adds what a schema cannot say: the RE2 subset of ``regex`` (1.3), the reason
registry ``ccg/registry/reasons.yaml`` (1.9: N-9, N-17, N-27), contract kinds that are core or declared
(N-10), ``id`` equal to the directory name (N-2), cross-references (lifting, query, slot), query joins
(N-14), the ``onUnknown`` defaults including the topic cells (O-16), signature samples and negatives
(N-22) and the severity row derivation (N-24). ``pack_errors`` returns every problem it finds;
``validate_pack`` raises ``PackError`` when there is one.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import jsonschema
import yaml

HERE = Path(__file__).parent
SCHEMA_PATH = HERE / "schemas" / "rule-pack-v1.schema.json"
BUNDLE_SCHEMA_PATH = HERE / "schemas" / "bundle-v1.schema.json"
REGISTRY_PATH = HERE / "registry" / "reasons.yaml"

# Spec3 section 6's contract kinds; a pack adds more only through ``vocab`` (N-10).
CORE_KINDS = frozenset({"http", "mcp_tool", "rpc", "topic", "data_entity", "env_var", "config_key", "field", "library_api"})
JOIN_FIELDS = frozenset({"id", "symbolId", "parentId", "callSiteId", "unitId", "targetId", "fileId", "point", "cond", "use", "def"})
SEV_PREFIX = {"injection": "INJ", "authorization": "AUTHZ", "configuration": "RCFG", "secrets": "SEC"}
LEVELS = ["low", "medium", "high", "critical"]
EDGES = {"provides": "PROVIDES", "consumes": "CONSUMES", "reads": "READS", "writes": "WRITES"}


class PackError(ValueError):
    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("; ".join(messages))


# ---------------------------------------------------------------- loading

@functools.cache
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@functools.cache
def _bundle_defs() -> dict:
    return json.loads(BUNDLE_SCHEMA_PATH.read_text())["$defs"]


class _StrictLoader(yaml.SafeLoader):
    """A pack is data with string keys: a duplicate key or a non-string key (``on:`` is True in YAML 1.1) is an error."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for k, _ in node.value:
            key = self.construct_object(k, deep=True)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(None, None, f"non-string key {key!r}", k.start_mark)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", k.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


def load_yaml(text: str):
    return yaml.load(text, Loader=_StrictLoader)


def load_pack(path: Path) -> dict:
    """Load ``cpg/packs/<id>/pack.yaml`` and validate it against its directory name; raises ``PackError``."""
    path = Path(path)
    try:
        pack = load_yaml(path.read_text())
    except yaml.YAMLError as e:
        raise PackError([f"{path}: {e}"]) from e
    validate_pack(pack, dirname=path.parent.name)
    return pack


def load_registry_rows() -> list[dict]:
    rows = load_yaml(REGISTRY_PATH.read_text())
    for r in rows:
        if set(r) != {"code", "status", "on"}:
            raise PackError([f"registry row keys {sorted(r)}"])
    return rows


@functools.cache
def _registry() -> dict[str, dict]:
    return {r["code"]: r for r in load_registry_rows()}


def reason_row(code: str) -> dict | None:
    """The registry row of ``code``; a parameterized code (``severity-rule:INJ-1``) resolves to its ``name:<param>`` row."""
    reg = _registry()
    if code in reg:
        return reg[code] if ":" not in code else None  # a literal `x:<p>` is the template, not a usable code
    head, sep, param = code.partition(":")
    if sep and param:
        for k, row in reg.items():
            if k.startswith(head + ":<"):
                return row
    return None


def allows(row: dict, edge: str, kind: str) -> bool:
    """Does the row's ``on`` set contain ``EDGE`` (any kind) or ``EDGE kind``?"""
    return edge in row["on"] or f"{edge} {kind}" in row["on"]


# ---------------------------------------------------------------- the RE2 subset (spec 1.3)

def regex_problem(rx: str) -> str | None:
    """Why ``rx`` is outside the RE2 subset (lookaround, backreference, atomic or possessive group, length), or None."""
    if len(rx) > 256:
        return "longer than 256 characters"
    try:
        re.compile(rx)
    except re.error as e:
        return f"not a regex: {e}"
    i, n, in_class = 0, len(rx), False
    while i < n:
        c = rx[i]
        if c == "\\":
            nxt = rx[i + 1: i + 2]
            if (nxt.isdigit() and nxt != "0") or nxt in ("k", "g"):
                return "backreference"
            i += 2
            continue
        if in_class:
            in_class = c != "]"
        elif c == "[":
            in_class = True
            if rx.startswith("[^]", i):
                i += 2
            elif rx.startswith("[]", i):
                i += 1
        elif c == "(" and rx.startswith("(?", i):
            if rx.startswith(("(?=", "(?!", "(?<=", "(?<!"), i):
                return "lookaround"
            if rx.startswith("(?>", i):
                return "atomic group"
            if rx.startswith("(?P=", i):
                return "backreference"
            if rx.startswith("(?(", i):
                return "conditional group"
        elif c in "*+?}" and rx[i + 1: i + 2] == "+":
            return "possessive quantifier"
        i += 1
    return None


# ---------------------------------------------------------------- helpers

def _walk(node, path=()):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (i,))


def _where(path) -> str:
    return ".".join(str(p) for p in path) or "pack"


def _scalars(m) -> list[str] | None:
    """The literal strings a matcher accepts when it is a bare string, ``exact`` or a list of those; else None."""
    if isinstance(m, str):
        return [m]
    if isinstance(m, dict) and set(m) == {"exact"}:
        return [m["exact"]]
    if isinstance(m, list):
        out = []
        for x in m:
            s = _scalars(x)
            if s is None:
                return None
            out += s
        return out
    return None


def _clauses(match: dict):
    if "anyOf" in match:
        for c in match["anyOf"]:
            yield from c.items()
    else:
        yield from match.items()


_SLOT_REF = re.compile(r"\{([A-Za-z_]\w*)\}")


def _always_resolved(expr: dict) -> bool:
    """An expression that never yields an unresolved alternative: a literal, a regex group, the matched symbol's name."""
    if any(k in expr for k in ("literal", "group", "target")):
        return True
    return "firstOf" in expr and any(_always_resolved(e) for e in expr["firstOf"])


def _nested_unknown_modes(expr: dict):
    if "onUnknown" in expr:
        yield expr["onUnknown"]
    for e in expr.get("firstOf", []):
        yield from _nested_unknown_modes(e)


def default_on_unknown(edge: str, kind: str) -> str:
    """Spec 1.4's per-slot default, with O-16's topic cell: ``provides`` of kind topic degrades to ``config-derived``."""
    if edge == "provides":
        return "config-derived" if kind == "topic" else "drop"
    return "drop" if edge == "writes" else "partial"


def implied_reason(mode: str, kind: str) -> str | None:
    """The registry code an unresolved alternative gets under ``mode`` (spec 1.4); None for ``drop``."""
    if mode == "drop":
        return None
    if mode == "partial":
        return "partial-key" if kind == "http" else "runtime-built"
    return mode  # runtime-built, config-derived


# ---------------------------------------------------------------- the validator

def pack_errors(pack: dict, *, dirname: str | None = None, negatives: list[str] | None = None) -> list[str]:
    """Every problem with ``pack`` as a list of messages (empty when valid).

    ``dirname`` is the pack's directory name (N-2); ``negatives`` are log lines no ``signatures`` regex may match (N-22).
    """
    validator = jsonschema.Draft202012Validator(load_schema())
    errs = [f"{_where(e.absolute_path)}: {e.message[:200]}" for e in sorted(validator.iter_errors(pack), key=lambda e: list(map(str, e.absolute_path)))]
    if errs:
        return errs  # the semantic checks below assume the structure the schema guarantees
    return _Checker(pack, dirname, negatives or []).run()


def validate_pack(pack: dict, *, dirname: str | None = None, negatives: list[str] | None = None) -> None:
    errs = pack_errors(pack, dirname=dirname, negatives=negatives)
    if errs:
        raise PackError(errs)


class _Checker:
    def __init__(self, pack, dirname, negatives):
        self.p, self.dirname, self.negatives = pack, dirname, negatives
        self.errs: list[str] = []
        self.kinds = CORE_KINDS | {v["kind"] for v in pack.get("vocab", [])}

    def err(self, where, msg):
        self.errs.append(f"{where}: {msg}")

    def run(self) -> list[str]:
        p = self.p
        if self.dirname is not None and p["id"] != self.dirname:
            self.err("id", f"{p['id']!r} is not the directory name {self.dirname!r}")
        for path, node in _walk(p):
            if isinstance(node, dict) and isinstance(node.get("regex"), str) and (why := regex_problem(node["regex"])):
                self.err(_where(path + ("regex",)), why)
        for body in ("rules", "mounts", "liftings", "urlBindings", "queries", "signatures"):
            ids = [x["id"] for x in p.get(body, [])]
            for dup in sorted({i for i in ids if ids.count(i) > 1}):
                self.err(body, f"duplicate id {dup!r}")
        self.vocab()
        self.kind_refs()
        self.bundle_enums()
        for i, q in enumerate(p.get("queries", [])):
            self.query(q, f"queries.{i}")
        for i, m in enumerate(p.get("mounts", [])):
            self.mount(m, f"mounts.{i}")
        for i, r in enumerate(p.get("rules", [])):
            self.rule(r, f"rules.{i}")
        for i, s in enumerate(p.get("signatures", [])):
            self.signature(s, f"signatures.{i}")
        self.rows()
        return self.errs

    # -- vocabulary and kinds
    def vocab(self):
        for i, v in enumerate(self.p.get("vocab", [])):
            t = v.get("transport", "")
            if t.startswith("inhouse-") and t != f"inhouse-{self.p['id']}":
                self.err(f"vocab.{i}.transport", f"{t!r} must be 'inhouse-{self.p['id']}'")

    def kind_ref(self, kind, where):
        if kind not in self.kinds:
            self.err(where, f"kind {kind!r} is neither core nor declared in vocab")

    def kind_refs(self):
        for i, l in enumerate(self.p.get("liftings", [])):
            self.kind_ref(l["role"]["kind"], f"liftings.{i}.role.kind")
        for i, l in enumerate(self.p.get("link", [])):
            self.kind_ref(l["kind"], f"link.{i}.kind")
        for i, u in enumerate(self.p.get("urlBindings", [])):
            for k in u["kinds"]:
                self.kind_ref(k, f"urlBindings.{i}.kinds")

    # -- mounts
    def mount(self, m, where):
        u = m.get("unmounted")
        if u:
            row = reason_row(u["reason"])
            if row is None:
                self.err(f"{where}.unmounted.reason", f"{u['reason']!r} is not in the reason registry")
            else:
                if "PROVIDES" not in row["on"]:
                    self.err(f"{where}.unmounted.reason", f"{u['reason']!r} is not valid on PROVIDES")
                if row["status"] != u["status"]:
                    self.err(f"{where}.unmounted", f"status {u['status']!r} disagrees with {u['reason']!r} ({row['status']})")

    # -- matcher operands that name a closed set of the bundle schema
    def bundle_enums(self):
        """``resolution`` (any call clause), ``form`` of a decorator clause, ``kind`` of a launch clause: literals must be in the bundle enum."""
        defs = _bundle_defs()
        for path, node in _walk(self.p):
            if not isinstance(node, dict):
                continue
            parent = path[-1] if path else None
            for field, (record, prop) in (("resolution", ("callsite", "resolution")), ("form", ("decorator", "form")), ("kind", ("launchFact", "kind"))):
                if field not in node or (field == "form" and parent != "decorator") or (field == "kind" and parent != "launch"):
                    continue
                allowed = defs[record]["properties"][prop]["enum"]
                for lit in _scalars(node[field]) or []:
                    if lit not in allowed:
                        self.err(_where(path + (field,)), f"{lit!r} not in {allowed}")

    # -- rules
    def rule(self, r, where):
        p = self.p
        queries = {q["id"]: q for q in p.get("queries", [])}
        liftings = {l["id"] for l in p.get("liftings", [])}
        for subject, clause in _clauses(r["match"]):
            if subject == "evidence" and p["kind"] != "infra-signatures":
                self.err(f"{where}.match", "the evidence subject belongs to infra-signatures packs")
            if subject == "lifted" and clause["lifting"] not in liftings:
                self.err(f"{where}.match.lifted", f"lifting {clause['lifting']!r} is not declared")
            if subject == "query":
                q = queries.get(clause)
                if q is None:
                    self.err(f"{where}.match.query", f"query {clause!r} is not declared")
                elif "handler" not in q["yield"]:
                    self.err(f"{where}.match.query", f"query {clause!r} yields no 'handler' column")
        slots = r.get("extract", {})
        columns = None
        for subject, clause in _clauses(r["match"]):
            if subject == "query" and clause in queries:
                columns = set(queries[clause]["yield"])
        for slot, expr in slots.items():
            for path, node in _walk(expr):
                if isinstance(node, dict) and "yield" in node and (columns is None or node["yield"] not in columns):
                    self.err(f"{where}.extract.{slot}", f"yield {node['yield']!r} is not a column of the rule's query")
        used: dict[str, list[tuple[str, str]]] = {s: [] for s in slots}
        for j, em in enumerate(r["emit"]):
            ((name, e),) = em.items()
            if name in EDGES:
                self.emission(r, name, e, slots, used, f"{where}.emit.{j}.{name}")
        for slot, expr in slots.items():
            if "config-derived" in _nested_unknown_modes(expr) and not used[slot]:
                self.err(f"{where}.extract.{slot}", "onUnknown config-derived on a slot no key uses")

    def slot_mode(self, expr, edge, kind):
        modes = list(_nested_unknown_modes(expr))
        return modes[0] if modes else default_on_unknown(edge, kind)

    def emission(self, r, name, e, slots, used, where):
        edge, kind = EDGES[name], e["kind"]
        self.kind_ref(kind, f"{where}.kind")
        refs = _SLOT_REF.findall(e["key"])
        for ref in refs:
            if ref == "mount":
                if not self.p.get("mounts"):
                    self.err(f"{where}.key", "{mount} needs a mounts declaration")
            elif ref not in slots:
                self.err(f"{where}.key", f"slot {{{ref}}} is not extracted by the rule")
            else:
                used[ref].append((edge, kind))
        slot_modes = {ref: self.slot_mode(slots[ref], name, kind) for ref in refs if ref in slots}
        for ref, mode in slot_modes.items():
            code = implied_reason(mode, kind)
            if code is None:
                continue
            row = reason_row(code)
            if row is None or not allows(row, edge, kind):
                self.err(f"{where}.key", f"slot {{{ref}}}: onUnknown {mode} gives reason {code!r}, which is not valid on {edge} {kind}")
        if "reason" in e:
            row = reason_row(e["reason"])
            if row is None:
                self.err(f"{where}.reason", f"{e['reason']!r} is not in the reason registry")
            else:
                if not allows(row, edge, kind):
                    self.err(f"{where}.reason", f"{e['reason']!r} is not valid on {edge} {kind}")
                if "status" in e and row["status"] != e["status"]:
                    self.err(f"{where}.status", f"{e['status']!r} disagrees with {e['reason']!r} ({row['status']})")
        if e.get("status") == "confirmed":
            for ref, mode in slot_modes.items():
                if mode != "drop" and not _always_resolved(slots[ref]):
                    self.err(f"{where}.status", f"confirmed, but slot {{{ref}}} can be unresolved (onUnknown {mode}); use drop")

    # -- queries (spec 1.7)
    def query(self, q, where):
        defs = _bundle_defs()
        aliases: dict[str, str] = {}
        for i, b in enumerate(q["from"]):
            w = f"{where}.from.{i}"
            if b["as"] in aliases:
                self.err(w, f"duplicate alias {b['as']!r}")
            props = defs[b["record"]]["properties"]
            joined = False
            for f, v in b["where"].items():
                if f not in props:
                    self.err(f"{w}.where", f"{b['record']} has no field {f!r}")
                if isinstance(v, str) and v.startswith("$"):
                    alias, _, fld = v[1:].partition(".")
                    if alias not in aliases:
                        self.err(f"{w}.where.{f}", f"{v!r} refers to an alias not bound before this one")
                    elif fld.startswith("args."):
                        if "args" not in defs[aliases[alias]]["properties"]:
                            self.err(f"{w}.where.{f}", f"{aliases[alias]} has no args")
                    else:
                        if fld not in defs[aliases[alias]]["properties"]:
                            self.err(f"{w}.where.{f}", f"{aliases[alias]} has no field {fld!r}")
                        elif f in JOIN_FIELDS and fld in JOIN_FIELDS:
                            joined = True
                        else:
                            self.err(f"{w}.where.{f}", f"join {f} = {v} is not on id-typed fields")
            if i > 0 and not joined:
                self.err(w, f"binding {b['as']!r} has no join equality")
            aliases[b["as"]] = b["record"]
        for col, ref in q["yield"].items():
            alias, _, fld = ref[1:].partition(".")
            if alias not in aliases:
                self.err(f"{where}.yield.{col}", f"{ref!r} names an unbound alias")
            elif not fld.startswith("args.") and fld not in defs[aliases[alias]]["properties"]:
                self.err(f"{where}.yield.{col}", f"{aliases[alias]} has no field {fld!r}")
            elif fld.startswith("args.") and "args" not in defs[aliases[alias]]["properties"]:
                self.err(f"{where}.yield.{col}", f"{aliases[alias]} has no args")

    # -- signatures (spec 8.1)
    def signature(self, s, where):
        sid, cls, rt = s["id"], s["class"], s["runtime"]
        prefix = f"{rt}-"
        ok = sid == prefix + cls or (cls == "third-party-exception" and re.fullmatch(re.escape(prefix) + r"third-party(-[a-z0-9-]+)?", sid))
        if not ok:
            self.err(f"{where}.id", f"{sid!r} is not '{rt}-{cls}' (or '{rt}-third-party[-<library>]' for third-party-exception)")
        if not re.search(s["regex"], s["sample"]):
            self.err(f"{where}.sample", "does not match its own regex")
        for line in self.negatives:
            if re.search(s["regex"], line):
                self.err(f"{where}.regex", f"matches the negative {line!r}")
                break

    # -- severity rows (spec 9.1)
    def rows(self):
        seen = set()
        for i, r in enumerate(self.p.get("rows", [])):
            w = f"rows.{i}"
            if r["level"] != LEVELS[r["rank"]]:
                self.err(w, f"level {r['level']!r} disagrees with rank {r['rank']}")
            if r["id"] != f"{SEV_PREFIX[r['finding']]}-{4 - r['rank']}":
                self.err(w, f"id {r['id']!r} is not the derived {SEV_PREFIX[r['finding']]}-{4 - r['rank']}")
            if (r["finding"], r["rank"]) in seen:
                self.err(w, f"a second row for ({r['finding']}, rank {r['rank']})")
            seen.add((r["finding"], r["rank"]))
