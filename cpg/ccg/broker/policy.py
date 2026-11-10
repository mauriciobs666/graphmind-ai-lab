"""Mode resolution and the request rule table (C-1001 P1-29).

Spec: ``docs/plans/connected-code-graph-egress-spec3.md`` section 2.4 (closed enumerations), 2.4a (components per
run kind and per purpose), 3.2 (mode resolution), 3.3 (the rule table, first match wins, and the ``R0`` checks) and
3.5 (the closed deny vocabulary). Everything here is a pure function of its arguments: no clock, no file, no
network. The broker (`ccg.broker.server`) supplies the loaded system, the run and the parsed request, and acts on
the `Verdict`; address validation after an allow (section 3.4) is `ccg.broker.addr`'s.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from ccg.broker import framing, structured
from ccg.manifest import ManifestError, Manifest, canonical_endpoint

if TYPE_CHECKING:  # pragma: no cover
    from ccg.broker.runs import Run

COMPONENTS = ("build.git", "build.frontend", "build.indexer", "build.pkgmgr", "answer.interpret",
              "answer.summarize", "answer.pr-adapter", "access.idp", "access.check", "vulndb.refresh")
PURPOSES = ("git-fetch", "dependency-download", "model-call", "idp", "access-check", "pr-addon", "vulndb-refresh",
            "import-name-lookup")
PAYLOAD_CLASSES = ("none", "repo-names-and-identity", "repo-derived", "vuln-feed-request")
METHODS = ("GET", "POST")
RUN_KINDS = ("refresh", "answer", "vulndb")

# section 2.4a, first table: the component prefixes a run kind admits.
RUN_KIND_COMPONENTS = {
    "refresh": ("build.", "access."),
    "answer": ("answer.", "access."),
    "vulndb": ("vulndb.",),
}
# section 2.4a, second table.
COMPONENT_PURPOSES = {
    "build.git": ("git-fetch",),
    "build.frontend": ("dependency-download", "import-name-lookup"),
    "build.indexer": ("dependency-download", "import-name-lookup"),
    "build.pkgmgr": ("dependency-download", "import-name-lookup"),
    "answer.interpret": ("model-call",),
    "answer.summarize": ("model-call",),
    "answer.pr-adapter": ("pr-addon",),
    "access.idp": ("idp",),
    "access.check": ("access-check",),
    "vulndb.refresh": ("vulndb-refresh",),
}

# section 3.5: exactly these 21 values.
DENY_REASONS = (
    "token-invalid", "enum-invalid", "component-not-in-run-kind", "component-purpose-mismatch", "codebases-empty",
    "codebases-in-vulndb", "codebases-not-subset", "endpoint-not-canonical", "path-not-allowed",
    "method-body-mismatch", "body-too-large", "resolve-failed", "forbidden-address", "address-check-unavailable",
    "not-in-p1:git-fetch", "not-in-p1:controlled-download", "not-in-p1:import-name-service", "payload-shape",
    "system-local-only-model", "sticky-local-only", "no-rule-matched",
)
# section 3.3's table, in evaluation order.
RULES = ("R0-INVALID", "R2-GIT-OWN-REMOTE", "R8-CONTROLLED-REGISTRY", "R9-IMPORT-NAME-SERVICE", "R1-LOCAL-ENDPOINT",
         "R3-VULNDB", "R4-NAMED-SERVICE", "R5-SYSTEM-LOCAL-MODEL", "R6-STICKY-LOCAL", "R7-HOSTED-SERVICE",
         "R99-DEFAULT-DENY")

REPO_ID_RE = re.compile(r"[a-z0-9._-]{1,256}")  # the audit schema's `repoId`
MAX_PATH_CHARS = 256
# The manifest's own `paths` pattern (egress-v1.schema.json): a caller path that cannot be an allow-list path is junk.
PATH_RE = re.compile(r"/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*")
MAX_ENDPOINT_CHARS = 300


# ------------------------------------------------------------------ the loaded system


@dataclass(frozen=True)
class LoadedSystem:
    """One manifest as the broker holds it: the egress section, each repository's default mode, a digest of both."""

    system: str
    modes: Mapping[str, str]
    egress: Mapping[str, Any]
    digest: str

    @classmethod
    def from_manifest(cls, m: Manifest) -> "LoadedSystem":
        modes = {r.repo_id: r.policy.egress for r in m.repos}
        for rid in modes:
            if not REPO_ID_RE.fullmatch(rid):
                raise ManifestError(f"repoId {rid!r} does not fit the audit schema's repoId (1-256 of [a-z0-9._-])")
        blob = json.dumps({"egress": m.egress, "modes": modes}, sort_keys=True, separators=(",", ":"))
        return cls(m.system, modes, m.egress, hashlib.sha256(blob.encode()).hexdigest())

    def lookup(self, endpoint: str):
        """``(section, entry)`` of the entry whose canonical endpoint equals ``endpoint``, else None (exact match)."""
        for section in ("local", "outbound", "hosted"):
            for e in self.egress[section]:
                if e["endpoint"] == endpoint:
                    return section, e
        return None

    def idp_endpoints(self) -> set[str]:
        return {e["endpoint"] for e in self.egress["outbound"] if e["kind"] == "idp"}


# ------------------------------------------------------------------ mode resolution (section 3.2)


def resolve_modes(system: LoadedSystem, codebases: list[str], override: str | None):
    """``(modes, override_refused, system_local_only)``; ``modes`` rows are the policy record's ``modes[]``."""
    rows, refused = [], False
    for c in codebases:
        default = system.modes[c]
        if default == "local-only":
            if override == "hosted":
                eff, rule, refused = "local-only", "P2-OVERRIDE-REFUSED", True
            else:
                eff, rule = "local-only", "P1-DEFAULT"
        else:
            if override == "local":
                eff, rule = "local-only", "P3-OVERRIDE-LOCAL"
            else:
                eff, rule = "hosted", "P1-DEFAULT"
        rows.append({"repoId": c, "default": default, "effective": eff, "rule": rule})
    effective = {r["repoId"]: r["effective"] for r in rows}
    local_only = any(
        (effective[rid] if rid in effective else mode) == "local-only" for rid, mode in system.modes.items()
    )
    return rows, refused, local_only


# ------------------------------------------------------------------ request parsing (section 5.1: junk becomes null)


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()  # never raises on a caller's string


def _string_free(value: Any) -> bool:
    """False if any string inside ``value`` (keys included) is not UTF-8 encodable: such a value is junk whatever
    its predicate says, and nothing downstream may be asked to encode it."""
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            try:
                v.encode("utf-8")
            except UnicodeEncodeError:
                return False
        elif isinstance(v, dict):
            stack.extend(v.keys())
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return True


def sanitize(raw: Any, repo_count: int) -> tuple[dict, dict]:
    """``(clean, rejected)``: each caller field that does not fit its record field is None, with the digest of its
    raw JSON in ``rejected``, so caller input can never make a record invalid (section 5.1)."""
    req = raw if isinstance(raw, dict) else {}
    target = req.get("target") if isinstance(req.get("target"), dict) else {}
    rejected: dict[str, str] = {}

    def pick(name: str, value: Any, ok) -> Any:
        if _string_free(value) and ok(value):
            return value
        rejected[name] = _digest(value)
        return None

    def endpoint_ok(v: Any) -> bool:
        if not isinstance(v, str) or len(v) > MAX_ENDPOINT_CHARS:
            return False
        try:
            return canonical_endpoint(v) == v
        except ManifestError:
            return False

    def codebases_ok(v: Any) -> bool:
        return (isinstance(v, list) and len(v) <= repo_count
                and all(isinstance(c, str) and REPO_ID_RE.fullmatch(c) for c in v))

    clean = {
        "component": pick("component", req.get("component"), lambda v: isinstance(v, str) and v in COMPONENTS),
        "purpose": pick("purpose", req.get("purpose"), lambda v: isinstance(v, str) and v in PURPOSES),
        "payloadClass": pick("payloadClass", req.get("payloadClass"),
                             lambda v: isinstance(v, str) and v in PAYLOAD_CLASSES),
        "method": pick("target.method", target.get("method"), lambda v: isinstance(v, str) and v in METHODS),
        "path": pick("target.path", target.get("path"), lambda v: isinstance(v, str) and len(v) <= MAX_PATH_CHARS
                    and PATH_RE.fullmatch(v) is not None),
        "endpoint": pick("endpoint", req.get("endpoint"), endpoint_ok),
        "codebases": pick("codebases", req.get("codebases"), codebases_ok),
    }
    # R0 check 6 asks whether the caller declared codebases at all, which a junk declaration still did.
    raw_cb = req.get("codebases")
    clean["_cb_declared"] = raw_cb is not None and raw_cb != []
    return clean, rejected


# ------------------------------------------------------------------ evaluation (section 3.3)


@dataclass(frozen=True)
class Verdict:
    rule: str
    decision: str  # allow | deny
    reason: str | None
    mode: str | None  # M of section 3.3; None only for token-invalid
    section: str | None = None  # the matched allow-list section of an allow
    entry: Mapping[str, Any] | None = None


def request_mode(run: "Run", clean: Mapping[str, Any]) -> str:
    """``M`` (section 3.3). A declaration that cannot be trusted (invalid) never narrows a refresh run's scope."""
    if run.run_kind == "vulndb":
        scope: list[str] = []
    elif run.run_kind == "answer":
        scope = list(run.codebases)
    else:
        cb = clean["codebases"]
        ok = isinstance(cb, list) and bool(cb) and len(set(cb)) == len(cb) and all(c in run.effective for c in cb)
        scope = list(cb) if ok else list(run.codebases)
    return "local-only" if any(run.effective[c] == "local-only" for c in scope) else "hosted"


def _r0(run: "Run | None", system: LoadedSystem, clean: Mapping[str, Any], body_bytes: bytes | None):
    """The first failing R0 check's reason, else None. Section 3.3, checks 1-11 in order."""
    if run is None:
        return "token-invalid"
    c = clean
    if c["component"] is None or c["purpose"] is None or c["payloadClass"] is None or c["method"] is None:
        return "enum-invalid"
    if not c["component"].startswith(RUN_KIND_COMPONENTS[run.run_kind]):
        return "component-not-in-run-kind"
    if c["purpose"] not in COMPONENT_PURPOSES[c["component"]]:
        return "component-purpose-mismatch"
    cb = c["codebases"]
    if cb == [] and run.run_kind != "vulndb":
        return "codebases-empty"
    if (cb or c["_cb_declared"]) and run.run_kind == "vulndb":
        return "codebases-in-vulndb"
    if cb is None or len(set(cb)) != len(cb) or any(x not in run.codebases for x in cb):
        return "codebases-not-subset"
    if c["endpoint"] is None:
        return "endpoint-not-canonical"
    hit = system.lookup(c["endpoint"])
    if hit is not None and c["path"] not in hit[1]["paths"]:
        return "path-not-allowed"
    if c["method"] == "GET":
        if body_bytes is not None or c["payloadClass"] != "none":
            return "method-body-mismatch"
    elif body_bytes is None:
        return "method-body-mismatch"
    if body_bytes is not None and len(body_bytes) > framing.MAX_REQUEST_BYTES:
        return "body-too-large"
    return None


_NOT_IN_P1 = {
    "git-fetch": ("R2-GIT-OWN-REMOTE", "not-in-p1:git-fetch"),
    "dependency-download": ("R8-CONTROLLED-REGISTRY", "not-in-p1:controlled-download"),
    "import-name-lookup": ("R9-IMPORT-NAME-SERVICE", "not-in-p1:import-name-service"),
}
_NAMED_KIND = {"idp": "idp", "access-check": "access-check", "pr-addon": "pr-addon"}


def evaluate(system: LoadedSystem, run: "Run | None", clean: Mapping[str, Any], body: Any,
             body_bytes: bytes | None) -> Verdict:
    """Section 3.3: the first matching row. ``run`` is None for an unknown or expired token."""
    mode = request_mode(run, clean) if run is not None else None
    reason = _r0(run, system, clean, body_bytes)
    if reason is not None:
        return Verdict("R0-INVALID", "deny", reason, mode)
    purpose = clean["purpose"]
    if purpose in _NOT_IN_P1:  # R2, R8, R9 sit before R1 in P1 (section 3.3, notes)
        rule, why = _NOT_IN_P1[purpose]
        return Verdict(rule, "deny", why, mode)
    hit = system.lookup(clean["endpoint"])
    section, entry = hit if hit else (None, None)
    if section == "local":
        return Verdict("R1-LOCAL-ENDPOINT", "allow", None, mode, section, entry)
    if (run.run_kind == "vulndb" and purpose == "vulndb-refresh" and clean["component"] == "vulndb.refresh"
            and clean["method"] == "GET" and clean["payloadClass"] == "none" and section == "outbound"
            and entry["kind"] == "vulndb"):
        return Verdict("R3-VULNDB", "allow", None, mode, section, entry)
    if purpose in _NAMED_KIND and section == "outbound" and entry["kind"] == _NAMED_KIND[purpose]:
        if clean["payloadClass"] == "repo-names-and-identity" and structured.valid_identity_body(
                body, set(system.modes), system.idp_endpoints()):
            return Verdict("R4-NAMED-SERVICE", "allow", None, mode, section, entry)
        return Verdict("R4-NAMED-SERVICE", "deny", "payload-shape", mode)
    if purpose == "model-call" and run.system_local_only:
        return Verdict("R5-SYSTEM-LOCAL-MODEL", "deny", "system-local-only-model", mode)
    if mode == "local-only":
        return Verdict("R6-STICKY-LOCAL", "deny", "sticky-local-only", mode)
    if mode == "hosted" and purpose == "model-call" and section == "hosted":
        return Verdict("R7-HOSTED-SERVICE", "allow", None, mode, section, entry)
    return Verdict("R99-DEFAULT-DENY", "deny", "no-rule-matched", mode)
