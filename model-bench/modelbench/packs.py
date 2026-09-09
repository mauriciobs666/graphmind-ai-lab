"""Pack identity as the report sees it, and S2's real pack loader.

Design: `docs/plans/small-model-benchmarking.md` §3.3, §3.4.4a and Appendix A. S1 shipped no pack
loader — only the *reference* a report is handed (`PackRef`), the pre-registered metric family
(`PackMetrics`), and the two rules a report must not be able to violate whatever a manifest says
(the `metrics` block and the `sampling` contract's structural half). This module now adds S2's
loader on top of that, unchanged:

* **`Pack`** — a loaded pack, `contentHash: str` total (`load_pack`'s postcondition). `Pack.ref()`
  is the §3.3 totality boundary: a `PackRef` built from a loaded pack never has `contentHash is
  None`, where one built by `pack_ref_from_manifest` alone always does.
* **`load_pack`** / **`content_hash`** — read a pack directory and hash its bytes (excluding
  `PROVENANCE.md`), never the declared version alone (§3.3 — "declared versions get forgotten; a
  hash cannot").
* **`validate_pack`** — `[]` means valid, matching `Fingerprint.validate()`'s shape. Three
  independent axes, per §4 S2's done-condition: the `sampling` contract (structural, reusing
  `check_sampling_contract` rather than re-implementing it, plus the row-count identity and `-ml`
  §3.4 Rule 6's `replicatesPerScript > 1` rejection), `callSurface` derived from
  `environment.requires` (§3.4.4a, rejecting a pack declaring neither or both of `lmstudio-chat` /
  `lmstudio-embeddings`), and the AST import allowlist over every `.py` file the pack ships.

**Not this module's job**, named so the boundary is checked rather than assumed: `run` cross-
checking a pack's derived `callSurface` against a model's catalog `type`, and `run` calling
`validate_pack`'s AST check and failing closed, are the runner/CLI unit's (§3.3, §3.4.4a) — this
module only builds the check and makes it callable. `modelbench.tooling` (the allowlist's second
member) and `modelbench.lmstudio` are separate S2 units and are not read here; the AST check
compares import *names* syntactically and never imports what it inspects.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Mapping, NamedTuple

from modelbench.stats import ALPHA_FAMILY

#: §3.3's plugin seam: a pack module may import stdlib and this one modelbench module — nothing
#: else. `modelbench.tooling` need not exist for this check to run: it compares import *names*
#: syntactically (an AST walk) and never imports what it inspects.
_ALLOWED_PACK_IMPORT_MODULE = "modelbench.tooling"

#: Python 3.10+'s own stdlib module-name registry — no hand-maintained list to fall out of date.
_STDLIB_MODULE_NAMES: frozenset[str] = frozenset(sys.stdlib_module_names)


class PackConfigError(ValueError):
    """A manifest that cannot be honestly reported from. Raised, never warned."""


class PackMetrics(NamedTuple):
    """The pre-registered verdict family (§3.3).

    `verdictMetrics` controls **inference** — its length *is* the multiplicity correction's `k`.
    `headlineMetric` controls **presentation** — what a reader is entitled to read as "the" number,
    and it may legitimately be `None` (`guard-judge` has two co-equal errors and no headline).
    """

    verdictMetrics: tuple[str, ...]
    headlineMetric: str | None

    @property
    def k(self) -> int:
        return len(self.verdictMetrics)

    @property
    def alpha_family(self) -> float:
        """The **unadjusted** α — the *floor's*, whatever *k* is (`-ml` v1.6 §7.1, M-ML-6).

        The floor asserts *"below this, nothing can reach significance at any observed outcome"*,
        which is true only at the loosest Holm step a member can face. Computed at α/k it is `7/n`
        and the sentence is false: at n=40 a rank-2 member with b=6, c=0 reaches p=0.031 and clears
        its own 0.05 step, 15.0 pp below the 17.5 pp such a floor would print.
        """
        return ALPHA_FAMILY

    @property
    def alpha_mdd(self) -> float:
        """The **family-adjusted** α, `α/k` — the *MDD's* (§3.3). Not optional, and not the floor's.

        The MDD promises power whatever rank the member draws under Holm, so it takes the tightest
        step. This is also the *pre-registration* α that `stats.verdict`'s third precondition pins.
        """
        return self.alpha_family / self.k


class PackRef(NamedTuple):
    """A pack's identity plus what a report must print and resolve (Appendix A, extended).

    Beyond Appendix A's five fields this carries the `sampling` contract's two declarations
    (§3.3), because `report.py` resolves the analysis-unit id from `analysisUnit` and **no call
    site chooses it**. Appendix A predates §3.3's v1.4 `sampling` block; without these two fields
    the resolution has nowhere to come from.
    """

    packId: str
    packVersion: str
    #: `None` until S2's `load_pack` computes it. `""` was indistinguishable from a hash that
    #: failed to compute, in a field whose whole job is identity (review m-5). Nothing reads it at
    #: S1: the AC-3 banner reads each run's own recorded `fingerprint.packContentHash`, which is
    #: the right source at this stage.
    contentHash: str | None
    role: str
    metrics: PackMetrics
    pairingKey: tuple[str, ...]
    analysisUnit: str
    #: `sampling.seed` (§3.3) — the pack's declared resample seed, and the **only** home for it.
    #: `report.py` carried it as the literal `20260902`, duplicating a manifest field this type
    #: had no room for, so the pack's own declaration could not reach the decision it governs
    #: (review P3-5). **Its object is `-ml` §3.2d's continuous-metric bootstrap** — MRR and
    #: `sep_z`, which have no closed form at any tolerable cost — and no longer the paired binary
    #: table, whose interval became exact and takes no seed at all (`-ml` v1.11 §3.4 Rule 4, §4
    #: S1e Table D). P3-5's finding is satisfied rather than reversed: the same predicate, that a
    #: seed is named only where a resample actually decided, now selects the continuous verdicts.
    #: **No default:** a seed conjured by omission reproduces nothing, and it is the same
    #: defaulting shape `-ml` §3.4 Rule 2 refuses for the design effect. **Discriminator:** the
    #: field stays only while the embedder pack's MRR verdict is a committed deliverable.
    seed: int

    @property
    def analysisUnitIndex(self) -> int:
        """Where the analysis-unit id sits in an `ItemResult.pairingKey`. Always 0, by the rule."""
        return self.pairingKey.index(self.analysisUnit)

    @property
    def label(self) -> str:
        return f"{self.packId}@{self.packVersion}"


def metrics_from_manifest(block: Mapping[str, Any]) -> PackMetrics:
    """Parse and enforce §3.3's `metrics` block. Raises `PackConfigError`, never repairs."""
    verdicts = block.get("verdictMetrics")
    if not verdicts:
        raise PackConfigError("metrics.verdictMetrics is absent or empty")
    if "headlineMetric" not in block:
        raise PackConfigError(
            "metrics.headlineMetric key is absent; omission is not the same statement as null, "
            "and only null is a decision (plan §3.3)"
        )
    headline = block["headlineMetric"]
    if headline is not None and headline not in verdicts:
        raise PackConfigError(
            f"metrics.headlineMetric {headline!r} is not a member of verdictMetrics "
            f"{tuple(verdicts)!r}"
        )
    return PackMetrics(verdictMetrics=tuple(verdicts), headlineMetric=headline)


def check_sampling_contract(ref: PackRef) -> None:
    """§3.3's structural route: the analysis unit is `pairingKey[0]`, outermost → innermost."""
    if not ref.pairingKey:
        raise PackConfigError("sampling.pairingKey is empty")
    if ref.analysisUnit != ref.pairingKey[0]:
        raise PackConfigError(
            f"sampling.analysisUnit {ref.analysisUnit!r} is not pairingKey[0] "
            f"{ref.pairingKey[0]!r}; the analysis unit is the outermost component of the "
            "pairing key, by rule (plan §3.3)"
        )


def _ref_from_manifest_fields(
    manifest: Mapping[str, Any], *, content_hash: str | None, label: str
) -> PackRef:
    """Build a `PackRef` from a manifest's `sampling` / `metrics` blocks (§3.3).

    Shared by `pack_ref_from_manifest` (S1's manifest-only read, `content_hash=None`) and
    `Pack.ref()` (S2's loaded-pack totality boundary, `content_hash` never `None`), so the two
    routes cannot drift on what counts as a valid `sampling` block — one rule, one place it can
    break.
    """
    sampling = manifest.get("sampling") or {}
    pairing_key = tuple(sampling.get("pairingKey") or ())
    if not pairing_key:
        raise PackConfigError(f"{label}: sampling.pairingKey is absent or empty")
    if "analysisUnit" not in sampling:
        raise PackConfigError(f"{label}: sampling.analysisUnit is absent")
    if "seed" not in sampling:
        raise PackConfigError(
            f"{label}: sampling.seed is absent; the resample seed is what makes a "
            "bootstrap-decided verdict reproducible, and every comparison takes that path until "
            "a determinism probe establishes the design effect by construction (-ml §3.2d)"
        )
    ref = PackRef(
        packId=manifest["packId"],
        packVersion=manifest["packVersion"],
        contentHash=content_hash,
        role=manifest["role"],
        metrics=metrics_from_manifest(manifest.get("metrics") or {}),
        pairingKey=pairing_key,
        analysisUnit=sampling["analysisUnit"],
        seed=sampling["seed"],
    )
    check_sampling_contract(ref)
    return ref


def pack_ref_from_manifest(path: Path | str) -> PackRef:
    """Read `pack.json` into a `PackRef`.

    **This is a manifest read, not S2's pack loader.** It does no content hashing, no AST import
    walk, no data-file row-count identity check and no provenance check — those are `load_pack` /
    `validate_pack` (plan §3.6a, S2). It exists because S1 ships `compare`, and a comparison cannot
    resolve its analysis unit or its verdict family without the manifest's `sampling` and `metrics`
    blocks. `contentHash` is `None` here — "not loaded", not "empty" (review m-5/P2-5): at S1 the
    authoritative hash for a comparison is the one each run recorded in its own fingerprint, which
    is what the AC-3 banner reads.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return _ref_from_manifest_fields(manifest, content_hash=None, label=str(path))


# --------------------------------------------------------------------------------------------
# S2: the real pack loader (plan §3.3, §3.4.4a, §4 S2)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pack:
    """A loaded pack — `pack.json` plus its directory, with its bytes hashed (§3.3).

    `contentHash` is **total** here, unlike `PackRef.contentHash: str | None`: a loaded pack
    always knows its bytes, because `load_pack` cannot construct one without reading every file in
    the pack directory to compute `content_hash(root)`. `Pack.ref()` is what carries that totality
    across into a `PackRef` — the §3.3 boundary this module's tests assert rather than assume.
    """

    packId: str
    packVersion: str
    role: str
    contentHash: str
    manifest: Mapping[str, Any]
    root: Path

    def data_path(self, key: str) -> Path:
        """Resolve a `data.<key>` manifest entry (§3.3, e.g. `"conversations"`) to a real path."""
        data = self.manifest.get("data") or {}
        if key not in data:
            raise PackConfigError(f"{self.packId}: data.{key} is absent from the manifest")
        return self.root / data[key]

    def load_tool_module(self) -> ModuleType:
        """Import `tools.module` (§3.3) from the pack root via `importlib`, not `sys.path`.

        The pack's own `tools.entrypoint` name is left for the caller to `getattr` — this method's
        job is only the import, which is what `validate_pack`'s AST check exists to make safe
        before this ever runs against an untrusted pack.
        """
        tools = self.manifest.get("tools") or {}
        module_rel = tools.get("module")
        if not module_rel:
            raise PackConfigError(f"{self.packId}: tools.module is absent from the manifest")
        module_path = self.root / module_rel
        module_name = f"modelbench_pack_{self.packId.replace('-', '_')}_tools"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise PackConfigError(f"{self.packId}: cannot load tool module at {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def ref(self) -> PackRef:
        """The §3.3 totality boundary: `contentHash` is never `None` on this path."""
        return _ref_from_manifest_fields(
            self.manifest, content_hash=self.contentHash, label=str(self.root)
        )


def _pack_relative_paths(root: Path) -> list[str]:
    """Every file's POSIX-style relative path under `root`, sorted, excluding `PROVENANCE.md` and
    `__pycache__` (a load-tool-module side effect, never pack content — plan §3.3 names only
    `PROVENANCE.md`, but a bytecode cache the loader itself writes cannot be part of a pack's
    identity without making the hash depend on whether some prior process happened to import it)."""
    paths: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "PROVENANCE.md":
            continue
        paths.append(rel)
    return sorted(paths)


def content_hash(root: Path) -> str:
    """SHA-256 over the sorted relative paths and bytes of every pack file but `PROVENANCE.md`.

    Declared versions get forgotten; a hash cannot (§3.3). Each entry contributes its relative
    path and its bytes, both NUL-delimited, so two packs cannot collide by concatenation (a file
    named `"a"` holding `"bc"` hashes differently from `"ab"` holding `"c"`).
    """
    root = Path(root)
    hasher = hashlib.sha256()
    for rel in _pack_relative_paths(root):
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update((root / rel).read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_pack(root: Path) -> Pack:
    """Read `<root>/pack.json` and hash the pack directory into a `Pack` (§3.3, §4 S2).

    This is the loader, not the validator: it raises `PackConfigError` only when the manifest
    cannot even be identified (unreadable JSON, or missing `packId`/`packVersion`/`role`). Every
    deeper contract — `sampling`, `callSurface`, the AST import allowlist — is `validate_pack`'s,
    run separately so a caller can load a pack and still see *every* problem at once rather than
    stopping at the first.
    """
    root = Path(root)
    manifest_path = root / "pack.json"
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PackConfigError(f"{root}: pack.json is absent") from None
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PackConfigError(f"{root}: pack.json is not valid JSON ({exc})") from exc
    missing = [key for key in ("packId", "packVersion", "role") if key not in manifest]
    if missing:
        raise PackConfigError(f"{root}: pack.json is missing {', '.join(missing)}")
    return Pack(
        packId=manifest["packId"],
        packVersion=manifest["packVersion"],
        role=manifest["role"],
        contentHash=content_hash(root),
        manifest=MappingProxyType(dict(manifest)),
        root=root,
    )


# --------------------------------------------------------------------------------------------
# validate_pack: the sampling contract, callSurface, and the AST import allowlist
# --------------------------------------------------------------------------------------------


def derive_call_surface(requires: Any) -> str | None:
    """§3.4.4a's declared discriminator: `lmstudio-chat` → `"chat"`, `lmstudio-embeddings` →
    `"embeddings"`. Returns `None` when the requirement list names neither or both — the exact
    ambiguity `validate_pack` must reject, factored out so the derivation itself is unit-testable
    apart from the message it produces."""
    tokens = requires if isinstance(requires, (list, tuple)) else []
    surfaces = set()
    if "lmstudio-chat" in tokens:
        surfaces.add("chat")
    if "lmstudio-embeddings" in tokens:
        surfaces.add("embeddings")
    if len(surfaces) != 1:
        return None
    return surfaces.pop()


def _call_surface_problems(pack: Pack) -> list[str]:
    environment = pack.manifest.get("environment") or {}
    requires = environment.get("requires") or []
    if derive_call_surface(requires) is not None:
        return []
    return [
        f"{pack.packId}: environment.requires must declare exactly one of "
        f"'lmstudio-chat' / 'lmstudio-embeddings' (plan §3.4.4a); found {list(requires)!r}"
    ]


def _row_count_identity_problems(pack: Pack, sampling: Mapping[str, Any]) -> list[str]:
    """§3.3's data-driven route: `scripts × replicatesPerScript` rows, `scripts` distinct
    `analysisUnit` values, each appearing exactly `replicatesPerScript` times.

    Reads rows from `data.conversations` — the plan's own manifest key for this, not an invented
    one: `docs/plans/small-model-benchmarking.md`'s one `sampling`-bearing manifest literal
    (`"data": {"conversations": "conversations.jsonl", ...}`, §3.3) declares it beside the
    matching `sampling` block, and no other key is named anywhere for "the data file" the rule
    speaks of. **An earlier version of this function read `sampling.dataFile` instead** — a key
    this module invented and no plan-conformant manifest ever carries, which made the whole route
    silently unreachable on every real pack shape (coordinator finding, 2026-09-09, confirmed:
    line 435 already settles the key). Fixed to `data.conversations` so the check fires on the
    manifest shape the plan actually specifies.

    Skips (returns `[]`) when `scripts` / `replicatesPerScript` / `analysisUnit` /
    `data.conversations` is absent: without a rows file naming the analysis unit per row, there is
    nothing this route can check, and the structural route is what a pack of a different shape
    (no `data.conversations` key at all — every item-level pack) gets instead.
    """
    scripts = sampling.get("scripts")
    replicates = sampling.get("replicatesPerScript")
    data = pack.manifest.get("data") or {}
    data_file = data.get("conversations")
    analysis_unit = sampling.get("analysisUnit")
    if (
        not isinstance(scripts, int)
        or isinstance(scripts, bool)
        or not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or not data_file
        or not analysis_unit
    ):
        return []

    rows_path = pack.root / data_file
    try:
        lines = rows_path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return [
            f"{pack.packId}: cannot read data.conversations {data_file!r} for the row-count "
            f"identity ({exc})"
        ]

    values = [row.get(analysis_unit) for row in rows]
    expected_total = scripts * replicates
    problems: list[str] = []
    if len(values) != expected_total:
        problems.append(
            f"{pack.packId}: data.conversations {data_file!r} holds {len(values)} rows, expected "
            f"scripts × replicatesPerScript = {scripts} × {replicates} = {expected_total} "
            "(row-count identity, plan §3.3)"
        )

    counts = Counter(values)
    if len(counts) != scripts:
        problems.append(
            f"{pack.packId}: sampling.analysisUnit {analysis_unit!r} has {len(counts)} distinct "
            f"values in {data_file!r} where sampling.scripts={scripts} are required (row-count "
            "identity, plan §3.3)"
        )
    else:
        wrong = {value: n for value, n in counts.items() if n != replicates}
        if wrong:
            problems.append(
                f"{pack.packId}: sampling.analysisUnit {analysis_unit!r} values in {data_file!r} "
                f"do not each appear sampling.replicatesPerScript={replicates} times "
                f"(row-count identity, plan §3.3): {sorted(wrong.items())!r}"
            )
    return problems


def _sampling_problems(pack: Pack) -> list[str]:
    sampling = pack.manifest.get("sampling") or {}
    problems: list[str] = []

    # Structural route (§3.3): reuse `check_sampling_contract` via `Pack.ref()` rather than
    # re-implementing the rule (impl review Pass 1, §4 item 6). `Pack.ref()` also runs
    # `metrics_from_manifest`, so a malformed `metrics` block surfaces here too rather than
    # silently short-circuiting the sampling checks below.
    try:
        pack.ref()
    except PackConfigError as exc:
        problems.append(str(exc))

    # `-ml` §3.4 Rule 6: only the one-level `cluster_bootstrap` exists, so a pack that needs the
    # two-level resample must fail validation rather than silently get the wrong interval.
    replicates = sampling.get("replicatesPerScript")
    if isinstance(replicates, int) and not isinstance(replicates, bool) and replicates > 1:
        problems.append(
            f"{pack.packId}: sampling.replicatesPerScript={replicates} > 1 is rejected; only the "
            "one-level cluster_bootstrap exists (-ml §3.4 Rule 6)"
        )

    problems.extend(_row_count_identity_problems(pack, sampling))
    return problems


def _import_allowed(module_name: str) -> bool:
    if not module_name:
        return False
    root = module_name.split(".", 1)[0]
    if root in _STDLIB_MODULE_NAMES:
        return True
    return (
        module_name == _ALLOWED_PACK_IMPORT_MODULE
        or module_name.startswith(_ALLOWED_PACK_IMPORT_MODULE + ".")
    )


def _tool_import_problems(pack: Pack) -> list[str]:
    """§3.3's AST walk: every pack `.py` file may import stdlib and `modelbench.tooling` only.

    A within-pack relative import (`from . import helper`, `level > 0`) is not "outside the
    allowlist" — it names the pack's own code, not a foreign package — and is left alone.
    """
    problems: list[str] = []
    for py_file in sorted(p for p in pack.root.rglob("*.py") if "__pycache__" not in p.parts):
        rel = py_file.relative_to(pack.root).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            problems.append(f"{pack.packId}: {rel} is not valid Python ({exc})")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _import_allowed(alias.name):
                        problems.append(
                            f"{pack.packId}: {rel} imports {alias.name!r}, which is outside the "
                            f"pack import allowlist (stdlib + {_ALLOWED_PACK_IMPORT_MODULE!r}, "
                            "plan §3.3)"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                module = node.module or ""
                if not _import_allowed(module):
                    problems.append(
                        f"{pack.packId}: {rel} imports from {module!r}, which is outside the "
                        f"pack import allowlist (stdlib + {_ALLOWED_PACK_IMPORT_MODULE!r}, "
                        "plan §3.3)"
                    )
    return problems


def validate_pack(pack: Pack) -> list[str]:
    """§4 S2's pack-integrity checks. `[]` means valid, matching `Fingerprint.validate()`'s shape.

    Three independent axes — a fixture can fail one, several, or none:

    * the `sampling` contract (§3.3): structural (`analysisUnit == pairingKey[0]`, via
      `check_sampling_contract`), the row-count identity, and `-ml` §3.4 Rule 6's
      `replicatesPerScript > 1` rejection;
    * `callSurface`, derived from `environment.requires` and rejected when neither or both of
      `lmstudio-chat` / `lmstudio-embeddings` are declared (§3.4.4a);
    * the AST import allowlist over every `.py` file the pack ships (§3.3).

    **Not here:** the `callSurface`-versus-catalog-`type` cross-check and the tool-calling
    eligibility gate are `run`'s (§3.4.4a, §3.6) — this function has no model catalog to check
    against and no LM Studio connection.
    """
    problems: list[str] = []
    problems.extend(_sampling_problems(pack))
    problems.extend(_call_surface_problems(pack))
    problems.extend(_tool_import_problems(pack))
    return problems
