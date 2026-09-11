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
* **`validate_pack`** — `[]` means valid, matching `Fingerprint.validate()`'s shape. Four
  independent axes, per §4 S2's done-condition: the `sampling` contract (structural, reusing
  `check_sampling_contract` rather than re-implementing it, plus the row-count identity and `-ml`
  §3.4 Rule 6's `replicatesPerScript > 1` rejection), `callSurface` derived from
  `environment.requires` (§3.4.4a, rejecting a pack declaring neither or both of `lmstudio-chat` /
  `lmstudio-embeddings`), `tools.module`'s own path (must resolve inside the pack root and end in
  `.py` — impl review Pass 12 P12-2/P12-3(i), the AST allowlist's own reach cannot exceed what it
  can see), and the AST import allowlist over every `.py` file the pack ships.

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
from typing import Any, Iterator, Mapping, NamedTuple

from modelbench.convo import _HISTORY_REPLAY_MODES, Conversation, PromptConfig, Turn
from modelbench.roles import MULTI_CALL_TURN_BY_ROLE, analysis_unit_field
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
    """§3.3's structural routes: the analysis unit is `pairingKey[0]`, outermost → innermost (i),
    and that outermost component is the *role's own* canonical analysis-unit field (iii, v1.25).

    Routes (i) and (ii) — the row-count identity, `_row_count_identity_problems`'s, not this
    function's — are both satisfied by any *self-consistent* naming: a pack can declare
    `analysisUnit == pairingKey[0]` and a matching row count while still naming the wrong field
    for its role (`P12-7` shipped exactly this as a positive control before `U76`'s fix). Route
    (iii) is the only one that catches that — `pairingKey[0]` must equal
    `roles.analysis_unit_field(ref.role)`, never merely *some* self-consistent name (plan §3.3).
    """
    if not ref.pairingKey:
        raise PackConfigError("sampling.pairingKey is empty")
    if ref.analysisUnit != ref.pairingKey[0]:
        raise PackConfigError(
            f"sampling.analysisUnit {ref.analysisUnit!r} is not pairingKey[0] "
            f"{ref.pairingKey[0]!r}; the analysis unit is the outermost component of the "
            "pairing key, by rule (plan §3.3)"
        )
    expected = analysis_unit_field(ref.role)
    if ref.pairingKey[0] != expected:
        raise PackConfigError(
            f"pairingKey[0] {ref.pairingKey[0]!r} is not role {ref.role!r}'s own analysis-unit "
            f"field {expected!r} (plan §3.3, route (iii), v1.25)"
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

        The pack's own `tools.entrypoint` name is left for the caller to `getattr`. `validate_pack`
        constrains this load two ways before it is safe to call: `_tool_module_problems` refuses a
        `tools.module` that resolves outside the pack root or does not end in `.py` (impl review
        P12-2 / P12-3(i)), and `_tool_import_problems`'s AST walk is a **coupling** rule over each
        file's own `import` statements — not a sandbox. `__import__(...)` and
        `importlib.import_module(...)` inside a pack module reach any importable name the walk
        never sees (impl review P12-3(ii), executed and confirmed), so a `validate_pack`-clean
        pack is one whose declared imports are in scope, never one whose code is safe to run
        untrusted.
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

    def iter_items(self) -> Iterator[Mapping[str, Any]]:
        """Yield each row of `data.items` (`items.jsonl`), parsed, in file order — the generic
        item-level pack data source (§3.8.1-3.8.3's `items.jsonl` copies). Pure data iteration; a
        role-specific typed parse is each role's own scorer's job, not this loader's (runner spec
        §9's own flag: this method is named for readability, not specified by the plan, and has no
        cross-cutting design stakes)."""
        with self.data_path("items").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def iter_scripts(self) -> Iterator[Conversation]:
        """Yield each row of `data.conversations` (`conversations.jsonl`) as a `convo.Conversation`,
        in file order — `tool-caller`'s own data source (§3.8.4). Same readability-only naming
        caveat as `iter_items` above."""
        with self.data_path("conversations").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Conversation(
                    scriptId=row["scriptId"],
                    shape=row["shape"],
                    replicate=row["replicate"],
                    turns=tuple(
                        Turn(seq=t["seq"], user=t["user"], expect=t.get("expect", {}))
                        for t in row["turns"]
                    ),
                    description=row.get("description", ""),
                    provenance=row.get("provenance"),
                )

    def find_script(self, script_id: str) -> Conversation:
        """One named script from `data.conversations`, by `scriptId` — §3.8.4's determinism-probe
        scripts (`sampling.determinismProbeScripts`). Raises `PackConfigError` if absent."""
        for script in self.iter_scripts():
            if script.scriptId == script_id:
                return script
        raise PackConfigError(f"{self.packId}: no script {script_id!r} in data.conversations")

    def ref(self) -> PackRef:
        """The §3.3 totality boundary: `contentHash` is never `None` on this path."""
        return _ref_from_manifest_fields(
            self.manifest, content_hash=self.contentHash, label=str(self.root)
        )

    def prompt_config(self) -> PromptConfig:
        """The pack's `prompt` manifest block (§3.3), parsed into a `PromptConfig` — the
        accessor `validate_pack` calls to catch a bad `prompt` block at validate time rather than
        at `drive`'s first call (impl review Pass 17, P17-5), and what a future caller (the
        runner unit, not yet built) reaches for instead of re-parsing the manifest itself.

        Raises `PackConfigError` when `historyReplay` is outside `convo._HISTORY_REPLAY_MODES` on
        a role that replays turns, or when `maxIterationsPerTurn` violates its role-scoping rule —
        required *iff* `roles.MULTI_CALL_TURN_BY_ROLE[role]`, forbidden otherwise (§3.3, v1.26).
        Every other `prompt` key is carried through as declared, unchecked: this route closes
        exactly the gap P17-5 named, not a general schema check nobody asked for.

        **`historyReplay` is checked under the same `MULTI_CALL_TURN_BY_ROLE` role-scoping as
        `maxIterationsPerTurn`, not unconditionally** (found live: a `validate`-clean, prompt-less
        item-level pack crashed `runner.run_pack` with an uncaught `PackConfigError`, since
        `prompt = self.manifest.get("prompt") or {}` defaults an absent `prompt` block to `{}`
        and this check then ran regardless of role). `historyReplay` is consumed nowhere on an
        item-level role's path: its one reader, `convo.assemble` (via `convo.drive`), is reached
        only from `runner._drive_conversations`, itself `tool-caller`-only — the four item-level
        roles are single-call by construction (`MULTI_CALL_TURN_BY_ROLE[role] is False`) and never
        replay a turn. The plan states role-scoping for `maxIterationsPerTurn` by name (v1.26) but
        not for `historyReplay`; this reuses that same column rather than inventing a second one,
        because on the pack's own role table the two conditions already coincide exactly — a role
        with turns to replay is, today, precisely the one role whose turn is a multi-call loop.

        **`systemPrompt` and `toolSchemas` are carried through as the manifest's own declared
        values — paths, not resolved content** — because resolving `prompt.*` paths against the
        pack root, the way `Pack.data_path` does for `data.*`, is a separate concern from the one
        this route closes, and `PromptConfig`'s own docstring assigns that resolution to whichever
        caller eventually drives a real turn; no such caller exists in this tree yet.
        """
        prompt = self.manifest.get("prompt") or {}
        multi_call = MULTI_CALL_TURN_BY_ROLE.get(self.role, False)

        history_replay = prompt.get("historyReplay")
        if multi_call and history_replay not in _HISTORY_REPLAY_MODES:
            raise PackConfigError(
                f"{self.packId}: prompt.historyReplay {history_replay!r} is not one of "
                f"{sorted(_HISTORY_REPLAY_MODES)!r} (plan §3.3)"
            )

        has_cap = "maxIterationsPerTurn" in prompt
        if multi_call and not has_cap:
            raise PackConfigError(
                f"{self.packId}: prompt.maxIterationsPerTurn is absent; role {self.role!r} runs "
                "a multi-call turn and the field is required (plan §3.3, v1.26)"
            )
        if not multi_call and has_cap:
            raise PackConfigError(
                f"{self.packId}: prompt.maxIterationsPerTurn is present but role {self.role!r} "
                "is not multi-call, where the field is forbidden (plan §3.3, v1.26)"
            )

        return PromptConfig(
            systemPrompt=prompt.get("systemPrompt"),
            toolSchemas=prompt.get("toolSchemas") or (),
            historyReplay=history_replay,
            representToolSchemasEachTurn=prompt.get("representToolSchemasEachTurn"),
            historyTurns=prompt.get("historyTurns"),
            maxIterationsPerTurn=prompt.get("maxIterationsPerTurn"),
            temperature=prompt.get("temperature"),
            maxTokens=prompt.get("maxTokens"),
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


#: The row-count identity's own manifest keys — `_row_count_identity_problems` consults exactly
#: this tuple to decide what to check, and the coverage probe (`tests/test_packs.py`, impl review
#: Pass 12 §4A) walks the same tuple to decide what to probe. One constant, not two lists that
#: happen to agree: a fifth key added here is checked by `_row_count_identity_field`/
#: `_row_count_identity_field_valid` below and probed by the test in the same edit, because both
#: read it from here.
ROW_COUNT_IDENTITY_KEYS: tuple[str, ...] = (
    "sampling.scripts",
    "sampling.replicatesPerScript",
    "sampling.analysisUnit",
    "data.conversations",
)

#: Human-readable "must be" clause per key, used only in the field-problem message.
_ROW_COUNT_IDENTITY_KEY_HINTS: dict[str, str] = {
    "sampling.scripts": "a plain int",
    "sampling.replicatesPerScript": "a plain int",
    "sampling.analysisUnit": "a non-empty string naming the identity's own grouping field",
    "data.conversations": (
        "a non-empty string path — a scripts-declaring pack is conversation-shaped and needs a "
        "rows file to check the identity against"
    ),
}

#: The one case this route is sanctioned to skip silently, each with the one-line reason it is
#: exempt (impl review Pass 12 §4A: "a module-level exemption constant carrying a one-line
#: reason"). `sampling.scripts` genuinely absent from the manifest — never merely invalid — is the
#: plan's own signal that a pack is item-level rather than conversation-shaped (§3.3): the
#: structural route (`check_sampling_contract`) covers that shape instead, and there is nothing
#: left here to check. Every other absence or wrong type on a `scripts`-declaring pack is a
#: reported problem, never a second silent case — round three of the same defect (P12-6) is closed
#: by having exactly **one** true exemption, not a predicate that quietly widens to cover more
#: than this constant names. The coverage probe asserts the *computed* silent-cell set equals this
#: constant exactly, so a stale entry here is as loud as a missing one.
ROW_COUNT_IDENTITY_EXEMPT_CELLS: dict[tuple[str, str], str] = {
    ("sampling.scripts", "absent"): (
        "no sampling.scripts at all is the item-level pack shape (no `scripts` declared, e.g. "
        "the four item-level fixtures); check_sampling_contract is what that shape gets instead, "
        "plan §3.3"
    ),
}


def _row_count_identity_field(
    key: str, sampling: Mapping[str, Any], data: Mapping[str, Any]
) -> Any:
    """Fetch one of `ROW_COUNT_IDENTITY_KEYS`'s dotted manifest paths (`"sampling.X"` /
    `"data.X"`)."""
    block, name = key.split(".", 1)
    source = sampling if block == "sampling" else data
    return source.get(name)


def _row_count_identity_field_valid(key: str, value: Any) -> bool:
    """Whether `value`, fetched for one of `ROW_COUNT_IDENTITY_KEYS`, is the declared type — a
    plain `int` for the two counts, a non-empty `str` for the two names/paths."""
    if key in ("sampling.scripts", "sampling.replicatesPerScript"):
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str) and bool(value)


def _row_count_identity_field_problem(pack: Pack, key: str, value: Any) -> str:
    return (
        f"{pack.packId}: {key} must be {_ROW_COUNT_IDENTITY_KEY_HINTS[key]}, got "
        f"{type(value).__name__} ({value!r}) (row-count identity, plan §3.3)"
    )


def _row_count_identity_problems(pack: Pack, sampling: Mapping[str, Any]) -> list[str]:
    """§3.3's data-driven route: `scripts × replicatesPerScript` rows, `scripts` distinct
    `analysisUnit` values, each appearing exactly `replicatesPerScript` times.

    Reads rows from `data.conversations` — the plan's own manifest key for this, not an invented
    one: `docs/plans/small-model-benchmarking.md`'s one `sampling`-bearing manifest literal
    (`"data": {"conversations": "conversations.jsonl", ...}`, §3.3) declares it beside the
    matching `sampling` block, and no other key is named anywhere for "the data file" the rule
    speaks of. **An earlier version of this function read `sampling.dataFile` instead** — a key
    this module invented and no plan-conformant manifest ever carries, which made the whole route
    silently unreachable on every real pack shape (S2 U73). U73's own fix then introduced a second
    round of the same shape: its `declares_scripts` predicate (`isinstance(scripts, int)`) skipped
    on `"scripts": "12"` just as it skipped on `scripts` truly absent, so a `scripts`-declaring
    pack could still evade the whole route by getting *any one* of its four fields' *type* wrong
    rather than by omitting `scripts` (impl review Pass 12, P12-6 — round three of the same class).

    **Exactly one case is sanctioned to skip silently: `sampling.scripts` genuinely absent from
    the manifest** (`ROW_COUNT_IDENTITY_EXEMPT_CELLS`) — the plan's own item-level-pack signal.
    Every other case — `scripts` present but the wrong type, or any of the other three keys absent
    or the wrong type on a `scripts`-declaring pack — is a reported problem below, generated from
    `ROW_COUNT_IDENTITY_KEYS`/`_row_count_identity_field_valid` rather than hand-checked one at a
    time, which is what let two of these branches (`replicatesPerScript`, then `scripts`'s own
    type) go unnoticed across two fix rounds.
    """
    if "scripts" not in sampling:
        return []

    data = pack.manifest.get("data") or {}
    values = {
        key: _row_count_identity_field(key, sampling, data) for key in ROW_COUNT_IDENTITY_KEYS
    }
    valid = {
        key: _row_count_identity_field_valid(key, values[key]) for key in ROW_COUNT_IDENTITY_KEYS
    }

    problems = [
        _row_count_identity_field_problem(pack, key, values[key])
        for key in ROW_COUNT_IDENTITY_KEYS
        if not valid[key]
    ]
    if not all(valid.values()):
        return problems

    scripts = values["sampling.scripts"]
    replicates = values["sampling.replicatesPerScript"]
    analysis_unit = values["sampling.analysisUnit"]
    data_file = values["data.conversations"]

    rows_path = pack.root / data_file
    try:
        lines = rows_path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return [
            f"{pack.packId}: cannot read data.conversations {data_file!r} for the row-count "
            f"identity ({exc})"
        ]

    values_seen = [row.get(analysis_unit) for row in rows]
    expected_total = scripts * replicates
    if len(values_seen) != expected_total:
        problems.append(
            f"{pack.packId}: data.conversations {data_file!r} holds {len(values_seen)} rows, "
            f"expected scripts × replicatesPerScript = {scripts} × {replicates} = "
            f"{expected_total} (row-count identity, plan §3.3)"
        )

    counts = Counter(values_seen)
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


def _tool_module_problems(pack: Pack) -> list[str]:
    """`tools.module`'s own path, checked before anything asks `load_tool_module` to run it
    (impl review Pass 12, P12-2 / P12-3(i); `_tool_import_problems` below is a different check —
    the *contents* of whatever `.py` files the AST walk finds, not whether `tools.module` itself
    is one of them or lives where the hash can see it).

    Two properties, both executed and confirmed missing before this fix (Appendix L.2/L.3(4)):

    * **containment (P12-2).** `"module": "../outside.py"` validated CLEAN, `load_tool_module`
      executed the outside file, and `content_hash` never moved when that file changed —
      falsifying §3.3's "pack code is part of the content hash, so a behavior change to a
      simulated tool is a version change like any other." The resolved path must stay inside the
      pack root.
    * **suffix (P12-3(i)).** `tools/sim.pyc` as `tools.module` validated CLEAN and the unscanned
      bytecode executed — `_tool_import_problems`'s walk globs `*.py` only, so anything else loads
      unscanned. A file-selection gap the AST check cannot see past, not a syntactic-versus-
      semantic one; closed by requiring the suffix that walk can actually scan.

    Absent `tools.module` is not this function's problem — a pack with no `tools` block (e.g.
    `guard-judge`) is never asked to load one, and `Pack.load_tool_module` already raises if one
    without the key is.
    """
    tools = pack.manifest.get("tools") or {}
    module_rel = tools.get("module")
    if not module_rel:
        return []
    if not isinstance(module_rel, str):
        return [
            f"{pack.packId}: tools.module must be a string path, got "
            f"{type(module_rel).__name__} ({module_rel!r}) (plan §3.3)"
        ]
    if not module_rel.endswith(".py"):
        return [
            f"{pack.packId}: tools.module {module_rel!r} must end in '.py'; the AST import "
            "allowlist only scans '*.py' files, so anything else loads unscanned (plan §3.3, "
            "impl review P12-3(i))"
        ]
    resolved_root = pack.root.resolve()
    resolved_module = (pack.root / module_rel).resolve()
    if not resolved_module.is_relative_to(resolved_root):
        return [
            f"{pack.packId}: tools.module {module_rel!r} resolves outside the pack root; pack "
            "code must live under the pack directory so it is covered by content_hash (plan §3.3, "
            "impl review P12-2)"
        ]
    return []


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


def _prompt_problems(pack: Pack) -> list[str]:
    """`prompt.historyReplay` and `prompt.maxIterationsPerTurn`'s role scoping (§3.3, v1.26;
    impl review P17-5), via `Pack.prompt_config`.

    Absent `prompt` is not this function's problem — same convention as `_tool_module_problems`
    for absent `tools.module`: a pack with no `prompt` block (every fixture but `valid`, today,
    and every non-tool-caller role in the plan's own examples) is never asked to build one.
    """
    if "prompt" not in pack.manifest:
        return []
    try:
        pack.prompt_config()
    except PackConfigError as exc:
        return [str(exc)]
    return []


def _scorer_problems(pack: Pack) -> list[str]:
    """The `"scorer"` key's own resolution — `modelbench.scoring.<name>` — checked at validate
    time instead of one step later at `run`, via `runner._load_item_scorer`'s
    `RunRefused(exitCode=4)` (§3.2/§7.1). Mirrors that function's own `importlib.import_module`
    call exactly, but returns a problem string here rather than raising.

    `pack.role == "tool-caller"` is not this function's problem — `runner.run_pack`'s own role
    branch never reaches `_load_item_scorer` for that role; it calls `_drive_conversations` ->
    `_load_conversation_scorer` instead, which resolves a `ConversationScorer` (a different kind,
    unrelated to `modelbench.scoring.<name>`) and today unconditionally raises
    `NotImplementedError` regardless of what string `"scorer"` names, because S5 hasn't built one
    yet. So a tool-caller pack's `"scorer"` value is not checked by this axis, same "not this
    function's problem" shape as the others below, but scoped by role rather than by key absence.

    Absent `"scorer"` is not this function's problem either — same convention as
    `_tool_module_problems` for absent `tools.module` and `_prompt_problems` for absent `prompt`:
    many fixtures under `tests/fixtures/packs/*/pack.json` have no `"scorer"` key at all (they
    test other axes), and a pack with no `"scorer"` is `_load_item_scorer`'s own problem to raise
    on, not this one's.
    """
    if pack.role == "tool-caller":
        return []
    name = pack.manifest.get("scorer")
    if not name:
        return []
    try:
        importlib.import_module(f"modelbench.scoring.{name}")
    except ImportError as exc:
        return [
            f"{pack.packId}: scorer {name!r} does not resolve to modelbench.scoring.{name} "
            f"({exc})"
        ]
    return []


def validate_pack(pack: Pack) -> list[str]:
    """§4 S2's pack-integrity checks. `[]` means valid, matching `Fingerprint.validate()`'s shape.

    Six independent axes — a fixture can fail one, several, or none:

    * the `sampling` contract (§3.3): structural (`analysisUnit == pairingKey[0]`, via
      `check_sampling_contract`), the row-count identity, and `-ml` §3.4 Rule 6's
      `replicatesPerScript > 1` rejection;
    * `callSurface`, derived from `environment.requires` and rejected when neither or both of
      `lmstudio-chat` / `lmstudio-embeddings` are declared (§3.4.4a);
    * `tools.module`'s own path — must resolve inside the pack root and end in `.py` (impl review
      Pass 12, P12-2 / P12-3(i));
    * the AST import allowlist over every `.py` file the pack ships (§3.3) — a coupling rule over
      each file's own `import` statements, not a sandbox (impl review P12-3(ii); see
      `Pack.load_tool_module`'s docstring);
    * the `prompt` block's `historyReplay` and `maxIterationsPerTurn`'s role scoping, via
      `Pack.prompt_config` — absent `prompt` is not a problem, same convention as absent
      `tools.module` (§3.3, v1.26; impl review Pass 17, P17-5);
    * the `"scorer"` key's own resolution to `modelbench.scoring.<name>` — absent `"scorer"` is
      not a problem, same convention as absent `tools.module` / `prompt` (mirrors
      `runner._load_item_scorer`, moving the failure from `run` time to validate time); scoped to
      the four item-level roles — `tool-caller` resolves a different, unbuilt `ConversationScorer`
      kind via `_load_conversation_scorer` instead, never through this path (`run_pack`'s own role
      branch).

    **Not here:** the `callSurface`-versus-catalog-`type` cross-check and the tool-calling
    eligibility gate are `run`'s (§3.4.4a, §3.6) — this function has no model catalog to check
    against and no LM Studio connection.
    """
    problems: list[str] = []
    problems.extend(_sampling_problems(pack))
    problems.extend(_call_surface_problems(pack))
    problems.extend(_tool_module_problems(pack))
    problems.extend(_tool_import_problems(pack))
    problems.extend(_prompt_problems(pack))
    problems.extend(_scorer_problems(pack))
    return problems
