"""The command surface. Plan §3.6a is the table this implements.

S1 shipped `compare` (including `--negative-control`), `index rebuild`, and the stored-records
half of `models --tested`. S2 adds `attest`, `validate`, and `run` — the runner-spec's own three
steps (`docs/plans/small-model-benchmarking-runner-spec.md` §7), of which this is the last:
`validate` and `run` wrap already-shipped machinery (`packs.validate_pack`, `runner.run_pack`) and
reimplement no check of their own. This closes S2.

**Exit codes are a closed set** (§3.6a, amended by the dispatch-failure note's §4(d)): `0`
whenever the tool ran and reported, *whatever the scores* — the requirements rule out pass/fail
gating, so a comparison that finds every stored record invalid still exits `0` and prints the
exclusion block. Non-zero is operational only: `2` bad arguments · `3` LM Studio unreachable ·
`4` invalid pack (`validate` failure, pack load error, the `callSurface`-versus-catalog-`type`
contradiction, the tool-calling eligibility gate — or, **after `run`'s artifacts are already
written**, a `tool-caller` conversation censored by a dispatch raise: `results/runs/<runId>.json`
and the `PACK DISPATCH FAILURES` disclosure land on disk *before* the process reports `4`, never
an aborted run, dispatch-failure note §4(d)) · `5` fingerprint incomplete, or `host.json`
absent/invalid/stale.

`validate --strict` is accepted but deliberately unimplemented (`_cmd_validate` raises
`NotImplementedError`): the runner-spec (§9) states plainly that `--strict`'s semantics are never
given anywhere in the plan, and names a `NotImplementedError` + citing `TODO` as the sanctioned way
to leave a genuinely open question open — an accepted flag that silently no-ops is explicitly ruled
out as a third option. Resolve or replace this per
`docs/plans/small-model-benchmarking-runner-spec.md` §9 once the plan states a ruling.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from modelbench import hostinfo
from modelbench.lmstudio import LMStudio
from modelbench.packs import PackConfigError, load_pack, pack_ref_from_manifest, validate_pack
from modelbench.report import compare_report
from modelbench.results import (
    RunResult,
    load_history,
    models_with_stored_results,
    rebuild_index,
    store,
)
from modelbench.runner import RunConfig, RunRefused, run_pack


class UnknownModelKey(ValueError):
    """`--models` named a key with no stored run for this pack (§3.6a's exit 2)."""


class AttestUsageError(ValueError):
    """A usage problem with `attest`'s inputs (§3.6a's exit 2) — a malformed `--set key=value`
    pair, an unrecognized key, a value that fails the field's own type (`hostRamGb` not an
    integer), or stdin running out while prompting for a field `--set` did not supply (review
    Pass 13, P13-6 — non-interactive `--set` is the sanctioned route, so this is normal usage,
    not abuse)."""


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_LMSTUDIO_UNREACHABLE = 3
EXIT_BAD_PACK = 4
EXIT_FINGERPRINT = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-bench",
        description=(
            "Measure one local model at a time against one versioned task pack. "
            "No CI hook, no pass/fail gate, no leaderboard."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_root(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--root",
            default=".",
            help="the model-bench directory holding packs/, results/ and reports/",
        )
        return p

    compare = with_root(sub.add_parser("compare", help="render a comparison for one pack"))
    compare.add_argument("--pack", required=True)
    compare.add_argument("--models", help="comma-separated model keys, in arm order")
    compare.add_argument("--session")
    compare.add_argument(
        "--negative-control",
        action="store_true",
        help="compare one stored run against a copy of itself (a smoke check: b=c=0 by "
        "construction, so it proves the mode is wired, not that the harness is sound)",
    )
    compare.add_argument("--out")

    index = with_root(sub.add_parser("index", help="the derived results/index.csv"))
    index.add_argument("action", choices=["rebuild"])

    models = with_root(sub.add_parser("models", help="models with stored results (FR-17a)"))
    models.add_argument("--tested", action="store_true", required=True)
    models.add_argument("--pack")
    models.add_argument("--role")

    attest = with_root(sub.add_parser("attest", help="write host.json, the operator-attested "
                                       "fingerprint half"))
    attest.add_argument("--api-base-url", default="http://localhost:1234")
    attest.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        dest="set_",
        help=f"one of {', '.join(hostinfo.ATTESTED_FIELD_NAMES)}, repeatable; unset fields are "
        "prompted for interactively",
    )

    validate = sub.add_parser(
        "validate", help="structural pack integrity — no LM Studio connection, no model catalog"
    )
    validate.add_argument("--pack", required=True, help="path to the pack directory")
    validate.add_argument(
        "--strict",
        action="store_true",
        help="not yet implemented (runner-spec §9) — raises NotImplementedError rather than "
        "silently no-op",
    )

    run = with_root(
        sub.add_parser("run", help="one model x one pack; validates first and fails closed")
    )
    run.add_argument("--pack", required=True, help="pack id under packs/ (plan §3.3 convention)")
    run.add_argument("--model", required=True)
    run.add_argument("--session")
    run.add_argument("--reference")
    run.add_argument("--warmup", type=int, help="extra untimed warm-up calls, additive")
    run.add_argument("--first-call-timeout", type=float)
    run.add_argument("--request-timeout", type=float)

    return parser


def _report_path(root: Path, pack_id: str) -> Path:
    """`reports/<pack-id>-<date>-<n>.md`, `<n>` a two-digit same-day sequence (plan §3.5).

    A same-day re-run is the normal case while a pack is being developed, and silently overwriting
    the earlier comparison is the one behaviour a tool built around durable history must not have.
    """
    directory = root / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    for n in range(1, 100):
        candidate = directory / f"{pack_id}-{stamp}-{n:02d}.md"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"more than 99 comparisons for {pack_id} on {stamp}")


def _select_arms(
    runs: Sequence[RunResult], *, models: str | None, session: str | None, negative_control: bool
) -> list[RunResult]:
    candidates = list(runs)
    if session is not None:
        candidates = [r for r in candidates if r.sessionId == session]
    if models:
        wanted = [m.strip() for m in models.split(",") if m.strip()]
        by_key = {r.modelKey: r for r in candidates}
        # A key with no stored run was silently dropped, which is how a user reached a one-arm
        # comparison — `--models cand,incumbnet` rendered a report asserting a reason that is
        # untrue of it. A typo in a model key is a usage error, not a comparison (review M-6).
        missing = [m for m in wanted if m not in by_key]
        if missing:
            raise UnknownModelKey(
                f"no stored run for {', '.join(repr(m) for m in missing)} in this pack "
                "(and session, if --session was given); "
                "`model-bench models --tested --pack <id>` lists what is stored"
            )
        candidates = [by_key[m] for m in wanted]
    if negative_control and candidates:
        # Two copies of ONE record, deliberately. The mode's own `--help` text and the report's
        # first banner both say why this cannot fail (`-ml` §9) — the report said nothing at all
        # until review P3-4, and this comment claimed otherwise while it did.
        return [candidates[0], candidates[0]]
    return candidates


def _cmd_compare(args: argparse.Namespace) -> int:
    root = Path(args.root)
    manifest = root / "packs" / args.pack / "pack.json"
    if not manifest.is_file():
        print(f"model-bench: no pack manifest at {manifest}", file=sys.stderr)
        return EXIT_BAD_PACK
    try:
        pack = pack_ref_from_manifest(manifest)
    except (PackConfigError, KeyError, ValueError) as exc:
        print(f"model-bench: invalid pack {args.pack}: {exc}", file=sys.stderr)
        return EXIT_BAD_PACK

    # The manifest's `packId`, not the pack **directory** name: they coincide by the §3.3
    # convention (`packs/<pack-id>/`) and nothing enforces it, so this was latent (review m-6).
    valid, invalid = load_history(root, packId=pack.packId)
    try:
        arms = _select_arms(
            valid,
            models=args.models,
            session=args.session,
            negative_control=args.negative_control,
        )
    except UnknownModelKey as exc:
        print(f"model-bench: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        markdown = compare_report(
            arms, pack=pack, invalid=invalid, negative_control=args.negative_control
        )
    except PackConfigError as exc:
        print(f"model-bench: invalid pack {args.pack}: {exc}", file=sys.stderr)
        return EXIT_BAD_PACK

    # The manifest's `packId`, not the pack **directory** name — the same distinction the
    # `load_history` call above draws, and the half Pass 1's m-6 left behind: `_report_path`'s
    # parameter is `pack_id` and its docstring promises `reports/<pack-id>-…` (review P3-14).
    target = Path(args.out) if args.out else _report_path(root, pack.packId)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {target}")
    return EXIT_OK


def _cmd_index(args: argparse.Namespace) -> int:
    print(rebuild_index(Path(args.root)))
    return EXIT_OK


def _cmd_models(args: argparse.Namespace) -> int:
    for key in models_with_stored_results(Path(args.root), packId=args.pack, role=args.role):
        print(key)
    return EXIT_OK


def _parse_set_flags(pairs: Sequence[str]) -> dict[str, str]:
    """`--set key=value`, repeated. Raises `AttestUsageError` on a malformed pair or an
    unrecognized key — `attested`'s four names are closed, not a place for a typo to silently add
    a fifth field `host.json`'s schema does not expect."""
    out: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise AttestUsageError(f"--set expects key=value, got {raw!r}")
        key, _, value = raw.partition("=")
        if key not in hostinfo.ATTESTED_FIELD_NAMES:
            raise AttestUsageError(
                f"--set {key!r} is not one of {', '.join(hostinfo.ATTESTED_FIELD_NAMES)}"
            )
        out[key] = value
    return out


def _coerce_attested_value(name: str, raw: str) -> Any:
    """`hostRamGb` is a number and `otherResidentWorkloads` a list (plan §3.4.4's schema); the
    other two are free-form strings, taken verbatim."""
    if name == "hostRamGb":
        try:
            return int(raw)
        except ValueError as exc:
            raise AttestUsageError(f"--set hostRamGb: {raw!r} is not an integer") from exc
    if name == "otherResidentWorkloads":
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def _gather_attested_fields(args: argparse.Namespace) -> dict[str, Any]:
    """The four operator-attested fields: from `--set`, or prompted for interactively when a
    field is not named there (§3.6a: "Prompts for the four operator-attested fields, ...
    non-interactive `--set k=v`").

    Raises `AttestUsageError` — never `EOFError` — when stdin has nothing left to give a prompt
    (review Pass 13, P13-6): `--set` is §3.6a's own non-interactive route, so a partly-specified
    non-interactive invocation is normal usage, not abuse, and every field still unset is named in
    one message rather than failing prompt by prompt.
    """
    set_values = _parse_set_flags(args.set_)
    fields: dict[str, Any] = {}
    unset: list[str] = []
    for name in hostinfo.ATTESTED_FIELD_NAMES:
        if name in set_values:
            raw = set_values[name]
        else:
            try:
                raw = input(f"{name}: ")
            except EOFError:
                unset.append(name)
                continue
        fields[name] = _coerce_attested_value(name, raw)
    if unset:
        raise AttestUsageError(
            f"no more input to prompt with, and {', '.join(unset)} "
            f"{'was' if len(unset) == 1 else 'were'} never given via --set"
        )
    return fields


def _cmd_attest(args: argparse.Namespace) -> int:
    root = Path(args.root)
    try:
        attested = _gather_attested_fields(args)
    except AttestUsageError as exc:
        print(f"model-bench: {exc}", file=sys.stderr)
        return EXIT_USAGE

    client = LMStudio(args.api_base_url)
    try:
        path = hostinfo.attest(
            root, api_base_url=args.api_base_url, attested=attested, client=client
        )
    except hostinfo.AttestProbeFailed as exc:
        print(f"model-bench: {exc}", file=sys.stderr)
        return EXIT_LMSTUDIO_UNREACHABLE
    except hostinfo.HostInfoError as exc:
        # `attest()`'s own defensive re-validation (e.g. an empty `--api-base-url`) — a usage
        # problem with the arguments given, not an LM Studio reachability failure (review Pass 13,
        # P13-5: this previously escaped as an uncaught traceback, exit 1).
        print(f"model-bench: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(f"wrote {path}")
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    """plan §3.6a's `validate` row: structural only — no LM Studio connection, no model catalog
    (`validate_pack`'s own docstring already states this scope, `packs.py` `:735-737`).

    `--strict` is named in §3.6a's table with no elaboration anywhere in the plan (runner-spec §9:
    "a genuine gap, not merely scattered"). Silently accepting it as a no-op is the one option the
    spec rules out, so it is a deliberate, documented deferral instead: raise, don't pretend.
    """
    if args.strict:
        raise NotImplementedError(
            "validate --strict: semantics are not decided anywhere in the plan (runner-spec §9, "
            "docs/plans/small-model-benchmarking-runner-spec.md §6.1/§9) — deliberately deferred "
            "rather than accepted as a silent no-op"
        )
    try:
        pack = load_pack(Path(args.pack))
    except (OSError, PackConfigError) as exc:
        print(f"model-bench: cannot load pack at {args.pack}: {exc}", file=sys.stderr)
        return EXIT_BAD_PACK
    problems = validate_pack(pack)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        return EXIT_BAD_PACK
    print(f"{pack.packId} {pack.packVersion} ({pack.role}): valid")
    return EXIT_OK


def _pack_root(root: Path, pack_id: str) -> Path:
    """`packs/<pack-id>/` under `--root` (§3.3's `packs/<pack-id>/` convention) — `run`'s own
    resolution of a bare `--pack <id>`, the counterpart to `_cmd_compare`'s inline
    `root / "packs" / args.pack / "pack.json"` lookup one level up (manifest read vs. full load)."""
    return root / "packs" / pack_id


def _cmd_run(args: argparse.Namespace) -> int:
    """plan §3.6a's `run` row (`:1918`): "calls validate first and fails closed" (spec §6.2).

    Every deeper refusal — LM Studio unreachable, the `callSurface`/catalog-`type` cross-check, the
    tool-calling eligibility gate, a stale attestation — is `run_pack`'s own capture-order logic
    (`runner.py`, already covered by its own offline suite); this wraps it: load the pack, validate
    it, build the `LMStudio` client from `host.json`'s own `apiBaseUrl` (`run` takes no
    `--api-base-url` of its own — §3.6a's flag list has none), and map `RunRefused` plus the
    dispatch-failure funnel onto §3.6a's exit codes.
    """
    root = Path(args.root)
    try:
        pack = load_pack(_pack_root(root, args.pack))
    except (OSError, PackConfigError) as exc:
        print(f"model-bench: cannot load pack {args.pack}: {exc}", file=sys.stderr)
        return EXIT_BAD_PACK

    problems = validate_pack(pack)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        return EXIT_BAD_PACK

    try:
        host = hostinfo.read_host_info(root)
    except hostinfo.HostInfoError as exc:
        print(f"model-bench: {exc}", file=sys.stderr)
        return EXIT_FINGERPRINT

    lmstudio = LMStudio(host["apiBaseUrl"])
    cfg = RunConfig(
        modelKey=args.model,
        sessionId=args.session,
        referenceKey=args.reference,
        warmupExtra=args.warmup or 0,
        firstCallTimeoutSeconds=args.first_call_timeout or 300.0,
        requestTimeoutSeconds=args.request_timeout or 120.0,
    )
    try:
        run, disclosures = run_pack(pack, cfg, lmstudio=lmstudio, root=root)
    except RunRefused as exc:
        print(f"model-bench: {exc}", file=sys.stderr)
        return exc.exitCode

    path = store(run, root)  # raises InvalidFingerprint on a runner defect — no bypass anywhere
                              # (plan §3.4.5 point 1); deliberately uncaught here.
    print(f"stored: {path}")

    if disclosures:
        print("PACK DISPATCH FAILURES")  # dispatch-failure note §4(d)'s funnel-head block
        for d in disclosures:
            print(f"  {d.scriptId} turn {d.turn}: {d.tool} — {d.reason}")
        return EXIT_BAD_PACK  # exit 4, AFTER store() has already written the record
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK

    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "attest":
        return _cmd_attest(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "run":
        return _cmd_run(args)
    return EXIT_USAGE  # pragma: no cover - argparse rejects unknown commands first
