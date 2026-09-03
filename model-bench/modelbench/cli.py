"""The command surface. Plan §3.6a is the table this implements.

S1 ships `compare` (including `--negative-control`), `index rebuild`, and the stored-records half
of `models --tested`. `attest`, `validate` and `run` are S2's and are deliberately absent.

**Exit codes are a closed set** (§3.6a): `0` whenever the tool ran and reported, *whatever the
scores* — the requirements rule out pass/fail gating, so a comparison that finds every stored
record invalid still exits `0` and prints the exclusion block. Non-zero is operational only:
`2` bad arguments · `3` LM Studio unreachable (S2) · `4` invalid pack · `5` fingerprint incomplete
or `host.json` stale.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

from modelbench.packs import PackConfigError, pack_ref_from_manifest
from modelbench.report import compare_report
from modelbench.results import RunResult, load_history, models_with_stored_results, rebuild_index


class UnknownModelKey(ValueError):
    """`--models` named a key with no stored run for this pack (§3.6a's exit 2)."""


EXIT_OK = 0
EXIT_USAGE = 2
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
        # Two copies of ONE record, deliberately: the mode's own docstring and the report say why
        # this cannot fail (`-ml` §9).
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
        markdown = compare_report(arms, pack=pack, invalid=invalid)
    except PackConfigError as exc:
        print(f"model-bench: invalid pack {args.pack}: {exc}", file=sys.stderr)
        return EXIT_BAD_PACK

    target = Path(args.out) if args.out else _report_path(root, args.pack)
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
    return EXIT_USAGE  # pragma: no cover - argparse rejects unknown commands first
