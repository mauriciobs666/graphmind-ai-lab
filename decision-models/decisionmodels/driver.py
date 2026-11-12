"""The campaign driver CLI, `python -m decisionmodels.driver` (plan
`docs/plans/typed-decision-models4.md` §3.5, plan6 `typed-decision-models6.md` §3.2
"Subcommands").

`init`, `resolve` and `status` (U5.6); `run`, `pause [--now]` and `resume` (U6.3, the runner and
stop switch in `decisionmodels.runner`); `start` (a full campaign's round 1); `install` (U5.7,
`decisionmodels.install`) and `smoke-fixture` (the label smoke's items, `decisionmodels.smoke`);
`finalize` (U5.10, `decisionmodels.finalize`). `next` and `ingest` are library entry points of
`campaign.Campaign` that `run` calls, listed for no session.

- `init --point <role> --phase smoke|pilot|full --root <dir> [--fixture <items.jsonl>]
  [--batch-timeout 45m] [--launch-fail-bound 6h]`
- `resolve --root <dir> --decision <text> [--skip-batch | --repin-instructions |
  --decline-topup | --reopen-topup | --probe-topup | --repin-model generator|labeler <id>]
  [--stakeholder-ref <text>]`
- `finalize --root <dir> --dataset-version <name> [--data-dir <dir>] [--rng-seed <n>]
  [--short]`: writes
  `<data-dir>/<role>/<name>/` (default: the component's `data/`)
- `install --root <dir> [--agents-dir <dir>]`
- `smoke-fixture --out <items.jsonl> [--count 50]`
- `start --root <dir>`: a `full` campaign's round 1 (`Campaign.begin_round(1)`: ⌈1.1 × target⌉
  seeds, queued as generation runs); `run` then spends tokens. A pilot starts with `pilot start`
- `run --root <dir>`: exits 0 when it stops (done, halt, usage stop, pause, a held lock, a
  `launchFailed` launch, a signal)
- `pause --root <dir> [--now]` and `resume --root <dir>`
- `status --root <dir>`

A refusal prints one line to stderr and exits 1. Every command but `run` has then written nothing
(`finalize`'s exceptions: a failed whole-point tranche recheck appends its `recheck-halt`, and a
cell under its floor appends `floor-short`);
`run` has appended nothing, and a refusal after it took the lock (the install record, the user
settings) leaves the reason in `store/refusal` for `status`. Its root and no-campaign refusals
come first and write nothing: `status` refuses a missing campaign the same way, and prints a root
refusal it computes itself.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import finalize, install, runner, smoke
from .campaign import DATA_DIR, PHASES, Campaign, CampaignRefused, parse_duration
from .points import ROLES


def _duration(text: str):
    try:
        return parse_duration(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decisionmodels.driver")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a campaign root and record its parameters")
    init.add_argument("--point", required=True, choices=ROLES)
    init.add_argument("--phase", required=True, choices=PHASES)
    init.add_argument("--root", required=True, type=Path)
    init.add_argument("--fixture", type=Path, help="smoke only: the label batch's items")
    init.add_argument("--batch-timeout", type=_duration, default="45m")
    init.add_argument("--launch-fail-bound", type=_duration, default="6h")

    resolve = commands.add_parser("resolve", help="clear a halt with a recorded decision")
    resolve.add_argument("--root", required=True, type=Path)
    resolve.add_argument("--decision", required=True)
    action = resolve.add_mutually_exclusive_group()
    action.add_argument("--skip-batch", action="store_true")
    action.add_argument("--repin-instructions", action="store_true")
    action.add_argument("--decline-topup", action="store_true",
                        help="a spend or unboundedCell halt: the stakeholder declines the top-up")
    action.add_argument("--reopen-topup", action="store_true",
                        help="no halt open, a decline in force: top-ups are planned again")
    action.add_argument("--probe-topup", action="store_true",
                        help="an unboundedCell halt: draw one 25-seed run of that stratum")
    action.add_argument("--repin-model", nargs=2, metavar=("ROLE", "MODEL"))
    resolve.add_argument("--stakeholder-ref", help="the second failure of finalize's whole-point "
                         "tranche recheck: who decided, and where it is recorded")

    inst = commands.add_parser("install", help="install the agent definitions and guard; record "
                                               "the declared expectations")
    inst.add_argument("--root", required=True, type=Path)
    inst.add_argument("--agents-dir", type=Path, default=install.AGENTS_DIR,
                      help="another directory of the same sources (the smoke's maxTurns copy)")

    fin = commands.add_parser("finalize", help="write the dataset version of a finished campaign")
    fin.add_argument("--root", required=True, type=Path)
    fin.add_argument("--dataset-version", required=True)
    fin.add_argument("--data-dir", type=Path, default=DATA_DIR)
    fin.add_argument("--rng-seed", type=int,
                     help="the split's RNG seed (default: a fresh one the ledger does not hold)")
    fin.add_argument("--short", action="store_true",
                     help="after resolve --decline-topup: finalize with short cells if each meets "
                          "its floor")

    fixture = commands.add_parser("smoke-fixture", help="write the label smoke's items.jsonl")
    fixture.add_argument("--out", required=True, type=Path)
    fixture.add_argument("--count", type=int, default=50)

    start = commands.add_parser("start", help="draw a full campaign's round 1 and queue its runs")
    start.add_argument("--root", required=True, type=Path)

    run = commands.add_parser("run", help="the unattended runner: one session per batch")
    run.add_argument("--root", required=True, type=Path)

    pause = commands.add_parser("pause", help="stop after the batch in flight (store/pause)")
    pause.add_argument("--root", required=True, type=Path)
    pause.add_argument("--now", action="store_true",
                       help="also stop the running runner and its child at once")

    resume = commands.add_parser("resume", help="remove store/pause")
    resume.add_argument("--root", required=True, type=Path)

    status = commands.add_parser("status", help="print the campaign's state")
    status.add_argument("--root", required=True, type=Path)
    return parser


def _smoke_fixture(out: Path, count: int) -> int:
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 1
    if count < 1:
        print("refused: --count must be positive", file=sys.stderr)
        return 1
    out.write_text(smoke.fixture_text(count), "utf-8")
    print(f"wrote {count} items to {out}")
    return 0


def _start(camp: Campaign) -> str:
    """Round 1 of a `full` campaign. Refuses, writing nothing, any other phase, a campaign with no
    `install` record and one whose round 1 is drawn; a crash between the draw and the queued runs
    is finished by `run` (`next` queues the stranded seeds, plan6 §3.8)."""
    with camp.locked():
        state = camp.state()
        if state.init["phase"] != "full":
            raise CampaignRefused(f"`start` is for a full campaign, not {state.init['phase']} "
                                  "(a pilot starts with `pilot start`)")
        if state.expectations is None:
            raise CampaignRefused("no `install` record (run `driver install`)")
        if state.rounds:
            stranded = len(state.unbatched_seeds())
            raise CampaignRefused(
                "round 1 is already drawn" + (f"; {stranded} seeds are not queued yet, "
                                              "`driver run` queues them" if stranded else ""))
        batches = camp.begin_round(1)
        seeds = len(camp.state().rounds[0]["seeds"])
        return f"round 1: {seeds} seeds drawn, {len(batches)} generation runs queued"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke-fixture":
        return _smoke_fixture(args.out, args.count)
    camp = Campaign(args.root)
    try:
        if args.command == "run":
            return runner.run(args.root)
        if args.command in ("pause", "resume"):
            lines = (runner.pause(args.root, now=args.now) if args.command == "pause"
                     else runner.resume(args.root))
            print("\n".join(lines))
        elif args.command == "start":
            print(_start(camp))
        elif args.command == "finalize":
            print("\n".join(finalize.finalize(camp, version=args.dataset_version,
                                              data_dir=args.data_dir, short=args.short,
                                              rng_seed=args.rng_seed)))
        elif args.command == "install":
            print("\n".join(install.install(args.root, agents_dir=args.agents_dir)))
        elif args.command == "init":
            camp.init(point=args.point, phase=args.phase, fixture=args.fixture,
                      batch_timeout=args.batch_timeout,
                      launch_fail_bound=args.launch_fail_bound)
            print(f"initialised {args.point} {args.phase} campaign at {camp.root}")
        elif args.command == "resolve":
            print("\n".join(camp.resolve(
                decision=args.decision, skip_batch=args.skip_batch,
                repin_instructions=args.repin_instructions, decline_topup=args.decline_topup,
                reopen_topup=args.reopen_topup, probe_topup=args.probe_topup,
                repin_model=tuple(args.repin_model) if args.repin_model else None,
                stakeholder_ref=args.stakeholder_ref)))
        else:
            print("\n".join(camp.status()))
    except CampaignRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
