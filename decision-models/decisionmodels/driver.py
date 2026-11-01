"""The campaign driver CLI, `python -m decisionmodels.driver` (plan
`docs/plans/typed-decision-models4.md` §3.5, plan6 `typed-decision-models6.md` §3.2
"Subcommands").

`init`, `resolve` and `status` (U5.6); `run`, `pause [--now]` and `resume` (U6.3, the runner and
stop switch in `decisionmodels.runner`). `install` (U5.7) and `finalize` (U5.10) join later;
`next` and `ingest` are library entry points of `campaign.Campaign` that `run` calls, listed for
no session.

- `init --point <role> --phase smoke|pilot|full --root <dir> [--fixture <items.jsonl>]
  [--batch-timeout 45m] [--launch-fail-bound 6h]`
- `resolve --root <dir> --decision <text> [--skip-batch | --repin-instructions |
  --repin-model generator|labeler <id>]`
- `run --root <dir>`: exits 0 when it stops (done, halt, usage stop, pause, a held lock, a
  `launchFailed` launch, a signal)
- `pause --root <dir> [--now]` and `resume --root <dir>`
- `status --root <dir>`

A refusal prints one line to stderr and exits 1. Every command but `run` has then written nothing;
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

from . import runner
from .campaign import PHASES, Campaign, CampaignRefused, parse_duration
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
    action.add_argument("--repin-model", nargs=2, metavar=("ROLE", "MODEL"))

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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    camp = Campaign(args.root)
    try:
        if args.command == "run":
            return runner.run(args.root)
        if args.command in ("pause", "resume"):
            lines = (runner.pause(args.root, now=args.now) if args.command == "pause"
                     else runner.resume(args.root))
            print("\n".join(lines))
        elif args.command == "init":
            camp.init(point=args.point, phase=args.phase, fixture=args.fixture,
                      batch_timeout=args.batch_timeout,
                      launch_fail_bound=args.launch_fail_bound)
            print(f"initialised {args.point} {args.phase} campaign at {camp.root}")
        elif args.command == "resolve":
            camp.resolve(decision=args.decision, skip_batch=args.skip_batch,
                         repin_instructions=args.repin_instructions,
                         repin_model=tuple(args.repin_model) if args.repin_model else None)
            print("resolved; the next `driver run` resumes")
        else:
            print("\n".join(camp.status()))
    except CampaignRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
