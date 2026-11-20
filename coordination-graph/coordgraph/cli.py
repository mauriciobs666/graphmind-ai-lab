"""The `coordgraph` command line (plan section 3.4). This step ships the offline subcommands:
`lint`, `migrate`, `facts`, `history-coverage`; the store-backed ones land with later steps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .discover import discover
from .history import coverage
from .lint import item_counts, lint_repo
from .migrate import TitleReviewRequired, WiError, analyze, apply_file, dry_run_report, wi_flush, wi_minter
from .parse.backlog import BacklogParseError, parse_backlog
from .scope import load_legacy_paths


def _backlogs(root: Path) -> list[str]:
    return [f.path for f in discover(root) if f.kind == "backlog"]


def cmd_lint(args, root: Path) -> int:
    paths = args.paths or None
    if args.count:
        counts = item_counts(root, paths)
        for path, n in counts.items():
            print(f"{path}\t{n}")
        print(f"total\t{sum(n for n in counts.values() if n > 0)}")
        return 1 if any(n < 0 for n in counts.values()) else 0
    findings = lint_repo(root, paths)
    for f in findings:
        print(f.render())
    errors = sum(f.severity == "error" for f in findings)
    print(f"lint: {errors} error(s), {len(findings) - errors} warning(s)", file=sys.stderr)
    return 1 if errors else 0


def cmd_migrate(args, root: Path) -> int:
    legacy_paths = load_legacy_paths(root)
    if args.apply:
        if len(args.paths) != 1:
            print("migrate --apply takes exactly one PATH (one file per run)", file=sys.stderr)
            return 2
        path = args.paths[0]
        if path not in _backlogs(root):
            print(f"{path} is not a covered backlog file", file=sys.stderr)
            return 2
        try:
            rf = apply_file(root, path, wi_minter(root), flush=lambda: wi_flush(root),
                            titles_reviewed=args.titles_reviewed)
        except TitleReviewRequired as exc:
            print(f"{path}: refusing to mint; {len(exc.flagged)} title(s) need review "
                  "(see `migrate --dry-run`; nothing was allocated):", file=sys.stderr)
            for it in exc.flagged:
                print(f"  line {it.line}: {it.title!r}: {'; '.join(it.issues)}", file=sys.stderr)
            print("fix the source text, or pass --titles-reviewed once the titles are acknowledged",
                  file=sys.stderr)
            return 2
        except WiError as exc:
            print(f"{path}: not written: {exc}", file=sys.stderr)
            return 1
        print(f"{path}: minted {len(rf.new_items)} id(s)")
        return 0
    paths = args.paths or _backlogs(root)
    for path in paths:
        if not (root / path).is_file():
            print(f"{path}: no such file", file=sys.stderr)
            return 2
        rf = analyze(path, (root / path).read_text(encoding="utf-8"))
        print(dry_run_report(rf, legacy_paths))
    return 0


def cmd_facts(args, root: Path) -> int:
    legacy_paths = load_legacy_paths(root)
    rows = []
    for path in _backlogs(root):
        try:
            doc = parse_backlog(path, (root / path).read_bytes(), legacy_paths)
        except BacklogParseError:
            continue
        for item in doc.items:
            for entry in item.touches or []:
                if not entry.startswith("fact:"):
                    continue
                body = entry[5:].strip()
                mode = ""
                if body.endswith(")") and "(" in body:
                    body, _, mode = body.rpartition("(")
                    mode = mode.rstrip(")").strip()
                token = body.strip().casefold()
                if args.like is None or args.like.casefold() in token:
                    rows.append((token, mode, item.ref, path))
    for row in sorted(set(rows)):
        print("\t".join(row))
    return 0


def cmd_history_coverage(args, root: Path) -> int:
    legacy_paths = load_legacy_paths(root)
    rows = coverage(root, _backlogs(root), legacy_paths)
    tm = tu = 0
    for r in rows:
        total = len(r.matched) + len(r.unmatched)
        rate = f"{100 * len(r.matched) / total:.0f}%" if total else "n/a"
        print(f"{r.path}\tmentioned {r.mentioned}\topen {r.still_open}\t"
              f"matched {len(r.matched)}\tunmatched {len(r.unmatched)}\t{rate}")
        if args.verbose:
            for tok in r.unmatched:
                print(f"    unmatched {tok}")
        tm += len(r.matched)
        tu += len(r.unmatched)
    pct = f"{100 * tm / (tm + tu):.0f}%" if tm + tu else "n/a"
    print(f"total\tmatched {tm}\tunmatched {tu}\t{pct}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coordgraph")
    sub = p.add_subparsers(dest="cmd", required=True)
    lint = sub.add_parser("lint", help="check backlog files")
    lint.add_argument("--count", action="store_true", help="print items per backlog file instead")
    lint.add_argument("paths", nargs="*")
    mig = sub.add_parser("migrate", help="one-time backlog layout reformat")
    mode = mig.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report and diff; allocates nothing")
    mode.add_argument("--apply", action="store_true", help="allocate IDs via `wi new` and rewrite ONE file")
    mig.add_argument("--titles-reviewed", action="store_true",
                     help="with --apply: acknowledge the titles `--dry-run` flagged (IDs are permanent)")
    mig.add_argument("paths", nargs="*")
    facts = sub.add_parser("facts", help="list fact tokens used in `Touches`")
    facts.add_argument("--like")
    hc = sub.add_parser("history-coverage", help="delivery-heading match rate over every history file")
    hc.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Config.from_env().repo_root
    handler = {"lint": cmd_lint, "migrate": cmd_migrate, "facts": cmd_facts,
               "history-coverage": cmd_history_coverage}[args.cmd]
    return handler(args, root)
