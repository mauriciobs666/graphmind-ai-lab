"""``wi`` command-line dispatcher.  Later steps add resolve/move/legacy/... here."""
import argparse
import sys
from pathlib import Path

from . import alloc, rules


def find_root(start):
    start = Path(start).resolve()
    for d in (start, *start.parents):
        if (d / ".work-items" / "project.json").exists():
            return d
    raise alloc.AllocError("no .work-items/project.json found in this directory or above")


def sync_headers(root, id_):
    """Header phase (step 6) -- deliberately a stub: writes nothing yet."""
    raise NotImplementedError("--sync: header writing arrives in plan step 6")


def build_parser():
    p = argparse.ArgumentParser(prog="wi")
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="allocate a work item and print its ID")
    n.add_argument("type", choices=["I", "F", "S", "T", "D", "K"])
    n.add_argument("--parent")
    n.add_argument("--slug")
    n.add_argument("--title")
    n.add_argument("--home")
    n.add_argument("--sync", action="store_true", help="then write family headers (step 6)")
    r = sub.add_parser("retire", help="abandon an item; its ID still resolves")
    r.add_argument("id")
    return p


def main(argv=None, cwd=None):
    args = build_parser().parse_args(argv)
    try:
        root = find_root(cwd or Path.cwd())
        if args.cmd == "new":
            id_ = alloc.new(root, args.type, parent=args.parent, slug=args.slug,
                            title=args.title, home=args.home)
            print(id_)
            if args.sync:
                sync_headers(root, id_)
        elif args.cmd == "retire":
            print(alloc.retire(root, args.id))
    except (rules.IllegalParent, rules.MalformedId, alloc.AllocError) as exc:
        print(f"wi: {exc}", file=sys.stderr)
        return 1
    return 0
