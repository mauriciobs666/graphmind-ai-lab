"""``wi`` command-line dispatcher.  Later steps add resolve/move/legacy/... here."""
import argparse
import json
import sys
from pathlib import Path

from . import alloc, legacy, move, project, resolve, rules
from .store import Store


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
    rs = sub.add_parser("resolve", help="map any name (ID, alias, slug, FR/AC citation) to the item")
    rs.add_argument("name")
    rs.add_argument("--json", action="store_true")
    mv = sub.add_parser("move", help="rename an item and its subtree under a new parent")
    mv.add_argument("id")
    mv.add_argument("newparent", help="parent ID, or 'none' for the project")
    pr = sub.add_parser("promote", help="turn a kaizen item into a feature")
    pr.add_argument("id")
    pr.add_argument("--to", required=True, choices=["F"])
    pr.add_argument("--parent", help="initiative to place the feature under (default: none)")
    ml = sub.add_parser("mint-legacy", help="give an old <home>:<name> item a scheme ID (idempotent)")
    ml.add_argument("spec", help="<home>:<name>, e.g. architect:K-004")
    ml.add_argument("--type", required=True, choices=["K", "F", "D"])
    ml.add_argument("--parent")
    ml.add_argument("--title")
    sub.add_parser("flush", help="copy journal events into the tracked registry")
    i = sub.add_parser("init", help="create .work-items/ in the current directory")
    i.add_argument("--key", required=True)
    i.add_argument("--components", nargs="*", default=[], help="space- or comma-separated")
    return p


def main(argv=None, cwd=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "init":
            comps = [c for part in args.components for c in part.split(",") if c]
            project.init(Path(cwd or Path.cwd()).resolve(), args.key, comps)
            return 0
        root = find_root(cwd or Path.cwd())
        if args.cmd == "new":
            id_ = alloc.new(root, args.type, parent=args.parent, slug=args.slug,
                            title=args.title, home=args.home)
            print(id_)
            if args.sync:
                sync_headers(root, id_)
        elif args.cmd == "retire":
            print(alloc.retire(root, args.id))
        elif args.cmd == "mint-legacy":
            print(legacy.mint(root, args.spec, args.type, parent=args.parent, title=args.title))
        elif args.cmd == "resolve":
            legacy.install(project.load_namespaces(root))
            rec = resolve.resolve(Store(root, alloc.load_key(root)).read_state(), args.name)
            print(json.dumps(rec, indent=2) if args.json else resolve.describe(rec))
        elif args.cmd == "move":
            print(move.move(root, args.id, args.newparent))
        elif args.cmd == "promote":
            print(move.promote(root, args.id, args.to, parent=args.parent))
        elif args.cmd == "flush":
            Store(root, alloc.load_key(root)).flush()
    except resolve.ResolveError as exc:
        print(f"wi: {exc}", file=sys.stderr)
        return exc.code
    except (rules.IllegalParent, rules.MalformedId, alloc.AllocError, project.InitError) as exc:
        print(f"wi: {exc}", file=sys.stderr)
        return 1
    return 0
