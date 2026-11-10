"""``wi`` command-line dispatcher.  Later steps add resolve/move/legacy/... here."""
import argparse
import json
import sys
from pathlib import Path

from . import alloc, headers, legacy, links, move, project, resolve, rules
from .store import Store


def find_root(start):
    start = Path(start).resolve()
    for d in (start, *start.parents):
        if (d / ".work-items" / "project.json").exists():
            return d
    raise alloc.AllocError("no .work-items/project.json found in this directory or above")


def sync_headers(root, id_=None):
    """Header phase: runs after the allocation lock is released; reports on stderr."""
    for path in headers.sync(root, id_):
        print(f"wi: header updated: {path.relative_to(root)}", file=sys.stderr)


def build_parser():
    p = argparse.ArgumentParser(prog="wi")
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="allocate a work item and print its ID")
    n.add_argument("type", choices=["I", "F", "S", "T", "D", "K"])
    n.add_argument("--parent")
    n.add_argument("--slug")
    n.add_argument("--title")
    n.add_argument("--home")
    n.add_argument("--sync", action="store_true", help="then write family headers")
    r = sub.add_parser("retire", help="abandon an item; its ID still resolves")
    r.add_argument("id")
    rs = sub.add_parser("resolve", help="map any name (ID, alias, slug, FR/AC citation) to the item")
    rs.add_argument("name")
    rs.add_argument("--json", action="store_true")
    mv = sub.add_parser("move", help="rename an item and its subtree under a new parent")
    mv.add_argument("id")
    mv.add_argument("newparent", help="parent ID, or 'none' for the project")
    mv.add_argument("--sync", action="store_true", help="then write family headers")
    pr = sub.add_parser("promote", help="turn a kaizen item into a feature")
    pr.add_argument("id")
    pr.add_argument("--to", required=True, choices=["F"])
    pr.add_argument("--parent", help="initiative to place the feature under (default: none)")
    pr.add_argument("--sync", action="store_true", help="then write family headers")
    hd = sub.add_parser("headers", help="family-document headers")
    hd.add_argument("action", choices=["sync"])
    hd.add_argument("id", nargs="?", help="item (default: every item with a slug)")
    ml = sub.add_parser("mint-legacy", help="give an old <home>:<name> item a scheme ID (idempotent)")
    ml.add_argument("spec", help="<home>:<name>, e.g. architect:K-004")
    ml.add_argument("--type", required=True, choices=["K", "F", "D"])
    ml.add_argument("--parent")
    ml.add_argument("--title")
    ml.add_argument("--home", help="component the item lives in (default: the legacy home if a component, else repo)")
    lk = sub.add_parser("link", help="relate items: concerns (one), fixed-by / relates (many)")
    lk.add_argument("id")
    lk.add_argument("rel")
    lk.add_argument("target", nargs="?")
    grp = lk.add_mutually_exclusive_group()
    grp.add_argument("--remove", action="store_true", help="remove the target from fixed-by/relates")
    grp.add_argument("--clear", action="store_true", help="empty concerns")
    st = sub.add_parser("set", help="set an attribute: milestone, home, touched, slug, where")
    st.add_argument("id")
    st.add_argument("assignment", help="attr=value (touched takes a comma-separated list)")
    sub.add_parser("flush", help="copy journal events into the tracked registry")
    i = sub.add_parser("init", help="create .work-items/ in the current directory")
    i.add_argument("--key", required=True)
    i.add_argument("--components", nargs="*", default=[], help="space- or comma-separated")
    return p


def main(argv=None, cwd=None):
    parser = build_parser()
    args = parser.parse_args(argv)
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
            print(legacy.mint(root, args.spec, args.type, parent=args.parent, title=args.title,
                              home=args.home))
        elif args.cmd == "resolve":
            legacy.install(project.load_namespaces(root))
            rec = resolve.resolve(Store(root, alloc.load_key(root)).read_state(), args.name)
            print(json.dumps(rec, indent=2) if args.json else resolve.describe(rec))
        elif args.cmd == "move":
            print(move.move(root, args.id, args.newparent))
            if args.sync:
                sync_headers(root, args.id)
        elif args.cmd == "promote":
            new_id = move.promote(root, args.id, args.to, parent=args.parent)
            print(new_id)
            if args.sync:
                sync_headers(root, new_id)
        elif args.cmd == "headers":
            sync_headers(root, args.id)
        elif args.cmd == "link":
            links.link(root, args.id, args.rel, args.target,
                       op="remove" if args.remove else None, clear=args.clear)
        elif args.cmd == "set":
            attr, eq, value = args.assignment.partition("=")
            if not eq:
                raise links.LinkError(f"expected attr=value, got {args.assignment!r}")
            links.set_attr(root, args.id, attr, value)
        elif args.cmd == "flush":
            Store(root, alloc.load_key(root)).flush()
    except resolve.ResolveError as exc:
        print(f"wi: {exc}", file=sys.stderr)
        return exc.code
    except (rules.IllegalParent, rules.MalformedId, alloc.AllocError, project.InitError) as exc:
        print(f"wi: {exc}", file=sys.stderr)
        return 1
    return 0
