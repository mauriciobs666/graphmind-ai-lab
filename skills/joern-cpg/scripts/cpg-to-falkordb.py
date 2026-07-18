#!/usr/bin/env python3
"""cpg-to-falkordb.py — Stage 4: turn a Joern neo4jcsv export into FalkorDB-dialect
Cypher and (optionally) load it.

Joern's `joern-export --format neo4jcsv` writes, nested per method under the
output dir:
    nodes_<LABEL>_header.csv / nodes_<LABEL>_data.csv   (:ID,:LABEL,<prop>[:type],...)
    edges_<TYPE>_header.csv  / edges_<TYPE>_data.csv     (:START_ID,:END_ID,:TYPE)
(The sibling *_cypher.csv files are Neo4j `LOAD CSV` scripts — not FalkorDB-usable.)

FalkorDB model (a sensible default; graph-dba owns real tuning):
  * every node gets a shared `:CpgNode` label PLUS its Joern type label, so edges
    can be matched by id without knowing the type label;
  * `CpgNode(id)` is indexed FIRST so edge MATCHes are cheap;
  * nodes/edges are created with UNWIND-batched CREATE (deduped by id / triple).

Output: a newline-delimited .cypher file (one runnable statement per line) — the
"export to Cypher" artifact. With --load, each statement is sent via
`redis-cli GRAPH.QUERY <graph>`. The loader REFUSES a non-empty graph: resetting
it (`redis-cli GRAPH.DELETE <graph>`) is a deliberate, destructive op left to the
operator (and caught by joern's destructive-ops guard), not hidden in here.

Usage:
  cpg-to-falkordb.py <export-dir> [-o load.cypher] [--graph cpg]
                     [--load] [--host localhost] [--port 6379]
                     [--batch 500] [--append]
Stdlib only.
"""
import argparse, csv, glob, json, os, subprocess, sys

csv.field_size_limit(1 << 24)  # CODE properties can be large


def parse_header(path):
    """Return list of (prop_name, kind) for a *_header.csv; kind in {int,string,array,skip}."""
    with open(path, newline="") as f:
        row = next(csv.reader(f), [])
    cols = []
    for c in row:
        name, _, typ = c.partition(":")
        if c in (":ID", ":LABEL", ":START_ID", ":END_ID", ":TYPE"):
            cols.append((c, "meta"))
        elif typ.endswith("[]"):
            cols.append((name, "array"))
        elif typ in ("int", "long"):
            cols.append((name, "int"))
        elif typ == "boolean":
            cols.append((name, "bool"))
        else:
            cols.append((name, "string"))
    return cols


def cypher_scalar(kind, val):
    if val == "":
        return None
    if kind == "int":
        try:
            return str(int(val))
        except ValueError:
            return json.dumps(val)
    if kind == "bool":
        # Joern :boolean columns carry "true"/"false"; emit real Cypher booleans
        # so predicates like `WHERE m.IS_EXTERNAL = false` work (else they'd be
        # string literals and never match a boolean).
        return "true" if val.strip().lower() == "true" else "false"
    if kind == "array":
        parts = [p for p in val.split(";") if p != ""]
        return "[" + ", ".join(json.dumps(p) for p in parts) + "]"
    return json.dumps(val)  # valid Cypher double-quoted string literal


def label_from(fname, prefix):
    b = os.path.basename(fname)
    return b[len(prefix):-len("_data.csv")]


def collect(export_dir):
    nodes = {}   # id -> (label, {prop: cypher_literal})
    edges = {}   # (start, end, type) -> True  (dedup)
    for hdr in glob.glob(os.path.join(export_dir, "**", "nodes_*_header.csv"), recursive=True):
        data = hdr[:-len("_header.csv")] + "_data.csv"
        if not os.path.exists(data):
            continue
        label = label_from(data, "nodes_")
        cols = parse_header(hdr)
        with open(data, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                nid, props = None, {}
                for (name, kind), raw in zip(cols, row):
                    if name == ":ID":
                        nid = raw
                    elif name == ":LABEL":
                        pass
                    elif kind == "meta":
                        pass
                    else:
                        lit = cypher_scalar(kind, raw)
                        if lit is not None:
                            props[name] = lit
                if nid is None or nid == "":
                    continue
                props["id"] = str(int(nid))
                nodes.setdefault(nid, (label, props))
    for hdr in glob.glob(os.path.join(export_dir, "**", "edges_*_header.csv"), recursive=True):
        data = hdr[:-len("_header.csv")] + "_data.csv"
        if not os.path.exists(data):
            continue
        etype = label_from(data, "edges_")
        with open(data, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 2 or row[0] == "" or row[1] == "":
                    continue
                edges[(str(int(row[0])), str(int(row[1])), etype)] = True
    return nodes, edges


def map_literal(props):
    return "{" + ", ".join(f"{k}: {v}" for k, v in props.items()) + "}"


def statements(nodes, edges, batch):
    yield "CREATE INDEX FOR (n:CpgNode) ON (n.id)"
    # nodes grouped by label so each batch shares one CREATE label set
    by_label = {}
    for nid, (label, props) in nodes.items():
        by_label.setdefault(label, []).append(props)
    for label, rows in by_label.items():
        for i in range(0, len(rows), batch):
            chunk = ", ".join(map_literal(p) for p in rows[i:i + batch])
            yield f"UNWIND [{chunk}] AS r CREATE (n:CpgNode:{label}) SET n = r"
    # edges grouped by type
    by_type = {}
    for (s, e, t) in edges:
        by_type.setdefault(t, []).append((s, e))
    for etype, pairs in by_type.items():
        for i in range(0, len(pairs), batch):
            chunk = ", ".join(f"{{s: {s}, e: {e}}}" for (s, e) in pairs[i:i + batch])
            yield (f"UNWIND [{chunk}] AS r "
                   f"MATCH (a:CpgNode {{id: r.s}}), (b:CpgNode {{id: r.e}}) "
                   f"CREATE (a)-[:{etype}]->(b)")


def redis_cli(host, port, *args):
    return subprocess.run(["redis-cli", "-h", host, "-p", str(port), *args],
                          capture_output=True, text=True)


def graph_nonempty(host, port, graph):
    """True iff the graph exists AND holds at least one node.

    Probe with GRAPH.RO_QUERY, not GRAPH.QUERY: a read via GRAPH.QUERY
    *materializes* an empty graph key as a side effect (live-verified on
    falkordb v4.18.11), whereas GRAPH.RO_QUERY on a non-existent graph returns
    `ERR Invalid graph operation on empty key` and creates nothing — so a
    missing graph reads as cleanly empty. Parse the count from the one
    pure-integer line of redis-cli output; never regex-scan the whole reply, or
    the `Query internal execution time: 0.179153 milliseconds` stat line's
    digits register as a phantom non-zero count.
    """
    r = redis_cli(host, port, "GRAPH.RO_QUERY", graph, "MATCH (n) RETURN count(n)")
    if "empty key" in (r.stdout + r.stderr).lower():
        return False  # graph doesn't exist yet → nothing to clobber
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.isdigit():
            return int(s) > 0
    return False


def main():
    ap = argparse.ArgumentParser(description="Joern neo4jcsv export -> FalkorDB Cypher / load")
    ap.add_argument("export_dir")
    ap.add_argument("-o", "--out", default="load.cypher")
    ap.add_argument("--graph", default="cpg")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--load", action="store_true", help="also send each statement to FalkorDB via redis-cli")
    ap.add_argument("--host", default=os.environ.get("FALKORDB_HOST", "localhost"))
    ap.add_argument("--port", default=os.environ.get("FALKORDB_PORT", "6379"))
    ap.add_argument("--append", action="store_true", help="allow loading into a non-empty graph")
    args = ap.parse_args()

    if not os.path.isdir(args.export_dir):
        sys.exit(f"cpg-to-falkordb: export dir not found: {args.export_dir}")

    nodes, edges = collect(args.export_dir)
    stmts = list(statements(nodes, edges, args.batch))
    with open(args.out, "w") as f:
        f.write("\n".join(stmts) + "\n")
    print(f"cpg-to-falkordb: {len(nodes)} nodes, {len(edges)} edges -> "
          f"{len(stmts)} Cypher statements written to {args.out}", file=sys.stderr)

    if not args.load:
        print(f"cpg-to-falkordb: not loading (no --load). Replay with:\n"
              f"  while IFS= read -r q; do redis-cli -h {args.host} -p {args.port} "
              f"GRAPH.QUERY {args.graph} \"$q\"; done < {args.out}", file=sys.stderr)
        return

    if not args.append and graph_nonempty(args.host, args.port, args.graph):
        sys.exit(f"cpg-to-falkordb: graph '{args.graph}' is not empty. Reset it first "
                 f"(destructive, escalates):\n  redis-cli -h {args.host} -p {args.port} "
                 f"GRAPH.DELETE {args.graph}\n...or pass --append to add to it.")

    failed = 0
    for i, q in enumerate(stmts):
        r = redis_cli(args.host, args.port, "GRAPH.QUERY", args.graph, q)
        if r.returncode != 0 or "(error)" in r.stdout.lower() or r.stderr.strip():
            # tolerate "index already exists" on the first statement
            if i == 0 and "already" in (r.stdout + r.stderr).lower():
                continue
            failed += 1
            sys.stderr.write(f"  stmt {i} failed: {(r.stdout + r.stderr).strip()[:200]}\n")
    print(f"cpg-to-falkordb: loaded into '{args.graph}' ({len(stmts) - failed}/{len(stmts)} "
          f"statements ok, {failed} failed)", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
