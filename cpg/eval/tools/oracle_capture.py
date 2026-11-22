#!/usr/bin/env python3
"""oracle_capture.py - runtime-oracle capture wrapper (P1-47; ``docs/plans/connected-code-graph-p1-ml.md`` §6.5).

    oracle_capture.py fence                      assert the capture fence on the CURRENT environment
    oracle_capture.py export --root R [--commit SHA]   git archive EVAL_COMMIT into R/oracle/src (user tree untouched)
    oracle_capture.py run --root R --log L -- CMD...   fence, then run CMD with the shims on PATH/PYTHONPATH
    oracle_capture.py normalize --log L --out O        raw log -> keyNorm edges (match_keys.normalize_contract)
    oracle_capture.py attribute-spa --raw R --dist D --out L   SPA fetch records -> shim-format log (source-mapped frames)
    oracle_capture.py summary --edges O                counts only

Fence (P1-47): FALKORDB_HOST and FALKORDB_PORT must be EXPLICIT (unset refuses: the named mutant); the effective
host is loopback; the port is none of 6379 (falkordb-dev), 6390 (ccg-falkordb), 16390 (ccg-falkordb-tw-16390);
CCG_ORACLE_RUN (the per-run owner token the runner generates) is set; and exactly one running container publishes
that port carrying the labels ccg-falkordb=1, ccg-falkordb.throwaway=1 AND ccg-oracle.run=<CCG_ORACLE_RUN>, with
every host binding of the port on 127.0.0.1 (a foreign container with only the two generic labels is refused).
Exit: 0 ok, 2 usage, 3 fence refused, 4 bad input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIMS = HERE / "oracle_shims"
REPO = HERE.parents[2]
FORBIDDEN_PORTS = (6379, 6390, 16390)
RUN_LABEL = "ccg-oracle.run"
RUN_ENV = "CCG_ORACLE_RUN"
LOOPBACK = ("127.0.0.1", "localhost", "::1")


class FenceRefused(Exception):
    pass


def _host_ips(ports_field: str, port: int) -> list[str]:
    """Host IPs publishing ``port`` in a ``docker ps`` Ports column (``127.0.0.1:16400->6379/tcp, [::]:16400->...``)."""
    ips = []
    for part in ports_field.split(", "):
        left, arrow, _ = part.partition("->")
        ip, _, hp = left.rpartition(":")
        if arrow and hp.split("-")[0] == str(port):
            ips.append(ip.strip("[]"))
    return ips


def _docker_labelled(port: int, run_id: str, run=subprocess.run) -> list[tuple[str, list[str]]]:
    """[(name, host IPs publishing ``port``)] of running containers carrying all three labels."""
    out = run(
        ["docker", "ps", "--filter", "label=ccg-falkordb=1", "--filter", "label=ccg-falkordb.throwaway=1",
         "--filter", f"label={RUN_LABEL}={run_id}", "--filter", f"publish={port}",
         "--format", "{{.Names}}\t{{.Ports}}"],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise FenceRefused(f"docker lookup failed: {out.stderr.strip()[:200]}")
    rows = [ln.split("\t", 1) for ln in out.stdout.splitlines() if ln.strip()]
    return [(r[0], _host_ips(r[1] if len(r) > 1 else "", port)) for r in rows]


def check_fence(env=None, containers=_docker_labelled) -> tuple[str, int, str]:
    """Return (host, port, container) or raise FenceRefused. ``containers(port, run_id)`` is injectable for tests."""
    env = os.environ if env is None else env
    host, port_s = env.get("FALKORDB_HOST"), env.get("FALKORDB_PORT")
    if not host or not port_s:
        raise FenceRefused("FALKORDB_HOST and FALKORDB_PORT must both be set explicitly "
                           "(defaults point at falkordb-dev 6379)")
    try:
        port = int(port_s)
    except ValueError:
        raise FenceRefused(f"FALKORDB_PORT is not an integer: {port_s!r}") from None
    if host not in LOOPBACK:
        raise FenceRefused(f"FALKORDB_HOST {host!r} is not loopback")
    if port in FORBIDDEN_PORTS:
        raise FenceRefused(f"FALKORDB_PORT {port} is a shared instance")
    run_id = env.get(RUN_ENV)
    if not run_id:
        raise FenceRefused(f"{RUN_ENV} (the per-run owner label value) must be set")
    found = containers(port, run_id)
    if len(found) != 1:
        raise FenceRefused(f"expected exactly one throwaway container (labels ccg-falkordb=1, "
                           f"ccg-falkordb.throwaway=1, {RUN_LABEL}=<run>) publishing {port}, "
                           f"found {[n for n, _ in found]}")
    name, ips = found[0]
    if not ips or any(ip != "127.0.0.1" for ip in ips):
        raise FenceRefused(f"{name} publishes {port} on {ips}, not only 127.0.0.1")
    return host, port, name


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def eval_commit() -> str:
    return (REPO / "cpg/eval/EVAL_COMMIT").read_text().strip()


def cmd_export(a) -> int:
    commit = a.commit or eval_commit()
    dest = Path(a.root).expanduser() / "oracle" / "src"
    if dest.exists() and any(dest.iterdir()):
        print(f"refusing: {dest} not empty", file=sys.stderr)
        return 4
    dest.mkdir(parents=True, exist_ok=True)
    arch = subprocess.Popen(["git", "-C", str(REPO), "archive", "--format=tar", commit], stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=arch.stdout)
    arch.stdout.close()
    if arch.wait() != 0 or tar.returncode != 0:
        print("export failed", file=sys.stderr)
        return 4
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(json.dumps({"exported": commit, "files": n, "dest": str(dest)}))
    return 0


def cmd_fence(_a) -> int:
    try:
        host, port, name = check_fence()
    except FenceRefused as e:
        print(f"FENCE REFUSED: {e}", file=sys.stderr)
        return 3
    print(json.dumps({"fence": "ok", "host": host, "port": port, "container": name}))
    return 0


def cmd_run(a) -> int:
    try:
        check_fence()
    except FenceRefused as e:
        print(f"FENCE REFUSED: {e}", file=sys.stderr)
        return 3
    if not a.cmd:
        print("no command", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["ORACLE_LOG"] = str(Path(a.log).resolve())
    env["ORACLE_ROOT"] = str(Path(a.root).expanduser().resolve() / "oracle" / "src")
    env["PATH"] = f"{SHIMS}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{SHIMS}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(SHIMS)
    return subprocess.call(a.cmd, env=env)


def cmd_normalize(a) -> int:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1]))
    import match_keys as M  # noqa: E402
    from urllib.parse import urlsplit
    edges = []
    for line in Path(a.log).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        path = urlsplit(r["path"]).path or "/"
        key = M.normalize_contract({"kind": "http", "keyNorm": f"{r['method']} {path}"}, "consumer")["keyNorm"]
        fr = r.get("frame")
        edges.append({"src": r["src"], "cls": r["cls"], "host": r["host"], "port": r["port"], "keyNorm": key,
                      "element": (fr["file"] + ("::" + fr["func"] if fr.get("func") else "")) if fr and fr.get("file") else None})
    Path(a.out).write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in edges), encoding="utf-8")
    print(json.dumps({"rows": len(edges)}))
    return 0


_B64 = {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")}


def _vlq(seg: str) -> list[int]:
    out, shift, val = [], 0, 0
    for ch in seg:
        d = _B64[ch]
        val += (d & 31) << shift
        if d & 32:
            shift += 5
        else:
            out.append(-(val >> 1) if val & 1 else val >> 1)
            shift = val = 0
    return out


def decode_map(sm: dict) -> list[list[tuple[int, int, int, int]]]:
    """Per generated line, sorted [(genCol, srcIdx, srcLine, srcCol)] (segments without a source are dropped)."""
    lines, src, sl, sc = [], 0, 0, 0
    for line in sm["mappings"].split(";"):
        gc, segs = 0, []
        for seg in line.split(",") if line else []:
            v = _vlq(seg)
            gc += v[0]
            if len(v) >= 4:
                src, sl, sc = src + v[1], sl + v[2], sc + v[3]
                segs.append((gc, src, sl, sc))
        lines.append(segs)
    return lines


def map_frame(maps: dict, bundle: str, line: int, col: int):
    """1-based line/col of a stack frame -> (source path, 1-based line) or None."""
    sm = maps.get(bundle)
    if not sm:
        return None
    segs = sm[1][line - 1] if 0 < line <= len(sm[1]) else []
    best = None
    for s in segs:
        if s[0] <= col - 1:
            best = s
    if best is None:
        return None
    return sm[0]["sources"][best[1]], best[2] + 1


_FRAME = __import__("re").compile(r"at (?:(?P<fn>[^ ]+) \()?(?P<url>https?://[^ )]+?):(?P<ln>\d+):(?P<col>\d+)\)?")
WRAPPERS = ("src/api/client.ts", "src/api/dispatch.ts")


def cmd_attribute_spa(a) -> int:
    """Pick, per fetch record, the first stack frame that maps to a source file that is not a transport wrapper
    (client.ts, dispatch.ts) and not node_modules: `src/api/endpoints.ts` in practice. Unmappable -> frame null."""
    dist = Path(a.dist)
    maps = {}
    for mp in dist.rglob("*.js.map"):
        sm = json.loads(mp.read_text(encoding="utf-8"))
        maps[mp.name[:-4]] = (sm, decode_map(sm))
    from urllib.parse import urlsplit
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for line in Path(a.raw).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u = urlsplit(r["url"])
            frame = None
            for m in _FRAME.finditer(r.get("stack", "")):
                hit = map_frame(maps, m.group("url").rsplit("/", 1)[-1], int(m.group("ln")), int(m.group("col")))
                if not hit:
                    continue
                src = hit[0].replace("\\", "/")
                src = src[src.index("src/"):] if "src/" in src else src
                if "node_modules" in src or src in WRAPPERS:
                    continue
                frame = {"file": "salesperson/" + src, "func": None}
                break
            out.write(json.dumps({"src": "spa", "method": r["method"], "host": u.hostname, "port": u.port,
                                  "path": u.path + ("?" + u.query if u.query else ""), "frame": frame, "cls": "app"},
                                 sort_keys=True, separators=(",", ":")) + "\n")
            n += 1
    print(json.dumps({"rows": n}))
    return 0


def cmd_summary(a) -> int:
    rows = [json.loads(l) for l in Path(a.edges).read_text().splitlines() if l.strip()]
    s = {"rows": len(rows), "testclient": sum(r["cls"] == "testclient" for r in rows),
         "unattributed": sum(r["element"] is None and r["cls"] != "testclient" for r in rows),
         "attributed": sum(r["element"] is not None and r["cls"] != "testclient" for r in rows),
         "distinctKeys": len({r["keyNorm"] for r in rows if r["cls"] != "testclient"}),
         "distinctEdges": len({(r["element"], r["keyNorm"]) for r in rows if r["element"] and r["cls"] != "testclient"})}
    print(json.dumps(s, sort_keys=True))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="verb", required=True)
    sp.add_parser("fence")
    e = sp.add_parser("export"); e.add_argument("--root", required=True); e.add_argument("--commit")
    r = sp.add_parser("run"); r.add_argument("--root", required=True); r.add_argument("--log", required=True)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    n = sp.add_parser("normalize"); n.add_argument("--log", required=True); n.add_argument("--out", required=True)
    q = sp.add_parser("attribute-spa"); q.add_argument("--raw", required=True); q.add_argument("--dist", required=True); q.add_argument("--out", required=True)
    s = sp.add_parser("summary"); s.add_argument("--edges", required=True)
    a = p.parse_args(argv)
    if a.verb == "run" and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    return {"fence": cmd_fence, "export": cmd_export, "run": cmd_run, "normalize": cmd_normalize,
            "summary": cmd_summary, "attribute-spa": cmd_attribute_spa}[a.verb](a)


if __name__ == "__main__":
    sys.exit(main())
