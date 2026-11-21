"""`hook-sync` and `retry-failed`: the Python half of the post-edit hook (plan section 3.2).

`hooks/post-tool-sync.sh` decides in bash whether anything mirrored changed and calls `hook-sync PATH...` in the
foreground; once the shell sees a retry is due it launches `retry-failed` detached. Both are routine commands in
`commands_store.confirmation` terms: the hook passes `--confirm coordination` as a flag (never the standing env),
and the connection goes through `store.open_graph`, so the guards of `store.require_shared` and
`store.require_shared_kb_url` apply unchanged.

Exit codes of `hook-sync`: 0 every file synced (or nothing to do), 2 at least one file failed or is pending (stderr
names each, the shell relays it to the agent), anything else is the command's own fault (the shell records the
files as failed). Failure state lives in `failed.json` with backoff 1, 2, 4, 8, then 15 minutes (`hookstate`);
a new edit (a different sha256) restarts the backoff. Imports no `wilib` or `identity`.
"""
from __future__ import annotations

import socket
import sys
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import commands_kb, commands_store, kbsync, store, sync
from .config import Config
from .discover import discover, kb_files
from .hookstate import HookState, backoff_s, relative

PING_TIMEOUT_S = 1.0  # fail fast (plan 3.2 step 5.1): FalkorDB 1 s, falkor-chat and the embedder 2 s each
HEALTH_TIMEOUT_S = 2.0
FIRST_RUN_RETRY_S = 60


def ping(host: str, port: int, timeout: float = PING_TIMEOUT_S) -> None:
    """RESP `PING`; raises `OSError` when nothing answers `+PONG` within `timeout`."""
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(b"*1\r\n$4\r\nPING\r\n")
        if not s.recv(16).startswith(b"+PONG"):
            raise OSError("no PONG")


def http_ok(url: str, timeout: float = HEALTH_TIMEOUT_S) -> None:
    """GET `url`; raises `OSError` (incl. `URLError`/`HTTPError`) unless it answers 2xx within `timeout`."""
    with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310 - the configured loopback endpoints
        pass


@dataclass
class Run:
    ok: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    deps: set[str] = field(default_factory=set)  # dependencies found down, named in the message
    deferred: dict[str, float] = field(default_factory=dict)  # retry only: path -> epoch s it may be tried again
    reached: bool = False  # FalkorDB answered and the guard let us in
    silent: bool = False  # first run, FalkorDB down: say nothing, retry later
    config_error: str | None = None  # first run, connection refused by a guard

    def fail_all(self, paths, reason: str, dep: str | None = None) -> None:
        for p in paths:
            self.failed[p] = reason
        if dep:
            self.deps.add(dep)


def process(args, cfg: Config, root: Path, paths: list[str], *, first_run: bool = False, retry: bool = False,
            now=None) -> Run:
    """Sync `paths` (repo-relative; every discovered file when `first_run`) and report each one's fate."""
    run = Run()
    key = getattr(args, "graph", None) or cfg.key
    confirm = commands_store.confirmation(args, cfg)
    kb_set = set(kb_files(root))
    if first_run:
        paths = [f.path for f in discover(root)] + sorted(kb_set)
    present = []
    for p in dict.fromkeys(paths):
        if (root / p).is_file():
            present.append(p)
        else:
            run.ok.add(p)  # a gone file is dropped from the failure state; `sync --all` prunes the graph
    try:
        ping(cfg.falkordb_host, cfg.falkordb_port)
    except OSError as exc:
        if first_run:
            run.silent = True
        else:
            run.fail_all(present, f"FalkorDB unreachable at {cfg.falkordb_host}:{cfg.falkordb_port} ({exc})",
                         "FalkorDB")
        return run
    try:
        db, graph = store.open_graph(cfg.falkordb_host, cfg.falkordb_port, key, confirm)
    except Exception as exc:  # noqa: BLE001 - a refused endpoint is one clear line
        if first_run:
            run.config_error = str(exc)
        else:
            run.fail_all(present, str(exc))
        return run
    run.reached = True
    coord = [p for p in present if p not in kb_set]
    kb = [p for p in present if p in kb_set]
    try:
        for o in sync.sync_paths(graph, sync.Parser.for_root(root), coord):
            if o.status == "failed":
                run.failed[o.path] = o.detail
            else:
                run.ok.add(o.path)
    except Exception as exc:  # noqa: BLE001 - the graph went away mid-run
        run.fail_all([p for p in coord if p not in run.ok and p not in run.failed], f"mirror graph: {exc}", "FalkorDB")
    if kb:
        _sync_kb(args, cfg, root, db, graph, kb, confirm, run, retry, now or time.time)
    return run


def _sync_kb(args, cfg, root, db, graph, kb_paths, confirm, run: Run, retry: bool, now) -> None:
    cfg_c = replace(cfg, shared_confirm=confirm)
    try:
        engine = commands_kb._engine(cfg_c, db, graph)  # noqa: SLF001 - the URL guard for the mirror's mode
        docs, _problems = commands_kb.load_docs(root)
        files = {r[0]: r[1] for r in engine._q(kbsync.READ_FILES)}  # noqa: SLF001
        rows, now_ms = engine.read_all()
    except Exception as exc:  # noqa: BLE001
        run.fail_all(kb_paths, str(exc))
        return
    by_path = {d.path: d for d in docs}
    differing = []
    for p in kb_paths:
        if p not in by_path:
            run.failed[p] = "registered KB file could not be read"
        elif by_path[p].sha256 == files.get(p):
            run.ok.add(p)
        else:
            differing.append(p)
    if retry:  # a section still inside its kb-finish cap belongs to kb-finish (plan 3.2)
        for p in list(differing):
            starts = [int(r["pendingSince"]) for r in rows
                      if r.get("path") == p and r.get("state") == "pending" and r.get("pendingSince")]
            live = [s for s in starts if now_ms - s <= kbsync.FINISH_CAP_MS]
            if live:
                run.deferred[p] = now() + (min(live) + kbsync.FINISH_CAP_MS - now_ms) / 1000
                differing.remove(p)
    if not differing:
        return
    for dep, url in (("falkor-chat", cfg.kb_url.rstrip("/") + "/health"),
                     ("the embedding endpoint", cfg.embed_url.rstrip("/") + "/models")):
        try:
            http_ok(url)
        except OSError as exc:
            run.fail_all(differing, f"{dep} unreachable ({url}: {exc})", dep)
            return
    ctx = kbsync.Context.build(docs, rows)
    pending = False
    for p in differing:
        try:
            outcomes = engine.sync_file(by_path[p], ctx, rows)
        except Exception as exc:  # noqa: BLE001 - one file's error must not lose the others
            run.failed[p] = f"{type(exc).__name__}: {exc}"
            continue
        bad = [o for o in outcomes if o.incomplete]
        if bad:
            run.failed[p] = "; ".join(f"{o.heading}: {o.status}" + (f" ({o.detail})" if o.detail else "")
                                      for o in bad)
            pending = pending or any(o.status == "pending" for o in bad)
        else:
            run.ok.add(p)
    if pending:
        commands_kb.detach_finish(cfg_c, confirm, getattr(args, "graph", None))


def record(state: HookState, root: Path, run: Run, now: float) -> None:
    """Fold a run into `failed.json` under `state.lock`: merge the shell's pending paths, clear what synced,
    back off what failed, then rewrite `retry-due`."""
    with state.locked():
        failed = state.load_failed()
        for p in state.take_pending():
            failed.setdefault(p, {"sha256": None, "error": "hook-sync failed; see hook.log", "attempts": 0,
                                  "nextRetryAt": now})
        for p in run.ok:
            failed.pop(p, None)
        for p, reason in run.failed.items():
            try:
                sha = sync.sha_of(root / p)
            except OSError:
                sha = None
            prev = failed.get(p)
            attempts = prev["attempts"] + 1 if prev and prev.get("sha256") in (sha, None) else 1
            failed[p] = {"sha256": sha, "error": reason, "attempts": attempts,
                         "nextRetryAt": now + backoff_s(attempts)}
        for p, due in run.deferred.items():
            if p in failed:
                failed[p]["nextRetryAt"] = max(failed[p].get("nextRetryAt", 0), due)
        state.save_failed(failed)


def _message(run: Run) -> str:
    lines = [f"coordgraph: {len(run.failed)} mirrored file(s) not synced:"]
    lines += [f"  {p}: {reason}" for p, reason in sorted(run.failed.items())]
    back = " and ".join(sorted(run.deps)) or "the cause"
    lines.append("this update is not complete — say so; it is retried automatically, or run "
                 f"`coordgraph sync <file>` once {back} is back")
    return "\n".join(lines)


def hook_sync_cmd(args) -> int:
    """`coordgraph hook-sync [--first-run] [PATH...]`: sync the paths the shell found changed."""
    cfg = Config.from_env()
    root = Path(cfg.repo_root)
    state = HookState(cfg.state_dir)
    now = time.time()
    paths = [r for r in (relative(root, p) for p in args.paths) if r]
    if not args.first_run and not paths:
        return 0
    run = process(args, cfg, root, paths, first_run=args.first_run)
    if run.silent:  # nothing reachable and nothing known to be edited: no warning, retried after retry-due
        state.write_retry_due(int(now) + FIRST_RUN_RETRY_S)
        return 0
    if run.config_error:
        state.write_retry_due(int(now) + FIRST_RUN_RETRY_S)
        print(f"coordgraph: {run.config_error}", file=sys.stderr)
        return 2
    record(state, root, run, now)
    if args.first_run:
        state.advance_stamp(args.stamp_at if args.stamp_at is not None else now)
    if run.failed:
        print(_message(run), file=sys.stderr)
        return 2
    return 0


def retry_failed_cmd(args) -> int:
    """`coordgraph retry-failed`: retry every due entry of `failed.json`; one instance at a time. Detached."""
    cfg = Config.from_env()
    root = Path(cfg.repo_root)
    state = HookState(cfg.state_dir)
    with state.retry_singleton() as got:
        if not got:
            return 0
        now = time.time()
        with state.locked():
            failed = state.load_failed()
            for p in state.take_pending():
                failed.setdefault(p, {"sha256": None, "error": "hook-sync failed; see hook.log", "attempts": 0,
                                      "nextRetryAt": now})
            state.save_failed(failed)
        due = [p for p, e in failed.items() if e.get("nextRetryAt", 0) <= now]
        if not due:
            return 0
        run = process(args, cfg, root, due, retry=True)
        record(state, root, run, now)
    for p, reason in sorted(run.failed.items()):
        print(f"retry-failed: {p}: {reason}")
    print(f"retry-failed: {len(due)} due, {len(run.ok)} settled, {len(run.failed)} failed again")
    return 0
