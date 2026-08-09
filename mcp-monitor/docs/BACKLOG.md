# mcp-monitor — Backlog

Living backlog, per the module documentation convention (root `AGENTS.md`) — no header block,
append-only-ish (items get struck through or annotated when resolved, not deleted).

## Open items

- **Persistent dedupe state across a restart.** Dedupe (per-watch matched-substring set, §6 of
  `docs/plans/mcp-monitor.md`) is in-memory only — a restart loses it, so a watch that has already
  fired can (depending on the watched tool) fire again after a restart, or fail to re-fire an item
  it never actually saw. Recommended for v1: accept in-memory-only (matches the existing
  "restart to apply config changes" posture in `docs/requirements/mcp-monitor.md`'s out-of-scope
  list). Flagged explicitly rather than silently resolved — plan §6, plan §13, review "Findings".
- **Unbounded dedupe-set growth.** The same per-watch in-memory dedupe set (above) grows for the
  life of the process with no eviction — a long-running watch with `repeat_trigger = false` against
  a busy/high-volume tool accumulates entries indefinitely. Likely benign at demo scale; the same
  class of trade-off as the item above, called out separately per the plan review's Minor finding
  (a) since §13 of the plan named the restart-loss risk but not this one.
- **Optional Docker packaging.** `docs/plans/mcp-monitor.md` §1: no containerization in v1 (no
  Joern/JVM-toolchain concern to isolate the way `cpg/mcp` has); `cpg/mcp/Dockerfile` +
  `docker-run.sh` are the template to follow if a real deployment need for mcp-monitor shows up.
- **Config hot-reload.** Out of scope per `docs/requirements/mcp-monitor.md` — restarting the
  process to apply a config change is accepted. Not attempted here.
- **Authentication / production hardening.** Out of scope per requirements — e.g. `launcher.py`
  hands a launched command the entire parent environment (`env={**os.environ, ...}`), which the
  plan review flagged as a trade-off worth naming rather than a defect (plan review "Suggestions").

## Cross-references (tracked elsewhere — do not duplicate here)

- **Turn-taking/backoff among multiple agents responding to the same trigger** —
  `kiro/docs/requirements/kiro-vision-followups.md` item 4. mcp-monitor is a client-side polling
  watcher; deciding how several triggered agents should coordinate is a separate, not-yet-resolved
  question tracked there.
- **Server-side real-time push (vs. client-side polling)** — `falkor-chat/docs/BACKLOG.md` K-018.
  mcp-monitor is a polling watcher by decision (`docs/requirements/mcp-monitor.md` decision log);
  a push mechanism on falkor-chat's side would be complementary, not a replacement for this
  component, and is tracked at that K-018 item, not here.
