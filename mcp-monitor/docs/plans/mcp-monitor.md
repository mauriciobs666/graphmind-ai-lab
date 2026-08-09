# mcp-monitor — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** — (M1)

Design for a brand-new standalone component per `mcp-monitor/docs/requirements/mcp-monitor.md`
(Status: Ready for design, stakeholder-confirmed 2026-08-08). All FR/AC/decision-log references
below are to that document — it is not paraphrased here beyond what's needed to justify a design
choice.

## 1. Language / runtime

**Python 3.12+, using the official `mcp` SDK, pinned `mcp>=1.28,<1.29`.**

Rationale:
- Both existing MCP integrations in this repo — `falkor-chat/server` (MCP server + an in-repo,
  test-only MCP *client* seam, `falkorchat/tools.py: McpToolClient`) and `cpg/mcp` (MCP server) —
  are Python on the same SDK line, verified against `mcp` 1.28.x
  (`falkor-chat/server/pyproject.toml`, `cpg/mcp/requirements.txt`). Following suit means
  mcp-monitor reuses a proven client library instead of hand-rolling MCP framing, and stays
  upgradeable in lockstep with the rest of the repo's MCP surface.
- **FR-2 (mcp-monitor as its own MCP client) is genuinely new ground** — `falkor-chat`'s
  `McpToolClient` is the only existing client code in-repo, and it is verified only against an
  **in-memory stub server** (`falkor-chat/server/tests/test_mcp_client.py`); wiring a real external
  server was explicitly deferred there. mcp-monitor is the first component to open a real
  stdio *and* a real Streamable-HTTP client connection in this repo. That client code is
  therefore written fresh here, not reused — see §2 for why its shape differs from
  `McpToolClient`'s.
- Async-native fits the problem shape directly: N independent watches, each polling on its own
  interval, each poll potentially spawning a parallel subprocess (FR-6, FR-9). `asyncio` handles
  all three (concurrent tasks, concurrent I/O-bound MCP calls, concurrent subprocess launches)
  without threads. Contrast with `falkor-chat`'s `McpToolClient`, which wraps its async
  `ClientSession` in a background-thread event loop *because* the executor that calls it is
  synchronous (`falkorchat/tools.py` docstring). mcp-monitor has no such synchronous host — its
  own `main()` is the event loop — so it uses `mcp.ClientSession` directly, no thread-wrapping
  needed. This is a deliberate divergence from the one in-repo client pattern, not an oversight.

No containerization: `cpg/mcp`'s Docker packaging exists because of Joern/JVM toolchain concerns
(content-hashed image, host `falkordb-dev` isolation) that don't apply here — mcp-monitor's only
dependency is the `mcp` SDK. It follows the **host-venv** half of that precedent
(`setup.sh`/`run.sh`), skipping the Docker half. Containerized deployment can be a later backlog
item if an actual deployment need arises; it is not required to satisfy any FR/AC.

## 2. Config file format

**TOML**, parsed with the standard-library `tomllib` (Python ≥3.11) — **no new dependency**.

FR-1 leaves the format open. The three candidates and why TOML wins for this shape of config:

| | JSON | YAML | TOML |
|---|---|---|---|
| Stdlib parse | yes | no (needs PyYAML) | yes (`tomllib`, read-only) |
| Comments | no | yes | yes |
| Regex-friendly strings | no — backslashes need doubling | yes (block/quoted scalars) | yes — literal strings `'...'` do zero escape processing |
| "list of independent watches" idiom | array of objects (fine, verbose) | list of mappings (fine) | array of tables `[[watch]]` (built for exactly this) |

TOML is the only option that adds no dependency (repo precedent leans minimal — `cpg/mcp`'s
`requirements.txt` carries exactly two runtime deps) *and* gives regex authors literal strings, so
a watch's `pattern = '@mcp-monitor\b'` never needs backslash-doubling the way a JSON string would.
`[[watch]]` array-of-tables is a direct match for "one or more independent watches" (FR-1/FR-6).
YAML was the closest runner-up; rejected only because it would be the first YAML dependency
anywhere in this repo (verified: no `.py`/`.toml`/`.txt` file outside `.venv` trees references
`yaml`) for a benefit (indentation-based nesting) this config doesn't need.

### Schema

```toml
# mcp-monitor config — one or more [[watch]] entries, each referencing a
# [server.<name>] connection block by name.

[server.falkor-chat]
transport = "http"
url = "http://localhost:8000/mcp"

[server.fake-test]
transport = "stdio"
command = ["python3", "fake_mcp_server/server.py"]
env = { FAKE_MCP_STATE_FILE = "/tmp/mcp-monitor-fake-state.json" }

[[watch]]
name = "falkor-chat-mention"          # identifier passed to the launched command (FR-5)
server = "falkor-chat"
tool = "read_messages"
args = { re = "demo-welcome", since = 0, limit = 200 }
interval_seconds = 5
pattern = '@mcp-monitor\b'
repeat_trigger = false                # FR-8 — suppress re-firing on an already-matched item
command = ["bash", "-c", "echo triggered: $MCP_MONITOR_WATCH_NAME"]

[[watch]]
name = "fake-server-demo"
server = "fake-test"
tool = "get_status"
args = {}
interval_seconds = 2
pattern = '"value":\s*"READY"'
repeat_trigger = true
command = ["scripts/handle_trigger.sh"]
```

Validation at load time (fail fast, before any polling starts): every `watch.server` resolves to a
`[server.*]` block; `interval_seconds > 0`; `pattern` compiles as a Python regex; `command` is a
non-empty array of strings; `repeat_trigger` is a bool (default `false` if omitted — the
conservative "already-handled" default, opt into repeats explicitly). A config error is a hard
startup failure with a clear message naming the offending watch — never a partially-running
process (FR-10's fail-soft posture is about *poll* failures at runtime, not config errors at
startup).

## 3. Poll / match / launch architecture

One `asyncio.Task` per `[[watch]]`, all created in `main()` and run concurrently
(`asyncio.gather`) for the process lifetime — this is FR-6 directly: N watches, N independent
loops, no coordination between them.

Each watch's own loop is **sequential** — poll, then sleep `interval_seconds`, then poll again;
a watch never has two polls in flight at once (there is no benefit to overlapping a tool's own
polls, and it would complicate ordering for no gain). Concurrency is *across* watches and *across
launched commands*, never within one watch's polling. Cadence is "sleep after completion," not
wall-clock-aligned — actual period is `interval_seconds + poll_duration`. Accepted simplification
(the requirements set no precision bound on interval; out-of-scope excludes production hardening).

Per-poll sequence:
1. Call the watch's tool with its configured (static) args via the shared server connection (§4).
2. On a raised exception (network error, tool error) — log at `WARNING` with watch/server/tool
   name and the exception, do **not** raise out of the loop, sleep, retry next cycle (FR-10,
   AC-7). The connection object marks itself for reconnect on the next call rather than caching a
   dead session forever (§4).
3. On success, flatten the `CallToolResult` to text (§4) and run `re.finditer(pattern, raw_text)`
   — **every** non-overlapping match in that poll's result is a separate candidate, not just the
   first. This matters because a single poll can legitimately contain several new items (e.g. two
   messages posted between polls, both matching).
4. For each match: compute a dedupe key (§5); if suppressed, skip it (log at `DEBUG`, not silence
   — still visible for troubleshooting a watch that "isn't firing" when it's actually correctly
   deduping). Otherwise, log the match at `INFO` (FR-11) and launch the command **without
   awaiting it** — `asyncio.create_task(_launch(...))` — so a slow or long-running triggered
   process never blocks the watch's own next poll, and two matches in the same poll (or across
   different watches) launch strictly in parallel (FR-9, AC-6). A supervisory task per launch
   awaits the subprocess and logs its exit code on completion (FR-11 — "triggered" alone isn't
   enough for after-the-fact debugging; knowing whether the launched process actually ran to
   success is the more useful signal).

mcp-monitor does not wait for launched commands at shutdown — they are fire-and-forget by design
(out of scope: "what the triggered command does once launched"). A `SIGINT`/`SIGTERM` handler
cancels the watch tasks; already-launched subprocesses are left running/orphaned, which is
consistent with the launched-CLI-is-not-this-feature's-concern framing in Out of scope.

## 4. MCP client layer

One connection per **named server** (the `[server.*]` block), shared by every watch that
references it — not one connection per watch — so two watches against the same falkor-chat
instance don't open two redundant sessions.

- `transport = "http"` → `mcp.client.streamable_http.streamablehttp_client(url)`.
- `transport = "stdio"` → `mcp.client.stdio.stdio_client(StdioServerParameters(command=..., env=...))`.

Both yield `(read, write[, ...])` streams wrapped in `mcp.ClientSession(read, write)` +
`await session.initialize()`, opened lazily on first use and kept open for reuse. On a call
failure, the session is discarded (not retried mid-call) and lazily reopened on the *next* poll
that needs it — this is what makes FR-10's "log and retry, never stop the watch" survive a server
restart, not just a transient RPC error.

Flattening a `CallToolResult` to the text the regex runs against mirrors the technique already in
`falkor-chat/server/falkorchat/tools.py` (`_content_to_text`): prefer `structuredContent`
(JSON-encoded) when the server returned it, else concatenate the text content blocks. This is the
"full raw tool result" of FR-5 — the whole flattened string, not a parsed/filtered subset.

## 5. Command payload delivery (FR-5)

**A single JSON document on stdin** is the canonical, complete payload:

```json
{
  "watch": "falkor-chat-mention",
  "server": "falkor-chat",
  "tool": "read_messages",
  "raw_result": "<the full flattened tool result text>",
  "matched_text": "@mcp-monitor",
  "matched_at": "2026-08-08T12:34:56Z"
}
```

Plus three environment variables carrying only the small identifiers, as a convenience for
trivial commands that don't want to parse JSON: `MCP_MONITOR_WATCH_NAME`,
`MCP_MONITOR_SERVER_NAME`, `MCP_MONITOR_TOOL_NAME`. The raw result and matched text are
**deliberately not** duplicated into env vars or argv — both have practical size/escaping limits
(OS argv length caps, shell quoting of arbitrary matched content, `ps`-visible env leakage) that
stdin JSON doesn't have. This is a hybrid, not two competing mechanisms: stdin is the only place
the *content* travels; env vars are pure convenience labels.

Launched via `asyncio.create_subprocess_exec(*command, stdin=PIPE, env={**os.environ, ...})` —
**argv array, never a shell string** (`command = [...]` in config, no `shell=True`). Two reasons:
nothing needs to be interpolated into the command line (the payload travels on stdin, not as
arguments), and argv-exec sidesteps shell-quoting ambiguity entirely for whatever operator-chosen
command runs. An operator who wants shell features (`&&`, pipes) can still put `["bash", "-c",
"..."]` in `command` themselves.

## 6. Repeat-trigger dedupe (FR-8)

Per-watch, in-memory set of dedupe keys, checked/updated at match time. **Dedupe key = the exact
matched substring** (`match.group(0)`), not a hash of the full raw result or a tool-specific
identifier — this stays generic across arbitrary tools without knowing anything about the tool's
result shape (no assumption that a message id, or any id at all, is present).

Known, accepted limitation: two genuinely different underlying items that happen to produce an
*identical* matched substring are indistinguishable to this dedupe and the second is suppressed
when `repeat_trigger = false`. Concretely, this can bite the falkor-chat demo watch itself if two
messages both literally contain `@mcp-monitor` with nothing else distinguishing the matched span —
the demo must use distinguishable trigger text if it wants to show two independent live triggers
on the mention watch (§8). The escape hatch is `repeat_trigger = true` for a watch that needs
every occurrence to fire regardless of text identity (AC-5 exercises exactly this switch).

**Open question, flagged rather than silently resolved:** dedupe state is in-memory only, lost on
restart. The out-of-scope list accepts "restart to apply config changes" for hot-reload, but says
nothing about whether already-fired state must survive a restart. For a cursor-advancing tool call
(e.g. falkor-chat's `read_messages` in cursor mode) this wouldn't matter — already-read messages
never reappear server-side. But §8 explains why the falkor-chat watch here deliberately does
*not* use cursor mode, and the fake test server (§7) has no concept of a cursor at all — for both,
a restart genuinely can cause a previously-fired match to be seen (and, if `repeat_trigger =
false`, silently *not* re-fire, or if state loss is treated as "unseen," to re-fire once more).
Recommend accepting in-memory-only for v1 (simplest, matches the existing restart-is-fine
posture) and filing persistent dedupe state as a `docs/BACKLOG.md` item rather than building it
now — but this is a real trade-off, not a non-issue, and the QA pass (coordination doc unit 5)
should know AC-5 is only asserting behavior *within one mcp-monitor run*, not across a restart.

## 7. Failure handling & logging (FR-10/FR-11)

Python `logging`, one line per event, to stdout (12-factor-friendly — operators redirect/tee as
they see fit; no built-in log file rotation, that's out of scope's "production hardening").
Every log line carries `watch=`, `server=`, `tool=` context via a `LoggerAdapter` per watch, so
`grep watch=falkor-chat-mention` isolates one watch's history — this is what "an operator can
verify after the fact that a watch is working" (FR-11) means concretely.

Levels: `WARNING` for a poll failure (AC-7); `INFO` for a match found, a command launched, and a
launched command's exit code (AC-2, AC-8); `DEBUG` for a suppressed repeat (visible on request,
not by default, so a correctly-deduping watch doesn't look silent under normal `INFO` operation).
A command that fails to even *spawn* (bad path, permission denied) is also logged at `ERROR` —
not named in FR-10/FR-11 explicitly, but squarely inside "an operator can debug it if not
working," so it's included rather than treated as scope creep.

## 8. The falkor-chat half of the demo (AC-3)

**Investigated and resolved a real design question here, not assumed.** falkor-chat's MCP tool
calls all run under one process-wide fixed identity —
`config.get_context()` returns `CallContext(ws=WS_ID, actor=USER_ID)` from env vars
(`falkor-chat/server/falkorchat/config.py`), the same context for every REST *and* MCP call, no
per-connection identity. `read_messages`' `isMention` flag (`repository.read_thread_since`) is
computed against that single fixed `actor` — i.e., "was **the configured user** mentioned," not
"was any arbitrary name mentioned." Using `isMention` as the AC-3 detection signal would either
(a) only detect mentions of whatever `FALKORCHAT_USER_ID` the demo server happens to run as — not
a name mcp-monitor's config controls — or (b) require reconfiguring that env var to the "agent"
identity, which would then apply to *every* caller against that server, including a human's own
REST/UI session, entangling mcp-monitor's config with falkor-chat's deployment config. Both are
worse than the alternative below, so `isMention` is **rejected** as the mechanism.

**Chosen mechanism: literal-text regex against the message body**, exactly mirroring what a human
"noticing an @mention" actually does (the requirements' own framing). The watch (§2's example)
polls `read_messages(re="demo-welcome", since=0, limit=200)` — explicit `since=0` deliberately
avoids the cursor-advance branch of `read_messages` (`re` given + `since` given → plain timestamp
read, cursor untouched — see the tool's own docstring in `falkorchat/mcp.py`), so mcp-monitor
never touches, races with, or depends on the shared actor's read cursor. This costs re-scanning
the whole thread's history every poll (mcp-monitor's own dedupe, §6, is what keeps that from
re-triggering) — acceptable at demo scale, and *decoupled from falkor-chat's identity model
entirely*, which is the more valuable property: mcp-monitor requires zero falkor-chat
configuration or code change to work (out-of-scope: "any change to falkor-chat's ... MCP server").

Demo mechanics: reuse the already-seeded `demo-welcome` thread (`kiro/.kiro/agents/
falkor-chat-demo.json`, `falkor-chat/scripts/seed_demo.sh`) so no new falkor-chat fixture is
needed. A human (or a small script calling `send_message` directly) posts a message whose body
contains the watch's configured pattern, e.g. `@mcp-monitor please wake up`; mcp-monitor detects
it on its next poll (≤ `interval_seconds` later) and launches the configured command carrying the
full message-list JSON + the matched text. This is a genuinely live, end-to-end demonstration
(AC-3's own wording) — start `falkor-chat`'s server, start `mcp-monitor` pointed at a config with
this watch, post the message, observe the launch (log line + the launched command's own visible
effect). It is not something a `pytest -m live` run can meaningfully assert unattended (there is
no falkor-chat server in CI), so it belongs as a documented, scripted **runbook**
(`mcp-monitor/scripts/demo_falkor_chat.sh`), not a pytest test — see §10.

## 9. The fake/test MCP server (FR-12)

A minimal stdio server, `mcp-monitor/fake_mcp_server/server.py`, using `mcp.server.fastmcp.FastMCP`
(the same library falkor-chat's own MCP tools and `cpg/mcp` use, so it's a five-minute build, not
new library surface) — one tool:

```python
@mcp.tool()
def get_status() -> dict:
    """Return whatever was last written to FAKE_MCP_STATE_FILE."""
```

It reads a small JSON state file (path from `FAKE_MCP_STATE_FILE`, default `{"value": "idle"}` if
absent) on every call. A companion `fake_mcp_server/set_state.py <value>` CLI overwrites that
file. This gives tests and the demo a way to **externally, deterministically flip** what the next
poll sees — set state to `READY`, wait one poll interval, assert the watch fired; flip back to
`IDLE` and to `READY` again to exercise AC-5's repeat-trigger on/off; register two watches against
it with different patterns/commands to exercise AC-6's parallel-launch guarantee. No FalkorDB, no
network, no external process beyond the fake server itself — this is what makes it usable in the
**automated** test suite (§10), unlike the falkor-chat half of the demo.

Deliberately **stdio**, where falkor-chat is **HTTP** — this is a second, independent axis of the
genericity proof (FR-7/AC-4 say "distinct MCP servers/tools"; using a different *transport* too,
not just a different tool signature, is a strictly stronger demonstration that mcp-monitor's
client layer is transport-agnostic, not just tool-agnostic). It lives at
`mcp-monitor/fake_mcp_server/`, a sibling of the main package — not buried under `tests/` —
because FR-12 names it a v1 deliverable in its own right (what AC-4 demonstrates against), not
merely test scaffolding, even though it is also used *by* tests.

## 10. File / module layout

```
mcp-monitor/
  README.md                    # NEW — entry doc (human): what it is, quickstart, config schema
  AGENTS.md                    # NEW — entry doc (agent): architecture pointers, conventions,
                                #   commands — the pair root AGENTS.md's Component docs table
                                #   expects (matches the salesperson/ and cpg/ precedent)
  pyproject.toml                # deps ["mcp>=1.28,<1.29"], dev extra [pytest, ruff, anyio]
  setup.sh                      # venv create/refresh — mirrors cpg/mcp/setup.sh
  run.sh                        # exec python -m mcp_monitor "$@" from the venv
  config.example.toml           # the §2 schema, documented — copy-and-edit starting point
  mcp_monitor/
    __init__.py
    __main__.py                 # CLI: --config PATH [--log-level LEVEL]
    config.py                   # TOML load + validation -> Watch/ServerConfig dataclasses
    client.py                   # per-server ClientSession lifecycle (§4), result flattening
    watch.py                    # per-watch poll/match/dedupe loop (§3, §6)
    launcher.py                 # subprocess launch: stdin JSON + env identifiers (§5), exit-code logging
    logging_setup.py            # LoggerAdapter-per-watch stdout logging (§7)
  fake_mcp_server/
    server.py                   # FR-12 minimal stdio server (§9)
    set_state.py                # state-flip CLI used by tests/demo
  tests/
    test_config.py               # schema validation, error messages
    test_matcher.py              # regex/dedupe logic, in isolation from any transport
    test_launcher.py             # stdin/env payload shape, argv-exec, exit-code logging (stubbed subprocess)
    test_watch_loop.py           # in-memory stub MCP server (mirrors falkor-chat's
                                  #   create_connected_server_and_client_session pattern) —
                                  #   fast, no real process/socket
    test_fake_server_integration.py  # REAL fake_mcp_server subprocess + real mcp-monitor loop;
                                      #   AC-4/AC-5/AC-6, fully automated (§9's "no external deps")
  scripts/
    demo_falkor_chat.sh          # AC-3 runbook (§8) — not a pytest test
  docs/
    requirements/mcp-monitor.md  # already exists
    plans/mcp-monitor.md         # this document
    plans/mcp-monitor-coordination.md  # already exists (teco)
    reviews/                     # unit 2 (analyst) writes mcp-monitor.md here
    test-plans/                  # unit 5 (qa-engineer)
    test-reports/                # unit 5 (qa-engineer)
    BACKLOG.md                   # NEW, seeded by the implementation unit (see below)
    HISTORY.md                   # NEW, first entry on M1 delivery
```

**Entry docs, explicitly:** `mcp-monitor/README.md` + `mcp-monitor/AGENTS.md` are the two files
root `AGENTS.md`'s Structure section and Component docs table should point to — the same pairing
used for `salesperson/` and `claude/`. Both are **new files the implementation unit creates**, not
this plan (this plan is design, not code/docs authorship) — but the file list above removes any
ambiguity about what must exist before root `AGENTS.md` gets its `mcp-monitor` row. Root
`AGENTS.md`'s Structure section and Component docs table update is itself in scope for whichever
unit lands `README.md`/`AGENTS.md` (per the coordination doc's unit 6 and the root convention that
adding a component's entry doc and registering it happen in the same change).

`docs/BACKLOG.md` should be seeded (by the implementation unit, not invented here) with at least:
the persistent-dedupe-state question (§6), config hot-reload and auth/hardening (both already
out-of-scope, but worth a placeholder so they're not forgotten rather than silently dropped), and
optional Docker packaging (§1). Turn-taking/backoff and server-side push are **not** duplicated
into this backlog — they're already tracked at `kiro/docs/requirements/kiro-vision-followups.md`
item 4 and `falkor-chat/docs/BACKLOG.md` K-018 respectively; mcp-monitor's BACKLOG.md should
cross-reference them, not restate them (avoids the same open question living in two places and
drifting).

## 11. Test strategy

Three tiers, only one of which is a manual runbook:

1. **Unit** (`test_config.py`, `test_matcher.py`, `test_launcher.py`) — no network, no
   subprocess (launcher tests stub `asyncio.create_subprocess_exec`). Fast, run on every change.
2. **Component, in-memory** (`test_watch_loop.py`) — the full poll→match→dedupe→launch loop
   driven against `mcp.shared.memory.create_connected_server_and_client_session` wrapping a tiny
   `FastMCP` stub server, exactly the technique `falkor-chat/server/tests/test_mcp_client.py`
   already uses for the same purpose (verifying client-side logic without a real transport). No
   real process, no real socket, deterministic.
3. **Component, real subprocess** (`test_fake_server_integration.py`) — launches the real
   `fake_mcp_server/server.py` as a stdio subprocess and a real mcp-monitor instance against it,
   drives state via `set_state.py`, asserts on the launched commands' observable side effects
   (e.g., a marker file each launched script appends its start timestamp to, so overlapping
   timestamps prove AC-6's parallelism, and a repeated `READY` flip proves AC-5's on/off switch).
   No FalkorDB, no LM Studio, no Docker — safe to run in the default suite, just slower than tier 1
   (real sleeps across real poll intervals; keep `interval_seconds` small, ~1-2s, in the test
   config to bound wall-clock cost).

`falkor-chat` end-to-end (AC-3) is **not** an automated test — no falkor-chat server exists in the
default test environment, and per §8, running it requires a live server + FalkorDB + a human (or
scripted) message post. It's `scripts/demo_falkor_chat.sh` plus a short manual-verification
checklist, matching `falkor-chat`'s own `-m live` precedent for "needs real infrastructure, run
deliberately, not by default." The QA pass (coordination doc unit 5) should walk this runbook and
record the result in the test report rather than expect it to show up as a green pytest line.

Testing uses `anyio`'s bundled pytest plugin (`@pytest.mark.anyio`) rather than adding
`pytest-asyncio` as a separate dependency — `anyio` is already a transitive dependency of `mcp`
itself, so this is zero net new dependencies for async test support.

## 12. Suggested implementation sequencing

The coordination doc (unit 3) asks the architect to recommend a split. Given the design above has
three genuinely separable pieces with different risk profiles, splitting is worth it:

- **3a. Core loop** (`mcp_monitor/` package: config, client, watch, launcher, logging) — the bulk
  of the risk (the new MCP-client ground, §1) and everything unit tiers 1-2 verify. Do this first;
  everything else is exercised through it.
- **3b. Fake test server + integration test** (`fake_mcp_server/`, `test_fake_server_integration.py`)
  — depends on 3a existing to run against, but is otherwise self-contained and low-risk (a FastMCP
  stdio server is a well-worn shape in this repo, `cpg/mcp/server.py` is a direct template for the
  mechanics if not the content). Delivers AC-4/AC-5/AC-6's automated proof.
- **3c. falkor-chat demo wiring** (`scripts/demo_falkor_chat.sh`, the example watch config,
  `mcp-monitor/README.md`'s quickstart) — depends on 3a; touches only mcp-monitor's own files
  (§8 already established zero falkor-chat-side changes are needed). Delivers AC-3's runbook.

3b and 3c have no file overlap and no dependency on each other, so they can run in parallel once
3a lands; 3a is a hard prerequisite for both. If a single `coder` unit is preferred instead
(simpler coordination, and the whole thing is not large), that's also reasonable — this split is
offered as the decomposition if teco wants parallelism, not a requirement to split.

## 13. Risks / open questions (flagged, not silently resolved)

- **Dedupe-state persistence across a restart (§6)** — genuinely unresolved; recommend
  in-memory-only for v1 with a BACKLOG.md note, but flag explicitly to QA that AC-5 is only
  verified *within one run*.
- **Dedupe key collisions on identical matched text from different items (§6)** — an accepted
  MVP simplification with a real, named failure mode: when two distinct items produce a
  byte-identical matched substring, the first occurrence fires and the second is suppressed as
  an apparent repeat when `repeat_trigger = false`. Fine for v1's demo scale; would need
  revisiting if mcp-monitor ever watches a high-volume, low-entropy-match tool in earnest.
- **Growing per-poll cost for the falkor-chat watch (§8)** — `since=0` re-reads the full thread
  every poll; fine at demo scale, not something to carry into any future "real" deployment of this
  watch without adding since-tracking (a natural BACKLOG.md follow-up, deliberately not built now
  per FR-1's static-args framing and the out-of-scope production-hardening exclusion).
- **No falkor-chat-side change needed** — confirmed, not assumed (§8's investigation): the
  literal-text mechanism requires zero new falkor-chat tool or schema change, so there is no
  blocker here for AC-3 as originally worried in the task brief. This risk is resolved, listed
  here so the resolution and its reasoning are visible to reviewers, not just the conclusion.
- **Command-spawn failures and orphaned processes at shutdown (§3, §7)** — both are handled
  (logged; accepted as fire-and-forget respectively) but are genuine, if minor, rough edges an
  analyst/QA pass should confirm are acceptable for v1's stated scope rather than assume away.
