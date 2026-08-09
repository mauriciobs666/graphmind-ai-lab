# mcp-monitor — Implementation Plan Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** — (M1)

Static review of `mcp-monitor/docs/plans/mcp-monitor.md` (441 lines, `architect`, Status: active)
against `mcp-monitor/docs/requirements/mcp-monitor.md` (FR-1..FR-12, AC-1..AC-8, Status: Ready for
design, stakeholder-confirmed 2026-08-08). No existing `mcp-monitor` code exists yet (`find
mcp-monitor -type f` returns only the three docs above) — the plan is reviewed on its own factual
and architectural merits, not diffed against an implementation.

## Verdict: **Approve with suggestions**

The plan is unusually well-researched: every falkor-chat factual claim it makes was independently
verified against the actual source (below) and checked out correct, the FR/AC coverage is
complete, and the repo-convention choices (TOML, `asyncio`, stdin-JSON, no Docker) are all argued
from real precedent rather than asserted. One concrete, confirmed defect (Major, §"Findings" below)
will break the FR-12/AC-4 fake-server deliverable exactly as the plan's own worked example is
written, and should be corrected before or during unit 3a/3b implementation — it does not require
sending the whole plan back through a new design pass. Two further findings (Moderate, Minor) are
real gaps worth closing; the rest are small completeness suggestions.

## Fact-checks performed against falkor-chat source (not taken on faith)

All of §8's load-bearing claims about falkor-chat verified independently:

- **`get_context()` returns one fixed `CallContext(ws=WS_ID, actor=USER_ID)`, same for every REST
  and MCP call** — confirmed, `falkor-chat/server/falkorchat/config.py:99-107`.
- **`isMention`/`read_thread_since` computes the mention flag against `me_id`, and `me_id` is
  always `ctx.actor`** (the single fixed actor, never a per-connection identity) — confirmed,
  `falkor-chat/server/falkorchat/repository.py:775-806` (`count(me) > 0 AS isMention` against
  `$meId`) called from `services.py:707/713/719/728` with `me_id=ctx.actor` in every branch. The
  plan's rejection of `isMention` as the AC-3 detection mechanism is factually grounded, not
  assumed.
- **Explicit `since=` bypasses the cursor-advance branch entirely** — confirmed,
  `falkor-chat/server/falkorchat/services.py:701-709`: `explicit_since` short-circuits to
  `read_thread_since` before the cursor/`advance` code path is ever reached. The plan's claim that
  `since=0` "never touches, races with, or depends on the shared actor's read cursor" is correct.
- **`read_messages(re=..., since=..., limit=...)` — `re` is the *thread id* parameter, not a
  regex** — confirmed, `falkor-chat/server/falkorchat/mcp.py:140-159`. The plan's example watch
  (`args = { re = "demo-welcome", since = 0, limit = 200 }`) uses it correctly.
  `http://localhost:8000/mcp` as the default MCP endpoint also checks out against
  `falkor-chat/scripts/start_server.sh:152-156`.
- **`_content_to_text` flattening (prefer `structuredContent`, else concatenate text blocks)** —
  confirmed verbatim in `falkor-chat/server/falkorchat/tools.py:388-402`.
- **`McpToolClient` wraps its `ClientSession` in a background-thread event loop because its host is
  synchronous, and is verified only against an in-memory stub, with a real external server
  explicitly deferred** — confirmed, `falkor-chat/server/tests/test_mcp_client.py:1-8` ("Wiring a
  real external server is deferred (§4)") and `falkorchat/tools.py:428-512`.
- **`mcp` pin `>=1.28,<1.29` matches both existing in-repo consumers** — confirmed,
  `falkor-chat/server/pyproject.toml:15` and `cpg/mcp/requirements.txt:10` both pin identically.
- **`anyio` is a transitive dependency of `mcp`, and ships its own `pytest11` plugin** — confirmed
  against the installed `mcp` 1.28.x metadata (`Requires-Dist: anyio>=4.5`) and
  `anyio-4.14.2.dist-info/entry_points.txt`'s `[pytest11]` entry.
- **`mcp.shared.memory.create_connected_server_and_client_session` exists and is the exact
  technique `falkor-chat`'s own client-seam test uses** — confirmed, same file as above.

No factual claim in the plan about falkor-chat's behavior was found to be wrong.

## FR / AC coverage — checked one by one

| Req | Covered by | Verdict |
|---|---|---|
| FR-1 (config file, per-watch server/tool/args/interval/regex/command) | §2 schema | ✓ |
| FR-2 (mcp-monitor is its own MCP client) | §1, §4 | ✓ |
| FR-3 (poll checked against regex) | §3 step 3 | ✓ |
| FR-4 (match → launch without waiting on a human) | §3 step 4, §5 | ✓ |
| FR-5 (raw result + matched text + watch/tool/server id delivered) | §5 (stdin JSON + env vars) | ✓ |
| FR-6 (multiple concurrent watches) | §3 (one `asyncio.Task` per `[[watch]]`, `gather`) | ✓ |
| FR-7 (genericity: ≥2 distinct servers/tools) | §8 (falkor-chat) + §9 (fake server) | ✓ |
| FR-8 (repeat-trigger is per-watch config) | §2 (`repeat_trigger`), §6 | ✓ |
| FR-9 (parallel launch, no blocking/skipping) | §3 step 4 (`create_task`, not awaited) | ✓ |
| FR-10 (poll failure logged, watch continues) | §3 step 2, §4 (discard-and-lazily-reopen) | ✓ |
| FR-11 (log failures + matches/triggers) | §7 (`WARNING`/`INFO`/`DEBUG` + `LoggerAdapter`) | ✓ |
| FR-12 (second server is a purpose-built fake, not an existing component) | §9 | ✓ |
| AC-1 (config → autonomous polling) | §3 | ✓ |
| AC-2 (match → launch with full payload) | §5 | ✓ |
| AC-3 (live falkor-chat demo) | §8 + `scripts/demo_falkor_chat.sh` runbook | ✓ (correctly scoped as a manual runbook, not a pytest assertion — no falkor-chat server exists in CI) |
| AC-4 (second server proves genericity) | §9, `test_fake_server_integration.py` | ✓ in design; **blocked by the Major finding below as literally specified** |
| AC-5 (repeat-trigger on/off) | §6, exercised via fake-server state flips | ✓ |
| AC-6 (parallel launch under an in-flight command) | §3, marker-file-timestamp proof | ✓ |
| AC-7 (poll failure: log, no crash, retry) | §3 step 2, §4 | ✓ |
| AC-8 (both failures and matches visible in logs) | §7 | ✓ |

Every FR and AC has a concrete design answer; nothing is silently unaddressed.

## Findings

### Major — `StdioServerParameters` field mismatch will break FR-12/AC-4 as literally specified

§4 writes: `transport = "stdio"` → `mcp.client.stdio.stdio_client(StdioServerParameters(command=...,
env=...))`, and §2's own worked example sets `command = ["python3", "fake_mcp_server/server.py"]`
for `[server.fake-test]`.

Checked against the installed `mcp` 1.28.x SDK (`mcp/client/stdio/__init__.py:72-94`):
`StdioServerParameters.command` is typed `str` (the executable only); the argument list is a
**separate** field, `args: list[str]`. Constructing `StdioServerParameters(command=["python3",
"fake_mcp_server/server.py"], ...)` passes a `list` where a Pydantic `BaseModel` expects a `str` —
this raises a `pydantic.ValidationError` at construction time, before any connection is attempted.

This is not a cosmetic nit: `[server.fake-test]` is exactly the connection FR-12's fake server and
AC-4's genericity proof depend on, and `test_fake_server_integration.py` (§11 tier 3) would fail at
setup, not at an assertion. The fix is mechanical — split the config's `command` array into the
executable (`StdioServerParameters.command`) and the rest (`StdioServerParameters.args`) when
building the connection for a `stdio` server block — but it must be applied; the plan's own
worked example does not compile as written. Note the config's **other** use of `command` (a
watch's launched-process command, §5, consumed by `asyncio.create_subprocess_exec(*command, ...)`)
is a genuinely different consumer with a genuinely different shape requirement (argv array — that
one is correct as written); the plan reuses one TOML array-of-strings shape for both without
flagging that the two downstream consumers need different structures. Whoever implements 3a (the
MCP client layer) or 3b (the fake server's own connection) should not copy §4's sketch literally.

### Moderate — shared per-server connection has no stated concurrency guard across watches

§4 deliberately shares one connection per **named server** across every watch that references it
("so two watches against the same falkor-chat instance don't open two redundant sessions") — a
scenario FR-6 explicitly anticipates ("different patterns against the same tool"). §4 also says a
call failure discards the session, which is lazily reopened "on the next poll that needs it." But
each watch's own poll loop runs as an independent `asyncio.Task` (§3) with no lock or coordination
mentioned around the shared connection object. If two watches share a `[server.*]` block and poll
concurrently, one watch's failure-triggered discard-and-reopen can race with the other watch's
in-flight call on the same (now-stale-or-being-replaced) session object. Neither §4 nor §13's risk
list mentions this. This doesn't need a large fix — an `asyncio.Lock` per server connection around
open/discard/reopen would close it — but as written it's an unaddressed gap in a scenario the
design explicitly says it supports. If the intended v1 answer is "no two watches share a server in
practice" that should be stated; right now the schema and prose both invite the shared case.

### Minor — unbounded dedupe-set growth not listed among the named risks

§6's per-watch dedupe is an in-memory set of matched substrings, checked/grown on every poll for
the life of the process. §13 names four risks (restart loss, key collisions, `since=0` replay cost,
orphaned processes) but not this one: a long-running watch with `repeat_trigger = false` against a
busy/high-volume tool accumulates dedupe-set entries indefinitely — the same *kind* of concern
`§13`'s other entries were careful to name explicitly. Likely benign at "demo scale" (the plan's own
standard for what's acceptable to defer), but the omission is inconsistent with how thoroughly the
rest of §13 catalogs exactly this class of trade-off. A one-line addition to §13 (or the
`BACKLOG.md` seed list) would close the gap.

### Minor — config validation doesn't cover transport-shape errors at load time

§2 states config validation is "fail fast, before any polling starts" and lists what's checked
(`watch.server` resolves, `interval_seconds > 0`, `pattern` compiles, `command` non-empty,
`repeat_trigger` is bool) — but doesn't mention validating that `[server.*].transport` is one of
the two supported values, or that the transport-appropriate field is present (`url` for `http`,
`command` for `stdio`). As written, a typo'd `transport = "htttp"` or a missing `url` would only
surface later, at first-connect time (inside the "log and retry" poll-failure path, per §4/FR-10) —
which is a materially different, and much quieter, failure mode than the "hard startup failure...
never a partially-running process" §2 promises for config errors. Worth folding into the same
load-time validation pass.

### Suggestions (non-blocking)

- **`env={**os.environ, ...}`** (§5) hands the launched command the entire parent environment. Not
  a defect — out-of-scope explicitly excludes "authentication/production hardening" — but it's the
  kind of trade-off §13 otherwise takes care to name explicitly (e.g., orphaned processes,
  command-spawn failures); a one-line acknowledgment would keep §13 internally consistent as the
  single place reviewers look for "what was traded away and why."
- Consider, later, whether `mcp-monitor/docs/manuals/` is warranted for the AC-3 runbook's
  human-facing steps once the component is real — the module documentation convention (root
  `AGENTS.md`) makes `manuals/` optional and owned by `tico`, so this is not a defect in the plan,
  just a forward note since AC-3 is explicitly a human-run live demonstration.

## Repo-convention conformance

- **Doc-kind / header-block / family-slug rules** (root `AGENTS.md`): the plan's own header block
  is correctly formed, and this review's placement (`docs/reviews/mcp-monitor.md`, same topic slug
  as `docs/requirements/mcp-monitor.md` → `docs/plans/mcp-monitor.md`) follows the required family
  rule. `docs/BACKLOG.md`/`docs/HISTORY.md` proposed as flat files directly under `docs/` (§10)
  matches the convention's explicit carve-out for those two names and mirrors `falkor-chat`'s
  reference layout. The `mcp-monitor/README.md` + `mcp-monitor/AGENTS.md` entry-doc pairing (§10)
  correctly mirrors the `salesperson/`/`claude/` precedent named in root `AGENTS.md`'s Structure
  section, and §10 correctly identifies that registering the new component in root `AGENTS.md`'s
  Structure section and Component docs table is part of whichever unit lands those files — the
  coordination doc's unit 6 agrees.
- **Language/runtime and config-format choices** are argued from real, checked precedent rather
  than preference: the `mcp` version pin matches both existing consumers exactly (verified above);
  the `asyncio`-direct-vs-thread-wrapped divergence from `falkor-chat`'s `McpToolClient` is
  justified by a real structural difference (no synchronous host here) rather than dismissed; TOML
  over JSON/YAML is argued on a concrete point (literal strings for regex, no new dependency) that
  holds up against both `cpg/mcp` (minimal-dependency precedent) and `falkor-chat/server` (existing
  `pyproject.toml` precedent, appropriately followed since mcp-monitor is a distributable package
  like `falkor-chat/server`, not a path-run script like `cpg/mcp`).
- **No containerization** is justified by contrast with `cpg/mcp`'s actual stated reason for
  containerizing (Joern/JVM toolchain, content-hashed image) rather than a blanket "Docker is
  unnecessary" — reasonable, and explicitly left as a backlog option rather than foreclosed.

## Summary

Twelve FRs and eight ACs are each traceable to a specific design decision; the plan's most
scrutiny-worthy section (§8, the falkor-chat `isMention`-vs-literal-regex investigation) holds up
completely against the actual source. The one concrete defect found (`StdioServerParameters`
command/args split) is real, verified against the installed SDK, and would break the FR-12/AC-4
deliverable exactly as the plan's own example is written — it should be corrected before or during
implementation of the stdio transport path (unit 3a) and the fake server's own wiring (unit 3b).
The shared-connection concurrency gap (Moderate) and the two Minor completeness gaps are worth
closing but don't require re-architecting anything. None of the findings change the overall
architecture's soundness — **approve with suggestions**.
