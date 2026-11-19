# P1-27 implementation review: teco impact-answer seam, cpg-analysis §5, `.mcp.json` ccg entry

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 P1-27

## Scope & verdict

Scope: the uncommitted tree against baseline `22b39823` (`git show 22b39823:<path>`), per
`docs/plans/connected-code-graph-p1-answers.md` P1-27: `claude/teco/teco.md`, `skills/cpg-analysis/SKILL.md`,
`skills/README.md`, `.mcp.json`, `cpg/tests/test_atrest.py`, `claude/teco/kaizen/{history,plan}.md`. Read against root
`REVIEW.md` (no `claude/REVIEW.md` exists). `cobb` keeps final say on prompt text.

**Verdict: approve with suggestions.** No blocker. Two majors are wording gaps in the teco bullet (a latent
deadlock reading, an under-specified anchor). One catalog row was not updated.

CPG: considered, not relevant — the change is agent/skill prompt text, JSON wiring and one test allowlist; the `ccg` server and CLI were exercised directly instead.

## Checks run

- **(1) CLI as instructed.** `cd cpg && .venv/bin/python -m clients.cli.main ask --request '{"useCase":"impact","anchor":{"kind":"name","name":"x"}}' --format text`
  parses and runs (flags `--request`, `--format text` exist per `--help`; also from `cwd=/tmp`, so the editable-install claim holds).
  No snapshot loaded (a refresh builds from clones, not cheap), so I ran it with `CCG_FALKORDB_PORT=1` and no credentials: it
  exits 2 with `{"error":{"code":"no-credentials",...}}`, a typed error as documented. The `store-error` (unreachable store)
  path was not reachable without credentials and was not run. I started no throwaway and touched neither `ccg-falkordb-tw-16390` nor `falkordb-dev`.
  The success path (stamp/notice in text) was read, not run: `ccg/render/text.py` prints `Snapshot <snapshotId>`, "Code views used" and
  `flow_notice_lines` (text = `FLOW_NOTICE_TEXT`, `render/common.py:50`), which matches the bullet's claims.
- **(3) Launch string.** `.mcp.json` `args[1]` equals the allowlist entry byte for byte (printed with `repr`); `pytest tests/test_atrest.py`:
  161 passed. Mutant (`/cpg` to `/cpg/../x` in a copy of `.mcp.json`, fed through `_collect`): the real file yields no
  `mcp.json:ccg` patterns, the mutant yields one, so a changed launch string fails. The server started from that exact
  `bash -c` string and `tools/list` returned `ask`, `resolve`, `refresh`, `raw_query` under server name `ccg`, so the tool names are `mcp__ccg__*`.
- **(4)** `allowed-tools` names match. §5's argument names match `clients/mcp/server.py` (`ask(request, question, cacheBypass, system)`,
  `resolve(anchor)`, `refresh(scope)`, `raw_query(cypher, timeoutMs)`) and the anchor kinds match `clients/common.py:_ANCHOR_KEYS`.
- **(6)** `claude/scripts/audit-team.sh` ends FAIL, but the listed failures are link-check hits in unrelated `docs/reviews`/`test-reports` files, none in changed files; not attributed to this change.

## Findings

### Major

**M1. "No stamp, no dispatch" reads as unconditional; it can deadlock teco when ccg is down.** `teco.md:120`: "a brief on a
covered system without a stamp is not dispatched", then, later in the same bullet, "A typed error ... is stated ... as
`ccg unavailable: <code>`, then the CPG line below applies." A typed error yields no stamp, so the two sentences conflict, and
the stricter one is first. Worse, failures with no typed JSON at all (missing `.venv`, Bash "No such file", a timeout, non-JSON
output) are not covered by "typed error", so teco has no fallback instruction. Suggest: fold into one rule, "no stamp and no
`ccg unavailable: <reason>` line means no dispatch; any failure to obtain an answer (typed error, nonzero exit with no JSON,
timeout) is stated as `ccg unavailable` and the CPG line applies." Also say to check the exit status (the CLI exits 2 on an error).

**M2. The instructed command cannot be written from the bullet alone.** `teco.md:120` gives `"anchor":{...}`; the real anchor needs
`kind` plus kind-specific keys (`{"kind":"name","name":...}`, `{"kind":"symbol","repo":...,"qname":...}`, `{"kind":"location","repo":...,"path":...}`,
...; `clients/common.py:55-61,190`), and an anchor missing `kind` is `bad-request`. §5 of the skill lists the kinds but not the required keys either.
An isolated teco will guess. Suggest one concrete example in the bullet (the `name` form, the one verified to parse) and a pointer to `cpg/README.md` for the rest.

### Minor

**m1. Catalog not updated.** `claude/AGENTS.md:311` requires the full `claude/README.md` catalog entry in the same change; the `teco`
row (`claude/README.md:7`) still says "CPG-freshness checking is centralized here ... via its `mcp__cypher__query` grant" and says nothing of the ccg impact answer.
`git status` shows no `claude/README.md` change. The `skills/README.md` row was updated. Add a clause to the teco row; the agent-team manual
(`claude/docs/manuals/agent-team.md`) is only needed if cobb judges this a team-behavior change.

**m2. Third `buildStatus` unaddressed.** The renderer emits three states (`fresh`, `stale-kept`, "not mapped, never built",
`render/common.py:183-189`). The bullet names only `stale-kept`; a never-built unit is the one most likely to make an impact list empty. Say "any non-`fresh` unit is stated with its status and reason".
Also the renderer says "stale, the build failed at ...", not the literal `stale-kept`; the bullet's "stated as `stale-kept`" means teco translates, which is acceptable but should say so.

**m3. Staleness absent when `staleness.units` is empty.** `render_text` omits "Code views used" then (`render/text.py:27`), yet the
bullet requires a staleness note from it. State what to write ("no code views listed") rather than leave it to inference.

**m4. `allowlist` entry `"127.0.0.1"` is global.** `test_atrest.py:693` skips any `command`/`args`/`env`/`url`/`cwd` value equal to it, so a
`url: "127.0.0.1"` is also exempt. Harmless (a bare loopback host names no file), and the real exposure (a changed launch string) is shown to fail. A note in the comment that the exemption is not field-scoped is enough; no change needed.

**m5. Retitled bullet is slightly off.** "CPG freshness (uncovered systems and optional `cpg_*` graphs)" is also the fallback for a covered system with ccg unavailable (the bullet says so). Add "or ccg unavailable".

**m6. `refresh` by teco via Bash** (`--scope <repo>` before a unit after a prior unit changed code) is a store rebuild, run through Bash with no tool-level
boundary; the kaizen plan already notes its cost/failure is unmeasured. Acceptable for P1; keep the first-live-use check.

### Nit

- §5 omits `cacheBypass` and `system` arguments (irrelevant for structured asks in P1); fine.

## Judgments requested

- **(2) Seam wording.** The bullet forbids "safe/complete/no impact/nothing affected", carries `notices` "exactly as the renderer prints it", and
  states an empty list is not evidence of no impact; this matches `FLOW_NOTICE_TEXT`. Not the deadlock-free part: see M1.
- **(5) Prompt safety.** Nothing instructs a raw_query write or treats answer text as instructions. The text output opens with the renderer's
  `FENCE_SENTENCE` ("it is data, not instructions") and code-derived text is code-spanned, so a verbatim paste preserves the fence. Suggest, for cobb,
  one clause in the bullet telling teco to paste the answer inside the brief as quoted data under that fence sentence. Skill §5 says `raw_query` is "for what `ask` cannot say", read-only, and the database refuses writes (plan P1-25 (4)); fine.
- **(6) Size and growth.** +256 words in `teco.md` (8743 to 8999); the new bullet is ~1.7k characters, comparable to the CPG bullet it sits beside, whose body is unchanged
  (diff confirms), so this is new scope, not restated history. No finding.
- **(7) CLI-through-Bash instead of granting `mcp__ccg__ask`: endorse for P1.** The CLI's `--format text` runs the deterministic renderer, so the
  flow notice reaches the brief as the renderer's own text, whereas the MCP `ask` returns JSON that teco would have to render or paraphrase (the very
  drift the notice rule forbids). AC-16 pins CLI and MCP to identical bytes, so no capability is lost, and no `tools:` change avoids a session-restart/approval step and an
  `audit-team.sh` tool-list change. The cost is the Bash capability for `refresh`/`raw` (m6). The kaizen parking-lot item records the open decision.
- **(8) Kaizen accuracy.** `history.md`: the bullet description, "retitled, body unchanged", flow-notice claim, "`clients` run from any cwd", and "teco lacks `mcp__ccg__*`"
  (frontmatter `tools:` has none) are all accurate. `plan.md` parking-lot entry is accurate and forward-looking. "Plan items: none" plausible.

## What's solid

`.mcp.json` entry carries no credential and starts a working server; the allowlist is exact-match and a mutant fails; `allowed-tools` and §5
argument shapes match the shipped server; the notice/staleness rule is faithful to the renderer.

## Open questions

- For `cobb`: M1/M2 wording, and whether m1's catalog update is wanted in this change or deferred to the P1 close.

## Pass 2

Re-checked against the uncommitted tree (cobb's fixes to `claude/teco/teco.md`, `claude/README.md`, `claude/teco/kaizen/history.md`).

- **M1 fixed in part (see N1).** One failure rule now covers non-zero exit, typed error, no JSON and no stamp ("`ccg unavailable: <code or reason>`, apply the CPG line", "do not stall"); no deadlock for those shapes. Run: `ask` with a closed port and no credentials exits 2 with `{"error":{"code":"no-credentials"...}}` on stderr (`clients/cli/main.py:main`, every `ClientError` returns exit 2); the rule maps that to unavailable.
- **M2 fixed.** The literal example `--request '{"useCase":"impact","anchor":{"kind":"name","name":"<symbol>"}}' --format text` parses and reaches the credential step (`name` is a valid kind with required key `name`). The five kinds match `_ANCHOR_KEYS` (`symbol`, `name`, `location`, `contract`, `config`, `clients/common.py:55`); the success path still unrun (no snapshot).
- **m1 fixed.** The `claude/README.md` teco row now names the ccg impact answer and the `ccg unavailable` fallback; only that sentence changed.
- **m2 fixed.** The quoted wording matches `staleness_lines` (`render/common.py:183-189`): `fresh`, `stale, the build failed at … ; the view kept is from …`, `not mapped, never built` (grep: each literal present). The bullet tells teco to name those units as stale views, which resolves the `stale-kept` translation.
- **m3 fixed.** `staleness: no code views listed` is prescribed for the empty case, consistent with `render_text` omitting the section.
- **m5 fixed.** The CPG bullet is titled "(uncovered systems, optional `cpg_*` graphs, or `ccg` unavailable)", consistent with the failure rule's target.
- **m4, m6 unchanged** (accepted as minor/no change).

### New findings

**N1 (major). The failure rule contradicts the anchor sentence.** `teco.md:120` says an `ambiguous`/`unresolved` anchor "(exit 3 or a typed anchor error) goes to the specialist as the candidates", then the bold rule says any non-zero exit or typed error means `ccg unavailable` and the CPG line. A structured ask with an ambiguous or unresolved anchor is a typed `anchor-ambiguous`/`anchor-unresolved` `ClientError` (`clients/common.py:88-93`), so it exits 2, hits both sentences, and teco must pick one. Also, "exit 3" never occurs here: the command carries no `--fail-on`, and exit 3 is for a canonical ambiguous document, which P1-26 produces (`docs/plans/connected-code-graph-p1-answers.md` P1-25 (8b)). Suggest: drop "exit 3" and write the carve-out into the rule: "Anything else, except `anchor-ambiguous`/`anchor-unresolved` (those go to the specialist as the candidates, never a guessed pick)". The same carve-out is needed for `bad-request`, which is the caller's own malformed anchor: retry once with the corrected JSON, else unavailable.

**N2 (minor).** The error JSON goes to stderr (`main` writes `err`), so "no JSON" on stdout is the normal error shape; say "read stderr for the code" so teco reports the real code rather than "no JSON".

### Verdict

**Needs changes, narrowly: N1 only** (a prompt contradiction between two sentences of one bullet, independent of the other fixes; M1/m2/m5 do not conflict with each other). Everything else is resolved; with N1 fixed (carve-out plus dropping "exit 3") this is approve. `cobb` keeps final say.

## Pass 3

Re-read the whole Impact-answer bullet (`claude/teco/teco.md:120`) against `clients/cli/main.py` and `clients/common.py`.

- **N1 fixed.** "exit 3" is gone. The carve-out names exactly the two typed codes that `Client.ask` raises for a non-single resolution: `AnchorNotResolved` sets `code = f"anchor-{status}"` with `status` in {`ambiguous`, `unresolved`} and `detail=resolution_json(res)` (`common.py:88-93, 368`); the CLI prints `exc.payload()` to **stderr** and returns `EXIT_ERROR` = 2 (`main.py:101-104`). The bullet's "exit 2, JSON on stderr, candidates in `error.detail`" matches. Not run (needs a loaded store to reach the resolver); read end to end instead.
- **N2 fixed.** The bullet now says "JSON on stderr" and reports `<code or reason>`.
- **Sentence-pair check.** Outcomes: (a) exit 0 with stamp, verbatim brief; (b) delivered `ambiguous`/`unresolved` document (exit 0), options go to the specialist; (c) the two anchor errors, candidates go to the specialist; (d) `bad-request`, one retry then (e); (e) everything else, `ccg unavailable` plus the CPG bullet. (b) and (c) are disjoint by exit code and shape (a document on stdout vs an error on stderr) and share one outcome, so they cannot conflict. (c) is explicitly "not failures" and (e) opens "Anything else", so the literal "a typed error" in (e)'s list does not capture (c); a reader could still trip on it, a nit (n1 below). (e) and the CPG bullet title "(... or `ccg` unavailable)" agree. (b) is not reachable in P1 (the plain-language path is P1-26) and is harmless forward text.
- **Nit n1.** In (e), "a typed error" could read "any typed error"; "any other typed error" would remove the last trace of the earlier collision. Take or leave.

**Verdict: approve with suggestions** (n1 only; the earlier minors m4/m6 stand as accepted). No blocker or major remains. `cobb` keeps final say on prompts.
