# Review: K-099 embedder context and MCP `send_message` cap

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-099

## Pass 1 — MCP send_message cap

### Scope & verdict

**Reviewed:** K-099 item (3), the uncommitted working-tree diff (`git diff`, read-only) of
`server/falkorchat/{services,mcp,storefront_api}.py` and `server/tests/test_{services,mcp,storefront_api}.py`.
The baseline is the K-099 entry in `falkor-chat/docs/BACKLOG.md`, root `REVIEW.md` and
`falkor-chat/REVIEW.md`. **Out of scope:** `scripts/ensure_demo_models.*`,
`tests/test_ensure_demo_models_script.py` and `tests/test_decision_contract.py` (Pass 2 or other owners).

**Verdict: approve with suggestions.** No blockers or majors. The cap is in the right place and
uses the right precedent. All three caller claims hold. The storefront unreachability claim is
true and pinned. One test overclaims what it pins (F1, minor).

CPG: used cpg_falkorchat — caller sweep for `post_message`/`post_agent_answer`/repo write methods, cross-checked by repo-wide grep.

**Checks run:**
- `tests/test_services.py tests/test_storefront_api.py`: **440 passed**.
- `tests/test_mcp.py` was **not run**, as the brief directed. Its three new tests were reviewed statically.
- **Shared-state warning.** `tests/test_storefront_api.py` is **not** review-safe. Its `seeded`
  fixture (`tests/test_storefront_api.py:662`) and `:3093` request `wf_repo`, which runs
  `MATCH (n) DETACH DELETE n` on `reference` (`tests/conftest.py:139-148`). After the run, a
  read-only `MATCH (n) RETURN labels(n), count(*)` on `reference` returns **no rows**. I don't know
  what `reference` held before the run. The published workflow defs, the catalog and the
  salesperson defs must be assumed gone. I did **not** re-seed it, because that would mutate shared
  state outside my role. Re-seed sequence (`falkor-chat/AGENTS.md`, Key scripts):
  `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh <wsId>` → `seed_demo.sh` → `seed_workflows.sh`,
  then for the demo `seed_catalog.sh` → `seed_salesperson.sh`. Verify with `verify_workflows.sh`,
  `verify_catalog.sh` and `verify_salesperson.sh`.

### Findings

**F1 (minor, verified by mutant) — the "before any repo call" test does not pin that the cap runs before validation.**
`test_post_message_rejects_text_over_max_text_len_before_any_repo_call` asserts
`repo.calls == []`. `FakeRepo` records only writes and a few reads. It does not record
`thread_exists` or the member lookup that `_validate_and_derive_role` (`services.py:911`) runs. I
moved the cap to *after* `_validate_and_derive_role(...)` (mutant M4 below). All three new
`test_services` tests stayed green. So the docstring's "rejected before validation reads or
writes" is not enforced. The stakes are low: nothing is written either way. But the error
precedence changes, so an over-long post to a missing thread would get `ThreadNotFoundError`, and
the docstring asserts an ordering no test holds.
**Fix:** add an over-length post to a thread that is *not* in `repo.threads`, and assert
`MessageTooLongError`. **Verified:** on the shipped code it raises `MessageTooLongError`. Under M4
it raises `ThreadNotFoundError`, so the test goes red (probe in the Appendix).

**F2 (nit) — the error text alone does not tell a model what to do.**
FastMCP wraps the message as `Error executing tool send_message: message text is N characters,
exceeding the 8000 limit`. The "split it into several messages" advice is only in the tool
description. Under `falkor-chat/REVIEW.md`, an error message is behavior. This one invites no
fabrication or bypass, so it is only a nit. An agent that skimmed the description would retry
better if the error itself said "split the body into several messages". Take it or leave it. If
you take it, keep the `exceeding the {MAX_TEXT_LEN} limit` substring that both tests match on.

### Brief items, answered

1. **Caller claim: holds (CPG + grep, verified).**
   - The CPG `CALL` sweep, excluding `tests/`, finds `post_message` at exactly
     `api.py:98` (`build_router.post_message`), `mcp.py` (`send_message`) and
     `storefront_api.py` (`build_storefront_router.post_message`). It finds `post_agent_answer`
     only at `responder.py:121` (`AgentResponder.maybe_respond`) and `tools.py:278`
     (`PostMessageTool.run`, the workflow `post_message` tool).
   - A repo-wide grep (`\.post_message(`, excluding tests/`deprecated/`/`.venv`) agrees. It also
     shows no script calling the service directly: `scripts/load_append.py` goes through REST.
   - The repository write methods are reached only through `_dispatch_write`'s
     `first_write`/`subsequent_write` arguments (`services.py:994-995`, `:1028-1029`).
   - Residual worth knowing: *external* agents (`mcp-monitor`-woken CLIs, the `kiro/` demo agent)
     post LLM-generated text through MCP `send_message`. They are now capped. That is the K-099
     intent, and the claim is about in-process replies, so it is accurate.
2. **What the MCP caller sees: consistent with `DocumentTooLargeError` (verified against `mcp` 1.28.1 source).**
   - `Tool.run` re-raises any exception as `ToolError(f"Error executing tool {name}: {e}")`
     (`mcp/server/fastmcp/tools/base.py:116-117`).
   - On the wire, the low-level server turns it into `CallToolResult(isError=True)` carrying that
     text (`mcp/server/lowlevel/server.py:589-590`).
   - `mcp.py` has no special handling for either error, so both take the identical path. The
     test's in-process `mcp.call_tool` → `pytest.raises(ToolError)` is the right altitude.
   - The docstring is accurate: 8000, nothing posted or scheduled (the cap precedes
     `_schedule_background`). It is pinned to the constant by
     `test_send_message_tool_description_states_the_current_max_text_len`.
3. **Storefront unreachability: true and pinned (verified by mutants).**
   - `PostMessageIn.text` has `max_length=MAX_MESSAGE_TEXT_LEN` = 2000 (`storefront_api.py:222,282`).
   - `body.text` reaches `services.post_message` unchanged (`:1240`).
   - Raising `MAX_MESSAGE_TEXT_LEN` to 8001 reddens the new pin test (SF1). Dropping the
     `SERVICE_ERRORS_UNREACHABLE` entry reddens the partition test (SF2).
   - Residual it would not catch: a future transform that prepends roughly 6000 chars to
     `body.text` before the call. This is implausible, so it is not a finding.
4. **Test quality: good, with F1 as the one gap.** Mutants I ran (file copied aside, edited,
   targeted tests run, restored from the copy, `__pycache__` cleared, file touched; sha256 identical
   before and after):

   | Mutant | Result |
   |---|---|
   | M1 `>` → `>=` (same length) | killed (`..._at_exactly_max_text_len_is_accepted`) |
   | M2 `> MAX_TEXT_LEN + 1` (off-by-one) | killed (`..._rejects_text_over_...`) |
   | M3 limit dropped from the message | killed (`..._rejects_text_over_...`) |
   | M4 cap moved after `_validate_and_derive_role` | **survived** (F1) |
   | SF1 `MAX_MESSAGE_TEXT_LEN = 8001` | killed (pin test) |
   | SF2 `SERVICE_ERRORS_UNREACHABLE` entry removed | killed (partition test) |

   The expected values come from `MAX_TEXT_LEN` itself. That is correct here: the contract is
   "equals the REST cap", not a literal.
5. **Live-spec sections affected (for the feature close, not a defect of this diff).**
   - `falkor-chat/docs/SERVER.md` §1.4, **Bounds** paragraph: "The document and batch caps are
     also enforced in the service, so an MCP caller is held to them too". It must now include
     message text.
   - `falkor-chat/docs/SERVER.md` §2.2, the **"No schema layer on MCP"** bullet: "only the
     service-enforced bounds apply: the document and batch caps hold". Same addition.
   - These two sections state the same fact. The close should update both or make one cite the
     other (root `REVIEW.md`, two-sections rule).
   - Optional: the §2.2 `send_message` tool row.
   - No change needed in §1.4's error envelope: on REST, `MessageTooLongError` would be a `400`,
     but Pydantic's `422` always answers first. No change needed in `DESIGN.md` or `QUERIES.md`.

### What's solid

- The check sits in `post_message`, before any write, with a message shaped exactly like
  `DocumentTooLargeError`'s (`services.py:1233-1237`).
- `len()` matches Pydantic's `max_length` semantics (characters).
- The MCP tests run `_schedule` synchronously, so "no embed scheduled" is actually observable
  rather than vacuously true.
- The boundary is tested on both sides (8000 accepted and embedded, 8001 refused).
- The storefront partition count went 13 → 14, consistently in both places.

### Open questions

- **Uncapped agent answers are still embedded.** `post_agent_answer` is uncapped on purpose, but
  `responder.py` self-embeds its answer (`responder.py:16-22`). A long responder answer could
  still exceed the embedder's 4096-token context. Item (1)'s over-context probe decides whether
  that matters. Flagging it so item (1)'s owner covers this path, not only MCP posts.
- **Other gaps on the MCP surface.** MCP `send_message` still accepts an empty body and more
  than 50 mentions, both of which REST rejects. They are outside K-099 (3). Should a backlog
  item cover them?

### Proposed REVIEW.md lines

None. F1 is a one-off, and the shared-state hazard above is a documentation gap in
`falkor-chat/AGENTS.md`'s "Probing shared graph state" subsection. That subsection names only
`test_services.py` as review-safe, which is correct. The brief's wider claim was the error. A
REVIEW.md rule would not change what a reviewer does. The fix belongs to whoever writes briefs:
`test_storefront_api.py` is review-safe only for nodes that don't request `seeded`, `wf_repo` or
`conn`.

## Pass 2 — embedder context

### Scope & verdict

**Reviewed:** K-099 items (1) and (2), the uncommitted diff (`git diff`, read-only) of:
- `scripts/ensure_demo_models.py`;
- `scripts/ensure_demo_models.sh` (the header);
- `server/tests/test_ensure_demo_models_script.py`;
- in `falkor-chat/AGENTS.md`, the `ensure_demo_models.sh` row and the demo-models paragraph.

The probe evidence is `falkor-chat/docs/test-reports/embedder-context-report.md`. Its
side-effecting live runs are taken as reported: I did not touch LM Studio.

**Verdict: approve with suggestions.** No blockers or majors.
- The probe supports silent head truncation, and the cut follows the loaded context.
- Warn-not-refuse is reasoned and consistent with `start_demo.sh`.
- Three new code paths have no test that can fail: the chat model's load-it-yourself line, the
  app hint, and the embedder check after a load. Six extra mutants survive (P2-1, P2-2).
- The probe's "never rejects" is not tied to the LM Studio build it ran on (P2-3).

CPG: considered, not relevant — the change is in `scripts/`, outside the `falkorchat` package the CPG covers.

**Checks run:**
- Test safety: I grepped the three files for `wf_repo`/`conn`/`wf_client`/`repo` fixtures and
  found none. They use only `tmp_path` and a loopback stub.
- `test_ensure_demo_models_script.py` + `test_start_demo_script.py` +
  `test_opencode_config_resolution.py`: **77 passed**, matching the report's 45 + 32.
- Mutants: ran on a copy of `ensure_demo_models.py`, restored afterwards; sha256 identical before
  and after. Positive control: M10 (`EMBED_CONTEXT_LENGTH = 8192`) gives 6 failed.

### Findings

**P2-1 (minor, verified by mutant) — no test covers the embedder check after a load.**
Under mutant MF (the `check_embedder_context` call dropped after a load), all 45 tests still pass.
- `test_one_missing_...`'s `"WARNING" not in out.stderr` is vacuous under MF.
- `test_missing_chat_model_loads_at_8192` uses `embed_ps=4096`, so it never expects a warning.

This uncovered case is the most likely real one after K-099: an embedder loaded at 8192 before
this change is still resident, and ministral is missing.
**Fix:** `_catalog(chat=None, embed=2048, embed_ps=8192)`. Assert exit 0, assert only `CHAT` is
loaded, and assert `loaded with context 8192, not 4096` in stderr. **Verified:** it passes on the
shipped code and fails under MF (Appendix).

**P2-2 (minor, verified by mutant) — the chat model's per-model context in the operator instructions is never asserted.**
`test_cli_absent_stops_with_instructions` has only the embedder missing. Two mutants survive:
- MA: the load-it-yourself command uses `EMBED_CONTEXT_LENGTH` for every model.
- MB: the app hint does the same.

The report's M5 kills only the opposite direction, where every model falls back to 8192. So under
MA/MB an operator with no CLI would be told to load ministral at 4096. Their next run would then
refuse on ministral's context.
**Fix:** with both models missing and `lms=False`, assert `lms load {CHAT} -c 8192 --gpu max` and
`{CHAT} 8192` alongside the embedder's two strings. **Verified:** it fails under MA and under MB
(Appendix).

**P2-3 (minor, the method is verified; the version-dependence comes from upstream sources) — "never a rejection" is stated without the build it holds for.**
The probe evidence for truncation is strong (item 1 below). But upstream llama.cpp's server
**rejects** an embed input longer than `n_ubatch` ("input is too large to process. increase the
physical batch size", the `slot.n_prompt_tokens > n_ubatch` check). So LM Studio's truncation is
a property of its runtime build, and an update could bring rejection back. That would bring back
K-099's original fear: the embed job fails, only a log line records it, and the message drops out
of vector search.

The report records no LM Studio app or llama.cpp runtime version. Both the script's warning text
and its docstring state truncation as plain fact.
**Fix:** record both versions in the report and scope the verdict to them ("on LM Studio X /
runtime Y"). In `ensure_demo_models.py`, have the docstring cite the report for the effect rather
than restating it unscoped. That makes a later re-probe a one-line comparison.

**P2-4 (nit) — the exit-0 wording now contradicts the warning.**
The docstring says "0 = demo set loaded with the demo settings". The no-op line reads "both demo
models already loaded with the demo settings — nothing to do". Both can now be followed by
`WARNING: … loaded with context 8192, not 4096`. Reword them to claim only what was checked, for
example "chat model at the demo context; embedder context checked below".

**P2-5 (nit) — untested edges of the advisory check.**
- Mutant MC (a bad `LMS_CLI` on the no-op path left uncaught) survives. It would turn a ready demo
  into exit 1, which the comment at `ensure_demo_models.py:398` exists to prevent.
- Mutant ME (0 or 2 instances matching the embedder's identifier) survives.
- A non-int `contextLength` (missing, or a string) prints "embeds only each input's first None
  tokens". Treat it as "not checked".

These are low stakes. One test with `LMS_CLI=/nonexistent` on the no-op path covers MC.

### Brief items, answered

1. **The probe supports truncation, and the cut follows the context (I read the evidence; I did not reproduce it).**
   - Every over-context call returned HTTP 200 with a 1024-dim vector. That includes 18,636 chars
     at a 512-token context and 94,469 chars at 4096.
   - Bit-identical vectors (cos 1.000000) for every input past a prefix are a strong signal of
     head truncation that needs no tokenizer. The suffix control (cos 0.46) rules out a constant
     or degenerate vector.
   - The binary-searched cut moves with the loaded context: 67 items at 512, 473 at 4096. The
     context ratio is 8. The item ratio is about 7, which is consistent because later items are
     longer.
   - The per-item token counts are estimates, as the report says. They aren't needed for the
     claim, which rests on cuts at two context points.
   - One hypothesis is indistinguishable: a cut at a batch size that LM Studio sets equal to the
     context. It doesn't matter in practice.
   - "Never rejects" is an inference beyond the tested build (P2-3).
   - The report's "every REST message fits whole at 4096" holds for Latin-script prose and for
     the dense filler it measured. Arbitrary 8,000-char text (CJK, emoji) can exceed one token
     per character. The effect is truncation, not failure, so it isn't a finding.
2. **Warn-not-refuse is consistent (traced).**
   - `start_demo.sh:251-263` fails only on a non-zero exit. A warning goes to stderr and bring-up
     continues, which matches the documented "never unloads or reloads".
   - The storefront never embeds, so a wrong embedder context can only cost VRAM (larger) or
     recall for the `:8200` server (smaller).
   - Edge paths:
     - No CLI: a note, exit 0 (tested).
     - Unreadable `ps` output: a note (tested). A non-zero `ps` exit, a timeout, an `OSError`
       and a `UnicodeDecodeError` (a `ValueError` subclass) are all caught.
     - Nothing-to-do path: still needs no CLI (tested), and a bad `LMS_CLI` there is caught
       (untested, P2-5).
   - The post-load path checks the embedder after `check_refusals` passes. That is the right order.
   - Wording caveat: P2-4.
3. **Test and mutant coverage: partial.**
   - The load command's per-model context is covered both ways (M1–M4, M10 per the report;
     M10 reproduced here).
   - The printed instructions are covered only for the embedder (P2-2).
   - The embedder check is covered on the no-op path but not after a load (P2-1).
   - Extra mutants I ran: MA, MB, MC, ME and MF survived. MD (the `ps` return code ignored)
     survived too, but it is close to an equivalent mutant: a failed `ps` with valid JSON on
     stdout is implausible.
4. **The `AGENTS.md` change is accurate and follows the convention.**
   - The row was rewritten in place, not appended to. The paragraph gained one parenthetical.
   - Both statements match the code: ministral 8192 refuses; the embedder at 4096 only warns, via
     `lms ps --json`.
   - The row dropped "shared with `start_server.sh`/…". That is a trim; the shlib path remains.
   - No line exceeds 700 chars (lines 87-88 already did at `HEAD` and are unchanged). The file
     grew from 2,788 to 2,812 words, already over the ~2,500 smell before this change; not
     attributable here.
   - Nit, left unraised: the 4096 now appears in both the row and the paragraph of the same file.
5. **What the responder's uncapped answers cost, now that we know LM Studio truncates (traced).**
   - `AgentResponder` self-embeds its answer (`responder.py:128`). The workflow `post_message`
     tool (`tools.py`) schedules no embed.
   - A responder answer longer than about 4,096 tokens (roughly 16k chars of prose) is still
     stored whole, still readable, and still found by full-text search (`QUERIES.md` §5,
     `db.idx.fulltext.queryNodes`).
   - Only its **vector** represents the head alone, so semantic and hybrid search can't match on
     the tail.
   - There is no error, no lost message and no failed job. The cost is a silent recall loss on
     rare, very long answers. The chat model's 8192 context, prompt included, bounds how often
     that happens. The storefront demo embeds nothing, so it is unaffected.
   - It becomes a failure only if P2-3's version risk lands (rejection). In that case the answer
     would vanish from vector search with only a log line.
   - Recommendation: no cap. Pin the version (P2-3).

### What's solid

- `context_for` is one decision point, shared by the load command and both printed hints. That's
  why the gaps are in tests, not in code.
- The check is read-only (`ps --json`) and every failure mode degrades to a note.
- The stub `lms` models the API's 2048 misreport separately from `ps`'s real context.
- The report is honest about method limits (`usage` zeros, no tokenizer). It discloses the `laya`
  eviction it may have caused instead of quietly reloading that model.

### Open questions

- **Probes on a shared LM Studio.** The probe's own `lms load` may have evicted `laya`
  (`developer.unloadPreviousJITModelOnLoad: true`), a model another concurrent workstream was
  using. Should a live probe on a shared LM Studio record `lms ps --json` before and after, and
  declare up front that it can evict JIT-loaded models? That's a process call for `teco`/`devops`.
  It isn't a defect of this diff: `ensure_demo_models` itself loads only when nothing foreign is
  resident.
- **Item (4)'s VRAM measurement.** If an embedder at 8192 measurably pages ministral, the
  warn-not-refuse trade-off deserves a second look. Item (4) is where that gets measured.

### Proposed REVIEW.md lines

None. P2-3 (version-pinning a runtime-behavior probe) is a natural candidate, but
`falkor-chat/REVIEW.md`'s "A measured comparison … pins its baseline to an explicit commit sha"
already covers the nearest class. One instance isn't enough to show a recurring pattern.

## Pass 3 — F1/F2 re-check

**Verdict: approve.** Pass 1's MCP cap is cleared. I ran only `tests/test_services.py` (288
passed). Mutants ran on a copy of `services.py`, restored afterwards; sha256 identical before and
after (`d2d60769…`).

- **F1 — fixed.** I re-ran M4 (the cap moved after `_validate_and_derive_role`). It is killed:
  the run fails with `AssertionError: repo.thread_exists touched before the length check`
  (`test_services.py:1833`). The implementer replaced my missing-thread case with
  `_UntouchableRepo`. It is stronger, because it catches any repo access ahead of the cap, not
  only an access that changes the error type. I checked whether it could pass vacuously; it
  can't:
  - `Services.__init__` (`services.py:769-782`) only stores the repo. Any init-time access would
    raise, which is red, not vacuous.
  - Nothing on the `post_message` path catches broadly (`services.py:902-1005`), so the
    `AssertionError` cannot be swallowed into a pass.
  - `hasattr` would not mask it either, since it catches only `AttributeError`.
  - Residual: implicit dunder lookups such as `bool(repo)` bypass `__getattr__`. That is not
    repo I/O, so it doesn't matter here.
- **F2 — fixed** (`services.py:986-987`, "; split it into several messages").

**P3-1 (nit, verified by mutant) — the split hint is pinned by no test.**
Blanking `"split it into several messages"` leaves `test_services.py` green. A grep of `tests/`
finds no assertion on it, and the `test_mcp.py` cap test matches only `exceeding the {MAX_TEXT_LEN}
limit`. Under `falkor-chat/REVIEW.md`, an error message is behavior, so it can now vanish
silently. **Fix:** add `split` to the existing `pytest.raises(..., match=...)` in either test.

## Pass 4 — P2 re-check

**Verdict: approve.** Pass 2's embedder-context change is cleared.
- I grepped the three permitted files for FalkorDB fixtures and found none. They pass: **85**.
- Mutants ran on a copy of `ensure_demo_models.py`, restored afterwards, with the file touched and
  `__pycache__` cleared. sha256 identical before and after (`191706c8…`).
- A gitignored `scripts/__pycache__/embedding_migration.cpython-312.pyc` was left behind. It is a
  bytecode file for a different module, not a change to the tree.
- I did no LM Studio loads or unloads.

- **P2-1 — fixed.** MF (the embedder check dropped after a load) is killed: 1 failed.
- **P2-2 — fixed.** MA (the load-it-yourself line at 4096) and MB (the app hint at 4096) are
  killed: 1 failed each.
- **P2-3 — fixed.**
  - The report's verdict now reads "On the builds above: silent head truncation, not a rejection".
  - The script docstring (step 6) and the WARNING text both say "on the LM Studio build probed in"
    and cite the report. The docstring adds a re-probe-after-update note.
  - I re-read the build evidence read-only, and every value matches the report:
    - `lms runtime ls` shows `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.67.0` in the ✓ column.
    - `lms version` shows `CLI commit: dcefd5a`.
    - `LM Studio/resources/app/package.json` has `"version": "0.4.26+4"`.
  - The report also gained a process note on probing a shared LM Studio. That note answers
    Pass 2's first open question.
- **P2-4 — fixed.**
  - The exit-code line now reads "0 = demo set loaded, nothing foreign, chat model at
    DEMO_CONTEXT_LENGTH (the embedder's context is only advised on)".
  - The no-op line now reads "chat model at the demo context — nothing to load (embedder context
    checked below)".
- **P2-5 — fixed.** These mutants are all killed:
  - MC (a bad `LMS_CLI` left uncaught on the no-op path): 1 failed.
  - ME with the instance-count guard turned into `== 0`, and the reverse case turned into `> 1`:
    1 failed each.
  - The non-int guard removed: 3 failed.
  - Its `bool` exclusion dropped: 1 failed.
  - The `lms ps` exit-code mutant stays untested, by agreement. It is close to an equivalent
    mutant (Pass 2, item 3).

Nothing new to report.

## Pass 5 — embedder at 2048

**Verdict: approve with suggestions.** The script, the tests, `AGENTS.md` and the
`MessageTooLongError` docstring are correct. Two documents still state 4096 claims that this
change makes stale (P5-1).
- I grepped the three permitted files for FalkorDB fixtures and found none. They pass: **86**.
- Mutants ran on a copy of `ensure_demo_models.py`, restored afterwards, with the file touched and
  `__pycache__` cleared. sha256 identical before and after (`86f9d69d…`).
- I did not touch LM Studio.

Stakeholder answer folded in: the embedder loads at 4096 → **2048** (2026-10-31). A message longer
than the window goes to K-100 chunking.

1. **The rationale matches the measurements.** I re-derived each number from the report's 2048
   tables:
   - Embedder VRAM while active: 1377 − 1025 = **352 MiB** (−25.6%). The script says "~350 MiB" ✓.
   - Ministral paging: 960 → 610 and 956 → 636, which is −350 / −320 MiB, or −36.5% / −33.5%. The
     script's "~35%" ✓. The report's "−34–36%" rounds slightly inward.
   - Cut points: 9,681 / 7,960 / 5,733 chars. The script's ~9.7k / ~8.0k / ~5.7k ✓.
   - `services.py:246-256`, the `MessageTooLongError` docstring only: the embedder at 2048 tokens,
     a dense `MAX_TEXT_LEN` message can exceed it, the head is embedded, chunking is K-100 ✓.
2. **The literal-2048 pins go red on drift.**
   - I planted `EMBED_CONTEXT_LENGTH = 4096`: **9 failed**.
   - New: 2048 now equals the API's embedder misreport. A regression that trusted the API instead
     of `lms ps` would confirm itself in production. I planted `ctx = want` in place of the `ps`
     value: **7 failed**. The tests still tell the two apart, because the warn cases use `ps`
     contexts other than 2048 (8192/4096/1024).
3. **Unbounded sweep: two stale claims remain.** I grepped the whole repo for `4096` and `4,096`
   (excluding `.git`, `.venv` and `deprecated/`) and mapped every hit (Appendix). The two stale
   hits are P5-1.
4. **`AGENTS.md` is within the convention.** The row and the paragraph clause each swapped one
   token in place. The word count stays at 2,812, and no new line exceeds 700 chars (only lines
   87-88 do, and they predate this change).

**P5-1 (minor, verified by grep) — two documents still state 4096 claims.**
- `falkor-chat/docs/BACKLOG.md:100` (K-100): "a reply past ~4,096 tokens is stored whole … its
  vector covers only the head". At 2048, that threshold is ~2,048 tokens.
  - In the same file, K-099's "`MAX_TEXT_LEN` = 8,000 chars (~2,000–2,700 tokens)" (line 76) is
    contradicted by the measured URL-dense cut. 5,733 chars filled 2048 tokens, so 8,000 such
    chars is about 2,860 tokens.
  - Owner: `teco`.
- `falkor-chat/docs/test-reports/embedder-context-report.md:292-294`: "It is a measurement only:
  `EMBED_CONTEXT_LENGTH` stays 4096". This is present tense, and the report's own "Adopted value"
  line now contradicts it. Make it past tense ("stayed 4096 during this run"). Owner: `devops`.

**P5-2 (nit) — the script comment's "every message stored today" is wider than what was measured.**
`ensure_demo_models.py:95-96` says 2048 "fits every document chunk and every message stored
today". The measurement covered only `ws:acme` messages (max 1,817) and `ws:agent-team` chunks
(max 1,139). Say "every input measured 2026-10-31" instead.

## Appendix

**Pass 5 sweep map** (`grep -rn '4096\|4,096'`, unbounded; every hit file → owner or ruling).
- **Stale, P5-1:** `falkor-chat/docs/BACKLOG.md:100`; `embedder-context-report.md:292-294`.
- **Correct as history:**
  - the rest of `embedder-context-report.md` (the 4096 A/B arm and the restore tables);
  - `embedder-context-coordination.md` (ledger rows);
  - `BACKLOG.md:76,84-85` (the original estimate and the reversal statement);
  - this review.
- **Correct test fixtures:** `test_ensure_demo_models_script.py:110` (the stub's no-`-c` default
  stands in for LM Studio's own default), `:299-412` (chat-model-at-4096 refusal cases) and `:588`
  (the warn parametrization). `docs/reviews/demo-model-ensure.md:40` describes those cases.
- **A different 4096, no change:**
  - the FalkorDB 4096-byte UNIQUE limit: `schemas.py:101-104`, `services.py:1690`,
    `test_services.py:2614`, `unique-constraint-oversized-value-crash-rca.md`,
    `oversized-indexed-property-guard-*.md`, `claude/graph-dba/falkordb-quirks.md`;
  - chat-model contexts: `local-model-ram-budget-ml.md:215`, `opencode/docs/manuals/local-llm.md`;
  - embedding vectors and numerics: `*.embeddings.json`, `model-bench/**`, `capacity-report.md`,
    the `archive/` plans, `docs/plans/connected-code-graph-*`, `claude/docs/plans/coordination-graph*`,
    `docs/reviews/connected-code-graph-p0-impl.md`, `test_retry.py`, `extract-transient-retry.md`,
    `salesperson-cart-product-id-rca.md`.

**Pass 2 kill probe** (`scratchpad/t/test_probe_p2.py`, imports the shipped `rig` fixture). On the shipped
code it gives 2 passed. Results under each mutant:

| Mutant | `post_load_path_warns_on_resident_embedder_at_8192` | `manual_instructions_name_each_models_own_context` |
|---|---|---|
| MF (post-load check dropped) | **fails** | passes |
| MA (load-it-yourself line all 4096) | passes | **fails** |
| MB (app hint all 4096) | passes | **fails** |

Upstream llama.cpp's rejection behavior: llama.cpp server commit
[89fea80](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/89fea80d298184d1cd93564f48e060d9f541f4b4)
(`n_prompt_tokens > n_ubatch` → "input is too large to process"); see also
[LocalAI PR 6344](https://github.com/mudler/LocalAI/pull/6344.patch).

**F1 probe** (`scratchpad/probe.py`, run against the shipped code and under M4):

```
repo = FakeRepo(); repo.members.add("u1"); svc = make_service(repo)   # thread NOT added
svc.post_message(CTX, thread_id="missing", text="x" * (MAX_TEXT_LEN + 1))
original: raises MessageTooLongError     M4: raises ThreadNotFoundError
```
