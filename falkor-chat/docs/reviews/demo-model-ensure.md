# Review — demo model ensure-loaded startup step (K-066 item 1)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-066

## Scope & verdict

**Reviewed:** the uncommitted working tree for K-066 "What remains (1)" (`falkor-chat/docs/BACKLOG.md`
K-066), meaning the new `falkor-chat/scripts/ensure_demo_models.py`, `falkor-chat/scripts/ensure_demo_models.sh` and
`falkor-chat/server/tests/test_ensure_demo_models_script.py`, plus the diffs to `falkor-chat/scripts/start_demo.sh`,
`falkor-chat/AGENTS.md` and `salesperson/README.md`. The baseline is the K-066 spec and the derivation
owners it reads (`falkorchat/modelconfig.py` `ModelGateway`, `falkorchat/proof_defs.py` `SALESPERSON_DEF`,
`config/models.json`).

**Done:** ran the test file (**12 passed**, 6.9 s). Ran `bash -n` on both shell scripts (OK). Ran one mutation
probe of my own (copied the file aside, restored it, confirmed with `cmp`; see m1). Ran the `>700`-char
check on the edited context files.
**Not done, per the brief:** anything against LM Studio, FalkorDB or `:8200`. The implementer's live 3/3
and "10 mutants caught" are **inherited, not re-verified**. I did not check whether `ws:demo` holds any
override (that needs a FalkorDB read).

**Verdict: approve with suggestions.** No blockers or majors. The step matches the spec: it reads
first, refuses before any load, loads only what is missing with `-c 8192 --gpu max`, never unloads,
does nothing on a second run, and finds the CLI through configuration. The findings below are a
test gap on one claimed path, two scoping limits worth stating, and operator-doc polish.

CPG: considered, not relevant — the change is a standalone operator script plus shell wiring. Its two code dependencies (`ModelGateway.resolve`, `SALESPERSON_DEF`) were read directly. There are no call-graph or impact questions a CPG would answer better.

## Findings

### Minor

**m1. The post-load re-check (foreign model appeared / wrong context after load) is untested.** I
verified this with a mutant. `ensure_demo_models.py:271` `check_refusals(demo, after)` → `pass`
still gives 12/12 green. The docstring (step 5) and the `.sh` header both claim "stops unless …
nothing foreign appeared", but no test drives that path. The only post-load test
(`test_load_that_does_not_show_up_in_the_api_stops`) covers `still_missing`, not the refusal
re-check. This is the one path that guards the check-then-load window: `:8200` JIT-loading `qwen`
during a multi-minute `lms load`. **Fix:** add a test whose stub `lms` also marks `QWEN` loaded on
`load`. Assert exit 1, the qwen message on stderr, and no `unload` calls. Optionally add a second
case where the stub loads the chat model at `-c 4096`, regardless of the flag it was given.

**m2. A workspace `step`-kind override on `ws:demo` would silently defeat the step.** Workspace
overrides are a **hard cap** over `config.model` (`modelconfig.py` module docstring, precedence
"workspace override → requested → default"). Suppose `ws:demo` ever carries an `agentModelOverride`
(this module's `step` kind, `modelconfig.py:89-100`). The ensure step would then confirm ministral
while the storefront JIT-loads the override model, which is the ~63% regime, undetected. The
docstring's "no FalkorDB dependency" reason is weak *inside `start_demo.sh`*, because FalkorDB is
already up at step 1. On a cold box, though, `ws:demo` doesn't exist until step 4, so a read there
would find nothing anyway. Also, `ensure_demo_models.py:15-17` claims the embedding pin "equals the
overlay default it was pinned from". That is true only at pin time: `pin_workspace_embedding_model.sh`
never re-pins an existing override (its header, line 18). The storefront never embeds, so that half
is harmless. **Fix (the owner chooses):** either (a) state the limit plainly in the docstring and the
AGENTS row ("a `ws:demo` model override is not seen; none is set by any script"), or (b) when
`ws:demo` exists, read `repository.read_model_overrides('demo')` read-only and refuse if a `step`
override is set. (a) is proportionate today.

**m3. Every refusal advertises the `DEMO_ENSURE_MODELS=0` bypass, including the foreign-model one
the step exists to enforce.** In `start_demo.sh:234-235` (the new step-3 block), the ERROR footer
prints "or DEMO_ENSURE_MODELS=0 for a non-LM-Studio provider" whatever the cause. An operator under
demo pressure who sees "qwen is loaded" plus a one-variable escape will take the escape. The opt-out
itself is justified. Without it, `start_demo.sh` would hard-fail for any non-LM-Studio provider,
because `/api/v0` is LM-Studio-only (the `HTTPError` branch, `ensure_demo_models.py:150-155`). It is
also fail-safe: only the exact string `0` skips it, and the skip prints a warning. **Fix:** give the
"not an LM Studio server" case its own exit code (for example 3). Have `start_demo.sh` mention the
bypass only for that code. The Python `HTTPError` message already names the bypass at the point
where it applies.

**m4. On this dev box, the default `FALKORCHAT_OPENCODE_CONFIG` stops bring-up at step 3. The
operator docs show the bare command.** Failing loudly here is **correct**. Before this change, the
same default sent the *server* to the same unreachable URL, so every storefront turn would have
failed after a "successful" bring-up. The measured runs all passed
`FALKORCHAT_OPENCODE_CONFIG=~/.config/opencode/opencode.local.json` explicitly
(`falkor-chat/docs/test-reports/salesperson-single-model-report.md:43,130`). The issue is
discoverability. The README's "Running the demo" block (`salesperson/README.md`) still shows
`./scripts/start_demo.sh` alone. The new AGENTS paragraph says "fails loudly when that URL is
unreachable" but not how to fix it. The repo already has a precedent:
`falkor-chat/scripts/start_agent_team.sh:156-170` resolves env var → `opencode.local.json` → shared
file, with an INFO/WARNING line. **Fix:** in this change, add one sentence to the README run
section and to the unreachable-API message ("on a box whose shared opencode.json is stale, set
`FALKORCHAT_OPENCODE_CONFIG` to a local copy, e.g. `~/.config/opencode/opencode.local.json`").
Adopting `start_agent_team.sh`'s three-tier fallback in `start_demo.sh` too (it would also cover
`start_server.sh`) is a separate scope decision: file it as its own backlog item rather than
folding it in here.

**m5. The `loading` state counts as present without being confirmed.** `loaded_models`
(`ensure_demo_models.py:169-171`) keeps anything not `not-loaded`. An embedder caught mid-load
(for example, after a timed-out earlier run whose server-side load continued) passes as present,
and exit 0 is "ready" without the load ever completing. For the chat model, `loading` most likely
carries no `loaded_context_length`, so it refuses with the confusing "loaded with context None".
That is safe, but the message misleads. The rule does **not** open an unload/reload hole, since
nothing destructive keys on it. **Fix:** treat `state == "loading"` as its own refusal ("a demo
model is still loading — wait and re-run"), or poll briefly. I have not verified that the v0 API
ever emits `loading`. If it doesn't, document the rule as defensive and leave it.

### Nit

- **n1. The embedder-context claim reads stronger than its consequence warrants.** Skipping the
  check is right: the API field is unreliable for embedders, and `lms ps --json` would make the
  no-op path need the CLI, which breaks "both loaded = no CLI". But the docstring
  (`ensure_demo_models.py:24-28`) spends four lines on probe evidence and none on what the skip
  costs: an embedder loaded at a wrong context passes undetected, which is harmless to the
  storefront because it never embeds (K-066). Say that, and trim the probe detail to a cite.
- **n2. The new AGENTS row reads as a second copy of the script header.**
  `falkor-chat/AGENTS.md:87` is 679 chars, just under the ~700 smell. The file grew from 2,604 to
  2,750 words, already past ~2,500. The paragraph *was* replaced rather than appended (good). The
  row repeats the derivation and flags the script header already documents. Trimming it to
  "K-066 ensure step; never unloads; refuses on a foreign model or wrong ministral context;
  details in the script header" fits the context-file convention better.
- **n3. Malformed `data` items crash with a traceback.** A `data` element that isn't a dict
  raises `AttributeError` at `m.get` (`ensure_demo_models.py:171`) instead of producing a
  `Refusal`. This is cosmetic, since it still exits non-zero.
- **n4. Loading shows no progress.** `lms load` output is captured (`ensure_demo_models.py:253`),
  so an operator sees one "loading:" line and then up to 300 s of silence. Consider a
  "this can take a minute" hint.

## What's solid

- **The model set comes from its owners.** `SALESPERSON_DEF`'s agent-step `config.model` through
  `ModelGateway.resolve("step", requested=…)`, at the same precedence rung the executor uses, plus
  the overlay `embedding` default. The URL is resolved from the same `FALKORCHAT_OPENCODE_CONFIG`
  the server gets, so the checked set is the served set (subject to m2). The single-server and
  single-element-chain refusals are cheap, correct guards against a role-based fallback chain.
  `SALESPERSON_DEF`'s only transition guard is `kind: cmp`, so no hidden LLM `guard`-kind call
  needs a model outside the set.
- **Ordering is right.** All refusals run before any load (tested:
  `test_wrong_context_stops_even_when_the_other_model_is_missing`). `--identifier <id>` keeps the
  loaded instance's API id equal to the id the server calls, and turns a racing duplicate load
  into a loud failure instead of `<id>:2`. The re-read catches a CLI wired to a different
  LM Studio (tested).
- **The `start_demo.sh` wiring is sound under `set -euo pipefail`.** `if ! …; then` does not trip
  `-e`. Step 3 runs after the venv exists (step 2), which the wrapper needs, and before any seeding.
  The opt-out is fail-safe (only `"0"` skips). `bash -n` is clean.
- **The tests drive the real `.sh` → venv Python path** through `subprocess`, with a stub HTTP
  server and a stub `LMS_CLI`, and a minimal `PATH` so the box's Windows interop `lms.exe` can't
  leak in. That is the right altitude for an operator script. The `start_demo.sh` branch itself is
  untested, which is acceptable for a five-line `if`.
- **Docs edits replace rather than append.** The old "pin the demo's models / load both with
  `lms load`" paragraph was rewritten in place in `falkor-chat/AGENTS.md`, and so was the matching
  README status sentence. Both new table rows are under 700 chars.
- **CLI lookup** (`LMS_CLI` → `lms` → `lms.exe`) only runs when a load is needed. A set-but-bad
  `LMS_CLI` refuses instead of silently falling through to `PATH`.

## Open questions

1. **Instance ids in `/api/v0/models` (inherited, not verified).** The foreign-instance test
   (`ministral-half`) assumes `/api/v0/models` lists each *loaded instance* under its identifier.
   If the v0 API lists one entry per downloaded model key instead, a second ministral instance or
   a demo model loaded under a custom identifier would go unseen. That would not be a safety
   break (nothing is unloaded), but the "foreign instance" refusal would be inert. Worth one
   read-only look the next time LM Studio is free to probe (not this run, per the brief).
2. **m4 scope.** Should `start_demo.sh` (and `start_server.sh`) adopt `start_agent_team.sh`'s
   `opencode.local.json` fallback? That is a stakeholder/devops call, and I recommend a separate
   backlog item.

## Pass 2

> 2026-10-22 · re-review of `devops`'s revision. The test file now has 22 tests: **22 passed**
> (12.6 s), which I re-ran. Three mutants, each restored and checked with `cmp`. A synthetic
> `set -euo pipefail` probe of the `ensure_rc` idiom. LM Studio's v0 REST docs read for the
> `loading` premise. No LM Studio, FalkorDB or `:8200` access.

### Dispositions

- **m1: fixed.** Two post-load tests (`test_foreign_model_appearing_during_the_load_stops_after_it`, `test_chat_model_landing_at_wrong_context_stops_after_load`). The pass-1 mutant (`ensure_demo_models.py:294` `check_refusals(demo, after)` → `pass`) now fails 2 tests.
- **m2: fixed (option a, documented).** Docstring `ensure_demo_models.py:16-20`, `start_demo.sh:33`, AGENTS row `falkor-chat/AGENTS.md:87`. The "at pin time / never re-pinned" wording is now accurate.
- **m3: fixed, with one caveat (new P2-m2 below).** `NotLMStudio` has `exit_code = 3`. The bypass hint is printed only on `ensure_rc -eq 3` (`start_demo.sh` step 3). Mutant `NotLMStudio(` → `Refusal(` at `:165` fails 1 test. `test_enforcing_refusals_exit_1_and_never_offer_the_bypass` covers 4 refusal kinds.
- **m4: fixed.** The unreachable message names `opencode.local.json` (`:175-177`, tested by `test_unreachable_message_names_the_fix`), as do the README run block and `falkor-chat/AGENTS.md:117`. The three-tier fallback remains a separate scope call (Open question 2).
- **m5: fixed for the pre-load read** (mutant: dropping the `still loading` append at `:207` fails 1 test). Its interaction with m1 is new finding P2-m1.
- **n1: fixed** (`:28-33` states the cost of the skip). **n2: fixed** (row 679 → 378 chars; file 2,750 → 2,720 words). **n3: fixed** (`:183-185`, tested). **n4: fixed** (`:274`).
- **Open question 1 (instance ids in `/api/v0/models`): still open, lower stakes.** The new evidence is unsaved testimony: `--identifier probe-embed` was listed as `probe-embed` and the key stayed `not-loaded`. LM Studio's v0 endpoint doc doesn't describe identifiers either way. It is plausible, but unverified.

### New findings

**P2-m1 (minor): after a successful load, a model in `loading` state is refused, and nobody
knows whether the state ever appears.**

What is known:
- LM Studio's v0 REST reference (`lmstudio.ai/docs/developer/rest/endpoints`) shows only `"not-loaded"` in its example and never names `loading`. The value is **undocumented**.
- The implementer's pass-1 live load ran *before* the `loading` rule existed. At that time `loading` counted as present and passed, so that run is **no evidence** about what the API reports right after `lms load` returns.
- `lms load` blocking until the model is resident is inherited behaviour that nobody here has checked. The stub CLI assumes it.

What follows:
- If the state can appear, the step is **fail-safe**. Nothing is unloaded or served, the message ("still loading — wait and re-run") is accurate, and a re-run becomes a no-op once the load settles.
- The cost is a spurious failure in front of an operator at demo bring-up. It is not a wrong demo, so this stays minor.

**Fix:**
- In the post-load path only, re-read up to a bound (for example ≤30 s at 1 s intervals) while any demo model reports `loading`, then call `check_refusals`. Keep the pre-load refusal as it is: there, `loading` means someone else's load is in flight.
- Test it with a stub CLI that leaves the model `loading` for the first N API reads.

**P2-m2 (minor): every HTTP error status is treated as "not LM Studio", so a real LM Studio
answering 5xx gets the bypass advice.**
- `read_models` (`ensure_demo_models.py:164-169`) maps *any* `HTTPError` to `NotLMStudio` (exit 3). `start_demo.sh` then prints "DEMO_ENSURE_MODELS=0 skips this step". That is the m3 failure mode coming back through a side door: a sick LM Studio gets told to skip the guard.
- A non-LM-Studio OpenAI-compatible server answers `/api/v0/models` with 404 (or 405).
- **Fix:** raise `NotLMStudio` only for 404/405. Any other status should be a plain `Refusal` (exit 1, "LM Studio answered HTTP <code> — check its server log"). Add one parametrised test for 404 → 3 and 500 → 1.

**On the requested classification check:** a 200 whose JSON has no `data` list → exit 3 is
**right**. The server answered and isn't speaking LM Studio's v0 shape, which is exactly what the
bypass exists for.

**P2-n1 (nit): a 200 with a body that isn't JSON is reported as "unreachable".**
- `json.load` raises `ValueError` inside the `with` block, and `:170` catches it as "unreachable" (exit 1). An HTML catch-all server (a SPA host answering 200 on every path) is reachable and not LM Studio.
- **Fix:** catch `json.JSONDecodeError` separately inside the `with` block and raise `NotLMStudio`, consistent with the no-`data` case.

**P2-n2 (nit): no test drives the `start_demo.sh` exit-3 branch.**
- The tests assert the Python's own stderr, not the shell branch. I checked the `cmd || ensure_rc=$?` idiom under `set -euo pipefail` with a synthetic script:
  - rc 0 continues;
  - rc 1 prints the fix message and exits 1;
  - rc 3 prints the bypass hint and exits 1.
- The capture is correct. The branch is small enough that leaving it untested is acceptable.

**Fixes checked against each other.** There are three couplings:
- **m5 × m1:** this is P2-m1.
- **m3 × the HTTPError mapping:** this is P2-m2.
- **n3 × m5:** malformed entries are rejected in `read_models` before `check_refusals` reads `state`, so a malformed entry can't masquerade as `loading`. No conflict.

### Verdict (pass 2)

**Approve with suggestions.** Every pass-1 finding is fixed or documented as chosen. The two new
minors are both fail-safe (P2-m1) or advisory (P2-m2), with no path to an unload or to a mis-served
demo. P2-m2 is the one I would take before commit, because it reopens the exact door m3 closed.

## Pass 3

> 2026-10-22 · narrow re-review of `tdd-engineer`'s P2 fixes. Stopping rule: report only a new
> blocker or major (could unload, serve with the wrong models, or wrongly block a correct
> bring-up); everything else is filed. Test file: **28 passed** (16.7 s), re-run by me. I also
> probed a malformed `baseURL` offline: two invalid configs, no network.

### Dispositions

- **P2-m1: fixed.** `_settle_after_load` (`ensure_demo_models.py:272-284`) re-reads while a demo model is `loading`, bounded by `SETTLE_S`/`SETTLE_INTERVAL_S`. It runs only after a load (`:318`). The pre-load `check_refusals` at `:291` still refuses `loading` immediately.
- **P2-m2: fixed.** `NOT_LMSTUDIO_HTTP_CODES = {404, 405}` (`:91`). Any other status becomes a plain `Refusal` (exit 1), so a sick LM Studio no longer gets the bypass advice.
- **P2-n1: fixed.** A 200 with a non-JSON body raises `NotLMStudio` (`:176-180`), the same classification as a 200 without `data`.
- **P2-n2: unchanged** (a nit, as filed).

### The fixes checked against each other

- **The settle loop masks nothing.** It returns its *last* read unchanged. `still_missing` and the full `check_refusals` (foreign set and chat context) then run on that read (`:319-326`). A foreign model that appears during the wait is in that read. A foreign model that is itself `loading` doesn't extend the wait, but `loaded_models` counts `loading` as present, so it is still refused. A wrong context is checked once nothing is `loading`, and a model still `loading` at the deadline is refused. The loop's only effect is time, at most `SETTLE_S`.
- **Removing `ValueError` from the unreachable handler lets no new traceback through.**
  - `json.load`'s `ValueError` (including `UnicodeDecodeError`) is caught by the inner handler.
  - A read timeout during `json.load` is an `OSError`; it propagates to the outer `(URLError, OSError)` handler.
  - The other `ValueError` source, `urlopen` on a bad scheme, can't be reached. `ModelGateway` rejects a scheme-less or non-http `baseURL` first. I verified this by running `127.0.0.1:9/v1` and `htp://…`, which both refuse cleanly with exit 1 and "invalid baseURL".
  - The one remaining escape, `http.client.IncompleteRead`/`HTTPException` (not an `OSError`), was already uncaught before this revision. It exits non-zero, so `start_demo.sh` still stops. Filed below, not raised.

### Filed, not fixed (below the stopping rule)

- A non-numeric `ENSURE_DEMO_MODELS_SETTLE_*` crashes at import. This is a test-only knob, and the crash still exits non-zero.
- No test pins that the wait happens *after the load only*. If the wait were moved before the load, the pre-load `loading` test would still pass, just 30 s slower.
- An `http.client.HTTPException` mid-body escapes as a traceback (non-zero exit; it predates this pass).
- P2-n2: no test drives the exit-3 branch of `start_demo.sh`. The idiom was verified synthetically in pass 2.

### Verdict (pass 3)

**Approve.** There are no blockers or majors. None of the three fixes can unload a model, serve a
demo with the wrong models loaded, or wrongly block a correct bring-up, and they compose correctly
with the pre-load `loading` refusal.
