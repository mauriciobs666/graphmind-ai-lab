# falkor-chat — review standing instructions

## What counts as serious

- Text the model reads (a system prompt, a tool description, a tool result, a hold or error
  message) is behavior. A change to it that invites a fabricated value, a bypass of a hold, or a
  wrong reply language is at least major unless a measurement covers it.

## Always check

- A change to a published workflow def's create-only fields (`config.tools`, `systemPrompt`,
  `config.model`) cannot be published under an old or uncommitted version name by any step in
  the order the plan runs them: a seed, `start_demo.sh`, or a re-seed after a default `pytest`
  wipes `reference`. Per `falkor-chat/AGENTS.md`, "Three traps when bumping `SALESPERSON_DEF`".
  For `salesperson` and `order-fulfillment`, `./scripts/verify_salesperson.sh` (check 6, also
  `start_demo.sh`'s preflight) shows such a publish only after the version is burned: in a plan
  review check the order; in an implementation review, a green run after the last seed is the
  evidence.
- A measured comparison (an eval, a live run, a re-run against a baseline) pins its baseline to
  an explicit commit sha, and no step of the run overwrites the file it compares against.
- In a spec check of an MCP tool's input contract, read the generated `inputSchema`
  (FastMCP `list_tools()`) and drive one violating `call_tool`, rather than reading the signature or
  docstring. A claim that an argument is untyped or unbounded, or that a bad item becomes its own
  receipt rather than failing the call, must hold in the schema and in the result.
- A change to how a model is loaded (context length, offload, quantization) in
  `scripts/` is checked for its effect on spec'd behavior (`DESIGN.md` §1.3, e.g. what an embed
  or a turn can cover), not only for the script's own contract.
- A `test_queries.sh` assertion runs a hand copy of a repository query, so it pins the copy, not the method. A contract claimed for the method needs a pytest over the real method that a mutant of the method turns red. A check that a copy matches the method reads the copy's value as bash evaluates it, from exactly one definition, never by parsing the source.
- A property added to a query that has alternative literal prefixes or branches (`$ingestedBy` / `$producedBy`) is asserted as written on every branch, not only on the default one.
- A change to an assertion helper in `scripts/test_queries.sh` re-runs `docs/reviews/test-queries-sigpipe-evidence/probe.sh` against the changed script, and the review reports its `TOTAL wrong verdicts` (0 required) together with one planted helper mutant that makes it non-zero.
- A value a compare-and-set checks (a version, a status, a key) is pinned by a test that injects a concurrent writer between the read the value came from and the CAS, ahead of any later read the code could substitute. A test that injects at the CAS itself cannot tell "carried from the read" from "re-read just before the CAS".
- A guard that proves a name is read (or unread) is checked with a planted comment-only and a
  string-only mention of the name. A regex or grep over source text counts prose: `config.X`
  is cited in docstrings across `falkorchat/`. A text match is a bound; count AST loads.

## Evidence bar

- A reported suite count says whether the tests that wipe `reference` ran (`wf_repo` tests,
  `./scripts/test_queries.sh`; `falkor-chat/AGENTS.md`, "Key scripts"). When shared state must
  survive, run the review-safe subset (`falkor-chat/AGENTS.md`, "Probing shared graph state
  without mutating it") and name what it left unrun. A run that wipes `reference` is reported
  with its re-seed sequence.
