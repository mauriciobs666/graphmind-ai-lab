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

## Evidence bar

- A reported suite count says whether the tests that wipe `reference` ran (`wf_repo` tests,
  `./scripts/test_queries.sh`; `falkor-chat/AGENTS.md`, "Key scripts"). When shared state must
  survive, run the review-safe subset (`falkor-chat/AGENTS.md`, "Probing shared graph state
  without mutating it") and name what it left unrun. A run that wipes `reference` is reported
  with its re-seed sequence.
