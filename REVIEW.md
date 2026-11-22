# Review standing instructions

## What counts as serious

- In a spec-update check, a contract the shipped diff changed that the live spec omits or
  contradicts is a blocker.

## Always check

- **Growth-stops check.** On a revision of a document that root `AGENTS.md`'s growth-stops bullet
  covers, say for any growth whether it is new scope or rework, and report rework growth,
  restated history and restated facts as findings. There is no numeric threshold.
- A plan for a component with a live spec states only the change against it, citing the affected
  spec sections. Report restated spec content as a finding. Where the plan depends on behavior
  that an open (not yet closed) feature changes, it cites that feature's plan; a missing citation
  is a finding.
- A spec update is checked against the shipped code (the diff from the feature's baseline sha),
  not against the plan. It also states no change history and keeps its section numbering (root
  `AGENTS.md`, live-spec bullet).
- Every check the work relies on (a test, a pin, a guard, a gate's acceptance command, a live
  probe) can both fail and pass. Name the mutant, planted input or positive control that turns it
  red. Its expected values are never read from the constant or function it checks. A check that
  cannot fail, or cannot pass, is a finding.
- A "no X remains" check is an unfiltered sweep, and every hit is mapped to an owner: the unit
  that edits it, or an explicit no-change ruling. A filtered or path-limited sweep, or a pattern
  that matches only the already-known sites, is a bound, not proof. Run a prescribed check
  command verbatim, from the working directory it states.
- A narrowing based on "did not reproduce" re-runs the entry's original invocation, with the same
  parameters (for a tool behaviour: the same flags, `offset`/`limit` included), and records which
  form was run. A negative from a different call form is a bound, not a refutation.
- In a spec-update check, read the lines the update left unchanged too. A contradiction
  carried over from the previous text is a finding, the same as a new one.
- In a live-spec check, a fact stated in two sections of the spec set is a finding, and every
  copy is checked against the code, not only the one the change edited.
- A whole-component spec check (a catch-up or a first spec) builds its inventory from every
  shipped file, scripts and operator tooling included. It never builds it from a git-log window:
  a document edited after a feature landed can still omit that feature.
- Whether a finding should become a standing rule. If it should, propose it in your review as a
  `### Proposed REVIEW.md lines` block naming the target file and heading. Each line is an
  addition or a prune, and the review's prose states its source: the finding ID or, for a
  recurring class, the review paths and finding IDs you read. Propose a line only if it changes
  what a reviewer does. A rule that a test, CI job or hook already enforces is not added, and a
  prune names the test, CI job or hook that now enforces it. With an addition, also propose
  pruning any line in that file that is now enforced elsewhere.
- A guard that enumerates a schema or tree (OpenAPI, JSON Schema, AST) to find "every X" is checked
  with one planted X for each composition axis the format allows (nested object, array element,
  map value, `$ref`, `anyOf`/`allOf`), not only the shapes in use today. A walker that recurses
  into only some axes is a bound, not proof.
- A method or plan that runs LLM work as Claude Code subagents names what its transcript audit checks beyond tool calls: the spawn prompt (against a hashed template), `agentType`, and the injected `instructions`/`gitStatus` context. A tool-call-only audit is a finding.
- When a unit's code validates the shape of a record field that a later unit produces, the plan's record row states the shape, and the producer's done-condition feeds its output through that validator (a round trip over every role and call kind). A validator the producer is never run against fails only at the end of a live run.
- For a record another unit consumes (a ledger event, a dataset row, a key binding), mutate each field's value at its write site. A field whose value no test pins, with the expected value computed independently, is a finding.
- For code that stops a process tree (a group kill, a grace period, a timeout), plant a descendant that survives the first signal and assert it is gone afterwards; assert the grace by elapsed time. Run timing-dependent tests under CPU oversubscription before calling them stable.
- A plan step that relies on agent-harness behaviour (frontmatter fields, hook firing, transcript fields, CLI flags, tool size limits) cites a live probe or transcript for each premise. The probe runs under the production invocation's exact flags and environment. Each premise has a named pass/fail route, and a worst-case-sized smoke that goes red if the premise is false.
- A control command that shares a single-flight lock with the long-running process it controls (stop, pause, signal) is checked for reachability while that process holds the lock.
- In a plan for unattended operation, every refusal or halt the plan adds is checked against what an ordinary user action on the same machine changes (config edits, logins, other sessions). Each refusal must also show up in a status command, not only in a run's exit line.
- A fail-closed path is traced to its consumer. The guard's own failure must reach a class that halts and is distinguishable from a policy decision. A fail-closed default that the consumer counts as the actor's fault turns a broken guard into silent attrition.
- A row that keys on a missing record (no hook entry, no result, no field) is checked against every producer of that absence: a kill, batched or parallel writes, and validation before the recorder runs. Checking only the producer that motivated the row is not enough.
- In a live-spec check, each absolute word in the spec (*only*, *never*, *always*, *every*, *exactly one*, *the one*) is verified by an unfiltered grep or enumeration over the whole component, including modules added after the sentence was first written. A claim checked only against the module it was written about is a bound, not proof.
- A surviving mutant called behaviour-equivalent is checked against concurrency as well as the serial suite. If it widens a condition to a state that another actor can write (a non-terminal status, an unlocked row), plant a concurrent write in that window before accepting the equivalence.
- When two programs read the same config file (a shell parser and compose's dotenv, a script and a loader), probe each grammar axis either reader treats specially (quoting, inline comments, `$` expansion, `export`, repeated keys, CRLF, trailing delimiters, edge whitespace that one reader trims) and require that each one either refuses or renders identically in both. Parity on the shipped file alone is a bound, and so is a reviewer-proposed character class until it has been run through the same probe.
- A procedure that waits on a status flag before an irreversible step names the state in which the flag already reads "done" before the work has run (scheduled, queued, never-run defaults). It requires a counter or timestamp that proves the work ran.
- A plan step that restarts a shared stateful service has five parts. It verifies before the stop that the state was saved. It copies the saved state out. It records an inventory of every graph or key before the stop and compares it after the restart. It names a rollback that cannot persist the wrong state over the copy. Its rehearsal covers the branch where the restart succeeds with wrong data, not only the branch where it fails to start. A verify step that checks one workspace is a bound, not proof.
- A fallback written as `<command> || <fallback>` for a peer that may not answer is checked against a hang, not only an error. The probe freezes the peer (for example `docker pause`) and asserts the fallback runs within a stated time.
- A parser of free-text documents that are revised in place is run over the whole real corpus and compared with an independent probe (last statement in document order; a regex recount), differences listed per file; expected values in its corpus test come from that probe, not from the function's own output.
- A one-time rewriter of real documents is checked by a word-multiset diff, old minus new, over every real input; each lost word is mapped to an authorising decision. Planted cells whose shape the rewriter does not recognise (a bold-wrapped value, a parenthetical) must survive verbatim.
- A replace/retry flow that relies on 'new first, old removed once ready' is checked against what the *write itself* does to the old record (auto-supersede, uniqueness, cascade), and against every failure outcome of the new one; the assertion is that the visible-to-readers count never drops below its starting value.
- A shell/awk/glob check is tested with a positive control (an input that must pass) as well as a negative one; a `case` pattern over a joined string is run on the exact joined value.
- A status word that reports from static configuration (`registered`, `enabled`, `active`) is checked against every setting that overrides it in the consuming tool (a global off switch, a handler `type`/`if` narrowing, scope precedence), one planted settings file each; and its plumbing argument is mutated at every call site, not only the one the live test drives.
- A live spec's description of a validator or guard ('asserts the property sets of §7') is checked by feeding it a record that carries every optional property the writer can emit, not only the fields present in the real corpus; a pass on the real tree is a bound.
- A test double that gains a read-only entry point (`ro_query`) rejects write statements through it; a passthrough to the write-capable method is a check that cannot fail.
- When a plan deletes a test helper, check every caller of that helper, including calls made through other helpers, and check it hardest for the tests the plan keeps as guards.
- When a plan drops a named element, build its close list from an unfiltered grep of the element's name **and** its prose synonyms across the live spec and docstrings. A grep for the field name alone is a bound.
- When a removal drops several independent elements (a clause and the column it feeds, a field and its stamp), mutate each element alone and name the check each one reddens. A mutant that restores them together proves only that some check catches the pair.

## Do not report

- In a spec-update check, a spec that departs from the plan where it matches the shipped code.
  The plan is not the baseline.

## Evidence bar

- State whether each finding was verified (run, traced end to end, checked against docs) or
  inferred. A fix you suggest is a claim: run it, and check its reach (callers, inbound
  citations) before you propose it.
