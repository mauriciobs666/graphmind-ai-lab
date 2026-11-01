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

## Do not report

- In a spec-update check, a spec that departs from the plan where it matches the shipped code.
  The plan is not the baseline.

## Evidence bar

- State whether each finding was verified (run, traced end to end, checked against docs) or
  inferred. A fix you suggest is a claim: run it, and check its reach (callers, inbound
  citations) before you propose it.
