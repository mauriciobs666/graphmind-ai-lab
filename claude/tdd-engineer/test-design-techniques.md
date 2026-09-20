# Test-design techniques — on-demand

> **On-demand knowledge base for `tdd-engineer`.** Situational test-design techniques — proving an
> assertion against a mutant beyond "it went red," CLI/marker wiring gaps a function-level test
> can't see, and fixture-shape traps — consulted when that specific situation arises, not part of
> the everyday red→green→refactor loop. Guards over other code's *text* (AST readers, lint rules)
> are a different problem and live in `guard-testing-techniques.md`; fixture design for a
> code-**computed** value lives in `estimator-test-fixtures.md`; ordinary unit practice lives in
> `tdd-engineer.md`.
>
> Origin: extracted 2026-09-17 from `tdd-engineer.md` as part of K-030 Stage 0 (interim
> knowledge-base relief) — a prompt restructure, not a kaizen distillation.

## A test passing for the right reason still needs proof against a coincidental pass

Failing for the right reason shows the test *can* fail, not that it can reject a *wrong*
implementation. Wherever a coincidence could satisfy the assertion — any same-type exception, when
an earlier-order check raises it too; any substring of a generated string; any ordering of
positionally-paired columns — break the exact code the test pins and confirm it goes red. Assert
the raiser's own message or the exact full string, never a token some other path could also
produce.

## A surviving mutant on a duplicated string is not always a weak test — mutate the drift, not the copy

Restoring an identical copy over a duplicated string is equivalent by construction, so a mutant
that just re-duplicates the string proves nothing: mutate the *drift* instead — edit the canonical
copy, leave the duplicate stale. A guard whose downstream twin raises the same sentence is
separable only by *ordering*, never by `match=`, so pick an input the later layer rejects
differently.

## Never defer a mutation as needing a lucky input — use a degenerate one

A degenerate input — an all-identical sample, a single-element collection — usually puts the
boundary there by construction, deterministically, so the mutant dies at one fixture instead of
never being tried.

## An identity assertion can't pin a serialization decision — assert on the serialized form itself

An **identity** assertion (`from_dict(to_dict(x)) == x`) is invariant under any change *both*
halves share, so it cannot pin a serialization decision — omit-vs-null, key name, ordering —
however many records it round-trips. Assert on the serialized form itself instead
(`assert "k" not in to_dict(x)`).

## A spec-mandated CLI mode can be fully unit-tested while its argparse wiring is silently skipped

Testing `run_check_tables_shape(...)` or an equivalent handler as a plain function proves the logic
works; it proves nothing about whether `_build_parser()`/`main()` ever declared the flag that
reaches it. When a spec names a literal command line as a test step, drive that exact command line
at least once — a function-level test suite passing 100% is not evidence the CLI surface it's meant
to expose actually works.

## Gate optional/slow tests with the runner's tag or marker mechanism, not a bare reachability check

When a test's dependency is normally present in the run environment (a live LLM endpoint, a local
service), an `if not reachable(): skip` guard won't opt it out of the default run — it fires
silently and the "fast, network-free" suite quietly starts making real calls. Register a
marker/tag and exclude it by default; keep a reachability check *inside* the test too, but only as
the "don't fail for environmental reasons" net, not as the opt-out gate.

## A fixture uniform on the rule's own anchor dimension proves nothing about the rule as documented

When the rule under test is *positional* — anchored to a line/string boundary, or to one position
in a list/collection ("the last accepted item", "the first match") — vary that exact position
across the corpus (first, middle, last) and check what the *consumer* does with the whole result,
not just the one element the rule binds. That dimension is not always a position: a collection
whose *length* a helper derives a constant from, and a sample whose *spread* a transform scales,
are the same trap, and a uniform fixture hides the transform completely. Verify the claim actually
matches what the code enforces.

## When merging two callers onto one shared helper, compare what each caller does with the result

A consumer that re-validates the result before acting and a consumer that acts on it directly do
not share a tolerance/safety contract, however identical their inputs appear — an overly permissive
shared parser can silently widen the *acting* consumer's exposure. Test each caller from its own
risk profile, not the shared implementation's.

## A hand-built "rejected alternative" mutant can reconstruct a different bug than the one being proven load-bearing

Refines `tdd-engineer.md`'s own "when a plan explicitly rejected an alternative, the mutant is that
alternative" rule at the construction step: getting the reconstruction subtly wrong doesn't fail
loud — it silently builds a *third* thing, and the correctness suite reacts to that instead. Worked
case: a rejected design was folding two
ingestor-resolution branches into one shared query text carrying a possibly-`NULL` parameter —
rejected for a query-plan regression, not a correctness one (the design note proves it with a live
`EXPLAIN`: `Node By Index Scan` when the parameter carries a value, `Node By Label Scan` + `Filter`
when it's `NULL`; `falkor-chat/docs/plans/agent-team-ingestion-graph.md` §2.2/§2.3). Hand-reconstructing
that single-query shape by gating the merged `CASE`/`WITH` on the `OPTIONAL MATCH`-bound variable
(`pa IS NOT NULL`) instead of on the parameter itself (`$producedBy IS NOT NULL`) looks like the same
mutant but silently reintroduces a different, unrelated bug — an actor-fallback regression — and
reddens several correctness tests instead of leaving correctness alone. That isn't evidence the
design decision is load-bearing; it's evidence the mutant is wrong. Diagnose which mutant you
actually built by the mechanism the design note itself used to justify the rejection, not by which
tests go red: verify the *intended* regression directly (here, a live `EXPLAIN` on the mutated query
showing the plan degradation) with the full correctness suite still green. A mutant that reddens
correctness tests instead needs fixing before it proves anything.
