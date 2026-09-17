# Plan-authoring techniques — on-demand

> **On-demand knowledge base for `architect`.** Situational plan-authoring/revision techniques —
> verification discipline for grep-based checks, completeness claims, plan-revision sweeps, and a
> handful of specific design-review traps — consulted when that specific situation arises during
> plan authoring or revision, not needed for every plan. General plan-writing doctrine lives in
> `architect.md`.
>
> Origin: extracted 2026-09-17 from `architect.md` as part of K-030 Stage 0 (interim knowledge-base
> relief) — a prompt restructure, not a kaizen distillation.

## Verify a hook gate by what it pattern-matches, not by its stated intent

When a plan step's verification depends on a `PreToolUse` hook firing, check what the hook
actually pattern-matches (the command text, not the intent) before treating the prompt as a gate —
a destructive operation wrapped inside a script can bypass a hook that only greps literal command
strings.

## Prose that disowns a value defeats a grep that hunts it

When a plan prescribes its own grep-based verification ("this name must appear with one
spelling"), write the explanatory prose to name the differing *segment or shape*, never to
re-quote the wrong literal — a grep cannot tell a disowning mention from a live one, so a revision
note that spells out what it is replacing makes the check unpassable.

## A completeness claim must be derived, not transcribed — and its check must be able to fail

An enumeration written by hand beside the invariant it instantiates drifts silently and survives
review, because a reviewer checks the list against the adjacent prose, not against the set it
claims to complement: write the derivation (a set difference over the owning section), not the
list. And an edit list pinned by one grep is a no-forgetting guarantee over the sites carrying that
token, never completeness — camelCase production and snake_case tests are disjoint vocabularies,
so name a second command for the sites the token never reaches; a residual asserted over a token
the edit does not retire is satisfied by construction.

## Compress by pointer only what nothing else cites literally

When revising a plan, "see the prior version, unchanged" is safe for rationale and discussion —
but content another section cites by reference (an exact Cypher block, an exact command) must
stay: compressed away, an isolated-context implementer has nothing concrete left to execute.

## Verify textual identity before prescribing one blanket find-and-replace

Before prescribing one blanket find-and-replace across N near-identical files or sections, verify
all N are actually textually identical first — a mix of past-tense and forward-looking/prescriptive
language means one substitution is wrong for some of them.

## Reconciling a diverging delegated note: rewrite in place, then sweep every downstream reference by grep, not recall

When a delegated `-graph.md`/`-ml.md` note diverges from a premise the plan left open pending live
verification, reconcile by rewriting the design section in place with a short revision note, then
sweep every downstream reference for the old premise — delegation table, stage file lists, AC test
table, risks section, and every other structurally-identical enumeration, including a routing or
authority table your revision note never reasoned through. A partial find-and-replace leaves the
plan internally inconsistent for the analyst gate. A step row has two independently binding
columns: sweeping the DONE-CONDITIONS is not enough, because the SCOPE column is the *build
instruction* — a trim that misses it leaves the removed contract still commissioned, often a few
lines above a done-condition that now contradicts it. So sweep by **grep, not by recall**: grep
every removed or renamed term across the whole plan and rule on each hit individually, rather than
editing the sites you reasoned about.

## A wrong mapping cell shared between the plan and a delegated note is fixed by deletion, not by splitting the row

When a plan and a `-ml`/`-graph` note both state the same mapping (mechanism → denominator, or any
other shared table) and a review finds one cell wrong, delete the plan's duplicate column and cite
the note instead of patching the one visibly-wrong row — a second home for one mapping is the
defect itself, and a row split transcribes the error rather than retiring it, leaving the same
latent mistake on every other row.

## A prior plan's Appendix/skeleton description of shipped code is a claim, not a fact

A plan can be internally consistent — a later section agrees with its own Appendix — while the
code it describes never caught up with an earlier ruling. Before specifying downstream work that
depends on a claimed shape, diff the actual module source directly; checking the plan's own
sections against each other proves nothing about whether the code matches either of them.

## A plan row's own word count is not a proxy for how specified that unit is

A short row that says "stated there, deliberately not restated here" and cites a design-doc
subsection can carry more real weight than a long row that restates prose already covered
elsewhere. Judge completeness by resolving the citation, never by the row's own length.

## A bypassable permission/allow-list design needs an architectural fix, not a tighter pattern

When a permission/allow-list design is found bypassable via pattern-matching, prefer an
architectural fix over a tighter pattern: move argument/command construction for the
smuggling-prone class into a code-reviewed wrapper keyed by an enum/allow-list lookup, rather than
tightening the glob/regex that was bypassed — a tighter pattern is still the same matcher class and
is a floor (useful as defense-in-depth), not the fix that closes the hole.

## A component's own stated hard constraint applies to every proposal in that component, no exceptions

A component's own stated hard constraint (e.g. "zero runtime dependencies") applies to every
proposal in that component, including a human-invoked one-off never on a run path — an absolute
rule admits only its own named reversal trigger, not a scope carve-out invented because the new
use looks like an edge case.

## A shared-file-ownership finding is a claim to verify, not a grant to transcribe

Before amending a plan's ownership rows off a review's shared-file-collision list, trace the actual
gating mechanism (the router's own top-level route list, not the review's enumeration) — a named
unit may route through a different, not-yet-built mechanism entirely (a bottom sheet, not a
route), which changes the right fix per unit rather than licensing one uniform grant. And when the
fix is real, sweep every row that touches the file, not just the rows the triggering review named
— a prior fix already missed an identical-shape row this way, because that row's own text had
already committed the file without anyone re-checking it against the ownership map.

## Specify how to pin a deliberately-open design decision, not just that it is open

When a plan defers a design choice to a later stage or another agent, prescribe encoding the
deferral in the implementation itself: name the mechanism (exception/class) up front, state in its
own docstring exactly what is decided versus still open and to whom the decision is owed, and add
one test whose docstring says plainly that it is the test that changes once the decision lands.
Leaving the deferral as plan prose alone lets an implementer resolve it silently either way, with
nothing to flag that it happened.
